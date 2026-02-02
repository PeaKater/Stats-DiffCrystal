import time
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from pathlib import Path
import os
import json
import copy
import argparse
import random
from common.model_utils import get_model
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_ddp(rank, world_size, master_addr='127.0.0.1', master_port='29512'):
    os.environ.setdefault('MASTER_ADDR', master_addr)
    os.environ.setdefault('MASTER_PORT', master_port)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def set_dataloader_epoch(model, epoch):
    # 给分布式采样器设置epoch，保证shuffle不一致
    if hasattr(model, 'train_dataloader') and hasattr(model.train_dataloader, 'sampler'):
        sampler = model.train_dataloader.sampler
        if hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)
    if hasattr(model, 'val_dataloader'):
        vd = model.val_dataloader
        if hasattr(vd, '__iter__'):  # 可能是list
            for dl in vd if isinstance(vd, list) else [vd]:
                if hasattr(dl, 'sampler') and hasattr(dl.sampler, 'set_epoch'):
                    dl.sampler.set_epoch(epoch)

def ddp_worker(rank, world_size, cfg):
    setup_ddp(rank, world_size)
    try:
        if cfg.train.deterministic:
            torch.manual_seed(cfg.train.random_seed + rank)
            torch.cuda.manual_seed(cfg.train.random_seed + rank)
            torch.cuda.manual_seed_all(cfg.train.random_seed + rank)
            np.random.seed(cfg.train.random_seed + rank)
            random.seed(cfg.train.random_seed + rank)    

        os.makedirs(cfg.output_dir, exist_ok=True)

        model = get_model(cfg)
        model.is_master = (rank == 0)
        model.init(expname=cfg.expname, test_wb=False)
        #torch.cuda.empty_cache()
        model.train_start()

        ddp_model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

        for e in range(model.current_epoch, cfg.train.pl_trainer.max_epochs):
            tick = time.time()
            model.train()
            model.train_epoch_start(e)
            set_dataloader_epoch(model, e)

            for batch_idx, batch in enumerate(model.train_dataloader):
                batch = batch.to(model.device)
                teacher_forcing = (model.current_epoch <= model.hparams.teacher_forcing_max_epoch)

                # 前向：必须走 ddp_model(...) 才能触发梯度同步
                outputs = ddp_model(batch, teacher_forcing, training=True)

                # 计算损失与记录，使用原模型的方法
                log_dict, loss = model.compute_stats(batch, outputs, prefix='train')
                model.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, prefix='train')

                model.optimizer.zero_grad()
                loss.backward()
                model.clip_grad_value_()
                model.optimizer.step()
                model.train_step_end(e)

            # 训练epoch结束：聚合日志并仅master写出
            model.train_epoch_end(e, test_wb=False)

            if e % cfg.logging.check_val_every_n_epoch == 0:
                #torch.cuda.empty_cache()
                model.eval()
                model.val_epoch_start(e)

                outs = []
                with torch.no_grad():
                    for val_batch_idx, val_batch in enumerate(model.val_dataloader):
                        val_batch = val_batch.to(model.device)
                        outputs = ddp_model(val_batch, teacher_forcing=False, training=False)
                        log_dict, val_loss = model.compute_stats(val_batch, outputs, prefix='val')
                        model.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True, prefix='val')
                        outs.append(val_loss.detach())

                        model.val_step_end(e)

                # 验证日志聚合 + 仅master写出
                model.val_epoch_end(e, test_wb=False)

                # 全局平均 val loss 给 scheduler（所有rank都用同一个值 step，保持同步）
                if cfg.optim.use_lr_scheduler and len(outs) > 0:
                    local_val = torch.mean(torch.stack(outs)).to(model.device)
                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(local_val, op=dist.ReduceOp.SUM)
                        local_val = local_val / dist.get_world_size()
                    if cfg.optim.lr_scheduler._target_=='ReduceLROnPlateau':
                        model.scheduler.step(local_val)
                    elif cfg.optim.lr_scheduler._target_=='CosineAnnealingWarmRestarts':
                        model.scheduler.step(e)

            model.train_val_epoch_end(e)
            if model.is_master:
                print(f"\tTraining time: {time.time() - tick} s")

            # 仅master判断早停，然后广播给所有rank
            stop_flag = torch.tensor(0, device=model.device)
            if model.is_master and model.early_stopping(e):
                stop_flag = torch.tensor(1, device=model.device)
            if dist.is_available() and dist.is_initialized():
                dist.broadcast(stop_flag, src=0)
            if stop_flag.item() == 1:
                break

        model.train_end(e)

    except KeyboardInterrupt:
        if rank == 0:
            print("Training interrupted. Cleaning up...")
        torch.cuda.empty_cache()
    finally:
        cleanup_ddp()

def main(cfg):
    # 单机多卡：根据可见GPU数量决定是否DDP
    world_size = torch.cuda.device_count()
    if world_size >= 2:
        mp.spawn(ddp_worker, args=(world_size, cfg), nprocs=world_size, join=True)
    else:
        # 单卡原始流程
        if cfg.train.deterministic:
            torch.manual_seed(cfg.train.random_seed)
            torch.cuda.manual_seed(cfg.train.random_seed)
            torch.cuda.manual_seed_all(cfg.train.random_seed)
            np.random.seed(cfg.train.random_seed)
            random.seed(cfg.train.random_seed)    
        
        os.makedirs(cfg.output_dir, exist_ok=True)
        model = get_model(cfg)
        model.is_master = True
        model.init(expname=cfg.expname, test_wb=False)
        torch.cuda.empty_cache()
        model.train_start()
        try:
            for e in range(model.current_epoch, cfg.train.pl_trainer.max_epochs):
                tick = time.time()
                model.train()
                model.train_epoch_start(e)
                for batch_idx, batch in enumerate(model.train_dataloader):
                    loss = model.training_step(batch.to(model.device), batch_idx)
                    model.optimizer.zero_grad()
                    loss.backward()
                    model.clip_grad_value_()
                    model.optimizer.step()
                    model.train_step_end(e)
                model.train_epoch_end(e, test_wb=False)

                if e % cfg.logging.check_val_every_n_epoch == 0:
                    torch.cuda.empty_cache()
                    model.eval()
                    model.val_epoch_start(e)

                    with torch.no_grad():
                        outs = []
                        for val_batch_idx, val_batch in enumerate(model.val_dataloader):
                            val_out = model.validation_step(val_batch.to(model.device), val_batch_idx)
                            outs.append(val_out.detach())
                            model.val_step_end(e)
                    model.val_epoch_end(e, test_wb=False)

                    if cfg.optim.use_lr_scheduler and cfg.optim.lr_scheduler._target_=='ReduceLROnPlateau':
                        model.scheduler.step(torch.mean(torch.stack([x for x in outs])))
                    elif cfg.optim.use_lr_scheduler and cfg.optim.lr_scheduler._target_=='CosineAnnealingWarmRestarts':
                        model.scheduler.step(e)
                
                model.train_val_epoch_end(e)
                print(f"\tTraining time: {time.time() - tick} s")
                if e % 10 == 0:
                    pass
                if model.early_stopping(e):
                    break
        except KeyboardInterrupt:
            print("Training interrupted. Cleaning up...")
            torch.cuda.empty_cache()
        finally:    
            model.train_end(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--expname', type=str, default='e3gnn')
    parser.add_argument('--predict_property', default=False)
    parser.add_argument('--predict_property_class', default=False)
    parser.add_argument('--early_stop', type=int, default=800)
    args = parser.parse_args()
    OmegaConf.clear_resolvers()
    OmegaConf.register_new_resolver("now", lambda x: time.strftime(x))
    cfg = OmegaConf.load(args.config_path)
    cfg.output_dir = args.output_path
    cfg.expname = args.expname
    if args.predict_property is not None:
        cfg.model.predict_property = args.predict_property
    if args.predict_property_class is not None:
        cfg.model.predict_property_class = args.predict_property_class      
    cfg.data = OmegaConf.load("./conf/data/"+cfg.data+".yaml")
    cfg = OmegaConf.create(OmegaConf.to_container(OmegaConf.create(OmegaConf.to_yaml(cfg)), resolve=True))
    cfg.data.early_stopping_patience_epoch = args.early_stop
    main(cfg)