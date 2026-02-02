from typing import Any, Dict

import numpy as np
import math
import copy
import json
import os
import glob
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter
from tqdm import tqdm

from common.datamodule import CrystDataModule
from common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume,
    frac_to_cart_coords, min_distance_sqr_pbc, bound_frac, 
    lattice_params_to_matrix_torch, lattice_params_from_matrix)

from gnn.embeddings import MAX_ATOMIC_NUM, KHOT_EMBEDDINGS
import torch.distributed as dist



class BaseModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.hparams = cfg.model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hparams.data = cfg.data
        self.current_epoch = 0 
        self.logs = {'train':[], 'val':[], 'test':[]}
        self.train_checkpoint_path = None
        self.val_checkpoint_path = None
        self.min_val_loss = float('inf')
        self.min_val_epoch = self.hparams.mul_cost_epoch
        self.train_log = None
        self.val_log = None
        self.test_log = None
        self.model_name = "model"
        self.datamodule = None
        self.is_master = True
    # 初始化优化器 调度器 数据模块    
    def init(self, load=False, test=False, expname='Stats-DiffCrystal', sample=False, test_wb=False):
        self.init_optimizer()
        self.init_scheduler()
        self.to(self.device)

        checkpoint = self.load_checkpoint(load)

        if checkpoint==False:
            #print("No checkpoint: Save hparams.yaml")
            with open(self.cfg.output_dir + "/hparams.yaml", "w") as f:
                f.write(OmegaConf.to_yaml(cfg=self.cfg))

        self.init_datamodule(load)

        self.init_dataloader(load, test)

    def init_optimizer(self):
        opt_name = self.cfg.optim.optimizer.get('_target_', 'AdamW')
        opt_kwargs = {k: v for k, v in self.cfg.optim.optimizer.items() if k != '_target_'}
        name_lc = str(opt_name).lower()
        if name_lc == 'adamw':
            self.optimizer = torch.optim.AdamW(self.parameters(), **opt_kwargs)
        elif name_lc == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters(), **opt_kwargs)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")
    def init_scheduler(self):
        if self.cfg.optim.lr_scheduler._target_=='ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **{k:v for k, v in self.cfg.optim.lr_scheduler.items() if k!='_target_'})
        elif self.cfg.optim.lr_scheduler._target_=='CosineAnnealingWarmRestarts':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, **{k:v for k, v in self.cfg.optim.lr_scheduler.items() if k!='_target_'})
        
        
    def init_datamodule(self, load):
        if self.cfg.data.datamodule._target_=='CrystDataModule':
            #print(f"Set up datamodule")
            self.datamodule = CrystDataModule(**{k:v for k, v in self.cfg.data.datamodule.items() if k!='_target_'})

            if load!=True:
                #print(">>> save scalers")
                self.lattice_scaler = self.datamodule.lattice_scaler.copy()
                self.param_decoder.lattice_scaler = self.datamodule.lattice_scaler.copy()
                self.scaler = self.datamodule.scaler.copy()
                torch.save(self.datamodule.lattice_scaler, self.cfg.output_dir  + '/lattice_scaler.pt')
                torch.save(self.datamodule.scaler, self.cfg.output_dir  + '/prop_scaler.pt')

        if load==True:
            #print(">>> Load scalers")
            self.lattice_scaler = torch.load(self.cfg.output_dir  + '/lattice_scaler.pt')
            self.scaler = torch.load(self.cfg.output_dir  + '/prop_scaler.pt')   
     

    def init_dataloader(self, load, test):
        if self.datamodule is not None:
            if load:
                if test:
                    self.datamodule.setup('test')
                    self.test_dataloader = self.datamodule.test_dataloader()[0]
                else:
                    self.datamodule.setup()
                    self.test_dataloader = self.datamodule.val_dataloader()[0]
            else:
                self.datamodule.setup("fit")
                self.train_dataloader = self.datamodule.train_dataloader()
                self.val_dataloader = self.datamodule.val_dataloader()[0]
                self.test_dataloader = None

    def train_start(self):
        if self.is_master:
            print(">>>TRAINING START<<<")
        pass

    def train_end(self, e):
        if self.is_master:
            print(">>>TRAINING END<<<")
        self.logging(e)
        if self.is_master:
            self.train_checkpoint_path = self.save_checkpoint(model_checkpoint_path=self.train_checkpoint_path, suffix="train")     

 # 梯度裁剪
    def clip_grad_value_(self):
        #torch.nn.utils.clip_grad_value_(self.parameters(), clip_value=0.5)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

    def train_step_end(self, e):
        # for examination
        pass

    def val_step_end(self, e):
        # for examination
        pass

    def train_epoch_start(self, e):
        self.clear_log_dict()
        self.train_log = None
        self.val_log = None
        self.test_log = None

    def val_epoch_start(self, e):
        pass    

    def train_epoch_end(self, e, test_wb=False):
        log_dict = {'epoch':e}
        log_dict.update({k:np.mean([x[k].item() if torch.is_tensor(x[k]) else x[k] for x in self.logs['train']]) for k in self.logs['train'][0].keys()})
        # 分布式聚合（对每个数值做 all_reduce 平均）
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            for k, v in list(log_dict.items()):
                if k == 'epoch':
                    continue
                t = torch.as_tensor(v, device=self.device, dtype=torch.float32)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                log_dict[k] = (t / world_size).item()
        # 写文件与wandb仅master
        if self.is_master:
            with open(self.cfg.output_dir + "/train_metrics.json", 'a') as f:
                f.write(json.dumps({k:v for k, v in log_dict.items()}))
                f.write('\r\n')
        self.train_log = log_dict

    def val_epoch_end(self, e, test_wb=False):
        log_dict = {'epoch':e}
        log_dict.update({k:np.mean([x[k].item() if torch.is_tensor(x[k]) else x[k] for x in self.logs['val']]) for k in self.logs['val'][0].keys()})
        # 分布式聚合
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            for k, v in list(log_dict.items()):
                if k == 'epoch':
                    continue
                t = torch.as_tensor(v, device=self.device, dtype=torch.float32)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                log_dict[k] = (t / world_size).item()
        # 写文件与wandb仅master
        if self.is_master:
            with open(self.cfg.output_dir + "/val_metrics.json", 'a') as f:
                f.write(json.dumps({k:v for k, v in log_dict.items()}))
                f.write('\r\n')
        self.val_log = log_dict

        if self.is_master and self.val_log['val_loss'] < self.min_val_loss:
            self.min_val_loss = self.val_log['val_loss']
            self.min_val_epoch = e
            self.val_checkpoint_path = self.save_checkpoint(model_checkpoint_path=self.val_checkpoint_path, suffix="val")

    def test_epoch_end(self):
        log_dict = {}
        log_dict.update({k:np.mean([x[k].item() if torch.is_tensor(x[k]) else x[k] for x in self.logs['test']]) for k in self.logs['test'][0].keys()})

        with open(self.cfg.output_dir + "/test_metrics.json", 'a') as f:
            f.write(json.dumps({k:v for k, v in log_dict.items()}))
            f.write('\r\n')

        self.test_log = log_dict

    def train_val_epoch_end(self, e):

        if e % self.cfg.logging.log_freq_every_n_epoch == 0:
            self.logging(e)

        if e % self.cfg.checkpoint_freq_every_n_epoch == 0:
            self.train_checkpoint_path = self.save_checkpoint(model_checkpoint_path=self.train_checkpoint_path, suffix="train")

        self.current_epoch += 1

    def load_checkpoint(self, load):
        checkpoint = False
        ckpts = list(glob.glob(f'{self.cfg.output_dir}/*={self.model_name}=*.ckpt'))   
        print(ckpts, f'{self.cfg.output_dir}/*={self.model_name}=*.ckpt')     
        if len(ckpts) > 0:

            ckpt_epochs = np.array([int(ckpt.split('/')[-1].split('.')[0].split('=')[1]) for ckpt in ckpts])
            ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
            print(f">>>>> Load model from checkpoint {ckpt}")

            if torch.cuda.is_available():
                ckpt = torch.load(ckpt)
            else:
                ckpt = torch.load(ckpt, map_location=torch.device('cpu'))
            
            self.current_epoch = ckpt['epoch'] + 1
            
            print(f">>>>> Update model from train checkpoint") 
            self.load_state_dict(ckpt['model_state_dict'])
            
            try:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except:
                print("ERROR: Loading state dict")

            if self.cfg.optim.use_lr_scheduler:
                try:
                    self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                except:
                    print("ERROR: Loading scheduler")                

            self.train_checkpoint_path = f"{self.cfg.output_dir}/epoch={ckpt['epoch']}={self.model_name}=train.ckpt"

            ckpts = list(glob.glob(f'{self.cfg.output_dir}/*={self.model_name}=val.ckpt'))
            # list(self.cfg.output_dir.glob(f'*={self.model_name}=val.ckpt'))

            if len(ckpts) > 0:

                ckpt_epochs = np.array([int(ckpt.split('/')[-1].split('.')[0].split('=')[1]) for ckpt in ckpts])
                ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
                
                print(f">>>>> Load val model from checkpoint {ckpt}")                

                if torch.cuda.is_available():
                    ckpt = torch.load(ckpt)
                else:
                    ckpt = torch.load(ckpt, map_location=torch.device('cpu'))                

                if load:
                    print(f">>>>> Update model from val checkpoint") 
                    self.load_state_dict(ckpt['model_state_dict'])

                self.min_val_epoch = ckpt['epoch']
                self.min_val_loss = torch.tensor(ckpt['val_loss'])

                print("min val epoch: ", self.min_val_epoch)
                print("min val loss: ", self.min_val_loss.item())            

                self.val_checkpoint_path = f"{self.cfg.output_dir}/epoch={ckpt['epoch']}={self.model_name}=val.ckpt"
            checkpoint = True
        else:
            print(f">>>>> New Training")        

        return checkpoint

    def save_checkpoint(self, model_checkpoint_path, suffix="val", logs={}):
        model_checkpoint = {
            'model_state_dict':copy.deepcopy(self.state_dict()), 
            'optimizer_state_dict':copy.deepcopy(self.optimizer.state_dict()), 
            'scheduler_state_dict':copy.deepcopy(self.scheduler.state_dict()),
            'epoch':self.current_epoch, 
            'train_loss':self.train_log['train_loss'], 
            'val_loss':self.val_log['val_loss'] if self.val_log else None 
        }

        model_checkpoint.update(logs)

        new_model_checkpoint_path = f"{self.cfg.output_dir}/epoch={self.current_epoch}={self.model_name}={suffix}.ckpt"
        if new_model_checkpoint_path != model_checkpoint_path:
            if model_checkpoint_path and os.path.exists(model_checkpoint_path):
                try:
                    os.remove(model_checkpoint_path)
                except FileNotFoundError:
                    pass
            print("Save model checkpoint: ", new_model_checkpoint_path)
            print("\tmodel checkpoint val loss: ", model_checkpoint['val_loss'])
            torch.save(model_checkpoint, new_model_checkpoint_path)

        return new_model_checkpoint_path

    def early_stopping(self, e):
        if e - self.min_val_epoch > self.cfg.data.early_stopping_patience_epoch:
            print("Early stopping")
            return True

        return False
    
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        teacher_forcing = (
                self.current_epoch <= self.hparams.teacher_forcing_max_epoch)
        outputs = self(batch, teacher_forcing, training=True)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='train')
        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            prefix='train'
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False, training=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='val')
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            prefix='val'
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False, training=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='test')
        self.log_dict(
            log_dict,
            prefix='test'
        )
        return loss

    def compute_stats(self, batch, outputs, prefix):
        num_atom_loss = outputs['num_atom_loss']
        lattice_loss = outputs['lattice_loss']
        coord_loss = outputs['coord_loss']
        type_loss = outputs['type_loss']
        kld_loss = outputs['kld_loss']
        composition_loss = outputs['composition_loss']
        # 元素计数损失：将每晶体的元素计数与GT计数做MAE
        def _count_loss_from_probs(pred_comp_per_atom, target_atom_types, batch_idx):
            # 目标转换为0-based
            tgt0 = target_atom_types - 1
            # 保证 batch_idx 形状与设备正确
            batch_idx = batch_idx.view(-1).to(pred_comp_per_atom.device)
            # 计算每原子的预测概率
            probs = F.softmax(pred_comp_per_atom, dim=-1)
            # 每晶体聚合概率并乘以原子数，得到预测计数的期望
            per_graph_prob = scatter(probs, batch_idx, dim=0, reduce='mean')
            num_atoms_per_graph = scatter(torch.ones_like(batch_idx, dtype=torch.float32), batch_idx, dim=0, reduce='sum')
            counts_pred = per_graph_prob * num_atoms_per_graph.unsqueeze(-1)
            # GT计数：one-hot后按晶体聚合
            num_cls = pred_comp_per_atom.size(-1)
            tgt_oh = F.one_hot(tgt0, num_classes=num_cls).float()
            counts_gt = scatter(tgt_oh, batch_idx, dim=0, reduce='sum')
            # MAE
            return torch.mean(torch.abs(counts_pred - counts_gt))

        sigma = torch.exp(self.param_log_sigma)
        count_loss = _count_loss_from_probs(outputs['pred_composition_per_atom'], outputs['target_atom_types'], batch.batch)
        warm_start = int(getattr(self.hparams, 'count_warmup_start', 20))
        warm_span = int(getattr(self.hparams, 'count_warmup_span', 40))
        progress = max(0.0, min(1.0, (self.current_epoch - warm_start) / float(max(1, warm_span))))
        w_count = getattr(self.hparams, 'cost_count', 1.0) * progress

        if 'x0pred' in outputs:
            pred_cart = frac_to_cart_coords(outputs['x0pred'], outputs['pred_lengths'], outputs['pred_angles'], batch.num_atoms)
            gt_cart = frac_to_cart_coords(batch.frac_coords, batch.lengths, batch.angles, batch.num_atoms)
            d2 = min_distance_sqr_pbc(pred_cart, gt_cart, batch.lengths, batch.angles, batch.num_atoms, device=self.device)
            cart_coord_loss = torch.mean(torch.sqrt(d2 + EPSILON))
        else:
            cart_coord_loss = torch.tensor(0.0, device=self.device)
        w_cart = float(getattr(self.hparams, 'cost_cart', 1.0))

        loss = (
            0.5/(sigma[0]**2) * self.hparams.cost_type * type_loss +      
            0.5/(sigma[1]**2) * self.hparams.cost_coord * coord_loss +    
            w_cart * cart_coord_loss +
            self.hparams.cost_natom * num_atom_loss +
            self.hparams.cost_lattice * lattice_loss +
            self.hparams.cost_composition * composition_loss +
            w_count * count_loss +
            self.hparams.beta * kld_loss 
            +2.0 * torch.sum(torch.log(sigma))
        )

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_natom_loss': num_atom_loss,
            f'{prefix}_lattice_loss': lattice_loss,
            f'{prefix}_coord_loss': coord_loss,
            f'{prefix}_cart_coord_loss': cart_coord_loss,
            f'{prefix}_type_loss': type_loss,
            f'{prefix}_kld_loss': kld_loss,
            f'{prefix}_composition_loss': composition_loss,
            f'{prefix}_count_loss': count_loss,
        }

        if prefix != 'train':
            
            # evaluate num_atom prediction.
            pred_num_atoms = outputs['pred_num_atoms'].argmax(dim=-1)
            num_atom_accuracy = (pred_num_atoms == batch.num_atoms).sum() / batch.num_graphs

            """ # evalute lattice prediction.
            pred_lengths_and_angles = outputs['pred_lengths_and_angles']
            self.lattice_scaler.match_device(pred_lengths_and_angles)
            scaled_preds = self.lattice_scaler.inverse_transform(
                pred_lengths_and_angles)
            pred_lengths = scaled_preds[:, :3]
            pred_angles = scaled_preds[:, 3:]

            if self.hparams.data.lattice_scale_method == 'scale_length':
                pred_lengths = pred_lengths * \
                               batch.num_atoms.view(-1, 1).float() ** (1 / 3) """
            # 改为直接使用解码器输出的物理长度与角度，避免 9→6 维度不匹配
            pred_lengths = outputs['pred_lengths']
            pred_angles = outputs['pred_angles']
            
            # 注意：lengths 已在解码器中按 scale_length 做了恢复，这里不再重复缩放
            lengths_mard = mard(batch.lengths, pred_lengths)
            angles_mae = torch.mean(torch.abs(pred_angles - batch.angles))

            pred_volumes = lengths_angles_to_volume(pred_lengths, pred_angles)
            true_volumes = lengths_angles_to_volume(
                batch.lengths, batch.angles)
            volumes_mard = mard(true_volumes, pred_volumes)

            # evaluate atom type prediction.
            pred_atom_types = outputs['pred_atom_types']
            target_atom_types = outputs['target_atom_types']
            type_accuracy = pred_atom_types.argmax(
                dim=-1) == (target_atom_types - 1)
            type_accuracy = scatter(type_accuracy.float(
            ), batch.batch, dim=0, reduce='mean').mean()

            loss = coord_loss+0.5 * outputs['coord_loss']

            log_dict.update({
                f'{prefix}_loss': loss,
                f'{prefix}_natom_accuracy': num_atom_accuracy,
                f'{prefix}_lengths_mard': lengths_mard,
                f'{prefix}_angles_mae': angles_mae,
                f'{prefix}_volumes_mard': volumes_mard,
                f'{prefix}_type_accuracy': type_accuracy,
            })

        return log_dict, loss    
    
    def logging(self, e):
        if self.is_master:
            print(f"Epoch {e:5d}:")
            print(f"\tTrain Loss:{self.train_log['train_loss']}")
            if self.val_log:
                print(f"\tVal Loss:{self.val_log['val_loss']}")
            print(f"\tLR:{self.optimizer.param_groups[0]['lr']}")

    def log_dict(self, log_dict, prefix, on_step=False, on_epoch=False, prog_bar=False):
        self.logs[prefix].append(log_dict)

    def clear_log_dict(self):
        for x in self.logs:
            self.logs[x] = []
        self.train_log = []
        self.val_log = []


    def kld_reparam(self, hidden):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return mu, log_var, z

    def shortest_side(self, x):
        neg = x < 0.
        dcm = x % ((-1)**neg)
        p = (-1) ** (dcm < (0.5*(-1)**neg))
        x_fold = p * torch.div(x, (-1)**neg, rounding_mode='floor') + dcm
        return x_fold

    def generate_rand_init(self, pred_composition_per_atom, pred_lengths,
                           pred_angles, num_atoms, batch):
        rand_frac_coords = torch.rand(num_atoms.sum(), 3,
                                      device=num_atoms.device)
        pred_composition_per_atom = F.softmax(pred_composition_per_atom,
                                              dim=-1)
        rand_atom_types = self.sample_composition(
            pred_composition_per_atom, num_atoms)
        return rand_frac_coords, rand_atom_types

    def sample_composition(self, composition_prob, num_atoms):
        """
        Samples composition such that it exactly satisfies composition_prob
        """
        batch = torch.arange(
            len(num_atoms), device=num_atoms.device).repeat_interleave(num_atoms)
        assert composition_prob.size(0) == num_atoms.sum() == batch.size(0)
        composition_prob = scatter(
            composition_prob, index=batch, dim=0, reduce='mean')

        all_sampled_comp = []

        for comp_prob, num_atom in zip(list(composition_prob), list(num_atoms)):
            comp_num = torch.round(comp_prob * num_atom)
            atom_type = torch.nonzero(comp_num, as_tuple=True)[0] + 1
            atom_num = comp_num[atom_type - 1].long()

            sampled_comp = atom_type.repeat_interleave(atom_num, dim=0)

            # if the rounded composition gives less atoms, sample the rest
            if sampled_comp.size(0) < num_atom:
                left_atom_num = num_atom - sampled_comp.size(0)

                left_comp_prob = comp_prob - comp_num.float() / num_atom

                left_comp_prob[left_comp_prob < 0.] = 0.
                left_comp = torch.multinomial(
                    left_comp_prob, num_samples=left_atom_num, replacement=True)
                # convert to atomic number
                left_comp = left_comp + 1
                sampled_comp = torch.cat([sampled_comp, left_comp], dim=0)

            sampled_comp = sampled_comp[torch.randperm(sampled_comp.size(0))]
            sampled_comp = sampled_comp[:num_atom]
            all_sampled_comp.append(sampled_comp)

        all_sampled_comp = torch.cat(all_sampled_comp, dim=0)
        assert all_sampled_comp.size(0) == num_atoms.sum()
        return all_sampled_comp

    def predict_num_atoms(self, z):
        return self.fc_num_atoms(z)

    def predict_property(self, z):
        self.scaler.match_device(z)
        return self.scaler.inverse_transform(self.fc_property(z))

    def predict_property_class(self, z):
        return torch.stack([self.fc_property_class[i](z) for i in range(self.len_prop_classes)], -1)

    def predict_lattice(self, z, num_atoms):
        self.lattice_scaler.match_device(z)
        pred_lengths_and_angles = self.fc_lattice(z)  # (N, 6)
        scaled_preds = self.lattice_scaler.inverse_transform(
            pred_lengths_and_angles)
        pred_lengths = scaled_preds[:, :3]
        pred_angles = scaled_preds[:, 3:]
        if self.hparams.data.lattice_scale_method == 'scale_length':
            pred_lengths = pred_lengths * num_atoms.view(-1, 1).float() ** (1 / 3)
        # <pred_lengths_and_angles> is scaled.
        return pred_lengths_and_angles, pred_lengths, pred_angles

    def predict_composition(self, z, num_atoms):
        z_per_atom = z.repeat_interleave(num_atoms, dim=0)
        pred_composition_per_atom = self.fc_composition(z_per_atom)
        return pred_composition_per_atom
    
    def smoothing_cross_entropy(self, pred, target, smoothing=0.05):
        num_class = pred.size(-1)
        smooth_label = (1-smoothing)*F.one_hot(target, num_class).float()+smoothing/num_class
        log_prob = F.log_softmax(pred, dim=1)
        return -torch.sum(log_prob * smooth_label, dim=-1)
    
    def ss_loss(self, pred_out, target, t, batch_idx):
        loss = self.diffalgo.kl(target, pred_out, t)
        loss = scatter(loss, batch_idx, reduce='mean').mean()
        return loss

    def num_atom_loss(self, pred_num_atoms, num_atoms):
        #return F.cross_entropy(pred_num_atoms, num_atoms)
        return self.smoothing_cross_entropy(pred_num_atoms,num_atoms).mean()

    def lattice_loss(self, pred_lengths_and_angles, batch):
        # 解析九维：前 3 为长度，后 6 为 (cos,sin) 对
        raw = pred_lengths_and_angles
        pred_lengths = raw[:, :3]
        cs = torch.tanh(raw[:, 3:])  # 保证落在 [-1,1]
        cos_alpha, sin_alpha, cos_beta, sin_beta, cos_gamma, sin_gamma = cs.unbind(dim=1)
        angles_rad = torch.stack([
            torch.atan2(sin_alpha, cos_alpha),
            torch.atan2(sin_beta, cos_beta),
            torch.atan2(sin_gamma, cos_gamma),
        ], dim=1)
        pred_angles_deg = angles_rad * (180.0 / math.pi)

        true_lengths = batch.lengths
        true_angles_deg = batch.angles

        # 若训练采用了按 N^(1/3) 缩放长度，这里与评测一致地恢复
        if getattr(self.hparams.data, 'lattice_scale_method', None) == 'scale_length':
            scale = batch.num_atoms.view(-1, 1).float() ** (1.0 / 3.0)
            pred_lengths = pred_lengths * scale

        # 1) 相对长度误差（无量纲）+ Huber
        rel_len_err = (pred_lengths - true_lengths) / (true_lengths + EPSILON)
        len_loss = F.smooth_l1_loss(rel_len_err, torch.zeros_like(rel_len_err), reduction='mean')

        # 2) 角度周期性损失改进：使用 Cosine 距离的平方形式增强梯度
        dtheta = (pred_angles_deg - true_angles_deg) * math.pi / 180.0
        dtheta = (dtheta + math.pi) % (2 * math.pi) - math.pi
        cos_loss = (1.0 - torch.cos(dtheta))
        sin_sq_loss = torch.sin(dtheta) ** 2
        cos_loss_sq = cos_loss ** 2
        w_cos_sq = float(getattr(self.hparams, "lattice_cos_sq_weight", 0.05))
        ang_loss = (cos_loss + 0.1 * sin_sq_loss + w_cos_sq * cos_loss_sq).mean()

        # 3) 几何一致性（Gram）
        pred_L = lattice_params_to_matrix_torch(pred_lengths, pred_angles_deg)
        true_L = lattice_params_to_matrix_torch(true_lengths, true_angles_deg)
        pred_G = torch.matmul(pred_L.transpose(-1, -2), pred_L)
        true_G = torch.matmul(true_L.transpose(-1, -2), true_L)
        diff_G = pred_G - true_G
        num = torch.norm(diff_G, dim=(1, 2))
        den = torch.norm(true_G, dim=(1, 2)) + EPSILON
        gram_loss = (num / den).mean()
        gram_clip = float(getattr(self.hparams, "lattice_gram_clip", 2.0))
        gram_loss = torch.clamp(gram_loss, min=0.0, max=gram_clip)

        w_len = float(getattr(self.hparams, "lattice_len_weight", 1.2))
        w_ang = float(getattr(self.hparams, "lattice_ang_weight", 2.5))
        w_grm = float(getattr(self.hparams, "lattice_gram_weight", 0.5))

        total = w_len * len_loss + w_ang * ang_loss + w_grm * gram_loss
        return total
    

    def composition_loss(self, pred_composition_per_atom, target_atom_types, batch_idx):
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(pred_composition_per_atom,
                               target_atom_types, reduction='none')
        #loss = self.smoothing_cross_entropy(pred_composition_per_atom, target_atom_types)
        return scatter(loss, batch_idx, reduce='mean').mean()


    def type_loss(self, pred_atom_types, target_atom_types,
                  sigma_type_t, batch_idx):
        # 1-based -> 0-based
        target = target_atom_types - 1
        # 从配置读取超参：焦点指数 gamma 与温度缩放
        gamma = float(getattr(self.hparams, 'type_focal_gamma', 0.5))
        temperature = float(getattr(self.hparams, 'type_temp', 1.0))
        # 计算带温度的 log_softmax 与概率
        logits = pred_atom_types / max(temperature, 1e-6)
        log_prob = F.log_softmax(logits, dim=-1)
        prob = log_prob.exp()
        # 目标类别的 log(p_t) 与 p_t
        log_pt = log_prob.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = prob.gather(1, target.unsqueeze(1)).squeeze(1).clamp_(1e-6, 1.0 - 1e-6)
        loss_vec = -log_pt
        # 可选：类别权重 alpha_t（缓解类别不平衡）
        w = getattr(self, 'type_class_weights', None)
        if w is not None:
            w = w.to(pred_atom_types.device)
            alpha_t = w[target]
            loss_vec = loss_vec * alpha_t
        # 焦点调制
        loss_vec = loss_vec * ((1.0 - pt) ** gamma)
        # 时间噪声归一化
        loss_vec = loss_vec / sigma_type_t

        return scatter(loss_vec, batch_idx, reduce='mean').mean()

    def kld_loss(self, mu, log_var):
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        return kld_loss

    def sample(self, num_samples, ld_kwargs):
        z = torch.randn(num_samples, self.hparams.hidden_dim,
                        device=self.device)
        samples = self.langevin_dynamics(z, ld_kwargs)
        return samples


class StatsDiffCrystal(BaseModel):
    def __init__(self, encoder, param_decoder, diffalgo, diffnet, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.hparams = cfg.model
        self.hparams.data = cfg.data
        self.hparams.algo = "Stats-DiffCrystal"
        self.model_name = "Stats-DiffCrystal"
        self.recenter = cfg.model.recenter
        self.logs = {'train': [], 'val': [], 'test': []}
        self.T = self.hparams.num_noise_level
        
        self.encoder = encoder
        self.param_decoder = param_decoder
        self.diffalgo = diffalgo
        self.diffnet = diffnet      

        self.fc_mu = nn.Linear(self.hparams.latent_dim,
                               self.hparams.latent_dim)
        self.fc_var = nn.Linear(self.hparams.latent_dim,
                                self.hparams.latent_dim)
        
        init_log_sigma = torch.log(torch.tensor(1.0/torch.sqrt(torch.tensor(2.0))))
        self.param_log_sigma = nn.Parameter(torch.full((2,), init_log_sigma))
        self.embedding = torch.zeros(100, 92)
        for i in range(100):
            self.embedding[i] = torch.tensor(KHOT_EMBEDDINGS[i + 1])

        # obtain from datamodule.
        self.lattice_scaler = None
        self.scaler = None        
        try:
            self.multi_t_K = cfg.model.multi_t_K
        except Exception:
            self.multi_t_K = 1

    def forward(self, batch, teacher_forcing, training, model_classifier=None):
        torch.set_printoptions(threshold=torch.inf)
        batch_idx = batch.batch
        atom_types = batch.atom_types
        num_atoms = batch.num_atoms
        frac_coords = batch.frac_coords
        
        # Encode        
        hidden = self.encoder(batch)
        mu, log_var, z = self.kld_reparam(hidden)
        
        # Decode lattice, num_atoms, and composition
        (pred_num_atoms, pred_lengths_and_angles, pred_lengths, pred_angles,
         pred_composition_per_atom) = self.param_decoder(
            z, batch.num_atoms, batch.lengths, batch.angles, teacher_forcing)
        lattices = lattice_params_to_matrix_torch(pred_lengths, pred_angles)
        
        # Perturb features
        composition_probs = F.softmax(pred_composition_per_atom.detach(), dim=-1)
        K = int(self.multi_t_K) if training else 1
        coord_losses = []
        type_losses = []

        for _ in range(K):
            noisy_feats, times = self.diffalgo.perturb_sample(
                frac_coords, atom_types, composition_probs, num_atoms
            )
            r_t, A_t = noisy_feats
            t_type = times[1].repeat_interleave(num_atoms, dim=0)#.squeeze(-1)
            sigma_type_t = self.diffalgo.type_sigmas[t_type]

            # Predict noise and atomic types
            Z_t = torch.multinomial(A_t, num_samples=1).squeeze(1) + 1
            px0, A_theta = self.diffnet(
                z, times[0], Z_t, r_t, lattices, num_atoms, batch_idx
            )
            x0pred = self.diffalgo.from_domain(px0)
            x0pred_last = x0pred

            # coord/type loss for this t
            coord_losses.append(self.ss_loss(x0pred, frac_coords, times[2], batch_idx))
            type_losses.append(self.type_loss(A_theta, atom_types, sigma_type_t, batch_idx))

            last_rand_r_t = r_t
            last_A_theta = A_theta

        coord_loss = torch.stack(coord_losses).mean() if K > 1 else coord_losses[0]
        type_loss = torch.stack(type_losses).mean() if K > 1 else type_losses[0]
        
        ## decoder_stats losses
        num_atom_loss = self.num_atom_loss(pred_num_atoms, num_atoms)
        lattice_loss = self.lattice_loss(pred_lengths_and_angles, batch)
        composition_loss = self.composition_loss(
            pred_composition_per_atom, batch.atom_types, batch_idx)
        
        ## KLD loss
        kld_loss = self.kld_loss(mu, log_var)

        return {
            'num_atom_loss': num_atom_loss,
            'lattice_loss': lattice_loss,
            'composition_loss': composition_loss,
            'coord_loss': coord_loss,
            'type_loss': type_loss,
            'kld_loss': kld_loss,
            'pred_num_atoms': pred_num_atoms,
            'pred_lengths_and_angles': pred_lengths_and_angles,
            'pred_lengths': pred_lengths,
            'pred_angles': pred_angles,
            'pred_atom_types': A_theta,
            'pred_composition_per_atom': pred_composition_per_atom,
            'target_frac_coords': batch.frac_coords,
            'target_atom_types': batch.atom_types,
            'rand_frac_coords': r_t,
            'x0pred': x0pred_last,
            'z': z,
        }
 
    @torch.no_grad()  
    def ss_sampling_procedure(self,z,ld_kwargs, gt_num_atoms=None,
                              gt_atom_tpyes=None, model_classifier=None, 
                              labels=None, is_watch=False):
        if ld_kwargs.save_traj:
            all_x0pred = []
            all_xt = []
            all_theta = []
            all_At = []
        num_atoms, _, lengths, angles, comp = self.param_decoder(z, gt_num_atoms)
        lattices = lattice_params_to_matrix_torch(lengths, angles)
        if gt_num_atoms is not None:
            num_atoms = gt_num_atoms
        batch_idx = torch.arange(
            len(num_atoms), device=num_atoms.device).repeat_interleave(num_atoms)
        
        #init xt
        f_t = self.diffalgo.result_distribution_sample(num_atoms)
        G_s_t = torch.zeros_like(f_t)
        G_c_t = torch.zeros_like(f_t)
        
        #obtain atom types
        A_t = F.softmax(comp, dim=-1)
        if gt_atom_tpyes is None:
            Z_t = self.sample_composition(A_t, num_atoms)
        else:
            Z_t = F.one_hot(gt_atom_tpyes-1, num_classes=MAX_ATOMIC_NUM)
            
        #predictor
        t_min = 1
        for j in tqdm(reversed(range(t_min, self.T)), total=(self.T-t_min),
                      disable=ld_kwargs.disable_bar):#disable=True):#
            
            # get t
            t = torch.tensor([j]*num_atoms.shape[0]).long().to(z.device)
            t_int = t.repeat_interleave(num_atoms, dim=0)
            
            # update Gt and ft
            G_c_t, G_s_t, theta = self.diffalgo.update_Gt(f_t, G_c_t, G_s_t, t_int, num_atoms)
            
            # Predict x0
            px0, A_t = self.diffnet(z, t, Z_t, theta, lattices, 
                                     num_atoms, batch_idx)
            x0pred = self.diffalgo.from_domain(px0)
            
            #sample ft ~ (x0p, t)
            if j>=0:
                f_t = self.diffalgo.reverse_step_sample(x0pred, t_int, num_atoms)
            else:
                tensor_list = []
                for _ in range(10):
                    f_t = self.diffalgo.reverse_step_sample(x0pred, t_int, num_atoms)
                    tensor_list.append(f_t)
                f_t = circular_median_tensor(torch.stack(tensor_list))
            
            if gt_atom_tpyes is None:
                Z_t = torch.argmax(A_t, dim=1) + 1
            
            if j%10==0 and is_watch:
                print(t[0].item(),end='\t')
                print(x0pred[0].tolist(),end='\t')
                print(Z_t[0].item())
                
            if ld_kwargs.save_traj:
                all_x0pred.append(x0pred)
                all_xt.append(f_t)
                all_theta.append(theta)
                all_At.append(A_t)
        if gt_atom_tpyes is not None:
            A_t = F.one_hot(gt_atom_tpyes - 1, num_classes=MAX_ATOMIC_NUM)

        output_dict = {'num_atoms': num_atoms, 'lengths': lengths, 'angles': angles,
                       'frac_coords': x0pred, 'atom_types': A_t,
                       'is_traj': False}
        
        if ld_kwargs.save_traj:
            output_dict.update(dict(
                all_x0pred=torch.stack(all_x0pred, dim=0),
                all_xt=torch.stack(all_xt, dim=0),
                all_theta=torch.stack(all_theta, dim=0),
                is_traj=True,
            ))
            
        return output_dict

def circular_median_tensor(tensor):

    bs, _ , _ = tensor.shape
    sorted_tensor, _ = torch.sort(tensor, dim=0)
    
    diffs = torch.cat(
        [sorted_tensor[1:]-sorted_tensor[:-1],
         (sorted_tensor[0:1]+1)-sorted_tensor[-1:]],
        dim=0
    )

    max_gap_indices = torch.argmax(diffs, dim=0)
    
    indices = torch.arange(bs, device=tensor.device).view(-1,1,1)
    rotated_indices = (indices-max_gap_indices[None,:,:]-1)%bs
    rotated_tensor = sorted_tensor.gather(0,rotated_indices)
    
    mid_idx = bs//2
    if bs%2==0:
        medians = (rotated_tensor[mid_idx-1]+rotated_tensor[mid_idx])/2
    else:
        medians = rotated_tensor[mid_idx]
    
    return medians
