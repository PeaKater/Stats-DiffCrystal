import random
from typing import Optional, Sequence
from pathlib import Path

#import hydra
import numpy as np
import omegaconf
# import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

# from cdvae.common.utils import PROJECT_ROOT
from common.data_utils import get_scaler_from_data_list
from common.dataset import CrystDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


class CrystDataModule:
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        scaler_path=None,
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

        self.get_scaler(scaler_path)
    
    # 数据集下载
    def prepare_data(self) -> None:
        # download only
        pass
    # 获取数据标量器
    def get_scaler(self, scaler_path):
        # Load once to compute property scaler
        if scaler_path is None:            
#            train_dataset = hydra.utils.instantiate(self.datasets.train)
            train_dataset = CrystDataset(**self.datasets.train)

            self.lattice_scaler = get_scaler_from_data_list(
                train_dataset.cached_data,
                key='scaled_lattice')

            self.scaler = [get_scaler_from_data_list(train_dataset.cached_data,key=prop) for prop in train_dataset.prop]
            # self.scaler = get_scaler_from_data_list(
            #     train_dataset.cached_data,
            #     key=train_dataset.prop)
        else:
            self.lattice_scaler = torch.load(
                Path(scaler_path) / 'lattice_scaler.pt')
            self.scaler = torch.load(Path(scaler_path) / 'prop_scaler.pt')
    # 用于构建数据集并分配数据标量器
    def setup(self, stage: Optional[str] = None):
        """
        construct datasets and assign data scalers.
        """
        if stage is None or stage == "fit":
            # self.train_dataset = hydra.utils.instantiate(self.datasets.train)
            self.train_dataset = CrystDataset(**self.datasets.train)
            self.val_datasets = [
                # hydra.utils.instantiate(dataset_cfg)
                CrystDataset(**dataset_cfg)
                for dataset_cfg in self.datasets.val
            ]

            self.train_dataset.lattice_scaler = self.lattice_scaler
            self.train_dataset.scaler = self.scaler
            for val_dataset in self.val_datasets:
                val_dataset.lattice_scaler = self.lattice_scaler
                val_dataset.scaler = self.scaler

        if stage is None or stage == "test":
            self.train_dataset = CrystDataset(**self.datasets.train)
            self.train_dataset.lattice_scaler = self.lattice_scaler
            self.train_dataset.scaler = self.scaler
            
            self.test_datasets = [
                # hydra.utils.instantiate(dataset_cfg)
                CrystDataset(**dataset_cfg)
                for dataset_cfg in self.datasets.test
            ]
            for test_dataset in self.test_datasets:
                test_dataset.lattice_scaler = self.scaler
                test_dataset.scaler = self.scaler
    
    # 训练数据加载
    def train_dataloader(self) -> DataLoader:
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(self.train_dataset, shuffle=True, drop_last=True)
            return DataLoader(
                self.train_dataset,
                shuffle=False,
                sampler=sampler,
                batch_size=self.batch_size.train,
                num_workers=self.num_workers.train,
                worker_init_fn=worker_init_fn,
                pin_memory=True,
                persistent_workers=(self.num_workers.train>0),
                prefetch_factor=(4 if self.num_workers.train>0 else None),
            )
        else:
            return DataLoader(
                self.train_dataset,
                shuffle=True,  # 由 False 改为 True
                batch_size=self.batch_size.train,
                num_workers=self.num_workers.train,
                worker_init_fn=worker_init_fn,
                pin_memory=True,
                persistent_workers=(self.num_workers.train>0),
                prefetch_factor=(4 if self.num_workers.train>0 else None),
            )

    # 验证数据加载
    def val_dataloader(self) -> Sequence[DataLoader]:
        if dist.is_available() and dist.is_initialized():
            return [
                DataLoader(
                    dataset,
                    shuffle=False,
                    sampler=DistributedSampler(dataset, shuffle=False, drop_last=True),
                    batch_size=self.batch_size.val,
                    num_workers=self.num_workers.val,
                    worker_init_fn=worker_init_fn,
                    pin_memory=True,
                    persistent_workers=(self.num_workers.val>0),
                    prefetch_factor=(4 if self.num_workers.val>0 else None),
                )
                for dataset in self.val_datasets
            ]
        else:
            return [
                DataLoader(
                    dataset,
                    shuffle=False,
                    batch_size=self.batch_size.val,
                    num_workers=self.num_workers.val,
                    worker_init_fn=worker_init_fn,
                    pin_memory=True,
                    persistent_workers=(self.num_workers.val>0),
                    prefetch_factor=(4 if self.num_workers.val>0 else None),
                )
                for dataset in self.val_datasets
            ]

    # 测试数据加载
    def test_dataloader(self) -> Sequence[DataLoader]:
        if dist.is_available() and dist.is_initialized():
            return [
                DataLoader(
                    dataset,
                    shuffle=False,
                    sampler=DistributedSampler(dataset, shuffle=False, drop_last=True),
                    batch_size=self.batch_size.test,
                    num_workers=self.num_workers.test,
                    worker_init_fn=worker_init_fn,
                    pin_memory=True,
                    persistent_workers=(self.num_workers.test>0),
                    prefetch_factor=(4 if self.num_workers.test>0 else None),
                )
                for dataset in self.test_datasets
            ]
        else:
            return [
                DataLoader(
                    dataset,
                    shuffle=False,
                    batch_size=self.batch_size.test,
                    num_workers=self.num_workers.test,
                    worker_init_fn=worker_init_fn,
                    pin_memory=True,
                    persistent_workers=(self.num_workers.test>0),
                    prefetch_factor=(4 if self.num_workers.test>0 else None),
                )
                for dataset in self.test_datasets
            ]

    # 返回包含类属性的字符串表示
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.datasets=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )


# @hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
# def main(cfg: omegaconf.DictConfig):
#     datamodule: pl.LightningDataModule = hydra.utils.instantiate(
#         cfg.data.datamodule, _recursive_=False
#     )
#     datamodule.setup('fit')
#     import pdb
#     pdb.set_trace()


# if __name__ == "__main__":
#     main()
