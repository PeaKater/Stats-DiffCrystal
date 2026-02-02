from omegaconf import DictConfig, OmegaConf

from stats_diffcrystal import models, diffusion_modules, encoders, decoders
from stats_diffcrystal.decoders import decode_stats

# 获取模型的超参
def get_hparams(hparams):
    hparams = {k: v for k, v in hparams.items() if k != "_target_"}
    return hparams

# 根据配置文件创建模型对象   
def get_model(cfg):
    # Encoder
    if cfg.encoder._target_ == "DimeNetpp":
        encoder = encoders.DimeNetppEncoder(**get_hparams(cfg.encoder))
    
    # Decode stats
    if cfg.param_decoder._target_ == "MLP":
        param_decoder = decode_stats.MLPDecodeStats(**get_hparams(cfg.param_decoder))

    # Noise model
    if cfg.diffalgo._target_ == "SSDDPM":
        diffalgo = diffusion_modules.SSDDPM(**get_hparams(cfg.diffalgo))
    
    # Decoder
    if cfg.diffnet._target_ == "E3GNN":
        diffnet = decoders.E3GNN(**get_hparams(cfg.diffnet))
    
    # Model
    if cfg.model._target_ == "StatsDiffCrystal":
        return models.StatsDiffCrystal(encoder, param_decoder, diffalgo, diffnet, cfg)
        