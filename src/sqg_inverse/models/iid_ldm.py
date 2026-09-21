import torch
import torch.nn as nn

from ..nn.time_conditioning import TimeConditionedModel
from ..nn.vit import ViT
from .rectified_flow import RectifiedFlow


class IIDLDM(nn.Module):

    def __init__(self, model_cfg, flow_cfg):
        super().__init__()
        cfg = dict(model_cfg)
        self.latent_channels = cfg.pop("latent_channels")
        cfg.pop("backbone")
        time_embed_dim = cfg.pop("time_embed_dim")
        base_model = ViT(
            in_channels=self.latent_channels,
            out_channels=self.latent_channels,
            mod_features=time_embed_dim,
            **cfg,
        )
        self.backbone = TimeConditionedModel(base_model, time_embed_dim)
        self.rf = RectifiedFlow(
            self.backbone,
            flow_cfg["time_sampling_mean"],
            flow_cfg["time_sampling_std"],
        )

    def forward(self, z):
        return self.rf(z)

    @torch.no_grad()
    def sample(self, batch_size, steps, time_warping, warp_param):
        return self.rf.sample(batch_size, steps, time_warping, warp_param)
