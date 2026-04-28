import torch
from omegaconf import DictConfig

from cell_segmentation.model.modules import UNet
from cell_segmentation.model.tiny_unet import UNet as TinyUNet

MODELS = {
    "unet": UNet,
    "tiny_unet": TinyUNet,
    # Задание 3: при выборе варианта с новой моделью добавьте её сюда
    # (например, "unet_se" → SEUNet).
}


def build_model(cfg: DictConfig) -> torch.nn.Module:
    model_cls = MODELS[cfg.model.name]
    return model_cls(
        n_channels=cfg.model.n_channels,
        n_classes=cfg.model.n_classes,
        channel_sizes=list(cfg.model.channel_sizes),
        bilinear=cfg.model.bilinear,
    )


def build_loss(cfg: DictConfig) -> torch.nn.Module:
    name = cfg.loss.name
    if name == "cross_entropy":
        return torch.nn.CrossEntropyLoss()
    # Задание 1: добавьте поддержку soft_dice (см. cell_segmentation.utils.losses.SoftDiceLoss).
    # Задание 3 (вариант с focal loss): добавьте сюда соответствующую ветку.
    raise ValueError(f"unknown loss: {name}")
