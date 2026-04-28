from typing import Any, cast

import lightning as L
import torch
import torchvision
from lightning.pytorch import Callback
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


def flatten_cfg(cfg: DictConfig | dict, prefix: str = "", sep: str = ".") -> dict:
    """Преобразует вложенный конфиг в плоский dict с точечными ключами —
    для логирования в Tensorboard HPARAMS. ``${...}`` интерполяции раскрываются.

    Списки → строки, ``None`` → ``"null"`` (HPARAMS не принимает None).
    """
    if not isinstance(cfg, dict):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    out: dict = {}
    for k, v in cfg.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_cfg(v, key, sep))
        elif isinstance(v, (list, tuple)):
            out[key] = str(v)
        elif v is None:
            out[key] = "null"
        else:
            out[key] = v
    return out


class PredictionPlotCallback(Callback):
    def __init__(self, nrow: int = 4, ncol: int = 2) -> None:
        super().__init__()
        self.nrow = nrow
        self.ncol = ncol
        self.images: list[Tensor] = []
        self.predictions: list[Tensor] = []

    def on_validation_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        self.images = []
        self.predictions = []

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        total_samples = self.nrow * self.ncol
        collected = sum(t.shape[0] for t in self.images)
        if collected >= total_samples:
            return
        needed = total_samples - collected
        images, _ = batch
        self.images.append(images[:needed].detach().cpu())
        self.predictions.append(outputs["predictions"][:needed].detach().cpu())

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not self.images:
            return
        images = torch.cat(self.images)
        predictions = torch.cat(self.predictions)
        predictions_grid = self.make_predictions_grid(images, predictions)
        trainer.logger = cast(TensorBoardLogger, trainer.logger)
        writer = cast(SummaryWriter, trainer.logger.experiment)
        writer.add_image(
            tag="predictions",
            img_tensor=predictions_grid,
            global_step=trainer.current_epoch,
        )

    def make_predictions_grid(self, images: Tensor, predictions: Tensor) -> Tensor:
        original_images_grid = torchvision.utils.make_grid(images, nrow=self.nrow, normalize=True)
        # draw_segmentation_masks requires uint8 input
        original_images_grid = (original_images_grid * 255).to(torch.uint8)
        masks_as_rgb = torch.nn.functional.one_hot(predictions.argmax(dim=1), num_classes=3)
        masks_grid = torchvision.utils.make_grid(
            masks_as_rgb.permute(0, 3, 1, 2)[:, 1:] > 0, nrow=self.nrow
        )

        predictions_grid = torchvision.utils.draw_segmentation_masks(
            original_images_grid,
            masks=masks_grid,
            alpha=0.4,
            colors=["red", "green"],
        )
        return predictions_grid
