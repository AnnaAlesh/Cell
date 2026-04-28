from typing import Any, cast

import lightning as L
import matplotlib

matplotlib.use("Agg")  # non-interactive backend; no GUI window during training

import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from lightning.pytorch import Callback  # noqa: E402
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from matplotlib.figure import Figure
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops.boxes import box_convert, nms

from cell_detection.data.constants import CLASS_LABELS
from cell_detection.data.transforms import cells_to_bboxes


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


def plot_image(image: Tensor, boxes: Tensor, class_labels: list[str]) -> Figure:
    colour_map = plt.get_cmap("tab20b")
    colors = [colour_map(i) for i in np.linspace(0, 1, len(class_labels))]

    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap="gray")
    h, w = image.shape
    for box in boxes:
        class_pred = int(box[0]) - 1
        x_c, y_c, bw, bh = box[2:]
        upper_left_x = (x_c - bw / 2) * w
        upper_left_y = (y_c - bh / 2) * h
        rect = patches.Rectangle(
            (upper_left_x, upper_left_y),
            bw * w,
            bh * h,
            linewidth=2,
            edgecolor=colors[class_pred],
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            upper_left_x,
            upper_left_y,
            s=class_labels[class_pred],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[class_pred], "pad": 0},
        )
    return fig


class DetectionPlotCallback(Callback):
    """Логирует в Tensorboard несколько примеров с предсказанными рамками после каждой эпохи."""

    def __init__(
        self,
        n_samples: int = 4,
        confidence_threshold: float = 0.2,
        nms_iou_threshold: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
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
        collected = sum(t.shape[0] for t in self.images)
        if collected >= self.n_samples:
            return
        needed = self.n_samples - collected
        images, _ = batch
        self.images.append(images[:needed].detach().cpu())
        self.predictions.append(outputs["predictions"][:needed].detach().cpu())

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not self.images:
            return
        images = torch.cat(self.images)
        predictions = torch.cat(self.predictions)
        anchors = pl_module.rescaled_anchors.detach().cpu()

        trainer.logger = cast(TensorBoardLogger, trainer.logger)
        writer = cast(SummaryWriter, trainer.logger.experiment)

        for i in range(images.shape[0]):
            boxes = cells_to_bboxes(predictions[i : i + 1], anchors, is_predictions=True)
            boxes = boxes[boxes[:, 1] > self.confidence_threshold]
            if boxes.shape[0] > 0:
                keep = nms(
                    box_convert(boxes[:, 2:], "cxcywh", "xyxy"),
                    boxes[:, 1],
                    iou_threshold=self.nms_iou_threshold,
                )
                boxes = boxes[keep]
            fig = plot_image(images[i, 0], boxes, CLASS_LABELS)
            writer.add_figure(
                tag=f"predictions/{i}", figure=fig, global_step=trainer.current_epoch
            )
            plt.close(fig)
