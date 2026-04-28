import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor, nn
from torchmetrics.detection import MeanAveragePrecision

from cell_detection.utils.losses import YOLOLoss
from cell_detection.utils.metrics import predictions_to_map_format


class DetectionModel(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        anchors: list[list[int]],
        image_size: int = 256,
        grid_size: int = 8,
        num_classes: int = 2,
        nms_iou_threshold: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-3,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = YOLOLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.nms_iou_threshold = nms_iou_threshold

        rescaled = torch.tensor(anchors, dtype=torch.float) / image_size * grid_size
        self.register_buffer("rescaled_anchors", rescaled)

        self.val_map = MeanAveragePrecision(
            iou_type="bbox", box_format="cxcywh", class_metrics=True
        )
        self.test_map = MeanAveragePrecision(
            iou_type="bbox", box_format="cxcywh", class_metrics=True
        )

    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int = 0
    ) -> dict[str, Tensor]:
        images, targets = batch
        predictions = self.model(images)
        loss = self.loss_fn(predictions, targets, self.rescaled_anchors)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "predictions": predictions}

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int = 0
    ) -> dict[str, Tensor]:
        images, targets = batch
        predictions = self.model(images)
        loss = self.loss_fn(predictions, targets, self.rescaled_anchors)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        preds_fmt, target_fmt = predictions_to_map_format(
            predictions.detach(),
            targets,
            self.rescaled_anchors,
            self.nms_iou_threshold,
        )
        self.val_map.update(preds_fmt, target_fmt)
        return {"loss": loss, "predictions": predictions}

    def test_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int = 0
    ) -> dict[str, Tensor]:
        images, targets = batch
        predictions = self.model(images)
        loss = self.loss_fn(predictions, targets, self.rescaled_anchors)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        preds_fmt, target_fmt = predictions_to_map_format(
            predictions.detach(),
            targets,
            self.rescaled_anchors,
            self.nms_iou_threshold,
        )
        self.test_map.update(preds_fmt, target_fmt)
        return {"loss": loss, "predictions": predictions}

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            self.val_map.reset()
            return
        result = self.val_map.compute()
        self.log("val/map", result["map"], prog_bar=True)
        self.log("val/map_50", result["map_50"])
        self.val_map.reset()

    def on_test_epoch_end(self) -> None:
        result = self.test_map.compute()
        self.log("test/map", result["map"])
        self.log("test/map_50", result["map_50"])
        self.test_map.reset()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
