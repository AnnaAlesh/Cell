import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor, nn

from cell_segmentation.utils.losses import SoftDiceLoss
from cell_segmentation.utils.metrics import create_segmentation_metrics


class SegmentationModel(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        n_classes: int = 3,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_metrics = create_segmentation_metrics(num_classes=n_classes, prefix="train")
        self.val_metrics = create_segmentation_metrics(num_classes=n_classes, prefix="val")
        self.test_metrics = create_segmentation_metrics(num_classes=n_classes, prefix="test")

    def _compute_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        if isinstance(self.loss_fn, SoftDiceLoss):
            # Для Dice loss нужны вероятности (применяем softmax),
            # для CrossEntropyLoss — логиты (softmax уже внутри).
            probs = F.softmax(predictions, dim=1)
            return self.loss_fn(probs, targets)
        return self.loss_fn(predictions, targets)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int = 0) -> dict[str, Tensor]:
        images, masks = batch
        predictions = self.model.forward(images)
        loss = self._compute_loss(predictions, masks)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.train_metrics.update(predictions.argmax(dim=1).detach(), masks)
        return {"loss": loss, "predictions": predictions}

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int = 0) -> dict[str, Tensor]:
        images, masks = batch
        predictions = self.model.forward(images)
        loss = self._compute_loss(predictions, masks)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_metrics.update(predictions.argmax(dim=1).detach(), masks)
        return {"loss": loss, "predictions": predictions}

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int = 0) -> dict[str, Tensor]:
        images, masks = batch
        predictions = self.model.forward(images)
        loss = self._compute_loss(predictions, masks)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.test_metrics.update(predictions.argmax(dim=1).detach(), masks)
        return {"loss": loss, "predictions": predictions}

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), prog_bar=True, on_step=False, on_epoch=True)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            self.log_dict(self.val_metrics.compute(), prog_bar=True, on_step=False, on_epoch=True)
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), prog_bar=True, on_step=False, on_epoch=True)
        self.test_metrics.reset()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
