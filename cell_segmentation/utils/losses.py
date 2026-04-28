import torch
from torch import Tensor, nn


class SoftDiceLoss(nn.Module):
    def __init__(self, num_classes: int = 3, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
        self.num_classes = num_classes

    def one_hot_encode(self, y_true: Tensor) -> Tensor:
        """Кодирует истинные метки в формат one-hot."""
        y_true = y_true.long()
        return torch.nn.functional.one_hot(y_true, num_classes=self.num_classes).permute(0, 3, 1, 2)

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        """Вычисляет Soft Dice Loss между предсказанными и истинными метками."""

        targets_one_hot = self.one_hot_encode(targets)

        axes = tuple(range(1, len(preds.shape) - 1))

        numerator = 2.0 * torch.sum(preds * targets_one_hot, dim=axes)
        denominator = torch.sum(preds**2 + targets_one_hot**2, dim=axes)

        dice_score = (numerator + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice_score.mean()
