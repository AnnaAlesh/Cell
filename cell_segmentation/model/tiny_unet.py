"""Задание 2: компактная архитектура (≤ 50k параметров).

Сейчас здесь просто реэкспорт базовой UNet из
``cell_segmentation.model.modules`` с ``channel_sizes=[8, 16, 32, 64]``
(см. ``config/train_tiny.yaml``) Ваша задача — переписать архитектуру в этом
файле так, чтобы количество параметров было ≤ 50k без существенной потери качества.

Подсказка: попробуйте заменить в модуле ``Down``
комбинацию ``MaxPool2d + DoubleConv`` на одну dilated conv с увеличенным stride:

    nn.Conv2d(in_channels, out_channels, kernel_size=3,
              stride=2, padding=2, dilation=2, bias=False)

В остальном — полная свобода творчества.
"""
from cell_segmentation.model.modules import UNet  # noqa: F401
