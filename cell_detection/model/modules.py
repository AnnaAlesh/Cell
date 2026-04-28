from torch import Tensor, nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.bn(self.conv(x)))


class TinyYOLO(nn.Module):
    def __init__(
        self, num_classes: int = 2, num_anchors: int = 1, in_channels: int = 1
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.in_channels = in_channels
        self.layers = nn.Sequential(
            CNNBlock(in_channels, 16, kernel_size=3, stride=2, padding=1, dilation=2),
            CNNBlock(16, 32, kernel_size=3, stride=2, padding=1, dilation=2),
            CNNBlock(32, 64, kernel_size=3, stride=2, padding=1, dilation=2),
            CNNBlock(64, 128, kernel_size=3, stride=2, padding=1, groups=8),
            CNNBlock(128, 256, kernel_size=3, stride=1, padding=1, groups=8),
            CNNBlock(256, 256, kernel_size=3, stride=1, padding=1, groups=16),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, num_anchors * (num_classes + 5), kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        B, _, W, H = x.shape
        # Перекладываем в формат (B, A, S, S, 5+C):
        # 5 = obj + cx + cy + w + h
        x = x.view(B, self.num_anchors, self.num_classes + 5, W, H)
        x = x.permute(0, 1, 3, 4, 2)
        return x
