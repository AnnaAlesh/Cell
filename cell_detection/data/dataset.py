from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.ops.boxes import box_convert

from cell_detection.data.transforms import boxes_to_cells


class YeastDetectionDataset(Dataset):
    def __init__(
        self,
        subset_dir: Path,
        anchors: list[list[int]],
        image_size: int = 256,
        grid_size: int = 8,
        ignore_iou_thresh: float = 0.5,
    ) -> None:
        super().__init__()
        self.subset_dir = Path(subset_dir)
        self.items = list((self.subset_dir / "inputs").glob("*.pt"))
        self.image_size = image_size
        self.grid_size = grid_size
        self.ignore_iou_thresh = ignore_iou_thresh
        self.rescaled_anchors = (
            torch.tensor(anchors, dtype=torch.float) / image_size * grid_size
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image_path = self.items[index]
        image = torch.load(image_path, weights_only=True).unsqueeze(0)
        # 0 класс — фон, прибавляем 1 к меткам объектов
        classes = torch.load(
            self.subset_dir / "classes" / image_path.name, weights_only=True
        ) + 1
        boxes = torch.load(
            self.subset_dir / "bounding_boxes" / image_path.name, weights_only=True
        )
        boxes = box_convert(boxes, "xyxy", "cxcywh") / self.image_size

        targets = boxes_to_cells(
            boxes, classes, self.rescaled_anchors, self.grid_size, self.ignore_iou_thresh
        )
        return image, targets
