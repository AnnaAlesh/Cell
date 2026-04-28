from pathlib import Path

import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

from cell_detection.data.dataset import YeastDetectionDataset


class DetectionDataModule(L.LightningDataModule):
    def __init__(
        self,
        datadir: Path,
        anchors: list[list[int]],
        image_size: int = 256,
        grid_size: int = 8,
        ignore_iou_thresh: float = 0.5,
        batch_size: int = 32,
        train_subdir: str = "train",
        val_subdir: str = "val",
        test_subdir: str = "test",
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.anchors = anchors
        self.image_size = image_size
        self.grid_size = grid_size
        self.ignore_iou_thresh = ignore_iou_thresh
        datadir = Path(datadir)
        self.train_dir = datadir / train_subdir
        self.val_dir = datadir / val_subdir
        self.test_dir = datadir / test_subdir

    def _build(self, subset_dir: Path) -> YeastDetectionDataset:
        return YeastDetectionDataset(
            subset_dir=subset_dir,
            anchors=self.anchors,
            image_size=self.image_size,
            grid_size=self.grid_size,
            ignore_iou_thresh=self.ignore_iou_thresh,
        )

    def setup(self, stage: str) -> None:
        if stage in ("fit", "validate"):
            self.val_dataset = self._build(self.val_dir)
        if stage == "fit":
            self.train_dataset = self._build(self.train_dir)
        elif stage == "test":
            self.test_dataset = self._build(self.test_dir)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
