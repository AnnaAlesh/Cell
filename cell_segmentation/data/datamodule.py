from pathlib import Path

import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

from cell_segmentation.data.dataset import YeastDataset


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        datadir: Path,
        batch_size: int,
        train_subdir: str = "train",
        val_subdir: str = "val",
        test_subdir: str = "test",
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        datadir = Path(datadir)
        self.train_dir = datadir / train_subdir
        self.val_dir = datadir / val_subdir
        self.test_dir = datadir / test_subdir

    def setup(self, stage: str) -> None:
        if stage in ("fit", "validate"):
            self.val_dataset = YeastDataset(self.val_dir)
        if stage == "fit":
            self.train_dataset = YeastDataset(self.train_dir)
        elif stage == "test":
            self.test_dataset = YeastDataset(self.test_dir)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
