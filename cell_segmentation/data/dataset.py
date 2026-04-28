from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset


class YeastDataset(Dataset):
    def __init__(self, subset_dir: Path) -> None:
        super().__init__()
        self.subset_dir = subset_dir
        self.items = list((self.subset_dir / "inputs").glob("*.pt"))

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image_path = self.items[index]
        image = torch.load(image_path, weights_only=True).unsqueeze(0)  # (W, H) -> (1, W, H)
        # 0 класс - фон, поэтому прибавляем 1 к меткам классов
        labels = torch.load(self.subset_dir / "classes" / image_path.name, weights_only=True) + 1
        masks = torch.load(self.subset_dir / "instances" / image_path.name, weights_only=True)

        # формируем маску: тензор размера W x H, где для каждого пиксела указан класс
        masks = ((masks * labels.view(-1, 1, 1)).sum(dim=0)).long()
        return image, masks

    def __len__(self) -> int:
        return len(self.items)
