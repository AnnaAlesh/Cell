from torchmetrics import Accuracy, JaccardIndex, MetricCollection
from torchmetrics.segmentation import DiceScore


def create_segmentation_metrics(num_classes: int, prefix: str) -> MetricCollection:
    return MetricCollection(
        {
            f"{prefix}/Accuracy": Accuracy(
                task="multiclass", num_classes=num_classes, average="macro"
            ),
            f"{prefix}/Dice": DiceScore(
                num_classes=num_classes,
                average="micro",
                input_format="index",
                compute_on_cpu=True,
            ),
            f"{prefix}/Jaccard": JaccardIndex(
                task="multiclass",
                num_classes=num_classes,
                average="macro",
                compute_on_cpu=True,
            ),
        }
    )
