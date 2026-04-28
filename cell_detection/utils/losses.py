import torch
from torch import Tensor, nn
from torchmetrics.functional.detection import intersection_over_union

from cell_detection.data.transforms import process_yolo_preds


class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred: Tensor, target: Tensor, anchors: Tensor) -> Tensor:
        # ниже входные тензоры будут меняться на месте, склонируем
        pred = pred.clone()
        target = target.clone()

        # разделяем рамки на содержащие объекты и не содержащие;
        # значение -1 (несколько объектов попали в одну ячейку) игнорируем
        obj = target[..., 0] == 1
        no_obj = target[..., 0] == 0

        scores, pred_boxes, logits = process_yolo_preds(pred, anchors)

        # no object loss: BCE на пустые ячейки
        no_object_loss = self.bce(scores[no_obj], target[..., 0:1][no_obj])

        # object loss: целью является IoU предсказанной и истинной рамки
        ious = intersection_over_union(pred_boxes[obj], target[..., 1:5][obj]).detach()
        object_loss = self.mse(scores[obj].sigmoid(), ious * target[..., 0:1][obj])

        # box coordinate loss: log-преобразование размеров
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        pred[..., 1:3] = pred[..., 1:3].sigmoid()
        target[..., 3:5] = torch.log(1e-6 + target[..., 3:5] / anchors)
        box_loss = self.mse(pred[..., 1:5][obj], target[..., 1:5][obj])

        # class loss
        class_loss = self.cross_entropy(logits[obj], target[..., 5][obj].long() - 1)

        return box_loss + object_loss + no_object_loss + class_loss
