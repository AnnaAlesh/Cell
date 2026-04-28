from torch import Tensor
from torchvision.ops.boxes import box_convert, nms

from cell_detection.data.transforms import cells_to_bboxes


def predictions_to_map_format(
    preds: Tensor,
    targets: Tensor,
    rescaled_anchors: Tensor,
    nms_iou_threshold: float = 0.3,
) -> tuple[list[dict], list[dict]]:
    """Готовит вход для torchmetrics MeanAveragePrecision.

    preds: (B, A, S, S, 5+C) сырые выходы модели
    targets: (B, A, S, S, 6) истинные значения
    Возвращает два списка по B элементов с полями {boxes, scores, labels} / {boxes, labels}.
    Все рамки в формате cxcywh, в долях от изображения.
    """
    predicted_boxes: list[dict] = []
    target_boxes: list[dict] = []
    for i in range(len(targets)):
        pred_b = cells_to_bboxes(preds[i : i + 1], rescaled_anchors, is_predictions=True)
        keep = nms(
            box_convert(pred_b[:, 2:], "cxcywh", "xyxy"),
            pred_b[:, 1],
            iou_threshold=nms_iou_threshold,
        )
        pred_b = pred_b[keep]
        predicted_boxes.append(
            {"boxes": pred_b[:, 2:], "scores": pred_b[:, 1], "labels": pred_b[:, 0].long()}
        )

        true_b = cells_to_bboxes(targets[i : i + 1], rescaled_anchors)
        true_b = true_b[true_b[:, 1] == 1]
        target_boxes.append({"boxes": true_b[:, 2:], "labels": true_b[:, 0].long()})
    return predicted_boxes, target_boxes
