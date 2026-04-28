import torch
from torch import Tensor


def iou_wh(wh1: Tensor, wh2: Tensor) -> Tensor:
    """IoU между рамками только по их размерам (без учёта позиции)."""
    intersection_area = torch.min(wh1[..., 0], wh2[..., 0]) * torch.min(wh1[..., 1], wh2[..., 1])
    box1_area = wh1[..., 0] * wh1[..., 1]
    box2_area = wh2[..., 0] * wh2[..., 1]
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area


def boxes_to_cells(
    boxes: Tensor,
    classes: Tensor,
    rescaled_anchors: Tensor,
    grid_size: int = 8,
    ignore_iou_thresh: float = 0.5,
) -> Tensor:
    """Переводит bbox представление в клеточное.

    Возвращает тензор (n_anchors, grid_size, grid_size, 6), где последняя ось:
    [obj_presence, cx, cy, w, h, class_id]. obj_presence = 1 если в клетке есть объект,
    0 — пусто, -1 — несколько объектов попали в клетку (игнорируем при обучении).
    """
    targets = torch.zeros((len(rescaled_anchors), grid_size, grid_size, 6))

    for box, class_label in zip(boxes, classes):
        iou_anchors = iou_wh(box[2:4], rescaled_anchors / grid_size)
        anchor_indices = iou_anchors.argsort(descending=True, dim=0)
        x, y, width, height = box

        has_anchor = False
        for anchor_idx in anchor_indices:
            s = grid_size
            i, j = int(s * y), int(s * x)
            anchor_taken = targets[anchor_idx, i, j, 0]

            if not anchor_taken and not has_anchor:
                x_cell, y_cell = s * x - j, s * y - i
                width_cell, height_cell = (width * s, height * s)
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                targets[anchor_idx, i, j, 0] = 1
                targets[anchor_idx, i, j, 1:5] = box_coordinates
                targets[anchor_idx, i, j, 5] = int(class_label)
                has_anchor = True
            elif not anchor_taken and iou_anchors[anchor_idx] > ignore_iou_thresh:
                targets[anchor_idx, i, j, 0] = -1

    return targets


def process_yolo_preds(preds: Tensor, rescaled_anchors: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Преобразование сырых выходов модели в:
    1) логит наличия объекта
    2) положение рамки относительно ячейки в формате cxcywh
    3) логиты классов
    """
    rescaled_anchors = rescaled_anchors.view(1, len(rescaled_anchors), 1, 1, 2)
    box_predictions = preds[..., 1:5].clone()
    box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
    box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * rescaled_anchors

    scores = preds[..., 0:1]
    return scores, box_predictions, preds[..., 5:]


def cells_to_bboxes(
    cells: Tensor, rescaled_anchors: Tensor, is_predictions: bool = False
) -> Tensor:
    """Переводит клеточное представление в плоский список рамок (-1, 6),
    где последняя ось: [class_id, score, cx, cy, w, h] в долях от изображения.
    """
    if is_predictions:
        scores, box_predictions, logits = process_yolo_preds(cells, rescaled_anchors)
        scores = torch.sigmoid(scores)
        best_class = torch.argmax(logits, dim=-1).unsqueeze(-1) + 1
    else:
        box_predictions = cells[..., 1:5].clone()
        scores = cells[..., 0:1]
        best_class = cells[..., 5:6]

    _, _, H, W, _ = cells.shape
    range_y, range_x = torch.meshgrid(
        torch.arange(H, dtype=cells.dtype, device=cells.device),
        torch.arange(W, dtype=cells.dtype, device=cells.device),
        indexing="ij",
    )
    x = torch.cat(
        [
            best_class,
            scores,
            (box_predictions[..., 0:1] + range_x[None, None, :, :, None]) / W,
            (box_predictions[..., 1:2] + range_y[None, None, :, :, None]) / H,
            box_predictions[..., 2:3] / W,
            box_predictions[..., 3:4] / H,
        ],
        -1,
    )
    return x.view(-1, 6)
