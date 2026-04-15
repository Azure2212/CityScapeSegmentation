import torch


def pixel_accuracy(outputs: torch.Tensor, masks: torch.Tensor) -> float:
    """Fraction of correctly classified pixels.

    Args:
        outputs: (B, C, H, W) raw logits
        masks:   (B, H, W)    integer class labels
    """
    preds = torch.argmax(outputs, dim=1)
    return (preds == masks).sum().item() / masks.numel()


def iou_score(
    outputs: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int = 20,
) -> tuple[float, float, dict]:
    """Mean IoU, Frequency-Weighted IoU, and per-class IoU for the batch.

    Mean IoU treats every class equally, which can be misleading when class
    sizes differ wildly (e.g. a tiny missed class pulls the average down as
    much as a large missed class).  Frequency-Weighted IoU (FWIoU) addresses
    this by weighting each class's IoU by the fraction of ground-truth pixels
    it occupies, so large classes matter more.

    Args:
        outputs:     (B, C, H, W) raw logits
        masks:       (B, H, W)    integer class labels
        num_classes: total number of classes

    Returns:
        (mean_iou, fw_iou, cls_ious) where cls_ious is {class_id: iou}
        for every class that had at least one pixel in union.
    """
    preds = torch.argmax(outputs, dim=1)
    ious        = []
    gt_counts   = []
    cls_ious    = {}   # {class_id: iou} for classes present in this batch
    for cls in range(num_classes):
        pred_inds   = preds == cls
        target_inds = masks == cls
        intersection = (pred_inds & target_inds).sum().item()
        union        = (pred_inds | target_inds).sum().item()
        gt_count     = target_inds.sum().item()
        if union > 0:
            cls_iou = intersection / union
            ious.append(cls_iou)
            gt_counts.append(gt_count)
            cls_ious[cls] = cls_iou

    if not ious:
        return 0.0, 0.0, {}

    mean_iou = sum(ious) / len(ious)

    total_gt = sum(gt_counts)
    if total_gt > 0:
        fw_iou = sum(iou * cnt for iou, cnt in zip(ious, gt_counts)) / total_gt
    else:
        fw_iou = mean_iou

    return mean_iou, fw_iou, cls_ious


def dice_score(outputs: torch.Tensor, masks: torch.Tensor, num_classes: int = 20, smooth: float = 1e-6) -> float:
    """Mean Dice score across classes present in the batch.

    Args:
        outputs:     (B, C, H, W) raw logits
        masks:       (B, H, W)    integer class labels
        num_classes: total number of classes
    """
    preds = torch.argmax(outputs, dim=1)
    scores = []
    for cls in range(num_classes):
        pred_inds   = preds == cls
        target_inds = masks == cls
        intersection = (pred_inds & target_inds).sum().item()
        union        = pred_inds.sum().item() + target_inds.sum().item()
        if union > 0:
            scores.append((2 * intersection + smooth) / (union + smooth))
    return sum(scores) / len(scores) if scores else 0.0
