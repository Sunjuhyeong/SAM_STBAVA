import torch
import torch.nn as nn

def F5_IoU_BCELoss(pred_mask, five_gt_masks):
    """
    binary cross entropy loss (iou loss) of the total five frames for multiple sound source segmentation

    Args:
    pred_mask: predicted masks for a batch of data, shape:[bs*5, 1, 224, 224]
    five_gt_masks: ground truth mask of the total five frames, shape: [bs*5, 1, 224, 224]
    """
    assert len(pred_mask.shape) == 4
    pred_mask = torch.sigmoid(pred_mask) # [bs*5, 1, 224, 224]
    loss = nn.BCELoss()(pred_mask, five_gt_masks)

    return loss

def IoULoss(pred_masks, gt_mask):
    """
    loss for multiple sound source segmentation

    Args:
    pred_masks: predicted masks for a batch of data, shape:[bs*5, 1, 224, 224]
    gt_mask: ground truth mask of the first frame (one-shot) or five frames, shape: [bs, 1, 1, 224, 224]
    """
    total_loss = 0
    iou_loss = F5_IoU_BCELoss(pred_masks, gt_mask)
    total_loss += iou_loss

    loss_dict = {}
    loss_dict['iou_loss'] = iou_loss.item()

    return total_loss, loss_dict
