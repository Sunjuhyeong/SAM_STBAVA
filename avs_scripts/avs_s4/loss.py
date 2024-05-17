import torch
import torch.nn as nn

def F1_IoU_BCELoss(pred_masks, first_gt_mask):
    """
    binary cross entropy loss (iou loss) of the first frame for single sound source segmentation

    Args:
    pred_masks: predicted masks for a batch of data, shape:[bs*5, 1, 224, 224]
    first_gt_mask: ground truth mask of the first frame, shape: [bs, 1, 1, 224, 224]
    """
    assert len(pred_masks.shape) == 4
    pred_masks = torch.sigmoid(pred_masks) # [bs*5, 1, 224, 224]
    indices = torch.tensor(list(range(0, len(pred_masks), 5)))
    indices = indices.cuda()

    first_pred = torch.index_select(pred_masks, dim=0, index=indices) # [bs, 1, 224, 224]
    assert first_pred.requires_grad == True, "Error when indexing predited masks"
    if len(first_gt_mask.shape) == 5:
        first_gt_mask = first_gt_mask.squeeze(1) # [bs, 1, 224, 224]
    first_bce_loss = nn.BCELoss()(first_pred, first_gt_mask)

    return first_bce_loss


def IoULoss(pred_masks, first_gt_mask):
    """
    loss for single sound source segmentation

    Args:
    pred_masks: predicted masks for a batch of data, shape:[bs*5, 1, 224, 224]
    first_gt_mask: ground truth mask of the first frame, shape: [bs, 1, 1, 224, 224]
    """
    total_loss = 0
    f1_iou_loss = F1_IoU_BCELoss(pred_masks, first_gt_mask)
    total_loss += f1_iou_loss
    loss_dict = {}
    loss_dict['iou_loss'] = f1_iou_loss.item()

    return total_loss, loss_dict
