import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS


# Penalizes the difference in "gradients" (edges and fine details) between prediction and target at different scales
@MODELS.register_module()
class GradMatchingScaleLoss(nn.Module):
    def __init__(self, scale_level=4):
        super(GradMatchingScaleLoss, self).__init__()

        self.t_valid = 0.0001
        self.scale_level = scale_level
    
    def forward(self, supervise_in_disparity, pred, target, valid_mask):
        loss = 0.0

        if supervise_in_disparity:
            pred = 1.0 / (pred + 1e-8)
            target = 1.0 / (target + 1e-8)

        for scale in range(self.scale_level):
            down_factor = 2 ** scale

            if down_factor > 1:
                # divisible by down_factor
                pad_h = (down_factor - target.shape[-2] % down_factor) % down_factor
                pad_w = (down_factor - target.shape[-1] % down_factor) % down_factor
                padding = (0, pad_w, 0, pad_h)
                gt_padded = F.pad(target, padding, mode="replicate")
                mask_padded = F.pad(valid_mask.float(), padding, mode="replicate")

                gt_scaled = F.avg_pool2d(gt_padded*mask_padded, down_factor)
                mask_scaled = F.avg_pool2d(mask_padded, down_factor)
            else:
                gt_scaled = target
                mask_scaled = valid_mask.float()
            
            if mask_scaled.sum() == 0:
                continue

            gt_scaled[mask_scaled > 0.0] = gt_scaled[mask_scaled > 0.0] / mask_scaled[mask_scaled > 0.0]
            mask_scaled[mask_scaled > 0.0] = 1.0

            mask_u = mask_scaled[:, :, :, 1:] * mask_scaled[:, :, :, :-1]
            mask_v = mask_scaled[:, :, 1:, :] * mask_scaled[:, :, :-1, :]

            num_valid = torch.sum(mask_u, dim=[1, 2, 3]) + torch.sum(mask_v, dim=[1, 2, 3])
            
            

            if down_factor > 1:
                pred_padded = F.pad(pred, padding, mode="replicate")
                pred_scaled = F.avg_pool2d(pred_padded, down_factor)
            else:
                pred_scaled = pred

            # Hardcode i_weight to 1
            i_weight = 1 
            residual = pred_scaled - gt_scaled

            # Penalizes the model's prediction introducing new edges (or missing edges) that are not present in the ground truth
            gradu_residual = torch.abs(residual[:, :, :, 1:] - residual[:, :, :, :-1])
            gradv_residual = torch.abs(residual[:, :, 1:, :] - residual[:, :, :-1, :])

            loss_u = mask_u * gradu_residual
            loss_v = mask_v * gradv_residual

            loss_u, loss_v = torch.nan_to_num(loss_u), torch.nan_to_num(loss_v)
            i_loss_u = torch.sum(loss_u, dim=[1, 2, 3]) / (num_valid + 1e-8)
            i_loss_v = torch.sum(loss_v, dim=[1, 2, 3]) / (num_valid + 1e-8)
            i_loss = i_loss_u + i_loss_v
            loss += i_weight * i_loss.mean()
        
        if loss == 0.0:
            return pred.sum() * 0.0
        return loss