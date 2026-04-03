import torch
from torch import nn
from mmengine.registry import MODELS

@MODELS.register_module()
class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, supervise_in_disparity, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        if supervise_in_disparity:
            print(f"SiLogLoss does not depend on supervise_in_disparity, but received supervise_in_disparity={supervise_in_disparity}")
            pred = 1.0 / (pred + 1e-8)
            target = 1.0 / (target + 1e-8)
        loss = 0.0
        if valid_mask.sum() == 0:
            return pred.sum() * 0.0
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss += torch.sqrt(torch.pow(diff_log, 2).mean() -
                    self.lambd * torch.pow(diff_log.mean(), 2))

        if loss == 0.0:
            return pred.sum() * 0.0
        return loss