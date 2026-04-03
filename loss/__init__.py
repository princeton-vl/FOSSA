from .silog import SiLogLoss
from .gradmatchingscaleloss import GradMatchingScaleLoss
from torch import nn
from mmengine.registry import MODELS

__all__ = ['SiLogLoss', 'GradMatchingScaleLoss', 'CombinedLoss']

@MODELS.register_module()
class CombinedLoss(nn.Module):
    def __init__(self, losses):
        super().__init__()
        self.losses = nn.ModuleDict()
        self.loss_cfgs = losses
        for loss_dict in losses:
            loss = MODELS.build(loss_dict['loss_config'])
            self.losses[loss.__class__.__name__] = loss
            # Store input_dict and weight for each loss
            loss_dict['instance'] = loss
        

    # Validate that all required inputs for the loss function are provided
    def validate_inputs(self, required_loss_inputs, provided_inputs):
        this_loss_inputs = {}
        
        for input_type in required_loss_inputs:
            if input_type not in provided_inputs or provided_inputs[input_type] is None:
                raise ValueError(f'Missing input type: {input_type}')
            this_loss_inputs[input_type] = provided_inputs[input_type]
        
                
        return this_loss_inputs

    def forward(self, supervise_in_disparity, pred=None, target=None, valid_mask=None):
        loss_dict = {}
        loss_dict['total'] = 0
        inputs = {
            'pred': pred,
            'target': target,
            'valid_mask': valid_mask,
            'supervise_in_disparity': supervise_in_disparity
        }
        for loss_dict_cfg in self.loss_cfgs:
            
            loss = loss_dict_cfg['instance']
            required_loss_inputs = loss_dict_cfg['loss_input']
            weight = loss_dict_cfg['loss_weight']

            loss_inputs = self.validate_inputs(required_loss_inputs, inputs)
            
            loss_value = loss(**loss_inputs) # Unpack loss inputs (pred, target, valid_mask) into keyword args
            name = loss.__class__.__name__
            loss_dict[name] = loss_value
            loss_dict['total'] += loss_value * weight
        return loss_dict['total']

def init_criterion(config):
    return MODELS.build(config['criterion']) # looks up the class with "type" field in the criterion