import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW
from engine.scheduler_function import scheduler_exp
import os
from huggingface_hub import hf_hub_download

__all__ = ['scheduler_exp']

def get_model_target_parameters(model, target_name):
    return [param for name, param in model.named_parameters() if target_name in name]

def get_model_rest_parameters(model, target_names):
    return [param for name, param in model.named_parameters() if not any(target_name in name for target_name in target_names)]

def init_optimizer_scheduler(config, model, **kwargs):
    if config['lr_groups'] is not None:
        target_names = []
        for group in config['lr_groups']:
            target_names.append(group['params'])
            group['params'] = get_model_target_parameters(model, group['params'])
            group['lr'] = config['lr'] * group['lr_scale']
        config['lr_groups'].append({'params': get_model_rest_parameters(model, target_names), 'lr': config['lr']})
    else:
        config['lr_groups'] = [{'params': model.parameters(), 'lr': config['lr']}]

    optimizer = AdamW(config['lr_groups'], lr=config['lr'], betas=(0.9, 0.999), weight_decay=0.01)
    lr_lambda_intializer = config['lr_lambda_intializer']
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_intializer(**kwargs))

    if 'resumed_from' in config and config['resumed_from'] is not None:
        if os.path.exists(config['resumed_from']):
            print(f"Loading resumed checkpoint from local path: {config['resumed_from']}")
            checkpoint = torch.load(config['resumed_from'], map_location='cpu')
        else:
            try:
                ckpt_path = hf_hub_download(
                    repo_id=f"venkatsubra/{config['resumed_from']}", # Access files at the public repo
                    filename="model.pth",
                )
                checkpoint = torch.load(ckpt_path, map_location='cpu')
            except Exception as e:
                raise FileNotFoundError(f"Resumed model not found at {config['resumed_from']} locally or on Hugging Face Hub at repo: venkatsubra/{config['resumed_from']}. Error: {str(e)}")
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    
        
        
        

    return optimizer, scheduler