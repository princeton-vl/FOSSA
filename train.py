import os
import logging
import wandb 
import torch
import torch.backends.cudnn as cudnn
import torchvision
from loss import init_criterion
from dataset import init_dataloader
from engine import init_optimizer_scheduler
from util.config import get_config
from util.train import to_cuda
from util.log import init_wandb, broadcast_wandb_dir, setup_logger, wandb_log_scalars
from util.val import validate
from util.dist import setup_distributed
from util.init import init_model
from util.vis import log_images
from util.util import run_model_on_sample, get_focal_stack_and_fd_list
import torch.distributed as dist
import random

def main(): 
    config = get_config('config/train.py')
    setup_distributed()
    cudnn.enabled = True
    cudnn.benchmark = True

    local_rank = int(os.environ["LOCAL_RANK"])
    config['local_rank'] = local_rank

    if local_rank == 0:
        init_wandb(config)
    broadcast_wandb_dir(config)
    logger = setup_logger(config['log_dir'], local_rank)

    train(config, logger)

def train(config, logger):
    model = init_model(config)
    criterion = init_criterion(config)
    train_loader, train_subset = init_dataloader(config, 'train')

    config['val_loader_config'] = config['val_loader_config_options'][config['val_loader_config_choice']]
    config['dataset']['val']['params']['args'].update({'val_loader_config': config['val_loader_config']}) # Necessary for InfinigenDefocus to get validation config options before it is instantiated

    val_loader, val_subset = init_dataloader(config, 'val')
    optimizer, scheduler = init_optimizer_scheduler(config, model, total_steps=config['lr_decay_max_epochs'] * len(train_loader))

    step = 0

    logger.info(f'Training with lr_decay_max_epochs: {config["lr_decay_max_epochs"]} epochs and training_epochs: {config["training_epochs"]}, {len(train_loader)} steps per epoch, {config["bs"] * torch.cuda.device_count()} samples per step')
    
    for epoch in range(config['lr_decay_max_epochs']):
        train_loader.sampler.set_epoch(epoch + 1)
        model.train()

        num_steps_this_epoch = 0 # Stop epochs early for debugging to log
        for batch_idx, sample in enumerate(train_loader):
            optimizer.zero_grad()
            sample = to_cuda(sample)

            depth = sample['gt'] # (B, 1, H, W)
            rgb = sample['rgb'] # batch_size x 3 x H x W tensor
            K = sample['K'] # batch_size x 3 x 3
            valid_mask = sample['valid_mask'].unsqueeze(1) # (B, 1, H, W)
            focal_stack = sample.get('focal_stack', None) # (B, N, 3, H, W)
            fd_list = sample.get('fd_list', None) # (B, N)

            if config['training_with_canonical_depth']:
                focal_stack, fd_list, canonicalized_depth = get_focal_stack_and_fd_list(
            rgb=rgb, depth=depth, K=K, depth_valid_mask=valid_mask, config=config, dataset_sampled_from=config['train_dataset'], training=True,
            dataset_focal_stack=focal_stack, dataset_fd_list=fd_list)
        
                pd = run_model_on_sample(model=model, focal_stack=focal_stack, fd_list=fd_list, evaluating_model_trained_with_canonical_depth=False, K=K)

                depth = canonicalized_depth
            else:
                raise ValueError("training_with_canonical_depth=False is not currently supported for training, set training_with_canonical_depth=True")
            assert pd.shape == depth.shape, f"Predicted depth shape {pd.shape} does not match ground truth depth shape {depth.shape}"

            if config['train_mask_used'] == 'depth_invalid_mask':
                # Use the depth invalid mask provided in the dataloader
                loss = criterion(supervise_in_disparity=config['supervise_in_disparity'], pred=pd, target=depth, valid_mask=valid_mask)
            else:
                raise ValueError(f"Unknown train_mask_used: {config['train_mask_used']}")
            


            if torch.isnan(loss):
                print(f"Warning: Loss is NaN at epoch {epoch}, step {step}. pd: {pd}, depth: {depth}, valid_mask: {valid_mask} fd: {fd_list} focal_stack: {focal_stack}")
                raise ValueError("Loss is NaN, stopping training.")

            loss.backward()
            optimizer.step()
            scheduler.step()

            if config['local_rank'] == 0:
                wandb_log_scalars({'loss': loss.item()}, step, 'train')
            step += 1
            num_steps_this_epoch += 1

            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")
        
        dist.barrier()

        print(f"epoch:{epoch} config['log_image_interval']:{config['log_image_interval']} config['local_rank']:{config['local_rank']}")
        if config['local_rank'] == 0 and epoch % config['log_image_interval'] == 0:
            print(f"I am Logging images for epoch {epoch} on rank {config['local_rank']}")
            log_images(model, config, train_subset, 'train', step, first_epoch=(epoch==0))
        else:
            print(f"Skipping logging images for epoch {epoch} on rank {config['local_rank']}")

        dist.barrier()
        if config['local_rank'] == 0 and epoch % config['validation_interval'] == 0:
            validate(model, config, val_loader, val_subset, step, first_epoch=(epoch==0))
        dist.barrier()
        if config['local_rank'] == 0 and epoch % config['save_checkpoint_interval'] == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
            }

            if epoch == config['training_epochs']: # Save a final checkpoint at the end of training
                logger.info(f"Reached epoch {config['training_epochs']}, saving checkpoint and stopping training.")
                save_path = os.path.join(config['log_dir'], 'checkpoints', f'{epoch}_final.pth')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(checkpoint, save_path)
                break
            else:
                save_path = os.path.join(config['log_dir'], 'checkpoints', f'{epoch}.pth')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(checkpoint, save_path)
        
        dist.barrier()

        # Refresh indices for multi-dataset loaders or DDFF12Loader_Train
        train_loader.dataset.refresh_indices()
    

if __name__ == '__main__':
    main()