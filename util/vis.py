import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm, Normalize
import torch
import random
import numpy as np
from util.log import wandb_log_images, wandb_log_focal_stack, wandb_log_coc_map
from util.train import to_cuda
from util.util import run_model_on_sample, get_focal_stack_and_fd_list


def visualize_rgb(rgb):
    def denormalize_image(image):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        return image * std + mean
    rgb = denormalize_image(rgb)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def get_normalizer_and_colorbar_ticks(depth, valid_mask):

    # Calculate depth range
    if valid_mask is None:
        depth_valid = depth
    else:
        depth_valid = depth[valid_mask != 0.0]
    depth_valid = depth_valid[np.isfinite(depth_valid)]
    if depth_valid.size == 0:
        depth_min, depth_max = 0.25, 0.75
        depth_normalizer = Normalize(vmin=depth_min, vmax=depth_max)
        colorbar_ticks = np.linspace(depth_min, depth_max, 5)
        return depth_normalizer, colorbar_ticks

    depth_positive = depth_valid[depth_valid > 0]
    num_ticks = 10
    if depth_positive.size > 0:
        # depth_min and max should be 2 and 98 percentile of depth_positive to avoid outliers dominating the colorbar
        depth_min = float(np.percentile(depth_positive, 2))
        depth_max = float(np.percentile(depth_positive, 98))
        if depth_min == depth_max:
            depth_max = depth_min * 1.01
        depth_normalizer = LogNorm(vmin=depth_min, vmax=depth_max)
        colorbar_ticks = np.logspace(np.log10(depth_min), np.log10(depth_max), num_ticks)
    else:
        depth_min = float(depth_valid.min())
        depth_max = float(depth_valid.max())
        if depth_min == depth_max:
            depth_max = depth_min + 1.0
        depth_normalizer = Normalize(vmin=depth_min, vmax=depth_max)
        colorbar_ticks = np.linspace(depth_min, depth_max, num_ticks)

    return depth_normalizer, colorbar_ticks # return vmin and vmax to make gt and pred have the same normalization

@torch.no_grad()
def log_images(model, config, subset, split, step, first_epoch=False):
    model.eval()
    for i in range(min(config['log_first_n_samples'], len(subset))):  # Fixed logging of first "log_first_n_samples" images in the subset
        
        sample = subset[i]
        sample = to_cuda(sample)

        depth = sample['gt'].unsqueeze(0) # batch_size x 1 x H x W tensor
        rgb = sample['rgb'].unsqueeze(0) # batch_size x 3 x H x W tensor
        K = sample['K'].unsqueeze(0) # batch_size x 3 x 3
        valid_mask = torch.as_tensor(sample['valid_mask']).unsqueeze(0).to(depth.device) # batch_size x 1 x H x W tensor
        focal_stack_input = sample.get('focal_stack', None)
        fd_list_input = sample.get('fd_list', None)
        if focal_stack_input is not None and focal_stack_input.dim() == 4:
            focal_stack_input = focal_stack_input.unsqueeze(0)
        if fd_list_input is not None and fd_list_input.dim() == 1:
            fd_list_input = fd_list_input.unsqueeze(0)
        
        # Focal stack is a torch tensor of shape [B, N, 3, H, W] where B is the batch size, N is the number of focal planes,
        # and H, W are the height and width of the images. Each focal plane is a blurred version of the original image.

        # Instance method to get dataset name from dataset object
        dataset_sampled_from = config['instantiated_train_dataset_object'].get_dataset_name(i) if split == 'train' else config['val_loader_config']['dataset_name']
        training = True if split == 'train' else False

        focal_stack, fd_list, _ = get_focal_stack_and_fd_list(
            rgb=rgb, depth=depth, depth_valid_mask=valid_mask, K=K, config=config, dataset_sampled_from=dataset_sampled_from, training=training,
            dataset_focal_stack=focal_stack_input, dataset_fd_list=fd_list_input)

        pd = run_model_on_sample(model=model, focal_stack=focal_stack, fd_list=fd_list, evaluating_model_trained_with_canonical_depth=(not training) and config['training_with_canonical_depth'], K=K)
        
        valid_mask = valid_mask.squeeze(0).cpu().numpy()
        rgb = rgb.squeeze(0)  # remove batch dimension for visualization
        depth = depth.squeeze(0) # remove batch dimension for visualization
        im_np = visualize_rgb(rgb.cpu().permute(1, 2, 0).numpy())
        gt_raw = depth.cpu().squeeze().numpy()
        normalizer, colorbar_ticks = get_normalizer_and_colorbar_ticks(gt_raw, valid_mask)
        pd_raw = pd.cpu().squeeze().numpy()

        wandb_log_images(
            { 'image': im_np, 'pd': pd_raw, 'gt': gt_raw },
            colorbar_ticks=colorbar_ticks,
            depth_normalizer=normalizer,
            step=step,
            split=split,
            index=i,
            depth_valid_mask=valid_mask,
            dataset_name=dataset_sampled_from,
            
        )   

        if first_epoch:
            assert focal_stack.dim() == 5, f"Expected focal_stack to have 5 dimensions [B, N, 3, H, W], but got {focal_stack.dim()} dimensions"
            B, N, C, H, W = focal_stack.shape # at logging time, B should be 1, but we treat it as B in case this changes
            assert B == fd_list.shape[0], f"Expected fd_list to have batch size {B}, but got {fd_list.shape[0]}"
            rand_log_batch = np.random.randint(0, B)

            focal_stack_to_log = focal_stack[rand_log_batch]  # [N, 3, H, W]
            wandb_log_focal_stack(focal_stack_to_log, fd_list=fd_list[rand_log_batch], step=step, split=split, index=i, dataset_name=dataset_sampled_from)

    assert len(subset) >= config['log_first_n_samples'], f"Expected subset to have at least {config['log_first_n_samples']} images to log random images, but got {len(subset)}"
    if len(subset) == config['log_first_n_samples']:
        return

    # Sammple by interpolating evenly between log_first_n_samples and len(subset)-1
    interpolated_sample = np.linspace(config['log_first_n_samples'], len(subset)-1, config['log_another_m_samples'], dtype=int)
    interpolated_samples = list(interpolated_sample)

    for j in interpolated_samples:
        sample = subset[j]
        sample = to_cuda(sample)


        depth = sample['gt'].unsqueeze(0) # batch_size x 1 x H x W tensor
        rgb = sample['rgb'].unsqueeze(0) # batch_size x 3 x H x W tensor
        K = sample['K'].unsqueeze(0) # batch_size x 3 x 3
        valid_mask = torch.as_tensor(sample['valid_mask']).unsqueeze(0).to(depth.device) # batch_size x 1 x H x W tensor
        focal_stack_input = sample.get('focal_stack', None)
        fd_list_input = sample.get('fd_list', None)
        if focal_stack_input is not None and focal_stack_input.dim() == 4:
            focal_stack_input = focal_stack_input.unsqueeze(0)
        if fd_list_input is not None and fd_list_input.dim() == 1:
            fd_list_input = fd_list_input.unsqueeze(0)
        
        # Focal stack is a torch tensor of shape [B, N, 3, H, W] where B is the batch size, N is the number of focal planes,
        # and H, W are the height and width of the images. Each focal plane is a blurred version of the original image.

        dataset_sampled_from = config['instantiated_train_dataset_object'].get_dataset_name(j) if split == 'train' else config['val_loader_config']['dataset_name']
        training = True if split == 'train' else False

        focal_stack, fd_list, _ = get_focal_stack_and_fd_list(
            rgb=rgb, depth=depth, depth_valid_mask=valid_mask, K=K, config=config, dataset_sampled_from=dataset_sampled_from, training=training,
            dataset_focal_stack=focal_stack_input, dataset_fd_list=fd_list_input)
        
        pd = run_model_on_sample(model=model, focal_stack=focal_stack, fd_list=fd_list, evaluating_model_trained_with_canonical_depth=(not training) and config['training_with_canonical_depth'], K=K)

        valid_mask = valid_mask.squeeze(0).cpu().numpy()
        rgb = rgb.squeeze(0)  # remove batch dimension for visualization
        depth = depth.squeeze(0) # remove batch dimension for visualization
        im_np = visualize_rgb(rgb.cpu().permute(1, 2, 0).numpy())
        gt_raw = depth.cpu().squeeze().numpy()
        normalizer, colorbar_ticks = get_normalizer_and_colorbar_ticks(gt_raw, valid_mask)
        pd_raw = pd.cpu().squeeze(0).squeeze(0).numpy()

        wandb_log_images(
            { 'image': im_np, 'pd': pd_raw, 'gt': gt_raw },
            colorbar_ticks=colorbar_ticks,
            depth_normalizer=normalizer,
            step=step,
            split=split,
            index=j,
            depth_valid_mask=valid_mask,
            dataset_name=dataset_sampled_from
        )

        if first_epoch:
            assert focal_stack.dim() == 5, f"Expected focal_stack to have 5 dimensions [B, N, 3, H, W], but got {focal_stack.dim()} dimensions"
            B, N, C, H, W = focal_stack.shape # at logging time, B should be 1, but we treat it as B in case this changes
            assert B == fd_list.shape[0], f"Expected fd_list to have batch size {B}, but got {fd_list.shape[0]}"
            rand_log_batch = np.random.randint(0, B)

            focal_stack_to_log = focal_stack[rand_log_batch]  # [N, 3, H, W]
            wandb_log_focal_stack(focal_stack_to_log, fd_list=fd_list[rand_log_batch], step=step, split=split, index=j, dataset_name=dataset_sampled_from)