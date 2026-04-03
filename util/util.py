from util.gen_focal_stack import gen_focal_stack, get_mixed_power_exp_psf_type_and_p
import torch
import random

def get_focal_stack_and_fd_list(
    rgb,
    depth,
    depth_valid_mask,
    K,
    config,
    dataset_sampled_from,
    training,
    dataset_focal_stack=None,
    dataset_fd_list=None,
):
    """
    Determine focal stack and focal distance (FD) list for a sample.

    Modes of operation:
    -------------------
    1. Dataset-provided:
        - Use `dataset_focal_stack` and `dataset_fd_list` directly.

    2. Depth-dependent FD list:
        - Generate FD list from ground-truth depth.
        - Each sample may have a different FD list.
        - Synthesize focal stack accordingly.

    3. Fixed FD list:
        - Use predefined FD list from config.
        - Synthesize focal stack accordingly.

    Args:
    -----
    rgb : torch.Tensor
        Shape: [B, 3, H, W]
        Input RGB image.

    depth : torch.Tensor
        Shape: [B, 1, H, W]
        Ground-truth depth map.

    depth_valid_mask : torch.Tensor
        Shape: [B, 1, H, W]
        Mask indicating valid depth values.

    K : torch.Tensor
        Shape: [B, 3, 3]
        Camera intrinsic matrix.

    config : dict
        Configuration dictionary containing:
            val_loader_config:
                - depth_dependent_fd_list (bool): flag to determine if FD list should be picked based on ground-truth depth values (different for each sample)
                - fd_list (List[float])

    dataset_sampled_from : str
        Name of dataset (used for dataset-specific logic).

    training : bool
        Whether model is in training mode.

    dataset_focal_stack : torch.Tensor, optional
        Shape: [B, N, 3, H, W]
        Precomputed focal stack from dataset.

    dataset_fd_list : torch.Tensor, optional
        Shape: [B, N]
        Precomputed focal distances from dataset.

        Note:
        -----
        Must be provided together with `dataset_focal_stack` if used.

    Returns:
    --------
    focal_stack : torch.Tensor
        Shape: [B, N, 3, H, W]

    fd_list : torch.Tensor
        Shape: [B, N]
    
    canonicalized_depth : torch.Tensor or None
        Shape: [B, 1, H, W]
        Canonicalized depth map if training with canonical depth, else (evaluation mode) returns None.
    """
    # assert that both dataset_focal_stack and dataset_fd_list are provided together if at all
    if dataset_focal_stack is not None or dataset_fd_list is not None:
        assert dataset_focal_stack is not None and dataset_fd_list is not None, "Both dataset_focal_stack and dataset_fd_list should be provided together if at all"


    canonicalized_depth = None
    if training:

        if dataset_focal_stack is None and dataset_fd_list is None:
            if config['train_random_fnumber_1_0_1_4_2_0_2_8_4_0']:
                fnumber_options = [1.0, 1.4, 2.0, 2.8, 4.0]
                chosen_fnumber = random.choice(fnumber_options)
            else:
                raise ValueError("Currently, training with random fnumber is the only supported mode when not using dataset-provided focal stack and fd list. Set train_random_fnumber_1_0_1_4_2_0_2_8_4_0 to True in config to enable this mode.")
            
            fd_list_params = {
                'power_inverse_sampling': config.get('train_power_inverse_sampling', False),
            }

            psf_type = config.get('psf_type', None)
            p = None
            if psf_type == 'mixed_power_exp_psf':
                p_distribution_type = config['mixed_power_exp_psf_p_distribution_type']
                psf_type, p = get_mixed_power_exp_psf_type_and_p(p_distribution_type)
            else:
                raise ValueError(f"Unsupported psf_type {psf_type} for training. Currently, only mixed_power_exp_psf is supported for training.")

            focal_stack, fd_list, _ = gen_focal_stack(depth, rgb, K, fnumber=chosen_fnumber, N=5, fd_list_params=fd_list_params, depth_valid_mask=depth_valid_mask, psf_type=psf_type, p=p)
            
        else:
            # Finetuning on DDFF provided focal stacks
            focal_stack, fd_list = dataset_focal_stack, dataset_fd_list

        if config['training_with_canonical_depth']:
            # Pass canonical FD list to the model and give canonical depth as the target
            width = torch.tensor(depth.shape[3]).to(depth.device)
            focal_length = torch.max(K[:,0,0], K[:,1,1])
            depth_scaling_factor = (width.view(-1, 1, 1, 1).expand_as(depth) / focal_length.view(-1, 1, 1, 1).expand_as(depth))

            # Overwrite the target depth to be canonicalized depth during training
            canonicalized_depth = depth * depth_scaling_factor

            # Width and depth scaling factor are the same except for shape ((B, 1, H, W) vs (B, N))
            fd_list_scaling_factor = (width.view(-1, 1).expand_as(fd_list) / focal_length.view(-1, 1).expand_as(fd_list))

            fd_list = fd_list * fd_list_scaling_factor  # [B, N]


    else:
        if dataset_focal_stack is None and dataset_fd_list is None:
            # Synthetically generate focus stack
            if (not config['val_loader_config'].get('depth_dependent_fd_list', False)) and config['val_loader_config'].get('fd_list') is None:
                raise ValueError(f"fd_list must be specified for {dataset_sampled_from} evaluation in val_config, or depth_dependent_fd_list must be True")

            # Set flags for focal stack generation based on config
            fd_list_params = {
                'depth_dependent': config['val_loader_config'].get('depth_dependent_fd_list', False),
                'fd_list': config['val_loader_config'].get('fd_list', None),
                }

            focal_stack, fd_list, _ = gen_focal_stack(depth, rgb, K, fnumber=config['val_loader_config']['fnumber'], N=config['val_loader_config']['focal_stack_size'], fd_list_params=fd_list_params, psf_type='gauss', p=None) # Only generate focal stack using GaussPSF when evaluation
        else:
            # Pull focal stack and fd list from dataset
            focal_stack, fd_list = dataset_focal_stack, dataset_fd_list
    
        if focal_stack is None or fd_list is None:
            raise ValueError("fd_list or focal_stack is None. Check the focal stack generation step for errors.")

        if config['training_with_canonical_depth']:
            # Convert fd_list to canonical fd_list
            width = torch.tensor(focal_stack.shape[-1]).to(depth.device).expand_as(fd_list)
            focal_length = torch.max(K[:,0,0], K[:,1,1]).expand_as(fd_list)  # adjust focal length for resizing
            scaling_factor = (width / focal_length)
            fd_list = fd_list * scaling_factor
    

    return focal_stack, fd_list, canonicalized_depth

def run_model_on_sample(model, focal_stack, fd_list, evaluating_model_trained_with_canonical_depth, K):
    """Runs the model on the provided focal_stack and fd_list, handling canonicalization"""

    pd = model(focal_stack, fd_list)

    # During training we keep the predicted depth in canonicalized space if using canonical depth
    if evaluating_model_trained_with_canonical_depth:
        # Revert canonicalization of predicted depth
        width = torch.tensor(focal_stack.shape[-1]).to(focal_stack.device).expand_as(pd)

        focal_length = torch.max(K[:,0,0], K[:,1,1]).expand_as(pd)
        scaling_factor = (width / focal_length)

        pd = pd / scaling_factor
        
        width_fd = torch.tensor(focal_stack.shape[-1]).to(focal_stack.device).expand_as(fd_list)
        focal_length_fd = torch.max(K[:,0,0], K[:,1,1]).expand_as(fd_list)  # adjust focal length for resizing
        fd_scaling_factor = (width_fd / focal_length_fd)
        fd_list = fd_list / fd_scaling_factor

    return pd