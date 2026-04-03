import os
import logging
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from dataset import init_dataloader
from util.config import get_config
from util.log import init_wandb, broadcast_wandb_dir, setup_logger
from util.dist import setup_distributed
from util.init import init_model
from util.metric import MetricTracker
from tqdm import tqdm
import zipfile

from util.train import to_cuda
from util.util import run_model_on_sample, get_focal_stack_and_fd_list

def main(): 
    config = get_config('config/val.py')
    setup_distributed()
    cudnn.enabled = True
    cudnn.benchmark = True

    local_rank = int(os.environ["LOCAL_RANK"])
    config['local_rank'] = local_rank
    print(f"config bs: {config['bs']}, local_rank: {local_rank}")
    assert config['bs'] == 1, "Batch size for validation must be 1"
    assert local_rank == 0, "Validation currently only supports single GPU"

    init_wandb(config)
    broadcast_wandb_dir(config)
    logger = setup_logger(config['log_dir'], local_rank)

    load_model_for_validation(config, logger)

    assert config['val_dataset'] == 'Zedd' and config['mode'] == 'test', "This script is specifically for testing on the Zedd dataset. Please set val_dataset to 'Zedd' and mode to 'test' in the config/val.py file."


def load_model_for_validation(config, logger):
    model = init_model(config)

    config['val_loader_config'] = config['val_loader_config_options'][config['val_loader_config_choice']]
    config['dataset']['val']['params']['args'].update({'val_loader_config': config['val_loader_config']})

    _, test_set = init_dataloader(config, 'val')

    save_zedd_outputs(model, config, test_set)

@torch.no_grad()
def save_zedd_outputs(model, config, test_set):
    model.eval()
    output_dir_base = config['zedd_output_dir']
    assert output_dir_base is not None, "Please specify an output directory for saving Zedd outputs in the config file under 'zedd_output_dir'"

    # Make a subdirectory for this model's outputs using the experiment name and encoder type
    output_dir = os.path.join(output_dir_base, f'{config["experiment_name"]}_{config["encoder"]}')
    os.makedirs(output_dir, exist_ok=True)
    pbar = tqdm(
        enumerate(test_set),
        total=len(test_set),
        desc="Validating",
        dynamic_ncols=True,
        leave=True
    )
    for i, sample in pbar:
        sample = test_set[i]
        sample = to_cuda(sample)

        # Valid and retrieved from Zedd test
        focal_stack_input = sample.get('focal_stack', None)
        fd_list_input = sample.get('fd_list', None)
        K = sample['K'].unsqueeze(0) # batch_size x 3 x 3

        # invalid and used as placeholders for shapes and device info
        rgb, depth, valid_mask = sample['rgb'], sample['gt'], sample['valid_mask']
        
        if focal_stack_input is not None and focal_stack_input.dim() == 4:
            focal_stack_input = focal_stack_input.unsqueeze(0)
        if fd_list_input is not None and fd_list_input.dim() == 1:
            fd_list_input = fd_list_input.unsqueeze(0)

        
        # Focal stack is a torch tensor of shape [B, N, 3, H, W] where B is the batch size, N is the number of focal planes,
        # and H, W are the height and width of the images. Each focal plane is a blurred version of the original image.

        # Instance method to get dataset name from dataset object
        dataset_sampled_from = 'Zedd'
        training = False

        focal_stack, fd_list, _ = get_focal_stack_and_fd_list(
            rgb=rgb, depth=depth, depth_valid_mask=valid_mask, K=K, config=config, dataset_sampled_from=dataset_sampled_from, training=training,
            dataset_focal_stack=focal_stack_input, dataset_fd_list=fd_list_input)

        pd = run_model_on_sample(model=model, focal_stack=focal_stack, fd_list=fd_list, evaluating_model_trained_with_canonical_depth=(not training) and config['training_with_canonical_depth'], K=K)
        
        pd_raw = pd.cpu().squeeze().numpy()

        # save pd_raw to disk as a .npy file
        
        output_path = os.path.join(output_dir, f'zedd_output_{(i+1):04d}.npy')
        np.save(output_path, pd_raw)

    zip_path = os.path.join(output_dir, "zedd_outputs.zip")

    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(output_dir):
            if fname.endswith(".npy"):
                fpath = os.path.join(output_dir, fname)
                zf.write(fpath, arcname=fname)  # store without full path
    
    for fname in os.listdir(output_dir):
        if fname.endswith(".npy"):
            os.remove(os.path.join(output_dir, fname))

    print(f"Zipped outputs to {zip_path} and deleted all .npy files from {output_dir}")

if __name__ == '__main__':
    main()