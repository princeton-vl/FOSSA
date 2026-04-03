from FOSSAModel import FOSSA
from dataset import DDFF12Loader_Val, Uniformat, InfinigenDefocus, Zedd, HAMMER
from config.validation_configs import val_loader_configs
import argparse
from huggingface_hub import snapshot_download
from pathlib import Path

parser = argparse.ArgumentParser()

# Model arguments
parser.add_argument('--encoder', type=str, required=True,
                    choices=['vits', 'vitb'], help='Choice of encoder architecture. "vits" for ViT Small, "vitb" for ViT Base. This should match the encoder used during training.')
parser.add_argument('--resumed_from', type=str, required=True, help='Path to the checkpoint to resume from. This should be a .ckpt file that was saved during training. If the checkpoint is on Hugging Face, you can specify it in the format "repo_id". If the checkpoint is local, specify the local path to the .ckpt file.')

# Validation dataset arguments
parser.add_argument('--val_loader_config_choice', type=str, default=None, help='Choice of validation dataset configuration. Should be one of the keys in val_loader_configs defined in config/validation_configs.py. Examples include "ibims", "infinigen_defocus", "diode", "zedd_test", "ddff12_val", and "hammer".')

# Logging arguments
parser.add_argument('--log_first_n_samples', type=int, default=1, help='Number of samples to log from the beginning of the validation set.')
parser.add_argument('--log_another_m_samples', type=int, default=30, help='Number of additional samples to log, evenly spaced throughout the validation set. These are in addition to the first n samples specified by log_first_n_samples.')
parser.add_argument('--project_name', type=str, default='depth_from_defocus')
parser.add_argument('--logging_turned_off', action='store_true', default=False)
parser.add_argument('--show_efficiency', action='store_true', default=False,
    help='Also measure and display model efficiency (memory + FLOPs) after validation.')
parser.add_argument('--zedd_test_output_dir', type=str, default=None, help = 'Directory to save model predictions on Zedd test set. Should be specified if val_loader_config_choice is set to a Zedd test config.')
                    


args, _unknown = parser.parse_known_args()

# derive values
project_name = args.project_name
base_log_dir = 'runs'
experiment_name = args.val_loader_config_choice + '_' + args.encoder + '_resumed_from_' + Path(args.resumed_from).stem.replace('/', '_') # create a unique experiment name based on the validation config, encoder, and checkpoint used
logging_turned_off = args.logging_turned_off
show_efficiency = args.show_efficiency
zedd_output_dir = args.zedd_test_output_dir

img_size=518
bs=1
local_rank=0

log_first_n_samples = args.log_first_n_samples
log_another_m_samples = args.log_another_m_samples
training_with_canonical_depth = True # set to true if this model was trained with canonical depth, false otherwise.
training=False

resumed_from = args.resumed_from
encoder = args.encoder


if args.encoder == 'vits':
    num_features = 64
    out_channels = [48, 96, 192, 384]
    num_layers_until_collapse=4
elif args.encoder == 'vitb':
    num_features = 128
    out_channels = [96, 192, 384, 768]
    num_layers_until_collapse=4
else:
    raise ValueError(f"Unknown encoder: {args.encoder}")

model=dict(
    target=FOSSA, 
    params=dict(
        encoder=args.encoder,
        features=num_features,
        out_channels=out_channels,
        last_layer='softplus',
        pe='fde_add',
        max_depth=1.0,
        temporal_fuse_method='mean_in_encoder',
        num_layers_until_collapse=4,
        fd_embed_function='none',
        turn_off_motion_module=False,
    ),
)



# Validation dataset configuration
val_loader_config_choice = args.val_loader_config_choice
val_loader_config_options = val_loader_configs

mode = 'val'
dataset_location = None # set to None for local datasets like HAMMER and DDFF12
if 'ibims' in args.val_loader_config_choice:
    val_dataset = 'iBims'
    val_dataset_object = Uniformat
    dataset_location = dict(
        repo_id="venkatsubra/Uniformat",
        subdir="ibims",
        source="huggingface",
        zip_filename="DefocusUniformat.zip",
        local_dir="dataset/datasets/defocus_uniformat/",

    )
    val_dataset_args = {}


elif 'infinigen_defocus' in args.val_loader_config_choice:
    val_dataset = 'InfinigenDefocus'
    val_dataset_object = InfinigenDefocus
    dataset_location = dict(
        repo_id="venkatsubra/InfinigenDefocus",
        subdir="",
        source="huggingface",
        zip_filename="InfinigenDefocus.zip",
        local_dir="dataset/datasets/infinigen_defocus",
    )
    val_dataset_args = {
    }
elif 'diode' in args.val_loader_config_choice:
    val_dataset = 'DIODE'
    val_dataset_object = Uniformat
    dataset_location = dict(
        repo_id="venkatsubra/Uniformat",
        subdir="diode",
        source="huggingface",
        zip_filename="DefocusUniformat.zip",
        local_dir="dataset/datasets/defocus_uniformat/",
    )
    val_dataset_args = {}
elif 'zedd' in args.val_loader_config_choice:
    val_dataset = 'Zedd'
    val_dataset_object = Zedd
    if 'test' in args.val_loader_config_choice:
        mode = 'test'
    dataset_location = dict(
        repo_id="venkatsubra/ZEDD",
        subdir="",
        source="huggingface",
        zip_filename="ZEDD.zip",
        local_dir="dataset/datasets/ZEDD",
    )
    val_dataset_args = {
    }
elif 'ddff12' in args.val_loader_config_choice:
    val_dataset = 'DDFF12Loader_Val'
    val_dataset_object = DDFF12Loader_Val
    val_dataset_args = {
        "hdf5_filename": "dataset/datasets/ddff12_val_generation/dfv_trainVal.h5", # path to the hdf5 file containing the DDFF12 validation set
        "stack_key": "stack_val",
        "disp_key": "disp_val",
        "n_stack": 5,
        "min_disp": 0.02,
        "max_disp": 0.28,
    } # parameters from eval_DDFF12.py default config
elif 'hammer' in args.val_loader_config_choice:
    val_dataset = 'HAMMER'
    val_dataset_object = HAMMER
    val_dataset_args = {
        "dir_data": "dataset/datasets/HAMMER", # path to the HAMMER dataset directory
    }
else:
    raise ValueError(f"Unknown val_loader_config: {args.val_loader_config_choice}")

if zedd_output_dir is not None and 'zedd_test' not in args.val_loader_config_choice:
    raise ValueError("zedd_test_output_dir should only be specified if val_loader_config_choice is set to a Zedd test config.")

val_dataset_args.update({'dataset_name': val_dataset})

dataset=dict(
    val=dict(
        target=val_dataset_object,
        params=dict(args=val_dataset_args, mode=mode),
    ),
)

# --- remove non-config globals so pretty_text doesn't break ---
del argparse, parser, args, _unknown