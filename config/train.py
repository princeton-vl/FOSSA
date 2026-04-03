from FOSSAModel import FOSSA
from dataset import MultiDataset, DDFF12Loader_Train, Zedd, DDFF12Loader_Val
from engine import scheduler_exp
from loss import SiLogLoss, GradMatchingScaleLoss
from config.validation_configs import val_loader_configs
import argparse

parser = argparse.ArgumentParser()

# Model arguments
parser.add_argument('--encoder', type=str, default='vits',
                    choices=['vits', 'vitb'])
parser.add_argument('--bs', type=int, default=2, help='batch size per GPU. If use 4 GPUs, actual batch size will be 8.')
parser.add_argument('--pretrained_or_resumed', type=str, default='resumed', choices=['pretrained', 'resumed'], help='Whether to start from a pretrained checkpoint or resume from a previous training checkpoint. "pretrained" means starting from a checkpoint that was trained on a different dataset or with a different configuration, while "resumed" means continuing training from a checkpoint that was saved during the current training process and we should load the optimizer/scheduler.')
parser.add_argument('--pretrained_from', type=str, default=None)
parser.add_argument('--resumed_from', type=str, default=None)

# Train dataset arguments
parser.add_argument('--augment', action='store_true', default=False, help='Whether to apply data augmentation during training. If set, random scaling, horizontal flipping, and randomly changed brightness, contrast, and saturation will be applied to the samples.')
parser.add_argument('--train_dataset', type=str, default=None, choices=['Hypersim+TartanAir', 'ddff12_train'], help='Name of the training dataset to use. This should match one of the dataset names defined in dataset/__init__.py. If you want to train on multiple datasets, you can specify them in a single string separated by "+", e.g. "Hypersim+TartanAir". Note that if you choose to train on multiple datasets, the training loop will sample from each dataset with equal probability, and the number of steps per epoch will be determined by the length of the longest dataset unless you specify a different total length using the --mixed_dataset_total_length argument (which is recommended to ensure consistent training across different combinations of datasets).')
parser.add_argument('--train_power_inverse_sampling', action='store_true', default=False, help='Whether to use power inverse sampling for the FD list sampled during training.')

# Validation dataset arguments
parser.add_argument('--val_loader_config_choice', type=str, default=None)

# Loss arguments

# Training arguments
parser.add_argument('--lr_decay_max_epochs', type=int, default=120, help="Number of epochs over which the learning rate will decay.")
parser.add_argument('--training_epochs', type=int, default=40, help='Total number of epochs to train for.')
parser.add_argument('--logging_turned_off', action='store_true', default=False)
parser.add_argument('--log_first_n_samples', type=int, default=1, help='Number of samples to log from the beginning of the validation set.')
parser.add_argument('--log_another_m_samples', type=int, default=30, help='Number of additional samples to log, evenly spaced throughout the validation set. These are in addition to the first n samples specified by log_first_n_samples.')

args, _unknown = parser.parse_known_args()

print(f"Parsed args: {args}")

# derive values
project_name = 'depth_from_defocus'
base_log_dir = 'runs'
experiment_name = args.train_dataset + '_' + args.encoder + '_pretrained_from_' + (args.pretrained_from if args.pretrained_from is not None else 'none') + '_resumed_from_' + (args.resumed_from if args.resumed_from is not None else 'none')
logging_turned_off = args.logging_turned_off

cli = 'train'
show_efficiency = False


lr=0.000005
img_size=518
bs=args.bs # This bs (which we trained on a bs of 2) is the per-GPU batch size, so the effective batch size is bs * number of GPUs = 2 * 4 = 8
lr_decay_max_epochs=args.lr_decay_max_epochs
training_epochs=args.training_epochs
local_rank=0
psf_type = "mixed_power_exp_psf"
mixed_power_exp_psf_p_distribution_type = 'log_uniform_2_32'

train_random_fnumber_1_0_1_4_2_0_2_8_4_0=True

training_with_canonical_depth = True
training=True

log_image_interval=5
validation_interval=5
save_checkpoint_interval=5

log_first_n_samples = args.log_first_n_samples
log_another_m_samples = args.log_another_m_samples

train_power_inverse_sampling=args.train_power_inverse_sampling
supervise_in_disparity=False
last_layer='softplus'

# Model configuration
if args.pretrained_or_resumed == 'pretrained':
    pretrained_from = args.pretrained_from
    resumed_from = None
else:
    pretrained_from = None
    resumed_from = args.resumed_from


if args.encoder == 'vits':
    num_features = 64
    out_channels = [48, 96, 192, 384]
elif args.encoder == 'vitb':
    num_features = 128
    out_channels = [96, 192, 384, 768]
else:
    raise ValueError(f"Unknown encoder: {args.encoder}")

model_type = 'FOSSA'
train_mask_used = 'depth_invalid_mask'
model_object = FOSSA

model=dict(
    target=model_object, 
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

mixed_dataset_total_length = None
# Train dataset configuration
if args.train_dataset == 'ddff12_train':
    train_dataset = 'DDFF12Loader_Train'
    train_dataset_object = DDFF12Loader_Train
    mixed_dataset_total_length = 66248 # use the same number of steps as hypersim for consistency with earlier experiments
# "+" indicates multi-dataset training
elif '+' in args.train_dataset:
    train_dataset = args.train_dataset
    train_dataset_object = MultiDataset
    mixed_dataset_total_length = 66248 # Hypersim's length, used for consistency with earlier experiments -- keep # of steps consistent
else:
    raise ValueError(f"Unknown dataset: {args.train_dataset}")


train_dataset_args = {
    "train_data_name": train_dataset,
    "mixed_dataset_total_length": mixed_dataset_total_length,
    "deterministic": False,
    "resize_height": 480,
    "resize_width": 640,
    "patch_height": 480,
    "patch_width": 640,
    "augment": args.augment,
    "random_rot_deg": 0.0,
    'min_valid_train_depth': 0.1,
    'min_valid_val_depth': 0.01,
    "random_scaling": True,
    "random_scaling_max": 1.5,
    "hypersim_max_depth_to_supervise_on": 1000.0,

    # DDFF12 specific parameters
    "hdf5_filename": "dataset/datasets/ddff12_val_generation/dfv_trainVal.h5",
    "stack_key": "stack_train",
    "disp_key": "disp_train",
    "n_stack": 5,
    "min_disp": 0.02,
    "max_disp": 0.28,
}


# Validation dataset configuration
val_loader_config_choice = args.val_loader_config_choice
val_loader_config_options = val_loader_configs


if 'ddff12' in args.val_loader_config_choice:
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
elif 'zedd' in args.val_loader_config_choice:
    assert 'test' not in args.val_loader_config_choice, "Test set evaluation for Zedd is currently not supported since the test set labels are not publicly available. Please choose a different val_loader_config_choice that does not contain 'zedd_test'."
    val_dataset = 'Zedd'
    val_dataset_object = Zedd
    dataset_location = dict(
        repo_id="venkatsubra/ZEDD",
        subdir="",
        source="huggingface"
    )
    val_dataset_args = {
    }
else:
    raise ValueError(f"Unknown val_loader_config: {args.val_loader_config_choice}")

val_dataset_args.update({'dataset_name': val_dataset})

dataset=dict(
    train=dict(
        target=train_dataset_object,
        params=dict(args=train_dataset_args, mode='train'),
    ),
    val=dict(
        target=val_dataset_object,
        params=dict(args=val_dataset_args, mode='val'),
    ),
)

# Loss configuration
loss_names = ['SiLogLoss', 'GradMatchingScaleLoss'] # add or remove loss function names from this list to change which losses are used during training. Make sure to also update the loss_coefficients list to have the same number of elements as this list, and to include an entry for each loss function in the known_loss_map dictionary below.
known_loss_map = {
    'SiLogLoss': SiLogLoss,
    'GradMatchingScaleLoss': GradMatchingScaleLoss,
}
loss_coefficients = [0.1, 1.0]
GMloss_scale_level = 2
if len(loss_names) != len(loss_coefficients):
    raise ValueError("The number of loss functions and coefficients must match.")

loss_object = []
for i, name in enumerate(loss_names):
    if name not in known_loss_map:
        raise ValueError(f"Unknown loss function: {name}")
    else:
        loss_object.append((name, loss_coefficients[i]))

# Loss_object is a list of tuples (loss_name, coefficient)
criterion=dict(
    type='CombinedLoss',
    losses=[
        {'loss_config': {
            'type': loss_name,
            **({'scale_level': GMloss_scale_level} if loss_name == 'GradMatchingScaleLoss' else {})
        }, 'loss_weight': coeff, 'loss_input': ['supervise_in_disparity', 'pred', 'target', 'valid_mask'],
        } for loss_name, coeff in loss_object
    ]
)

# Scheduler and optimizer configuration
lr_groups=[{'params': 'pretrained', 'lr_scale': 0.5},]
lr_lambda_intializer=scheduler_exp


# --- remove non-config globals so pretty_text doesn't break ---
del argparse, parser, args, _unknown
