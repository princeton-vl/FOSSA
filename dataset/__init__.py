import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Subset
from util.init import instantiate_class_from_config
from .uniformat import Uniformat
from .infinigen_defocus import InfinigenDefocus
from .zedd import Zedd
from .ddff12_val import DDFF12Loader_Val
from .hammer import HAMMER
from .ddff12_train import DDFF12Loader_Train
from .hypersim import Hypersim
from .tartanair import TartanAir
from .multidataset import MultiDataset

from PIL import Image

from importlib import import_module

__all__ = ['Uniformat', 'InfinigenDefocus', 'Zedd', 'HAMMER', 'DDFF12Loader_Val', 'DDFF12Loader_Train']
def init_dataloader(config, split):
    dataset = instantiate_class_from_config(config['dataset'][split])
    if split == 'train':
        # Allows pulling dataset name from dataset object later using get_dataset_name method
        config['instantiated_train_dataset_object'] = dataset
    sampler = DistributedSampler(dataset)

    sample_indices = list(range(0, len(dataset)))
    subset = Subset(dataset, indices=sample_indices)
    

    if split == 'train':
        dataloader = DataLoader(dataset, batch_size=config['bs'], pin_memory=True, num_workers=4, drop_last=True, sampler=sampler)
        return dataloader, subset
    elif split == 'val' or split == 'test':
        # No distributed sampler for val
        sampler = None
        dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=4, drop_last=True, sampler=sampler, shuffle=False)
        return dataloader, subset


def get(args, mode):
    if mode == "train":
        data_name = args.train_data_name
    elif mode == "val" or mode == "test":
        data_name = args.val_data_name
    else:
        raise NotImplementedError

    data_names = data_name.split("+")
    if len(data_names) == 1: # use the original dataset
        module_name = "dataset." + data_name.lower() # Load in vkitti.py if --train_data_name VKITTI
        dataset_name = data_name
    else:
        module_name = 'data.multidataset'
        dataset_name = 'MultiDataset'

    module = import_module(module_name)
    return getattr(module, dataset_name)(args, mode=mode) # Returns the initialized dataset with the args and mode


