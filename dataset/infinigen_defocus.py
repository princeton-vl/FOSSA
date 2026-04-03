import os
import json
import numpy as np
from .base import BaseDataset
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from pathlib import Path

class InfinigenDefocus(BaseDataset):
    def __init__(self, args, mode):
        super(InfinigenDefocus, self).__init__(args, mode)

        self.args = args
        self.mode = mode
        assert self.mode == 'val'

        self.height = 480
        self.width = 640

        self.resize = False
        self.dataset_folder = args.dataset_folder

        print('Loading Infinigen Defocus...')
        split_file = os.path.join(Path(__file__).parent, "splits", "infinigen_defocus", "val.json")
        with open(split_file, 'r') as f:
            self.filenames = json.load(f)['files']
        self.fd_list = args.val_loader_config.fd_list

        self.min_valid_val_depth = getattr(args, 'min_valid_val_depth', 0.01)
        self.max_valid_val_depth = getattr(args, 'max_valid_val_depth', 1000.0)
        self.fnumber = args.val_loader_config.fnumber
        self.num_fd = getattr(args, 'num_fd', 5)

        self.dataset_name = "InfinigenDefocus"
        self.use_focus_stack_from_dataset = args.val_loader_config.get('use_focus_stack_from_dataset', False)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        sample = self.filenames[idx]

        image = Image.open(self.dataset_folder + sample['image']).convert('RGB')
        depth = np.load(self.dataset_folder + sample['depth'])
        K = np.load(self.dataset_folder + sample['camera'])['K']
        # mask_invalid = data['mask_invalid'][0][0]  # Mask for invalid pixels
        depth = np.asarray(depth).astype(np.float32)
        # some pixels may be invalid
        depth_valid_mask = np.ones_like(depth, dtype=bool) # Make sure this is a boolean mask
        infinite_vals_mask = np.logical_or.reduce((np.logical_not(np.isfinite(depth)), np.isnan(depth), depth<self.min_valid_val_depth, depth>self.max_valid_val_depth)) # mask for infinite and nan values

        depth_valid_mask[infinite_vals_mask] = False

        depth[~depth_valid_mask] = 1000

        rgb = image

        dep = Image.fromarray(depth, mode='F')

        K = torch.Tensor(K)

        t_rgb = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        t_rgb_np_raw = T.Compose([
            self.ToNumpy(),
        ])

        t_dep = T.Compose([
            self.ToNumpy(),
            T.ToTensor()
        ])

        rgb = t_rgb(rgb)
        dep = t_dep(dep)

        output = dict()
        if self.use_focus_stack_from_dataset:
            focal_stack = []
            for fd in self.fd_list:
                defocus_image = Image.open(self.dataset_folder + sample[f'image_ap_{self.fnumber:.1f}_fd_{fd:.1f}']).convert('RGB')
                focal_stack.append(t_rgb(defocus_image))
        
            if self.resize:
                rgb = TF.resize(rgb, [self.height, self.width], interpolation=TF.InterpolationMode.BICUBIC)
                dep = TF.resize(dep, [self.height, self.width], interpolation=TF.InterpolationMode.NEAREST)
                depth_valid_mask = TF.resize(torch.tensor(depth_valid_mask).unsqueeze(0).float(), [self.height, self.width], interpolation=TF.InterpolationMode.NEAREST).squeeze(0).bool()
                K[0, :] = K[0, :] * (self.width / depth.shape[1])
                K[1, :] = K[1, :] * (self.height / depth.shape[0])
                focal_stack_resized = []
                for fs in focal_stack:
                    fs_resized = TF.resize(fs, [self.height, self.width], interpolation=TF.InterpolationMode.BICUBIC)
                    focal_stack_resized.append(fs_resized)
                focal_stack = focal_stack_resized

            output['focal_stack'] = torch.stack(focal_stack, dim=0)
            output['fd_list'] = torch.tensor(self.fd_list, dtype=torch.float32)
        
        output['rgb'] = rgb
        output['gt'] = dep
        output['K'] = K
        output['valid_mask'] = depth_valid_mask

        return output
