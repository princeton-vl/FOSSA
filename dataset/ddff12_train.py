#! /usr/bin/python3

import os
import numpy as np
from torch.utils.data import Dataset
import torchvision
import torch
import h5py
import scipy.io
from .base import BaseDataset
from PIL import Image

# code adopted from https://github.com/soyers/ddff-pytorch/blob/master/python/ddff/dataproviders/datareaders/FocalStackDDFFH5Reader.py


class DDFF12Loader_Train(BaseDataset):

    def __init__(self, args, mode):
        """
        Args:
            root_dir_fs (string): Directory with all focal stacks of all image datasets.
            root_dir_depth (string): Directory with all depth images of all image datasets.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # Disable opencv threading since it leads to deadlocks in PyTorch DataLoader
        self.args = args
        self.mode = mode
        if self.mode != 'train':
            raise NotImplementedError("DDFF12Loader_Train only supports training since validation and testing are done on the generated DDFF12 focal stacks and depth maps, not the original hdf5 files. If you want to use the original hdf5 files for validation/testing, you will need to modify the code to support that.")
        self.resize_width = args.resize_width
        self.resize_height = args.resize_height
        self.crop_size = (args.patch_height, args.patch_width)
        hdf5_filename = args.hdf5_filename
        stack_key = args.stack_key
        disp_key = args.disp_key
        n_stack = args.n_stack
        min_disp = args.min_disp
        max_disp = args.max_disp
        transform = None if not 'transform' in args else args.transform



        self.hdf5 = h5py.File(hdf5_filename, 'r')
        self.stack_key = stack_key
        self.disp_key = disp_key
        self.max_n_stack = 10
        self.ignore_train = True
        self.augment = args.get('augment', False)

        assert n_stack <= self.max_n_stack, 'DDFF12 has maximum 10 images per stack!'
        self.n_stack = n_stack
        self.disp_dist = torch.linspace(max_disp,min_disp, steps=self.max_n_stack)
        
        self.internal_length = self.hdf5[self.stack_key].shape[0]
        self.total_length = args.mixed_dataset_total_length
        self.deterministic = args.deterministic
        self.indices = self._generate_indices()

        self.dataset_name = "DDFF12Loader_Train"

    def __len__(self):
        return self.total_length
    
    def _generate_indices(self):
        if self.deterministic:
            np.random.seed(0)
        if self.internal_length <= self.total_length:
            
            # Guarantee that all samples are seen at least once
            dataset_indices = range(self.internal_length)
            remaining_indices_to_sample = self.total_length - self.internal_length

            if remaining_indices_to_sample > 0:
                # Sample remaining indices with replacement
                dataset_indices = list(dataset_indices) + list(
                    np.random.choice(range(self.internal_length), size=remaining_indices_to_sample, replace=True)
                )
        else:
            dataset_indices = np.random.choice(range(self.internal_length), size=self.total_length, replace=False)
            
        calib_mat_file_path = "dataset/datasets/ddff12_val_generation/third_part/IntParamLF.mat"
        assert os.path.exists(calib_mat_file_path), "DDFF12 intrinsics file not found at 'ddff12_val_generation/third_part/IntParamLF.mat!'"
        mat = scipy.io.loadmat(calib_mat_file_path)
        mat = np.squeeze(mat['IntParamLF'])
        K2 = mat[1]
        fxy = mat[2:4]
        if K2 >1983 or K2 < 1982:
            raise ValueError("DDFF12 intrinsics K2 value seems off, expected around 1982-1983.")
        flens = max(fxy)
        self.fsubaperture = 521.4052 # pixel
        self.baseline = K2/flens*1e-3 # meters
        # From page 7 of Hazirbas et al. -- https://arxiv.org/pdf/1704.01085
        K = np.array([[521.4, 0, 285.11],
                        [0, 521.4, 187.83],
                        [0, 0, 1]])
        self.K = torch.Tensor(K)

        return dataset_indices
    
    def refresh_indices(self):
        self.indices = self._generate_indices()

    def __getitem__(self, idx):
        # Create sample dict

        idx = self.indices[idx]
        # Loading FD list
        img_idx = np.array([0,2,4,6,9]) # using the same fd list as in validation

        sample = {'input': self.hdf5[self.stack_key][idx].astype(float), 'output': self.hdf5[self.disp_key][idx]}
        out_imgs = sample['input'][img_idx]
        out_imgs_np = sample['input'][img_idx].astype(np.uint8)  # [n_stack, H, W, 3]
        ddff12_stack_pil = [Image.fromarray(out_imgs_np[k], mode="RGB") for k in range(out_imgs_np.shape[0])]
        out_disp = sample['output']

        disp_dist = self.disp_dist[img_idx]

        dep = (self.baseline*self.fsubaperture)/out_disp # in meters
        fd_list = (self.baseline*self.fsubaperture)/disp_dist # in meters
        rgb = out_imgs_np[0].astype(np.uint8) 

        depth_valid_mask = (out_disp > 0.0).squeeze()

        dep = Image.fromarray(dep.astype(np.float32), mode='F')
        depth_valid_mask = Image.fromarray(depth_valid_mask.astype(np.uint8), mode='L')  # Convert boolean mask to uint8
        rgb = Image.fromarray(rgb, mode='RGB')
        

        rgb, dep, K, depth_valid_mask, rgb_np_raw, ddff12_stack = self.process(rgb, dep, self.K, depth_valid_mask, ddff12_focus_stack=ddff12_stack_pil)
        # reorder out_imgs to be [B, N, C, H, W] instead of [B, N, H, W, C]

        output = {'rgb': rgb, 'gt': dep, 'K': K, 'valid_mask': depth_valid_mask, 'fd_list': fd_list, 'focal_stack': ddff12_stack}
        
        return output

    def get_stack_size(self):
        return self.__getitem__(0)['input'].shape[0]
    
    def get_dataset_name(self, idx):
        return self.dataset_name

    class ToTensor(object):
        """Convert ndarrays in sample to Tensors."""

        def __call__(self, sample):
            # Add color dimension to depth map
            sample['output'] = np.expand_dims(sample['output'], axis=0)
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            sample['input'] = torch.from_numpy(sample['input'].transpose((0, 3, 1, 2))).float().div(255) #I add div 255
            sample['output'] = torch.from_numpy(sample['output']).float()
            return sample