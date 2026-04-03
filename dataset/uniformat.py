import os
import warnings
import numpy as np
from .base import BaseDataset
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from pathlib import Path
warnings.filterwarnings("ignore", category=UserWarning)

class Uniformat(BaseDataset):
    def __init__(self, args, mode="train"):
        super(Uniformat, self).__init__(args, mode)

        self.args = args
        self.mode = mode
        self.dataset_name = args.dataset_name

        if mode != 'test' and mode != 'val':
            raise NotImplementedError

        print('Loading Uniformat...')

        if 'DIODE' in self.dataset_name:
            self.sample_list = [os.path.join(args.dir_data_indoor, f) for f in os.listdir(args.dir_data_indoor) if f.endswith('.npy')] + [os.path.join(args.dir_data_outdoor, f) for f in os.listdir(args.dir_data_outdoor) if f.endswith('.npy')]
        else:
            self.sample_list = sorted([os.path.join(args.dir_data, f) for f in os.listdir(args.dir_data) if f.endswith('.npy')])

        self.center_crop = False

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        t_rgb = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        t_dep = T.Compose([
            T.ToTensor()
        ])

        # what is saved:
        # save_dict = {
        #     'gt': depth,
        #     'depth_filled': depth_filled,
        #     'rgb': rgb,
        #     'valid_mask': valid_mask,
        #     'K': K
        # }

        filedir = self.sample_list[idx]
        data_dict = dict(np.load(filedir, allow_pickle=True).item())

        rgb = t_rgb(Image.fromarray(data_dict['rgb'], mode='RGB'))
        K = torch.from_numpy(data_dict['K'].astype(np.float32))
        depth_valid_mask = data_dict['valid_mask']
        depth_filled = t_dep(data_dict['depth_filled'])

        if self.dataset_name == 'iBims':
            # transpose K for iBims to be consistent with other datasets
            K = K.T
        
        # crop to make the image such that it was taken with a 50mm focal length lens, then resize back to original size
        if self.center_crop:
            H, W = rgb.shape[1], rgb.shape[2]
            cropped_h, cropped_w = 260, 346
            top = (H - cropped_h) // 2
            left = (W - cropped_w) // 2
            rgb = TF.resize(TF.crop(rgb, top, left, cropped_h, cropped_w), (H, W), interpolation=T.InterpolationMode.BILINEAR)
            depth_filled = TF.resize(TF.crop(depth_filled, top, left, cropped_h, cropped_w), (H, W), interpolation=T.InterpolationMode.NEAREST)
            depth_valid_mask = TF.resize(TF.crop(torch.tensor(depth_valid_mask).unsqueeze(0).float(), top, left, cropped_h, cropped_w), (H, W), interpolation=T.InterpolationMode.NEAREST).squeeze(0).bool()
            K[0, :] = K[0, :] * (W / cropped_w)
            K[1, :] = K[1, :] * (H / cropped_h)
            K[0, 2] = K[0, 2] - left * (W / cropped_w)
            K[1, 2] = K[1, 2] - top * (H / cropped_h)

            print(f"cropped h: {cropped_h}, cropped w: {cropped_w}")

        # use depth_filled as gt for training
        # note that the values in depth_filled are the same as gt for valid pixels
        # others are filled by OMNI-DC, but those are only relevant for focus stack generation.
        output = {'rgb': rgb, 'gt': depth_filled, 'K': K, 'valid_mask': depth_valid_mask}
        
        return output # Assume uniformat only corresponds to 1 validation dataset