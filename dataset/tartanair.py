import numpy as np
from .base import BaseDataset
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode, Resize
import glob
from tqdm import tqdm
from pathlib import Path
dataset_folder = (Path(__file__).parent / "datasets").resolve()



class TartanAir(BaseDataset):
    def __init__(self, args, mode="train"):
        super(TartanAir, self).__init__(args, mode)

        self.args = args
        self.mode = mode
        self.crop_size = (args.patch_height, args.patch_width)

        self.resize_height = args.resize_height
        self.resize_width = args.resize_width

        if mode != 'train':
            raise NotImplementedError
        
        # adopted from depthanything v3
        self.removed_scenes = ['amusement', 'ocean']
        self.max_depth_per_scene = {
            'carwelding': 80.0,
            'hospital': 100.0,
            'office': 30.0,
            'office2': 30.0,
        }

        self.sample_list = []
        pattern = str(dataset_folder / "tartanair/*/*/*/image_left/*_left.png")
        images = glob.glob(pattern)
        print('Loading TartanAir...')
        for image in tqdm(images):
            scene, difficulty, P = image.split("/")[-5: -2]
            if scene in self.removed_scenes:
                continue
            i = image.split("/")[-1].split("_")[0]
            depth = str(dataset_folder / f"tartanair/{scene}/{difficulty}/{P}/depth_left/{i}_left_depth.npy")
            self.sample_list.append((scene, image, depth))

        self.K = np.array([
            [320.0, 0.0, 320.0],
            [0.0, 320.0, 240.0],
            [0.0, 0.0, 1.0]
        ])
        self.augment = self.args.augment

        self.dataset_name = "TartanAir"

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        scene_name, image_file, depth_file = self.sample_list[idx]

        if scene_name in self.max_depth_per_scene:
            max_depth = self.max_depth_per_scene[scene_name]
        else:
            max_depth = 200.0

        dep_raw = np.load(depth_file).astype('float32')
        # clamp depth to 1000.0
        dep_raw = np.clip(dep_raw, 0, 1000.0)

        # tartanair has some crazily large depth values. clamp it:
        depth_valid_mask = ((dep_raw > 0.0) & (dep_raw < max_depth)).astype(np.bool_)

        rgb = Image.open(image_file, mode='r')
        dep = Image.fromarray(dep_raw, mode='F')
        depth_valid_mask = Image.fromarray(depth_valid_mask.astype(np.uint8), mode='L')

        K = torch.from_numpy(self.K).float()

        rgb, dep, K, depth_valid_mask, _, _ = self.process(rgb, dep, K, depth_valid_mask)

        output = {'rgb': rgb, 'gt': dep, 'K': K, 'valid_mask': depth_valid_mask}

        return output
        
    def get_dataset_name(self, idx):
        return self.dataset_name
