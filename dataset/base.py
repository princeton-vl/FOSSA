import numpy as np
from importlib import import_module
from torch.utils.data import Dataset
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode, Resize

PATTERN_IDS = {
    'random': 0,
    'velodyne': 1,
    'sfm': 2
}

def get_center_crop_origin(image_size, crop_size):
    h_img, w_img = image_size
    crop_h, crop_w = crop_size
    i = (h_img - crop_h) // 2
    j = (w_img - crop_w) // 2
    return i, j

def _apply_to_stack(stack, fn):
    if stack is None:
        return None
    return [fn(im) for im in stack]

def get(args, mode):
    if mode == "train":
        data_name = args.train_data_name
    elif mode == "val" or mode == "test":
        data_name = args.val_data_name
    else:
        raise NotImplementedError

    data_names = data_name.split("+")
    if len(data_names) == 1: # use the original dataset
        module_name = 'dataset.' + data_name.lower() # Load in vkitti.py if --train_data_name VKITTI
        dataset_name = data_name
    else:
        module_name = 'dataset.multidataset'
        dataset_name = 'MultiDataset'

    module = import_module(module_name)
    return getattr(module, dataset_name)(args, mode=mode) # Returns the initialized dataset with the args and mode


class BaseDataset(Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode # training or validation
        self.max_depth_range = 100.0

    # Method that must be extended in the child class (like vkitti.py)
    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    # A workaround for a pytorch bug
    # https://github.com/pytorch/vision/issues/2194
    class ToNumpy:
        def __call__(self, sample):
            return np.array(sample)

    # Common method for all datasets to preprocess RGB images, depth maps, and camera intrinsics
    def process(self, rgb, dep, K, depth_valid_mask, random_crop=True, ddff12_focus_stack=None):
        args = self.args

        if self.augment and self.mode == 'train':
            # Random scaling of height and width
            if self.args.random_scaling:
                _scale = np.random.uniform(1.0, self.args.random_scaling_max)
            else:
                _scale = 1.0
            scaled_h = int(self.resize_height * _scale)
            scaled_w = int(self.resize_width * _scale)
            degree = np.random.uniform(-self.args.random_rot_deg, self.args.random_rot_deg)
            flip = np.random.uniform(0.0, 1.0)

            # 50% chance of horizontally flipping the image
            if flip > 0.5:
                rgb = TF.hflip(rgb)
                dep = TF.hflip(dep)
                depth_valid_mask = TF.hflip(depth_valid_mask)
                ddff12_focus_stack = _apply_to_stack(ddff12_focus_stack, TF.hflip)

            # Random rotation
            rgb = TF.rotate(rgb, angle=degree)
            dep = TF.rotate(dep, angle=degree)
            depth_valid_mask = TF.rotate(depth_valid_mask, angle=degree)
            ddff12_focus_stack = _apply_to_stack(ddff12_focus_stack, lambda im: TF.rotate(im, angle=degree))

            original_w, original_h = rgb.size
            scale_ratio_h, scale_ratio_w = scaled_h / original_h, scaled_w / original_w
            scale_ratio = max(scale_ratio_h, scale_ratio_w)
            scaled_h = int(scale_ratio * original_h)
            scaled_w = int(scale_ratio * original_w)
            # assert np.isclose(scale_ratio_h, scale_ratio_w, rtol=0.01), "only support resizing that keeps the original aspect ratio"

            resize_rgb = T.Resize((scaled_h, scaled_w))
            # Resize the depth map by filling in unknown values by plugging in the nearest known pixel value
            resize_dep = T.Resize((scaled_h, scaled_w), InterpolationMode.NEAREST)
            resize_depth_valid_mask = T.Resize((scaled_h, scaled_w), InterpolationMode.NEAREST)
            ddff12_focus_stack = _apply_to_stack(ddff12_focus_stack, lambda im: TF.resize(im, (scaled_h, scaled_w)))

            rgb = resize_rgb(rgb)
            dep = resize_dep(dep)
            depth_valid_mask = resize_depth_valid_mask(depth_valid_mask)


            random_crop = False # accidentally set to false during code development, but keep as it is for reproducibility of results in the paper
            if random_crop:
                # Random crop
                i, j, h, w = T.RandomCrop.get_params(
                    rgb, output_size=self.crop_size)
                rgb = TF.crop(rgb, i, j, h, w)
                dep = TF.crop(dep, i, j, h, w)
                depth_valid_mask = TF.crop(depth_valid_mask, i, j, h, w)
                ddff12_focus_stack = _apply_to_stack(ddff12_focus_stack, lambda im: TF.crop(im, i, j, h, w))

            else:
                cropping_op = T.CenterCrop(self.crop_size)
                rgb = cropping_op(rgb)
                dep = cropping_op(dep)
                depth_valid_mask = cropping_op(depth_valid_mask)
                ddff12_focus_stack = _apply_to_stack(ddff12_focus_stack, lambda im: cropping_op(im))

                i, j = get_center_crop_origin((scaled_h, scaled_w), self.crop_size)

            # adjust intrinsics
            K = K.clone()

            # adjust focal for resizing
            K[0] = K[0] * scale_ratio
            K[1] = K[1] * scale_ratio

            # adjust principal point for cropping
            K[0, 2] -= j
            K[1, 2] -= i

            # Randomly change brightness, contrast, and saturation
            t_rgb = T.Compose([
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

            t_rgb_np_raw = T.Compose([
                self.ToNumpy(),
            ])

            t_dep = T.Compose([
                self.ToNumpy(),
                T.ToTensor()
            ])

            rgb_final = t_rgb(rgb)
            dep = t_dep(dep) # 1 x H x W
            rgb_np_raw = t_rgb_np_raw(rgb)
            if ddff12_focus_stack is not None:
                stack_t = torch.stack([TF.to_tensor(im) for im in ddff12_focus_stack], dim=0)  # [N, C, H, W]
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                ddff12_focus_stack = (stack_t - mean) / std


        else:
            t_rgb = T.Compose([
                T.Resize((self.resize_height, self.resize_width)),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            t_rgb_np_raw = T.Compose([
                T.Resize((self.resize_height, self.resize_width)),
                T.CenterCrop(self.crop_size),
                self.ToNumpy()
            ])

            t_dep = T.Compose([
                T.Resize((self.resize_height, self.resize_width), InterpolationMode.NEAREST),
                T.CenterCrop(self.crop_size),
                self.ToNumpy(),
                T.ToTensor()
            ])

            t_depth_mask = T.Compose([
                T.Resize((self.resize_height, self.resize_width), InterpolationMode.NEAREST),
                T.CenterCrop(self.crop_size)
            ])

            

            original_w, original_h = rgb.size
            scale_ratio_h, scale_ratio_w = self.resize_height / original_h, self.resize_width / original_w
            assert np.isclose(scale_ratio_h, scale_ratio_w, rtol=0.01), "only support resizing that keeps the original aspect ratio"

            i, j = get_center_crop_origin((self.resize_height, self.resize_width), self.crop_size)

            K = K.clone()

            # adjust focal for resizing
            K[0] = K[0] * scale_ratio_h
            K[1] = K[1] * scale_ratio_w

            # adjust principal point for cropping
            K[0, 2] -= j
            K[1, 2] -= i

            rgb_final = t_rgb(rgb)
            dep = t_dep(dep)
            rgb_np_raw = t_rgb_np_raw(rgb)
            depth_valid_mask = t_depth_mask(depth_valid_mask)

        # Replaces NaN values with 0
        dep = torch.nan_to_num(dep)
        # Convert depth_valid_mask to a np array
        depth_valid_mask = np.array(depth_valid_mask).astype(bool)
        
        return rgb_final, dep, K, depth_valid_mask, rgb_np_raw, ddff12_focus_stack

    def refresh_indices(self):
        pass 