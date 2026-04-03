import numpy as np
from pathlib import Path
from .base import BaseDataset
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

def load_metadata(path: Path) -> dict:
    """Parse metadata.txt into {run_id: {hex_code, focus_distance}}."""
    metadata = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        run_id, hex_code, focus_distance = line.split('\t')
        metadata[run_id] = {
            'hex_code': hex_code,
            'focus_distance': float(focus_distance),
        }
    return metadata





class Zedd(BaseDataset):
    def __init__(self, args, mode):
        super(Zedd, self).__init__(args, mode)

        self.DATASET_ROOT = Path('dataset/datasets/ZEDD')

        self.METADATA_FILE = self.DATASET_ROOT / 'metadata.txt'

        self.CALIBRATION_RESULTS = load_metadata(self.METADATA_FILE)

        self.args = args
        self.mode = mode
        assert self.mode in ('val', 'test'), "Zedd dataset only supports val and test modes"

        if self.mode == 'val':
            self.height = 480
            self.width = 640
        else:
            self.height = 1216
            self.width = 1824

        self.resize = False

        split_dir = self.DATASET_ROOT / self.mode
        self.scene_folders = sorted(split_dir.iterdir())

        self.fd_list = args.val_loader_config.fd_list
        self.min_valid_val_depth = getattr(args, 'min_valid_val_depth', 0.01)
        self.max_valid_val_depth = getattr(args, 'max_valid_val_depth', 1000.0)
        self.fnumber = args.val_loader_config.fnumber
        self.num_fd = getattr(args, 'num_fd', 5)
        self.dataset_name = "Zedd"
        self.use_focus_stack_from_dataset = args.val_loader_config.get('use_focus_stack_from_dataset', False)

    def __len__(self):
        return len(self.scene_folders)

    def __getitem__(self, idx):
        scene_folder = self.scene_folders[idx]

        image_folder = scene_folder / 'focus_stack'
        gt_folder    = scene_folder / 'gt'

        # this is the all-in-focus image, which we treat as the "RGB" input
        image = Image.open(image_folder / 'img_run_5_motor_2371_aperture_F16.0.jpg').convert('RGB')

        K = np.loadtxt(gt_folder / 'K.txt')

        # test split gt folder contains only K.txt (no depth.npy in the public release).
        # we produce a dummy all-zeros depth and an all-False valid mask so downstream
        # code can run without branching on mode; these values must NOT be used for eval.
        if self.mode == 'test':
            depth = np.zeros((self.height, self.width), dtype=np.float32)
            depth_original_h, depth_original_w = self.height, self.width
        else:
            depth = np.load(gt_folder / 'depth.npy')
            depth_original_h, depth_original_w = depth.shape
            depth = np.asarray(depth).astype(np.float32)

        # resize and center-crop to target resolution if needed, adjusting K accordingly
        if image.height != self.height:
            new_width = int(image.width * (self.height / image.height))
            image = image.resize((new_width, self.height), Image.LANCZOS)
            if self.mode != 'test' and depth_original_h != self.height:
                depth = TF.resize(Image.fromarray(depth, mode='F'), (self.height, new_width), interpolation=TF.InterpolationMode.NEAREST)
                depth = np.asarray(depth).astype(np.float32)
            scale_x = new_width / depth_original_w
            scale_y = self.height / depth_original_h
            K[0, :] = K[0, :] * scale_x
            K[1, :] = K[1, :] * scale_y

        if image.width > self.width:
            left = (image.width - self.width) // 2
            right = left + self.width
            image = image.crop((left, 0, right, self.height))
            if self.mode != 'test':
                depth = depth[:, left:right]
            K[0, 2] = K[0, 2] - left

        # for val: derive valid mask from depth values.
        # for test: all-False mask since depth is a placeholder (no ground truth available).
        if self.mode == 'test':
            depth_valid_mask = np.zeros((self.height, self.width), dtype=bool)
        else:
            depth_valid_mask = np.ones_like(depth, dtype=bool)
            infinite_vals_mask = np.logical_or.reduce((
                np.logical_not(np.isfinite(depth)),
                np.isnan(depth),
                depth < self.min_valid_val_depth,
                depth > self.max_valid_val_depth,
            ))
            depth_valid_mask[infinite_vals_mask] = False

        t_rgb = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        t_dep = T.Compose([
            self.ToNumpy(),
            T.ToTensor()
        ])

        rgb = t_rgb(image)
        dep = t_dep(Image.fromarray(depth, mode='F'))
        K   = torch.Tensor(K)

        output = {'rgb': rgb, 'gt': dep, 'K': K, 'valid_mask': depth_valid_mask}

        # load the focus stack
        if self.use_focus_stack_from_dataset:
            if self.fd_list is None:
                raise ValueError("fd_list must be provided when use_focus_stack_from_dataset is True")
            fd_list = self.fd_list

            focal_stack = []
            fd_meters   = []
            for fd_id in fd_list:
                run_key  = f'run_{fd_id + 1}'
                hex_code = self.CALIBRATION_RESULTS[run_key]['hex_code']
                file_name = f'img_run_{fd_id+1}_motor_{hex_code}_aperture_F{self.fnumber:.1f}.jpg'
                defocus_image = Image.open(image_folder / file_name).convert('RGB')

                if defocus_image.height != self.height:
                    new_width = int(defocus_image.width * (self.height / defocus_image.height))
                    defocus_image = defocus_image.resize((new_width, self.height), Image.LANCZOS)

                if defocus_image.width > self.width:
                    left = (defocus_image.width - self.width) // 2
                    right = left + self.width
                    defocus_image = defocus_image.crop((left, 0, right, self.height))

                focal_stack.append(t_rgb(defocus_image))
                fd_meters.append(self.CALIBRATION_RESULTS[run_key]['focus_distance'])

            output['focal_stack'] = torch.stack(focal_stack, dim=0)
            output['fd_list']     = torch.tensor(fd_meters, dtype=torch.float32)

        return output