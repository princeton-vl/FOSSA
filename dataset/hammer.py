import io
import json
from pathlib import Path
import argparse
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from .base import BaseDataset


class HAMMER(BaseDataset):
    def __init__(self, args, mode):

        super(HAMMER, self).__init__(args, mode)

        self.args = args
        self.mode = mode
        assert self.mode == 'val'

        self.dataset_name = args.dataset_name

        repo_root = Path(__file__).resolve().parent.parent
        dir_data = args.dir_data
        dir_data = Path(dir_data)
        if not dir_data.is_absolute():
            dir_data = (repo_root / dir_data).resolve()

        self.height = 480
        self.width = 640

        print('Loading HAMMER...')

        index_path = dir_data / ".index.txt"
        self.sample_list = []
        with open(index_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                scene_name, frame_id = line.split("/")
                frame_dir = dir_data / line
                self.sample_list.append((frame_dir, scene_name, frame_id))

        print(f"HAMMER total frames: {len(self.sample_list)}")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        frame_dir, scene_name, frame_id = self.sample_list[idx]

        with open(frame_dir / "meta.json", "r") as f:
            meta = json.load(f)
        K = np.array(meta["intrinsics"], dtype=np.float32)

        rgb = Image.open(frame_dir / "image.jpg").convert("RGB")
        h, w = np.array(rgb).shape[:2]

        # intrincis are normalized
        K[0, :] *= w
        K[1, :] *= h
        K = torch.from_numpy(K).float()

        # https://github.com/microsoft/MoGe/blob/42cb86af5e0f8873ff966f758033fac30a86fa49/moge/utils/io.py#L86
        depth_path = frame_dir / "depth.png"
        data = depth_path.read_bytes()
        pil_image = Image.open(io.BytesIO(data))
        raw = np.array(pil_image)
        near = float(pil_image.info["near"])
        far = float(pil_image.info["far"])
        mask_nan, mask_inf = raw == 0, raw == 65535
        depth = (raw.astype(np.float32) - 1) / 65533.0
        depth = np.power(near, 1 - depth) * np.power(far, depth)
        if "unit" in pil_image.info:
            depth = depth * float(pil_image.info["unit"])
        depth[mask_nan] = np.nan
        depth[mask_inf] = np.inf
        
        depth_valid_mask = np.ones_like(depth, dtype=bool)
        depth_valid_mask[~np.isfinite(depth)] = False
        depth_valid_mask[depth <= 0] = False
        depth[~depth_valid_mask] = 1000  # set invalid depth to 1000 

        t_rgb = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        t_dep = T.Compose([T.ToTensor()])

        rgb = t_rgb(rgb)
        dep = t_dep(Image.fromarray(depth, mode="F"))

        output = {'rgb': rgb, 'gt': dep, 'K': K, 'valid_mask': depth_valid_mask}
        return output

    def get_dataset_name(self, idx):
        return self.dataset_name
