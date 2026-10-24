"""Camera frames kept on disk and decoded per window instead of up front.

The upstream `*_Img_Dataset` classes decode every frame of every trajectory into
a float32 tensor in `__init__`. One 96x96 frame costs 96*96*3*4 = 110 KB, so a
single camera of the full aligning train split (900 trajectories x ~126 frames)
is 12 GB and stacking (1000 x ~691) is 73 GB -- per camera, and the datasets
hold two. That is why the upstream classes carry debug slices
(`state_files[:3]`, `state_files[:100]`): they only fit in RAM at toy sizes.

This store keeps the sorted jpg paths instead and decodes the `window_size`
frames a sample actually needs inside the DataLoader worker. Frames come back as
uint8 (27 KB each); the agent moves them to the GPU and divides by 255 there, so
a batch costs 4x less host RAM and PCIe traffic than the float32 form.
"""
import glob
import os

import cv2
import numpy as np
import torch


class LazyImageStore:

    def __init__(self, data_dir: str, cameras=('bp-cam', 'inhand-cam')):
        self.data_dir = data_dir
        self.cameras = cameras
        self.paths = {cam: [] for cam in cameras}

    def add_trajectory(self, file_name: str) -> int:
        """Index one trajectory. Returns the number of usable frames."""
        n_frames = []
        for cam in self.cameras:
            imgs = glob.glob(os.path.join(self.data_dir, 'images', cam, file_name, '*'))
            imgs.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
            self.paths[cam].append(imgs)
            n_frames.append(len(imgs))

        # A trajectory whose cameras disagree in length is truncated to the
        # shorter one rather than crashing an epoch in a worker process.
        return min(n_frames)

    def load(self, cam: str, traj: int, start: int, end: int) -> torch.Tensor:
        """(end - start, 3, 96, 96) uint8, in the channel order cv2 reads."""
        frames = [
            cv2.imread(p).transpose((2, 0, 1))
            for p in self.paths[cam][traj][start:end]
        ]
        return torch.from_numpy(np.ascontiguousarray(np.stack(frames)))
