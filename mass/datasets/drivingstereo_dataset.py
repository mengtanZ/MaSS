from __future__ import absolute_import, division, print_function
import os
import numpy as np
import PIL.Image as pil
from imageio import imread
from .mono_dataset import MonoDataset

class DrivingstereoDataset(MonoDataset):
    """Superclass for the preprocessed datasets (including: DDAD, nuScenes, Waymo, DIML) loaders
    """

    def __init__(self, *args, **kwargs):
        super(DrivingstereoDataset, self).__init__(*args, **kwargs)

    def check_depth(self):
        return False

    def load_intrinsics(self, folder, frame_index):
        intrinsics = np.array([[1.16, 0, 0.5, 0], [0, 1.92, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        return intrinsics

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

    def get_depth(self, folder, frame_index, side, do_flip):
        depth_gt = np.load(self.get_depth_path(folder, frame_index, side)).astype(np.float32)
        if do_flip:
            depth_gt = np.fliplr(depth_gt)
        return depth_gt

    def get_psedo_depth(self, folder, frame_index, side, do_flip):
        psedo_depth = imread(self.get_psedo_depth_path(folder, frame_index, side)).astype(np.float32)
        if do_flip:
            psedo_depth = np.fliplr(psedo_depth)
        return psedo_depth

    def get_image_path(self, folder, frame_index, side):
        f_str = '{:06d}{}'.format(frame_index, self.img_ext)
        return os.path.join(self.data_path, folder, f_str)

    def get_depth_path(self, folder, frame_index, side):
        f_str = '{:06d}{}'.format(frame_index, '.npy')
        return os.path.join(self.data_path, folder, 'depth', f_str)

    def get_psedo_depth_path(self, folder, frame_index, side):
        f_str = '{:06d}{}'.format(frame_index, '.png')
        return os.path.join(self.data_path, folder, self.psedo_depth, f_str)
