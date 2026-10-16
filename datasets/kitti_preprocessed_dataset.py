from __future__ import absolute_import, division, print_function
import os
import numpy as np
import PIL.Image as pil
from imageio import imread
from .mono_dataset import MonoDataset

class KPreprocessedDataset(MonoDataset):
    """Superclass for the preprocessed datasets (including: DDAD, nuScenes, Waymo, DIML) loaders
    """

    def __init__(self, *args, **kwargs):
        self.image_size_cache = {}
        super(KPreprocessedDataset, self).__init__(*args, **kwargs)

    def check_depth(self):
        return False

    def get_original_image_size(self, folder, frame_index, side):
        key = (folder, frame_index, side)
        if key not in self.image_size_cache:
            image_path = self.get_image_path(folder, frame_index, side)
            with pil.open(image_path) as image:
                self.image_size_cache[key] = image.size
        return self.image_size_cache[key]

    def load_intrinsics(self, folder, frame_index):
        camera_file = os.path.join(self.data_path, folder, 'cam.txt')
        if os.path.exists(camera_file):
            camera = np.genfromtxt(camera_file)
            image_width, image_height = self.get_original_image_size(folder, frame_index, None)
            fx = camera[0, 0]
            fy = camera[1, 1]
            u0 = camera[0, 2]
            v0 = camera[1, 2]
            intrinsics = np.array([[fx, 0, u0, 0], [0, fy, v0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
            intrinsics[0, :] /= image_width
            intrinsics[1, :] /= image_height
            return intrinsics
        raise FileNotFoundError('Camera intrinsics file not found: {}'.format(camera_file))

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
        f_str = '{:010d}{}'.format(frame_index, self.img_ext)
        return os.path.join(self.data_path, folder, f_str)

    def get_depth_path(self, folder, frame_index, side):
        f_str = '{:010d}{}'.format(frame_index, '.npy')
        return os.path.join(self.data_path, folder, 'depth', f_str)

    def get_psedo_depth_path(self, folder, frame_index, side):
        f_str = '{:010d}{}'.format(frame_index, '.png')
        return os.path.join(self.data_path, folder, self.psedo_depth, f_str)
