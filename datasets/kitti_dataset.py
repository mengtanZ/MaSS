from __future__ import absolute_import, division, print_function
import os
import skimage.transform
import numpy as np
import PIL.Image as pil
from pathlib import Path
from numpy.typing import NDArray
from utils.kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset
from PIL import Image

class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI datasets loaders
    """

    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)
        self.K = np.array([[0.58, 0, 0.5, 0], [0, 1.92, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.full_res_shape = (1242, 375)
        self.side_map = {'2': 2, '3': 3, 'l': 2, 'r': 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])
        velo_filename = os.path.join(self.data_path, scene_name, 'velodyne_points/data/{:010d}.bin'.format(int(frame_index)))
        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

class KITTIRAWDataset(KITTIDataset):
    """KITTI datasets which loads the original velodyne depth maps for ground truth
    """

    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = '{:010d}{}'.format(frame_index, self.img_ext)
        image_path = os.path.join(self.data_path, folder, 'image_0{}/data'.format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split('/')[0])
        velo_filename = os.path.join(self.data_path, folder, 'velodyne_points/data/{:010d}.bin'.format(int(frame_index)))
        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')
        if do_flip:
            depth_gt = np.fliplr(depth_gt)
        return depth_gt
