"""Dataset protocols retained from the MaSS evaluation scripts."""
import csv
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from scipy import sparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import datasets
from .networks import MaSS_Encoder, DepthDecoder
from .networks.layers import disp_to_depth
from .utils.utils import readlines

PROTOCOLS = {
    'kitti': ('eigen', 0.001, 80),
    'ddad': ('ddad', 0.001, 200),
    'nyu': ('nyu', 0.001, 10),
    'nuscenes': ('nuscenes', 0.001, 75),
    'waymo': ('waymo', 0.001, 75),
    'drivingstereo': ('drivingstereo', 0.001, 75),
    'cityscapes': ('cityscapes', 0.001, 80),
    'diml': ('diml', 0.1, 10),
    'ibims': ('ibims', 0.001, 10),
    'sunrgbd': ('sunrgbd', 0.1, 10),
    '7scenes': ('7scenes', 0.1, 10),
    'diode': ('diode', 0.001, 75),
}
METRICS = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']


def compute_errors(gt, pred):
    thresh = np.maximum(gt / pred, pred / gt)
    return (np.mean(np.abs(gt - pred) / gt), np.mean((gt - pred) ** 2 / gt),
            np.sqrt(np.mean((gt - pred) ** 2)), np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2)),
            (thresh < 1.25).mean(), (thresh < 1.25 ** 2).mean(), (thresh < 1.25 ** 3).mean())


def post_process(left, right):
    _, h, w = left.shape
    x = np.tile(np.linspace(0, 1, w), (h, 1))
    left_mask = (1 - np.clip(20 * (x - 0.05), 0, 1))[None]
    right_mask = left_mask[:, :, ::-1]
    return right_mask * left + left_mask * right + (1 - left_mask - right_mask) * (left + right) / 2


def make_dataset(opt, filenames, height, width):
    args = (opt.data_path, filenames, height, width, [0], 1)
    if opt.dataset == 'kitti':
        return datasets.KITTIRAWDataset(*args, is_train=False, img_ext='.png' if opt.png else '.jpg')
    if opt.dataset == 'drivingstereo':
        return datasets.DrivingstereoDataset(*args, is_train=False, img_ext='.png')
    if opt.dataset == 'cityscapes':
        return datasets.CityscapesEvalDataset(*args, is_train=False)
    ext = '.jpg' if opt.dataset in {'nuscenes', 'waymo', 'ddad', 'sunrgbd'} else '.png'
    fmt = '{:04d}' if opt.dataset in {'nyu', 'ibims', 'diml'} else '{:06d}'
    return datasets.PreprocessedDataset(*args, is_train=False, img_ext=ext, name_format=fmt)


def gt_files(opt, count):
    root = Path(opt.gt_path or Path(opt.data_path) / 'depth')
    if not root.is_dir():
        raise FileNotFoundError(f'Ground-truth directory does not exist: {root}. Set DATA_ROOT or edit the paths in scripts/testing_submit.sh.')
    if opt.dataset == 'cityscapes':
        files = [root / f'{i:03d}_depth.npy' for i in range(count)]
    else:
        suffix = '.npz' if opt.dataset == 'ddad' else '.png' if opt.dataset in {
            'nyu', 'ibims', 'sunrgbd', '7scenes', 'diml', 'drivingstereo'} else '.npy'
        files = sorted(root.glob('*' + suffix))
    if len(files) != count:
        raise ValueError(f'{opt.dataset}: {count} split entries but {len(files)} ground-truth files in {root}')
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    return files


def load_gt(opt, path):
    if opt.dataset == 'ddad':
        return sparse.load_npz(path).toarray().astype(np.float16)
    if path.suffix == '.png':
        scale = {'nyu': 5000, 'ibims': 1000, 'sunrgbd': 5000, '7scenes': 5000,
                 'diml': 1000, 'drivingstereo': 256}[opt.dataset]
        return imageio.imread(path).astype(np.float32) / scale
    depth = np.load(path)
    if opt.dataset == 'diode':
        mask_root = Path(opt.data_path) / 'mask'
        masks = sorted(mask_root.glob('*.npy'))
        depths = sorted(path.parent.glob('*.npy'))
        if len(masks) != len(depths):
            raise ValueError('DIODE mask and depth counts differ')
        mask = np.load(masks[depths.index(path)])
        # Preserve the sorted mask/depth pairing of the prepared release data.
        depth = np.squeeze(depth)
        mask = np.squeeze(mask).astype(bool)
        if depth.shape != mask.shape:
            raise ValueError('DIODE mask and depth shapes differ')
        depth = np.where(mask, depth, 0)
    return depth


def aligned_errors(opt, disparity, gt):
    """Return per-image metrics and median scale under the original dataset protocol."""
    _, minimum, maximum = PROTOCOLS[opt.dataset]
    if opt.dataset == 'cityscapes':
        # Keep the dedicated Cityscapes script's interpolation, crop and clipping.
        height = int(round(gt.shape[0] * 0.75))
        gt = torch.as_tensor(gt[:height], dtype=torch.float32)
        disp = torch.as_tensor(disparity)[None, None]
        pred = 1 / F.interpolate(disp, (height, gt.shape[1]), mode='bilinear', align_corners=True)[0, 0]
        gt, pred = gt[256:, 192:1856], pred[256:, 192:1856]
        mask = torch.isfinite(gt) & (gt > minimum) & (gt < maximum)
        gt, pred = gt[mask], pred[mask]
        if not gt.numel():
            raise ValueError('Cityscapes image has no valid pixels')
        pred *= opt.pred_depth_scale_factor
        ratio = 1.0 if opt.disable_median_scaling else (torch.median(gt) / torch.median(pred)).item()
        pred = (pred * ratio).clamp(minimum, maximum)
        return compute_errors(gt.numpy(), pred.numpy()), ratio
    if opt.dataset in {'nuscenes', 'waymo'}:
        pred = 1 / cv2.resize(disparity, (opt.gt_width, opt.gt_height))
        x, y, depth = gt[:, 0], gt[:, 1], gt[:, 2]
        mask = np.isfinite(gt).all(axis=1) & (x >= 0) & (x < opt.gt_width) & (y >= 0) & (y < opt.gt_height) & (depth > minimum) & (depth < maximum)
        gt, pred = depth[mask], pred[y[mask].astype(int), x[mask].astype(int)]
    else:
        height, width = gt.shape[:2]
        pred = 1 / cv2.resize(disparity, (width, height))
        mask = np.isfinite(gt) & (gt > minimum) & (gt < maximum)
        if opt.dataset in {'kitti', 'nyu'}:
            crop = (np.array([0.40810811 * height, 0.99189189 * height, 0.03594771 * width, 0.96405229 * width]).astype(int)
                    if opt.dataset == 'kitti' else [45, 471, 41, 601])
            crop_mask = np.zeros(mask.shape, dtype=bool)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = True
            mask &= crop_mask
        gt, pred = gt[mask], pred[mask]
        # The original script caps these predictions before median scaling.
        # SUN-RGBD and 7-Scenes had an adjacent-string typo; both are included here.
        if opt.dataset in {'nyu', 'ibims', 'sunrgbd', '7scenes'}:
            pred = np.minimum(pred, 10)
    if not gt.size or not np.isfinite(pred).all() or np.any(pred <= 0):
        raise ValueError(f'{opt.dataset}: no valid pixels or invalid predicted depth')
    pred *= opt.pred_depth_scale_factor
    ratio = 1.0 if opt.disable_median_scaling else np.median(gt) / np.median(pred)
    return compute_errors(gt, pred * ratio), float(ratio)


def evaluate(opt):
    if opt.dataset not in PROTOCOLS:
        raise ValueError(f'Unknown dataset {opt.dataset}; choose from {list(PROTOCOLS)}')
    split = PROTOCOLS[opt.dataset][0]
    if opt.eval_split != split:
        raise ValueError(f'{opt.dataset} requires --eval_split {split}')
    if not opt.load_weights_folder:
        raise ValueError('--load_weights_folder must contain encoder.pth and depth.pth')
    device = torch.device('cpu' if opt.no_cuda else f'cuda:{opt.device_id}')
    folder = Path(opt.load_weights_folder).expanduser()
    enc_state = torch.load(folder / 'encoder.pth', map_location='cpu')
    dec_state = torch.load(folder / 'depth.pth', map_location='cpu')
    # Full MaSS checkpoints contain the frozen DINO backbone; no separate pretrain is needed.
    encoder = MaSS_Encoder(pretrained=False)
    decoder = DepthDecoder(encoder.num_ch_enc, scales=opt.scales)
    height, width = enc_state.pop('height'), enc_state.pop('width')
    if (opt.eval_height is None) != (opt.eval_width is None):
        raise ValueError('Specify both --eval_height and --eval_width')
    if opt.eval_height is not None:
        height, width = opt.eval_height, opt.eval_width
    if height <= 0 or width <= 0 or height % 32 or width % 32:
        raise ValueError('Evaluation height and width must be positive multiples of 32')
    enc_state.pop('use_stereo', None)
    encoder.load_state_dict(enc_state, strict=True)
    decoder.load_state_dict(dec_state, strict=True)
    encoder.to(device).eval()
    decoder.to(device).eval()
    filenames = readlines(Path(__file__).resolve().parent / 'splits' / split / 'test_files.txt')
    paths = gt_files(opt, len(filenames))
    loader = DataLoader(make_dataset(opt, filenames, height, width), batch_size=opt.batch_size,
                        shuffle=False, num_workers=opt.num_workers, pin_memory=device.type == 'cuda')
    errors, ratios, predictions = [], [], []
    index = 0
    print(f'Evaluation input: {height}x{width}; images: {len(filenames)}; checkpoint: {folder}', flush=True)
    with torch.no_grad():
        for batch in tqdm(loader, desc=f'Evaluating {opt.dataset}'):
            rgb = batch['color', 0, 0].to(device)
            if opt.post_process:
                rgb = torch.cat([rgb, torch.flip(rgb, [3])])
            features, _, _, _ = encoder(rgb)
            disp, _ = disp_to_depth(decoder(features)['disp', 0], opt.min_depth, opt.max_depth)
            disp = disp[:, 0].cpu().numpy()
            if opt.post_process:
                n = disp.shape[0] // 2
                disp = post_process(disp[:n], disp[n:, :, ::-1])
            for prediction in disp:
                error, ratio = aligned_errors(opt, prediction, load_gt(opt, paths[index]))
                errors.append(error)
                ratios.append(ratio)
                index += 1
            if opt.save_pred_disps:
                predictions.extend(disp)
    mean = np.mean(errors, axis=0)
    output = Path(opt.eval_out_dir or 'eval')
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / f'{opt.dataset}.csv'
    exists = csv_path.exists()
    with csv_path.open('a', newline='') as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(['model_name', 'weights', 'dataset', 'images', 'input_height', 'input_width', *METRICS])
        writer.writerow([opt.model_name, str(folder), opt.dataset, index, height, width, *mean.tolist()])
    if opt.save_pred_disps:
        np.save(output / f'disps_{opt.dataset}.npy', np.stack(predictions))
    print(' | '.join(METRICS))
    print(' | '.join(f'{value:.4f}' for value in mean))
    print(f'Median scale: {np.median(ratios):.4f}; results: {csv_path}')
    return mean
