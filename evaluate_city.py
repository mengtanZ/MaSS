"""Cityscapes entrypoint, retaining the dedicated crop/interpolation protocol."""
import os
from options import Options

if __name__ == '__main__':
    options = Options().parse()
    options.dataset = 'cityscapes'
    options.eval_split = 'cityscapes'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(options.vis_device_id)
    from evaluation import evaluate
    evaluate(options)
