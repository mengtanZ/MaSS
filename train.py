import os
from options import Options

if __name__ == '__main__':
    opts = Options().parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.vis_device_id)
    from trainer import Trainer
    Trainer(opts).train()
