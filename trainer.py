from __future__ import absolute_import, division, print_function
import io
import json
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tensorboardX import SummaryWriter
from kornia.geometry import depth_to_normals
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import datasets
import networks
from networks.layers import *
from networks.contrastive_loss import compute_contrastive_lossesv2
from networks.linear_warmup_cosine_annealing_warm_restarts_weight_decay import ChainedScheduler
from utils.utils import *
random.seed(640)
np.random.seed(640)
torch.manual_seed(640)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(640)

class Trainer:

    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"
        self.models = {}
        self.models_pose = {}
        self.parameters_to_train = []
        self.parameters_to_train_pose = []
        self.device = torch.device('cpu' if self.opt.no_cuda else 'cuda:{}'.format(self.opt.device_id))
        self.zero_tensor = (torch.tensor(0.0, device=self.device, requires_grad=False), torch.tensor(0.0, device=self.device, requires_grad=False), torch.tensor(0.0, device=self.device, requires_grad=False))
        self.num_scales = len(self.opt.scales)
        self.frame_ids = len(self.opt.frame_ids)
        self.num_pose_frames = 2
        assert self.opt.frame_ids[0] == 0, 'frame_ids must start with 0'
        self.use_pose_net = True
        self.models['encoder'] = networks.MaSS_Encoder(backbone_type='mass', freeze=True, weights_path=self.opt.dino_weights)
        self.models['depth'] = networks.DepthDecoder(self.models['encoder'].num_ch_enc, self.opt.scales)
        self.models['encoder'].to(self.device)
        self.parameters_to_train += list(self.models['encoder'].parameters())
        self.models['depth'].to(self.device)
        self.parameters_to_train += list(self.models['depth'].parameters())
        self.models_pose['pose_encoder'] = networks.ResnetEncoder(self.opt.num_layers, True, num_input_images=self.num_pose_frames)
        self.models_pose['pose'] = networks.PoseDecoder(self.models_pose['pose_encoder'].num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
        self.models_pose['pose_encoder'].to(self.device)
        self.parameters_to_train_pose += list(self.models_pose['pose_encoder'].parameters())
        if 'pose' in self.models_pose:
            self.models_pose['pose'].to(self.device)
            self.parameters_to_train_pose += list(self.models_pose['pose'].parameters())
        self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.lr[0], weight_decay=self.opt.weight_decay)
        if self.use_pose_net:
            self.model_pose_optimizer = optim.AdamW(self.parameters_to_train_pose, self.opt.lr[3], weight_decay=self.opt.weight_decay)
        self.model_lr_scheduler = ChainedScheduler(self.model_optimizer, T_0=int(self.opt.lr[2]), T_mul=1, eta_min=self.opt.lr[1], last_epoch=-1, max_lr=self.opt.lr[0], warmup_steps=0, gamma=0.9)
        self.model_pose_lr_scheduler = ChainedScheduler(self.model_pose_optimizer, T_0=int(self.opt.lr[5]), T_mul=1, eta_min=self.opt.lr[4], last_epoch=-1, max_lr=self.opt.lr[3], warmup_steps=0, gamma=0.9)
        if self.opt.load_weights_folder is not None:
            self.load_model()
        if self.opt.mypretrain is not None:
            print('--mypretrain is a legacy Lite-Mono option; MaSS uses --dino_weights and a randomly initialized spatial prior.')
        print('Training model named:\n  ', self.opt.model_name)
        print('Models and tensorboard events files are saved to:\n  ', self.opt.log_dir)
        print('Training is using:\n  ', self.device)
        dataset_names = self.opt.dataset.split(',')
        split_names = self.opt.split.split(',')
        data_paths = self.opt.data_path.split(',')
        assert len(dataset_names) == len(split_names) == len(data_paths), 'dataset, split, and data_path must have the same number of comma-separated elements'
        datasets_dict = {'kitti_pp': datasets.KPreprocessedDataset, 'ddad': datasets.PreprocessedDataset, 'nyu': datasets.PreprocessedDataset, 'nuscenes': datasets.PreprocessedDataset}
        train_datasets = []
        val_datasets = []
        for dataset_name, split_name, data_path in zip(dataset_names, split_names, data_paths):
            dataset_class = datasets_dict[dataset_name]
            with open(f'splits/{split_name}/train_files.txt', 'r') as f:
                train_filenames = f.read().splitlines()
            with open(f'splits/{split_name}/val_files.txt', 'r') as f:
                val_filenames = f.read().splitlines()
            img_ext = '.png' if self.opt.png else '.jpg'
            train_dataset = dataset_class(data_path, train_filenames, self.opt.height, self.opt.width, self.opt.frame_ids, 4, is_train=True, img_ext=img_ext, psedo_depth='psedo_depth')
            val_dataset = dataset_class(data_path, val_filenames, self.opt.height, self.opt.width, self.opt.frame_ids, 4, is_train=True, img_ext=img_ext, psedo_depth='psedo_depth')
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
        train_dataset = ConcatDataset(train_datasets)
        self.train_loader = DataLoader(train_dataset, self.opt.batch_size, True, num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = ConcatDataset(val_datasets)
        self.val_loader = DataLoader(val_dataset, self.opt.batch_size, True, num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)
        num_train_samples = len(train_dataset)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
        self.writers = {}
        for mode in ['train', 'val']:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))
        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)
        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // 2 ** scale
            w = self.opt.width // 2 ** scale
            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)
            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)
        self.depth_metric_names = ['de/abs_rel', 'de/sq_rel', 'de/rms', 'de/log_rms', 'da/a1', 'da/a2', 'da/a3']
        print('Using split:\n  ', self.opt.split)
        print('There are {:d} training items and {:d} validation items\n'.format(len(train_dataset), len(val_dataset)))
        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()
        for m in self.models_pose.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()
        for m in self.models_pose.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()
            if self.opt.max_steps > 0 and self.step >= self.opt.max_steps:
                self.save_model('weights_{}_step_{}'.format(self.epoch, self.step))
                break
        for writer in self.writers.values():
            writer.close()
        print('Completed optimizer steps: {}'.format(self.step))

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        print('Training')
        self.set_train()
        self.model_lr_scheduler.step()
        if self.use_pose_net:
            self.model_pose_lr_scheduler.step()
        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()
            outputs, losses = self.process_batch(inputs)
            self.model_optimizer.zero_grad()
            if self.use_pose_net:
                self.model_pose_optimizer.zero_grad()
            losses['loss'].backward()
            self.model_optimizer.step()
            if self.use_pose_net:
                self.model_pose_optimizer.step()
            duration = time.time() - before_op_time
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 20000
            late_phase = self.step % 1000 == 0
            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses['loss'].cpu().data)
                self.log('train', inputs, outputs, losses)
            self.step += 1
            if self.opt.max_steps > 0 and self.step >= self.opt.max_steps:
                break

    def extract_global_feature(self, feature_map):
        global_feat = F.adaptive_avg_pool2d(feature_map, (1, 1)).view(feature_map.size(0), -1)
        global_feat = F.normalize(global_feat, dim=1)
        return global_feat

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        if self.opt.contrastive_learning:
            features, semantic_features, structural_features, dino_features = self.models['encoder'](inputs['color_aug', 0, 0])
            outputs = self.models['depth'](features)
            outputs['semantic_descriptor'] = self.extract_global_feature(semantic_features[-1])
            outputs['structural_descriptor'] = self.extract_global_feature(structural_features[-1])
            outputs['dino_descriptor'] = self.extract_global_feature(dino_features[-1])
        else:
            features, semantic_features, structural_features, dino_features = self.models['encoder'](inputs['color_aug', 0, 0])
            outputs = self.models['depth'](features)
        pose_features = features
        outputs.update(self.predict_poses(inputs, pose_features))
        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)
        return (outputs, losses)

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        pose_feats = {f_i: inputs['color_aug', f_i, 0] for f_i in self.opt.frame_ids}
        pose_feature_cache = None
        for f_i in self.opt.frame_ids[1:]:
            if f_i != 's':
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]
                pose_inputs = [self.models_pose['pose_encoder'](torch.cat(pose_inputs, 1))]
                axisangle, translation = self.models_pose['pose'](pose_inputs)
                outputs['axisangle', 0, f_i] = axisangle
                outputs['translation', 0, f_i] = translation
                outputs['cam_T_cam', 0, f_i] = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=f_i < 0)
                zero_translation = translation[:, 0] * torch.tensor([0, 0, 0], dtype=translation.dtype, device=translation.device)
                outputs['cam_T_cam_purer', 0, f_i] = transformation_from_parameters(axisangle[:, 0], zero_translation, invert=f_i < 0)
        return outputs

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs['disp', scale]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode='bilinear', align_corners=False)
                source_scale = 0
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs['depth', 0, scale] = depth
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                if frame_id == 's':
                    T = inputs['stereo_T']
                else:
                    T = outputs['cam_T_cam', 0, frame_id]
                    R = outputs['cam_T_cam_purer', 0, frame_id]
                cam_points = self.backproject_depth[source_scale](depth, inputs['inv_K', source_scale])
                pix_coords = self.project_3d[source_scale](cam_points, inputs['K', source_scale], T)
                outputs['sample', frame_id, scale] = pix_coords
                outputs['color', frame_id, scale] = F.grid_sample(inputs['color', frame_id, source_scale], outputs['sample', frame_id, scale], padding_mode='border', align_corners=True)
                iter_key = ('cam_T_cam_iters', 0, frame_id)
                if not self.opt.disable_automasking:
                    outputs['color_identity', frame_id, scale] = inputs['color', frame_id, source_scale]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0
        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0
            disp = outputs['disp', scale]
            color = inputs['color', 0, scale]
            target = inputs['color', 0, source_scale]
            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs['color', frame_id, scale]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)
            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs['color', frame_id, source_scale]
                    identity_reprojection_losses.append(self.compute_reprojection_loss(pred, target))
                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    identity_reprojection_loss = identity_reprojection_losses
            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses
            if not self.opt.disable_automasking:
                identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape, device=self.device) * 1e-05
                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss
            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)
            if not self.opt.disable_automasking:
                outputs['identity_selection/{}'.format(scale)] = (idxs > identity_reprojection_loss.shape[1] - 1).float()
            loss += to_optimise.mean()
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-07)
            smooth_loss = get_smooth_loss(norm_disp, color)
            loss += self.opt.disparity_smoothness * smooth_loss / 2 ** scale
            total_loss += loss
            if scale == 0:
                losses['photometric_loss'] = loss
        total_loss /= self.num_scales
        if self.opt.contrastive_learning:
            losses_back = compute_contrastive_lossesv2(inputs, outputs)
            contrast_loss = losses_back['loss_total_contrastive']
            loss_structure_cl = losses_back['loss_structure_cl']
            loss_consistency = losses_back['loss_consistency']
            loss_orthogonality = losses_back['loss_orthogonality']
            losses['loss_structure_cl'] = loss_structure_cl
            losses['loss_consistency'] = loss_consistency
            losses['loss_orthogonality'] = loss_orthogonality
            total_loss += self.opt.contrast_weights * contrast_loss
            losses['contrastive_loss'] = contrast_loss
        if not inputs['psedo_depth'].shape == outputs['depth', 0, 0].shape:
            inputs['psedo_depth'] = F.interpolate(inputs['psedo_depth'], [self.opt.height, self.opt.width], mode='bilinear', align_corners=False)
        tgt_normal = depth_to_normals(outputs['depth', 0, 0], inputs['K', 0][:, :3, :3])
        tgt_pseudo_normal = depth_to_normals(inputs['psedo_depth'], inputs['K', 0][:, :3, :3])
        loss_normal = (tgt_normal - tgt_pseudo_normal).abs().mean()
        total_loss += self.opt.psedo_weights * loss_normal
        losses['normal_loss'] = loss_normal
        losses['loss'] = total_loss
        return losses

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = 'epoch {:>3} | lr {:.6f} |lr_p {:.6f} | batch {:>6} | examples/s: {:5.1f}' + ' | loss: {:.5f} | time elapsed: {} | time left: {}'
        print(print_string.format(self.epoch, self.model_optimizer.state_dict()['param_groups'][0]['lr'], self.model_pose_optimizer.state_dict()['param_groups'][0]['lr'], batch_idx, samples_per_sec, loss, sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        writer = self.writers[mode]
        for name, value in losses.items():
            writer.add_scalar(name, value, self.step)
        for j in range(self.opt.batch_size):
            writer.add_image(f'rgb/{j}', inputs['color', 0, 0][j].data, self.step)
            writer.add_image(f'disp/{j}', normalize_image(outputs['disp', 0][j]), self.step)
        if 'structure_sim_matrix' in outputs:
            sim = outputs['structure_sim_matrix'].detach().cpu().numpy()
            fig, ax = plt.subplots()
            im = ax.imshow(sim, cmap='plasma')
            plt.colorbar(im, ax=ax)
            ax.set_title('KL-based Structural Similarity')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            image = np.array(Image.open(buf))
            writer.add_image('similarity/structure', image.transpose(2, 0, 1), self.step, dataformats='CHW')
            plt.close(fig)
        if 'positive_mask' in outputs:
            mask = outputs['positive_mask'].detach().float().cpu().numpy()
            fig, ax = plt.subplots()
            im = ax.imshow(mask, cmap='gray')
            plt.colorbar(im, ax=ax)
            ax.set_title('Positive Mask (Thresholded Similarities)')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            image = np.array(Image.open(buf))
            writer.add_image('mask/positive', image.transpose(2, 0, 1), self.step, dataformats='CHW')
            plt.close(fig)
        if 'feature_sim_matrix' in outputs:
            feat_sim = outputs['feature_sim_matrix'].detach().cpu().numpy()
            fig, ax = plt.subplots()
            im = ax.imshow(feat_sim, cmap='plasma')
            plt.colorbar(im, ax=ax)
            ax.set_title('Feature Cosine Similarity')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            image = np.array(Image.open(buf))
            writer.add_image('similarity/feature', image.transpose(2, 0, 1), self.step, dataformats='CHW')
            plt.close(fig)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, 'models')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()
        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self, folder_name=None):
        """Save model weights to disk
        """
        if folder_name is None:
            folder_name = 'weights_{}'.format(self.epoch)
        save_folder = os.path.join(self.log_path, 'models', folder_name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, '{}.pth'.format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = False
            torch.save(to_save, save_path)
        for model_name, model in self.models_pose.items():
            save_path = os.path.join(save_folder, '{}.pth'.format(model_name))
            to_save = model.state_dict()
            torch.save(to_save, save_path)
        save_path = os.path.join(save_folder, '{}.pth'.format('adam'))
        torch.save(self.model_optimizer.state_dict(), save_path)
        save_path = os.path.join(save_folder, '{}.pth'.format('adam_pose'))
        if self.use_pose_net:
            torch.save(self.model_pose_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)
        assert os.path.isdir(self.opt.load_weights_folder), 'Cannot find folder {}'.format(self.opt.load_weights_folder)
        print('loading model from folder {}'.format(self.opt.load_weights_folder))
        models_to_load = list(self.opt.models_to_load)
        for n in models_to_load:
            print('Loading {} weights...'.format(n))
            path = os.path.join(self.opt.load_weights_folder, '{}.pth'.format(n))
            if n in self.models_pose:
                model_dict = self.models_pose[n].state_dict()
                pretrained_dict = torch.load(path, map_location=self.device)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
                model_dict.update(pretrained_dict)
                self.models_pose[n].load_state_dict(model_dict)
            elif n in self.models:
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path, map_location=self.device)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)
            else:
                print("Skipping {} weights because posenet='{}' does not use it".format(n, 'mono2'))
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, 'adam.pth')
        optimizer_pose_load_path = os.path.join(self.opt.load_weights_folder, 'adam_pose.pth')
        if os.path.isfile(optimizer_pose_load_path):
            print('Loading Adam weights')
            optimizer_dict = torch.load(optimizer_load_path, map_location=self.device)
            self.model_optimizer.load_state_dict(optimizer_dict)
            optimizer_pose_dict = torch.load(optimizer_pose_load_path, map_location=self.device)
            try:
                self.model_pose_optimizer.load_state_dict(optimizer_pose_dict)
            except ValueError as e:
                print('Skipping Adam pose weights because optimizer structure changed: {}'.format(e))
        else:
            print('Cannot find Adam weights so Adam is randomly initialized')
