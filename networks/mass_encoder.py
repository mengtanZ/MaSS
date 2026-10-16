import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torchvision.transforms as T
from dinoV2.dinov2.models import vit_base

class DINOv2Wrapper(nn.Module):
    def __init__(self, weights_path=None, freeze=True):
        super().__init__()
        self.model = vit_base(img_size=518, patch_size=14, init_values=1e-5,
                              ffn_layer='mlp', block_chunks=0, qkv_bias=True,
                              proj_bias=True, ffn_bias=True, num_register_tokens=4,
                              interpolate_offset=0.0, interpolate_antialias=True)
        if weights_path is not None:
            checkpoint = torch.load(weights_path, map_location='cpu')
            state = checkpoint.get('model', checkpoint)
            state = {k.removeprefix('pretrained.'): v for k,v in state.items() if not k.startswith('depth_head.')}
            report = self.model.load_state_dict(state, strict=False)
            missing = [k for k in report.missing_keys if not any(k.startswith(p) for p in ('spm.', 'injector.', 'extractor.', 'level_embed'))]
            if missing or report.unexpected_keys:
                raise RuntimeError(f'Incompatible DINOv2 weights: missing={missing}, unexpected={report.unexpected_keys}')
        self.patch_size = 14
        self.embed_dim = 768
        self.depth = 12
        if freeze:
            for name, param in self.model.named_parameters():
                if not any(k in name for k in ['spm', 'injector', 'level_embed', 'extractor']):
                    param.requires_grad = False
            self.model.eval()
        self.transform = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def forward(self, x):
        x = self.transform(x)
        return self.model.get_intermediate_layers(x, n=[3, 7, 11], reshape=True)


class Projector(nn.Module):

    def __init__(self, in_channels=768, scale=16):
        super().__init__()
        self.preprocess_1_16 = nn.Sequential(nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.preprocess_1_8 = nn.Sequential(nn.ConvTranspose2d(in_channels, 80, kernel_size=2, stride=2, padding=0), nn.BatchNorm2d(80), nn.ReLU(inplace=True))
        self.preprocess_1_4 = nn.Sequential(nn.ConvTranspose2d(in_channels, 48, kernel_size=4, stride=4, padding=0), nn.BatchNorm2d(48), nn.ReLU(inplace=True))

    def forward(self, features):
        feat_1_16 = self.preprocess_1_16(features[2])
        feat_1_8 = self.preprocess_1_8(features[1])
        feat_1_4 = self.preprocess_1_4(features[0])
        return [feat_1_4, feat_1_8, feat_1_16]

class Feature_Fuser(nn.Module):

    def __init__(self, num_ch_enc=[48, 80, 128], fusion_type='concat'):
        super().__init__()
        self.fusion_type = fusion_type
        self.num_ch_enc = num_ch_enc
        if fusion_type == 'concat':
            self.reduction_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(c * 2, c, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(c), nn.ReLU(inplace=True)) for c in num_ch_enc])
        else:
            self.reduction_layers = nn.ModuleList([nn.Identity() for _ in num_ch_enc])

    def forward(self, semantic_feats, structural_feats):
        fused_feats = []
        for i, (sem, stru) in enumerate(zip(semantic_feats, structural_feats)):
            if self.fusion_type == 'concat':
                fused = torch.cat([sem, stru], dim=1)
            elif self.fusion_type == 'add':
                fused = sem + stru
            else:
                raise ValueError(f'Unsupported fusion type: {self.fusion_type}')
            fused = self.reduction_layers[i](fused)
            fused_feats.append(fused)
        return fused_feats

class MaSS_Encoder(nn.Module):
    def __init__(self, backbone_type='mass', freeze=True, weights_path=None, pretrained=True):
        super().__init__()
        if backbone_type not in ('mass', 'mass_dinov2'):
            raise ValueError('This release contains the DINOv2 MaSS model only')
        self.dim = 128
        self.num_ch_enc = np.array([48, 80, 128])
        if pretrained:
            weights_path = weights_path or os.environ.get('MASS_DINO_WEIGHTS') or str(Path(__file__).resolve().parents[1] / 'weights/dinov2_vitb14_reg4_pretrain.pth')
            if not Path(weights_path).is_file():
                raise FileNotFoundError(f'DINOv2 weights not found: {weights_path}. See docs/WEIGHTS.md.')
        self.dino = DINOv2Wrapper(weights_path if pretrained else None, freeze)
        self.contextual_proj = Projector(in_channels=self.dino.embed_dim)
        self.structural_proj = Projector(in_channels=self.dino.embed_dim)
        self.fuser = Feature_Fuser()

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=14 / 16, mode='bilinear', align_corners=True)
        dino_features = self.dino(x)
        contextual_features = self.contextual_proj(dino_features)
        structural_features = self.structural_proj(dino_features)
        fused_features = self.fuser(contextual_features, structural_features)
        return fused_features, contextual_features, structural_features, dino_features
