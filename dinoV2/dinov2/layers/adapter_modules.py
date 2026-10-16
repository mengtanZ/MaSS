# MaSS release: removed the unused Lite-Mono wrapper and debugger import.
import logging
import math
import os
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from functools import partial
from timm.models.layers import DropPath
import torch.nn.functional as F

class DWConv(nn.Module):

    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21
        x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x

class ConvFFN(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Injector(nn.Module):

    def __init__(self, dim, num_heads=6, norm_layer=partial(nn.LayerNorm, eps=1e-06), init_values=0.0, with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.gamma = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)

    def forward(self, query, feat):
        """
        query: [B, N, C] - token embeddings
        feat:  [B, M, C] - external features to inject
        """

        def _inner_forward(query, feat):
            q = self.query_norm(query)
            k = self.feat_norm(feat)
            v = k
            attn_output, _ = self.attn(q, k, v)
            return query + self.gamma * attn_output
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)
        return query

class Extractor(nn.Module):

    def __init__(self, dim, num_heads=6, with_cffn=True, cffn_ratio=0.25, drop=0.0, drop_path=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-06), with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, query, feat, H, W):

        def _inner_forward(query, feat):
            q = self.query_norm(query)
            kv = self.feat_norm(feat)
            attn_out, _ = self.attn(q, kv, kv)
            query = query + attn_out
            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
            return query
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)
        return query

class SpatialPriorModule(nn.Module):

    def __init__(self, inplanes=64, embed_dim=384, with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.stem = nn.Sequential(*[nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False), nn.SyncBatchNorm(inplanes), nn.ReLU(inplace=True), nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False), nn.SyncBatchNorm(inplanes), nn.ReLU(inplace=True), nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False), nn.SyncBatchNorm(inplanes), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)])
        self.conv2 = nn.Sequential(*[nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False), nn.SyncBatchNorm(2 * inplanes), nn.ReLU(inplace=True)])
        self.conv3 = nn.Sequential(*[nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False), nn.SyncBatchNorm(4 * inplanes), nn.ReLU(inplace=True)])
        self.conv4 = nn.Sequential(*[nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False), nn.SyncBatchNorm(4 * inplanes), nn.ReLU(inplace=True)])
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):

        def _inner_forward(x):
            c1 = self.stem(x)
            c2 = self.conv2(c1)
            c3 = self.conv3(c2)
            c4 = self.conv4(c3)
            c1 = self.fc1(c1)
            c2 = self.fc2(c2)
            c3 = self.fc3(c3)
            c4 = self.fc4(c4)
            bs, dim, _, _ = c1.shape
            c2 = c2.view(bs, dim, -1).transpose(1, 2)
            c3 = c3.view(bs, dim, -1).transpose(1, 2)
            c4 = c4.view(bs, dim, -1).transpose(1, 2)
            return (c1, c2, c3, c4)
        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)
        return outs
