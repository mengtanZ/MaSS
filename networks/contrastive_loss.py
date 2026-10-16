import torch
import torch.nn.functional as F
import cv2

def differentiable_depth_histogram(depth_map, d_min=0, d_max=100, bins=20, a=1.0):
    """
    depth_map: [B,1,H,W]
    返回: [B, bins]，可微分深度分布
    """
    device = depth_map.device
    L = (d_max - d_min) / bins
    centers = d_min + L * (torch.arange(bins, device=device) + 0.5)
    depth_expanded = depth_map.view(depth_map.size(0), 1, -1)
    centers = centers.view(1, -1, 1)
    probs = torch.sigmoid((depth_expanded - centers + L / 2) / a) - torch.sigmoid((depth_expanded - centers - L / 2) / a)
    probs = probs.mean(dim=-1)
    return probs

def compute_semantic_dino_consistency_loss(outputs, temperature=0.1, eps=1e-08):
    """
    软一致性：保持 semantic_descriptor 与 dino_descriptor 的 batch-wise 相似度矩阵结构相近
    实现：对两组相似度矩阵做行级 softmax (概率分布)，然后计算对称 KL。
    inputs:
        outputs: dict 包含 "semantic_descriptor" [B,C1], "dino_descriptor" [B,C2] (已归一化)
    returns:
        loss (scalar)
    """
    sem_feat = outputs['semantic_descriptor']
    dino_feat = outputs['dino_descriptor']
    with torch.no_grad():
        sim_dino_orig = (dino_feat @ dino_feat.t()).detach()
    sim_dino_new = sem_feat @ sem_feat.t()
    loss_semantic_preserve = F.mse_loss(sim_dino_new, sim_dino_orig)
    loss = loss_semantic_preserve
    return loss

def compute_orthogonality_loss(outputs, dot_weight=1.0, crosscov_weight=1.0):
    """
    强制 semantic_descriptor 和 structural_descriptor 正交 / 去相关
    两个组成部分：
      1) per-sample dot: 每个样本的语义向量与结构向量的点积幅值越小越好
      2) cross-covariance: batch 维度下的语义-结构协方差矩阵的 Frobenius 范数越小越好
    inputs:
      outputs: dict 包含 "semantic_descriptor" 和 "structural_descriptor", 均为 [B, C] (优先 normalize)
      dot_weight, crosscov_weight: 权重
    returns:
      loss (scalar)
    """
    sem = outputs['semantic_descriptor']
    stru = outputs['structural_descriptor']
    B = sem.size(0)
    device = sem.device
    if B == 0:
        return torch.tensor(0.0, device=device)
    per_sample_dot = (sem * stru).sum(dim=1)
    loss_dot = per_sample_dot.abs().mean()
    loss = dot_weight * loss_dot
    return loss

def depth_as_structure_map(depth, downsample_ratio=8):
    """
    输入:
        depth: torch.Tensor, shape = [B, 1, H, W]

    输出:
        structure: torch.Tensor, shape = [B, 1, H//ratio, W//ratio]
    """
    depth_norm = depth.float() / 500
    structure = F.avg_pool2d(depth_norm, kernel_size=downsample_ratio, stride=downsample_ratio)
    return structure

def compute_structure_contrastive_loss(inputs, outputs, hist_bins=100, kl_threshold=0.05, temperature=0.1, eps=1e-08):
    depth_map = inputs['psedo_depth']
    structure_map = depth_as_structure_map(depth_map)
    batch_hist = differentiable_depth_histogram(structure_map)
    outputs['batch_hist'] = batch_hist
    B = batch_hist.size(0)
    kl_matrix = torch.zeros(B, B, device=batch_hist.device)
    for i in range(B):
        for j in range(B):
            kl_ij = F.kl_div((batch_hist[j] + 1e-08).log(), batch_hist[i], reduction='batchmean')
            kl_ji = F.kl_div((batch_hist[i] + 1e-08).log(), batch_hist[j], reduction='batchmean')
            kl_matrix[i, j] = 0.5 * (kl_ij + kl_ji)
    sim_matrix = torch.exp(-kl_matrix)
    outputs['structure_sim_matrix'] = sim_matrix
    pos_mask = sim_matrix > 1 - kl_threshold
    outputs['positive_mask'] = pos_mask
    features = outputs['structural_descriptor']
    features = F.normalize(features, dim=1)
    cos_sim = features @ features.t()
    outputs['feature_sim_matrix'] = cos_sim
    loss_cl = 0
    for i in range(B):
        pos_idx = pos_mask[i].nonzero(as_tuple=False).squeeze(1)
        if pos_idx.numel() == 0:
            continue
        pos_sim = cos_sim[i, pos_idx] / temperature
        all_sim = cos_sim[i, :] / temperature
        loss_i = -torch.log(torch.exp(pos_sim).sum() / torch.exp(all_sim).sum())
        loss_cl += loss_i
    loss_cl /= B
    return loss_cl

def compute_contrastive_lossesv2(inputs, outputs, weights=None, hist_bins=100, kl_threshold=0.01, temperature=0.1):
    """
    汇总所有 loss：
      - semantic-dino soft consistency (保持大模型语义结构)
      - orthogonality between semantic & structural
      - structural contrastive guided by depth-histogram
    weights: dict 可指定各 loss 权重，例如:
        weights = {"consistency": 1.0, "orthogonality": 1.0, "structure_cl": 1.0}
    返回 dict: losses 与 total_loss
    """
    if weights is None:
        weights = {'consistency': 1.0, 'orthogonality': 1.0, 'structure_cl': 1.0}
    loss_consistency = compute_semantic_dino_consistency_loss(outputs, temperature=temperature)
    loss_orth = compute_orthogonality_loss(outputs, dot_weight=1.0, crosscov_weight=1.0)
    loss_struct = compute_structure_contrastive_loss(inputs, outputs, hist_bins=hist_bins, kl_threshold=kl_threshold, temperature=temperature)
    total = weights['consistency'] * loss_consistency + weights['orthogonality'] * loss_orth + weights['structure_cl'] * loss_struct
    return {'loss_consistency': loss_consistency, 'loss_orthogonality': loss_orth, 'loss_structure_cl': loss_struct, 'loss_total_contrastive': total}
