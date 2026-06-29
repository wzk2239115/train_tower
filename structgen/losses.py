"""Multi-objective geometry losses for SDF / occupancy voxel fields.

The flow-matching velocity loss is reused from ``tower.train.losses`` (the
decoder predicts x0, the clean field). The remaining losses operate on the
predicted clean field vs the GT field:

* SDF L1 / L2 regression
* occupancy BCE (sign of SDF)
* Chamfer distance on surfaces extracted from predicted vs GT fields
* normal consistency (gradient-alignment on the zero level set)
* curvature smoothness (Laplacian of the SDF)
* topology / connectivity (small-component penalty)
* manufacturability (overhang / minimum-wall-thickness penalty)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:  # reuse the project's rectified-flow velocity loss
    from tower.train.losses import rectified_flow_velocity_loss as _rf_velocity_loss
except Exception:  # pragma: no cover - standalone fallback
    def _rf_velocity_loss(x_pred, z, t, x_clean, *, t_eps: float = 0.05):
        denom = (1.0 - t).clamp_min(t_eps)
        while denom.ndim < z.ndim:
            denom = denom.unsqueeze(-1)
        v_pred = (x_pred - z) / denom
        v_target = (x_clean - z) / denom
        return F.mse_loss(v_pred, v_target)


def fm_velocity_loss(x_pred, z, t, x_clean, *, t_eps: float = 0.05) -> torch.Tensor:
    """Rectified-flow velocity MSE (project convention: net predicts x0)."""
    return _rf_velocity_loss(x_pred, z, t, x_clean, t_eps=t_eps)


# --------------------------------------------------------------------------- #
# Field-level losses
# --------------------------------------------------------------------------- #


def sdf_l1_loss(pred_field: torch.Tensor, gt_field: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred_field, gt_field)


def sdf_l2_loss(pred_field: torch.Tensor, gt_field: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_field, gt_field)


def occupancy_bce_loss(pred_field: torch.Tensor, gt_field: torch.Tensor,
                       eps: float = 1e-2) -> torch.Tensor:
    """BCE on occupancy derived from SDF sign.

    occ = sigmoid(-sdf/eps), so the logit is simply ``-sdf/eps`` (autocast-safe).
    """
    logit = -pred_field / eps
    g_occ = (gt_field < 0).float()
    return F.binary_cross_entropy_with_logits(logit, g_occ)


# --------------------------------------------------------------------------- #
# Derivative-based regularizers (normal consistency, curvature smoothness)
# --------------------------------------------------------------------------- #


def _central_diff(field: torch.Tensor) -> torch.Tensor:
    """Gradient of a (B,1,D,H,W) field via central differences. Returns (B,3,D,H,W)."""
    assert field.ndim == 5 and field.shape[1] == 1
    gx = F.pad(field[:, :, 2:, :, :] - field[:, :, :-2, :, :], (0, 0, 0, 0, 1, 1))
    gy = F.pad(field[:, :, :, 2:, :] - field[:, :, :, :-2, :], (0, 0, 1, 1, 0, 0))
    gz = F.pad(field[:, :, :, :, 2:] - field[:, :, :, :, :-2], (1, 1, 0, 0, 0, 0))
    return torch.cat([gx, gy, gz], dim=1)


def normal_consistency_loss(pred_field: torch.Tensor, gt_field: torch.Tensor,
                            band: float = 0.05) -> torch.Tensor:
    """Cosine alignment of SDF gradients near the surface (the zero level set)."""
    g_pred = _central_diff(pred_field)
    g_gt = _central_diff(gt_field)
    mask = (gt_field.abs() < band).float()
    if mask.sum() < 1:
        return pred_field.sum() * 0.0
    p = F.normalize(g_pred, dim=1)
    g = F.normalize(g_gt, dim=1)
    cos = (p * g).sum(dim=1, keepdim=True)
    return -(cos * mask).sum() / mask.sum().clamp_min(1.0) + 1.0


def curvature_smooth_loss(pred_field: torch.Tensor) -> torch.Tensor:
    """Penalize the Laplacian of the SDF → smooth, manufacturable surfaces."""
    g = _central_diff(pred_field)
    lap = (
        F.pad(g[:, 0:1, 2:, :, :] - g[:, 0:1, :-2, :, :], (0, 0, 0, 0, 1, 1))
        + F.pad(g[:, 1:2, :, 2:, :] - g[:, 1:2, :, :-2, :], (0, 0, 1, 1, 0, 0))
        + F.pad(g[:, 2:3, :, :, 2:] - g[:, 2:3, :, :, :-2], (1, 1, 0, 0, 0, 0))
    )
    return lap.pow(2).mean()


# --------------------------------------------------------------------------- #
# Chamfer on surface points
# --------------------------------------------------------------------------- #


def _voxel_xyz(field: torch.Tensor, n: int) -> torch.Tensor:
    """World coords of each voxel center in [-1,1] for a (B,1,D,H,W) field."""
    B = field.shape[0]
    lin = torch.linspace(-1.0 + 1.0 / n, 1.0 - 1.0 / n, n, device=field.device)
    X, Y, Z = torch.meshgrid(lin, lin, lin, indexing="ij")
    return torch.stack([X, Y, Z], -1).reshape(1, n * n * n, 3).expand(B, -1, -1)


def chamfer_loss_from_field(pred_field: torch.Tensor, gt_surface: torch.Tensor,
                            n_sample: int = 1024) -> torch.Tensor:
    """Approx Chamfer between predicted surface points and GT surface points.

    ``pred_field``: (B,1,D,H,W). ``gt_surface``: (B,S,3). Sample predicted
    surface points from the near-zero band of the predicted field.
    """
    B, _, D, H, W = pred_field.shape
    flat = pred_field.reshape(B, -1)
    # sample surface voxel indices from the near-zero band
    xyz = _voxel_xyz(pred_field, D)  # (B, N, 3)
    losses = []
    for b in range(B):
        # distance estimate via |sdf|/|grad|
        g = _central_diff(pred_field[b:b + 1])[0]
        gnorm = g.norm(dim=0).reshape(-1) + 1e-6
        dist_est = flat[b].abs() / gnorm
        # weighted sample of near-surface voxels
        w = torch.exp(-(dist_est ** 2) / (2 * 0.03 ** 2))
        if w.sum() <= 0:
            continue
        idx = torch.multinomial(w, n_sample, replacement=True)
        pred_pts = xyz[b, idx]  # (n_sample,3)
        gt = gt_surface[b]  # (S,3)
        # bidirectional nearest-neighbor (downsample gt for speed)
        gt_s = gt[torch.randperm(gt.shape[0], device=gt.device)[:n_sample]]
        d1 = torch.cdist(pred_pts, gt_s).min(dim=1).values.mean()
        d2 = torch.cdist(gt_s, pred_pts).min(dim=1).values.mean()
        losses.append(d1 + d2)
    if not losses:
        return pred_field.sum() * 0.0
    return torch.stack(losses).mean()


# --------------------------------------------------------------------------- #
# Topology / connectivity & manufacturability (cheap differentiable proxies)
# --------------------------------------------------------------------------- #


def topology_loss(pred_field: torch.Tensor, min_thick: float = 0.04) -> torch.Tensor:
    """Encourage connected walls: penalize very thin solid slivers.

    Thin slivers = voxels that are solid (sdf<0) but adjacent to a strongly
    positive SDF (i.e. close to a wall that is too thin). Differentiable proxy:
    penalize high gradient magnitude *inside* the solid.
    """
    g = _central_diff(pred_field)
    gnorm = g.norm(dim=1, keepdim=True)
    solid = torch.clamp(-pred_field / min_thick, 0.0, 1.0)  # 1 deep inside
    return (gnorm * solid).mean()


def manufacturability_loss(pred_field: torch.Tensor, overhang_axis: int = 2,
                           max_overhang: float = 0.5) -> torch.Tensor:
    """Discourage unsupported overhangs along ``overhang_axis`` (print direction).

    Proxy: solid voxels whose *upstream* neighbour along the build axis is empty
    receive a penalty proportional to the lateral SDF gradient (steep overhang).
    """
    g = _central_diff(pred_field)
    lateral = [i for i in range(3) if i != overhang_axis]
    overhang = (g[:, lateral].norm(dim=1, keepdim=True)).clamp(max=2.0)
    solid = (pred_field < 0).float()
    # upstream (previous slice along build axis) occupancy
    up = F.pad(solid[:, :, :, :, :-1], (0, 1), value=0.0)
    unsupported = solid * (1.0 - up)
    return (unsupported * overhang).mean() * max_overhang


# --------------------------------------------------------------------------- #
# Aggregator
# --------------------------------------------------------------------------- #


def compute_all_losses(pred_field: torch.Tensor, gt_field: torch.Tensor,
                       z: torch.Tensor, t: torch.Tensor, x_clean: torch.Tensor,
                       gt_surface: torch.Tensor,
                       weights, *, t_eps: float = 0.05) -> tuple[torch.Tensor, dict]:
    """``pred_field``/``gt_field`` are (B,1,D,H,W) SDF channels.

    Returns (total_loss, log_dict).
    """
    w = weights
    logs: dict[str, torch.Tensor] = {}
    total = pred_field.new_zeros(())

    if w.fm:
        _l = fm_velocity_loss(pred_field, z, t, x_clean, t_eps=t_eps)
        logs["loss/fm"] = _l.detach()
        total = total + w.fm * _l
    if w.sdf_l1:
        _l = sdf_l1_loss(pred_field, gt_field)
        logs["loss/sdf_l1"] = _l.detach()
        total = total + w.sdf_l1 * _l
    if w.sdf_l2:
        _l = sdf_l2_loss(pred_field, gt_field)
        logs["loss/sdf_l2"] = _l.detach()
        total = total + w.sdf_l2 * _l
    if w.occupancy_bce:
        _l = occupancy_bce_loss(pred_field, gt_field)
        logs["loss/occ_bce"] = _l.detach()
        total = total + w.occupancy_bce * _l
    if w.normal_consistency:
        _l = normal_consistency_loss(pred_field, gt_field)
        logs["loss/normal"] = _l.detach()
        total = total + w.normal_consistency * _l
    if w.curvature_smooth:
        _l = curvature_smooth_loss(pred_field)
        logs["loss/curv"] = _l.detach()
        total = total + w.curvature_smooth * _l
    if w.chamfer:
        _l = chamfer_loss_from_field(pred_field, gt_surface)
        logs["loss/chamfer"] = _l.detach()
        total = total + w.chamfer * _l
    if w.topology:
        _l = topology_loss(pred_field)
        logs["loss/topo"] = _l.detach()
        total = total + w.topology * _l
    if w.manufacturability:
        _l = manufacturability_loss(pred_field)
        logs["loss/manuf"] = _l.detach()
        total = total + w.manufacturability * _l

    logs["loss/total"] = total.detach()
    return total, {k: float(v) for k, v in logs.items()}
