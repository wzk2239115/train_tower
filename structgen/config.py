"""Configuration for structgen (complex surface / topology generation)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BackboneConfig:
    """Multimodal condition backbone (Step-3.7-Flash on compute box).

    On the dev machine the Step-3.7-Flash snapshot is incomplete, so a CLIP
    ``proxy`` backbone is used to exercise the full pipeline. On the compute
    box, switch ``kind="stepfun"`` and point ``pretrained_path`` at the full
    weights + modeling code.
    """

    kind: str = "proxy"  # "proxy" | "stepfun" | "cached"
    pretrained_path: str | None = None  # Step-3.7-Flash dir on compute box
    text_emb_path: str | None = None  # precomputed prompt->emb dict (.pt), kind="cached"
    cond_dim: int = 768  # output condition token dim fed into the decoder
    n_cond_tokens: int = 32  # pooled/selected token count used for cross-attn
    freeze: bool = True  # backbone is a *pretrained component*, not trained
    image_size: int = 224  # sketch / reference image side


@dataclass
class DecoderConfig:
    """3D voxel velocity network (rectified flow on the SDF/occupancy field)."""

    grid_res: int = 64  # voxel grid resolution (cube)
    field_channels: int = 2  # [sdf, occupancy]; 1 = sdf only
    base_channels: int = 64  # U-Net feature width
    channel_mults: tuple[int, ...] = (1, 2, 4, 8)
    num_blocks: int = 2  # conv blocks per stage
    cond_dim: int = 768  # must match backbone.cond_dim
    n_cond_tokens: int = 32
    cross_attn: bool = True  # inject backbone tokens via cross-attention
    use_self_cond: bool = True  # ELF-style self-conditioning


@dataclass
class FlowConfig:
    """Rectified flow schedule (reuses tower.train.losses primitives)."""

    time_schedule: str = "logit_normal"
    p_mean: float = -0.8
    p_std: float = 0.8
    t_eps: float = 0.05
    noise_scale: float = 1.0
    n_sample_steps: int = 50  # Euler steps at inference


@dataclass
class LossWeights:
    sdf_l1: float = 1.0
    sdf_l2: float = 0.0
    occupancy_bce: float = 1.0
    chamfer: float = 0.5
    normal_consistency: float = 0.1
    curvature_smooth: float = 0.05
    topology: float = 0.05
    manufacturability: float = 0.05
    fm: float = 1.0  # rectified-flow velocity MSE (reused from tower)


@dataclass
class TrainConfig:
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 0.0
    max_steps: int = 5000
    warmup_steps: int = 200
    grad_clip: float = 1.0
    device: str = "cuda"
    amp: bool = True
    log_every: int = 20
    save_every: int = 1000
    out_dir: str = "outputs/structgen"


@dataclass
class StructGenConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    loss_weights: LossWeights = field(default_factory=LossWeights)
    train: TrainConfig = field(default_factory=TrainConfig)
    seed: int = 0

    # data synthesis
    num_samples: int = 2048  # synthetic structures to generate
    surface_samples: int = 2048  # points sampled on GT surface for Chamfer
    real_data_dir: str | None = None  # ABC .npz dir (from convert-abc); None=synthesize
