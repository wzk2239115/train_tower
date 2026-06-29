"""Tests for structgen: SDF synthesis, losses, meshing, decoder forward/backward."""

from __future__ import annotations

import torch

torch.manual_seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_tpms_synthesis():
    from structgen.data import sampler

    recipes = sampler.all_recipes()
    assert len(recipes) >= 20
    for rp in recipes[:6]:
        field = sampler.build_field(rp, 32)
        assert field.shape == (32, 32, 32)
        assert field.min() < 0 < field.max()  # has both solid and empty


def test_topologies_covered():
    from structgen.data import sampler

    kinds = {rp.topology for rp in sampler.all_recipes()}
    for t in ("gyroid", "diamond", "schwarz_p", "graded_gyroid", "voronoi", "lattice"):
        assert t in kinds


def test_losses_compute():
    from structgen.config import LossWeights
    from structgen import losses as L

    gt = torch.randn(2, 1, 24, 24, 24, device=DEVICE) * 0.3
    pred = (gt + torch.randn_like(gt) * 0.05).requires_grad_(True)
    z = torch.randn_like(gt)
    t = torch.tensor([0.3, 0.7], device=DEVICE)
    surf = torch.randn(2, 128, 3, device=DEVICE) * 0.5
    total, logs = L.compute_all_losses(pred, gt, z, t, gt, surf, LossWeights())
    assert torch.isfinite(total)
    assert total.requires_grad
    for k in ("loss/fm", "loss/sdf_l1", "loss/occ_bce", "loss/chamfer"):
        assert k in logs


def test_meshing_and_export(tmp_path):
    from structgen.data import sampler
    from structgen.model.meshing import sdf_to_mesh, export_mesh, write_obj

    field = sampler.build_field(sampler.all_recipes()[0], 40)
    mesh = sdf_to_mesh(field)
    assert mesh is not None and len(mesh.vertices) > 0 and len(mesh.faces) > 0
    stl = tmp_path / "p.stl"
    export_mesh(mesh, str(stl))
    assert stl.stat().st_size > 0
    write_obj(mesh, str(tmp_path / "p.obj"))


def test_decoder_forward_backward():
    from structgen.config import DecoderConfig, StructGenConfig
    from structgen.model.geometry_decoder import GeometryDecoder

    cfg = StructGenConfig(decoder=DecoderConfig(
        grid_res=16, base_channels=16, channel_mults=(1, 2), num_blocks=1,
        field_channels=2, cross_attn=False))
    dec = GeometryDecoder(cfg.decoder).to(DEVICE)
    field = torch.randn(2, 2, 16, 16, 16, device=DEVICE) * 0.2
    pooled = torch.randn(2, cfg.decoder.cond_dim, device=DEVICE)
    surf = torch.randn(2, 64, 3, device=DEVICE) * 0.5
    loss, logs = dec.decode_loss(field, pooled, None, surf, cfg.loss_weights, cfg.flow)
    loss.backward()
    out = dec.sample(pooled, None, cfg.flow, device=DEVICE)
    assert out.shape == (2, 2, 16, 16, 16)


def test_proxy_backbone_offline():
    from structgen.model.backbone import ProxyBackbone

    bb = ProxyBackbone(cond_dim=128, n_cond_tokens=4, image_size=64).to(DEVICE)
    sketch = torch.randn(2, 3, 64, 64, device=DEVICE)
    cond = bb(["a gyroid part", "a lattice part"], sketch=sketch)
    assert cond.tokens.shape[0] == 2
    assert cond.pooled.shape[1] == 128
    assert cond.tokens.shape[1] > 4  # text tokens + pooled query tokens


def test_full_minimal_loop(tmp_path):
    """Train a couple of steps, checkpoint, reload, sample — full closed loop."""
    from structgen.config import StructGenConfig, DecoderConfig, TrainConfig
    from structgen.data.dataset import StructGenDataset, collate_structgen
    from structgen.model.backbone import ProxyBackbone
    from structgen.model.geometry_decoder import GeometryDecoder
    from structgen.infer import load_trained, generate

    cfg = StructGenConfig(
        decoder=DecoderConfig(grid_res=16, base_channels=16, channel_mults=(1, 2),
                              num_blocks=1, field_channels=2, cross_attn=False),
        train=TrainConfig(batch_size=2, out_dir=str(tmp_path)),
    )
    bb = ProxyBackbone(cond_dim=cfg.decoder.cond_dim, n_cond_tokens=4,
                       image_size=64).to(DEVICE)
    dec = GeometryDecoder(cfg.decoder).to(DEVICE)
    ds = StructGenDataset(grid_res=16, surface_samples=128, num_samples=8,
                          image_size=64)
    batch = collate_structgen([ds[i] for i in range(2)])
    field = batch["field"].to(DEVICE)
    surf = batch["surface"].to(DEVICE)
    sketch = batch["sketch"].to(DEVICE)
    opt = torch.optim.AdamW(list(dec.parameters()) + list(bb.parameters()), lr=1e-3)
    for _ in range(2):
        opt.zero_grad()
        cond = bb(batch["prompt"], sketch=sketch)
        loss, _ = dec.decode_loss(field, cond.pooled, cond.tokens, surf,
                                  cfg.loss_weights, cfg.flow)
        loss.backward()
        opt.step()
    ckpt = tmp_path / "d.pt"
    torch.save({"decoder": dec.state_dict(), "step": 2,
                "cfg": __import__("dataclasses").asdict(cfg)}, ckpt)
    bb2, dec2 = load_trained(cfg, str(ckpt), DEVICE)
    out_path = tmp_path / "gen.stl"
    field_arr, mesh = generate(bb2, dec2, cfg, [batch["prompt"][0]],
                               sketch=batch["sketch"][:1].to(DEVICE),
                               device=DEVICE, out_mesh=str(out_path))
    assert field_arr.shape == (1, 16, 16, 16)
    assert out_path.exists()
