"""Definitive test: can the voxel decoder generate ONE real solid shape?

Loads one real ShapeNet occupancy (NRRD), overfits the flow-matching decoder to
reproduce EXACTLY that shape, then generates from noise and measures IoU.
  - IoU high (>0.5)  → geometry generation is sound (scale up to full data)
  - IoU low           → the decoder/flow itself is broken (real bug to fix)
"""
import gzip, os, time
import numpy as np
import torch
import torch.nn.functional as F

RES = 64
NRRD = "/tmp/opencode/nrrd_sample/nrrd_256_filter_div_128_solid/1006be65e7bc937e9141f9b58470d646/1006be65e7bc937e9141f9b58470d646.nrrd"


def read_nrrd_occ(path, res=64):
    raw = open(path, "rb").read()
    sep = raw.index(b"\n\n")
    arr = np.frombuffer(gzip.decompress(raw[sep + 2:]), dtype=np.uint8).reshape(4, 128, 128, 128)
    occ = (arr[:3].max(0) > 0).astype(np.float32)  # 128^3 {0,1}
    step = 128 // res
    return occ[::step, ::step, ::step]              # strided (no dilation)


def main():
    from structgen.model.voxelnnet import VoxelVelocityNet

    occ = read_nrrd_occ(NRRD, RES)
    print(f"GT occ {occ.shape} solid_frac={occ.mean():.3f}")
    gt = torch.from_numpy(occ)[None, None].cuda()   # (1,1,R,R,R)

    net = VoxelVelocityNet(field_channels=1, base_channels=64,
                           channel_mults=(1, 2, 4, 8), num_blocks=2,
                           cond_dim=64, use_self_cond=False, cross_attn=False).cuda()
    # learnable null condition (unconditional generation)
    null_pool = torch.nn.Parameter(torch.randn(1, 64).cuda() * 0.02)
    opt = torch.optim.AdamW([*net.parameters(), null_pool], lr=1e-4)
    pooled = null_pool

    steps = 1200
    t0 = time.time()
    for it in range(steps):
        # flow batch: z = (1-t) noise + t * occ
        t = torch.rand(1, device=gt.device).clamp(0.02, 0.98)
        noise = torch.randn_like(gt)
        z = (1 - t) * noise + t * gt
        x0 = net(z, t.expand(1), pooled, cond_tokens=None)
        # STABLE x0-prediction: directly regress occupancy (no 1/(1-t) blow-up).
        p = torch.sigmoid(x0)
        bce = F.binary_cross_entropy_with_logits(x0, gt)
        mse = ((p - gt) ** 2).mean()
        loss = bce + mse
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 300 == 0 or it == steps - 1:
            with torch.no_grad():
                pred = (torch.sigmoid(x0) > 0.5).float()
                iou = ((pred > .5) & (gt > .5)).sum() / ((pred > .5) | (gt > .5)).sum().clamp_min(1)
            print(f"it {it:4d} bce={bce.item():.4f} mse={mse.item():.4f} "
                  f"train_iou={iou.item():.3f} dt={ (time.time()-t0)/(it+1):.2f}s")

    # GENERATE from pure noise (Euler flow)
    print("\n=== generate from noise ===")
    with torch.no_grad():
        z = torch.randn_like(gt)
        N = 50; dt = 1 / N
        for i in range(N):
            tt = torch.full((1,), i * dt, device=z.device)
            x0 = net(z, tt, pooled, cond_tokens=None)
            z = z + (x0 - z) * dt
        gen = (torch.sigmoid(z) > 0.5).float()
    inter = ((gen > .5) & (gt > .5)).sum().item()
    union = ((gen > .5) | (gt > .5)).sum().item()
    iou = inter / max(union, 1)
    print(f"GENERATED IoU vs GT = {iou:.3f}  (gen solid_frac={gen.mean():.3f}, GT={gt.mean():.3f})")
    os.makedirs("outputs/structgen/overfit", exist_ok=True)
    np.save("outputs/structgen/overfit/gen_occ.npy", gen[0, 0].cpu().numpy())
    np.save("outputs/structgen/overfit/gt_occ.npy", gt[0, 0].cpu().numpy())
    print("saved -> outputs/structgen/overfit/{{gen,gt}}_occ.npy")


if __name__ == "__main__":
    main()
