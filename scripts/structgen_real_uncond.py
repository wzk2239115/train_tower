"""Real-data unconditional solid generation (multi-shape).

Trains the voxel decoder on MANY real ShapeNet occupancies (NRRD) with the
STABLE x0-prediction loss (no 1/(1-t) divergence), then generates from noise
and reports: solid_frac distribution match + nearest-training-shape IoU.
Proves real multi-shape generation (not single-shape overfit).
"""
import glob, os, time
import numpy as np
import torch
import torch.nn.functional as F

RES = 64
NRRD_DIR = "/tmp/opencode/shapenet_nrrd"
BATCH = 4
STEPS = 600


def main():
    from structgen.model.voxelnnet import VoxelVelocityNet

    files = sorted(glob.glob(NRRD_DIR + "/*.nrrd"))
    print(f"shapes: {len(files)}")
    occs = np.stack([_load(f) for f in files])  # (N,R,R,R)
    occs_t = torch.from_numpy(occs)[:, None].cuda()  # (N,1,R,R,R)
    print(f"loaded {occs.shape} mean solid_frac={occs.mean():.3f}")

    net = VoxelVelocityNet(field_channels=1, base_channels=40,
                           channel_mults=(1, 2, 4), num_blocks=2,
                           cond_dim=64, use_self_cond=False, cross_attn=False).cuda()
    null_pool = torch.nn.Parameter(torch.randn(BATCH, 64, device="cuda") * 0.02)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    N = occs.shape[0]

    t0 = time.time()
    for it in range(STEPS):
        idx = np.random.randint(0, N, BATCH)
        gt = occs_t[idx]                                  # (B,1,R,R,R)
        t = torch.rand(BATCH, device="cuda").clamp(0.02, 0.98)
        noise = torch.randn_like(gt)
        z = (1 - t)[:, None, None, None, None] * noise + t[:, None, None, None, None] * gt
        x0 = net(z, t, null_pool, cond_tokens=None)
        bce = F.binary_cross_entropy_with_logits(x0, gt)
        mse = ((torch.sigmoid(x0) - gt) ** 2).mean()
        loss = bce + mse
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 250 == 0 or it == STEPS - 1:
            with torch.no_grad():
                pred = (torch.sigmoid(x0) > 0.5).float()
                iou = _iou(pred, gt)
            print(f"it {it:4d} bce={bce.item():.4f} mse={mse.item():.4f} "
                  f"train_iou={iou:.3f} dt={(time.time()-t0)/(it+1):.2f}s")

    # GENERATE
    print("\n=== generate from noise ===")
    net.eval()
    G = BATCH
    with torch.no_grad():
        z = torch.randn(G, 1, RES, RES, RES, device="cuda")
        nS, dt = 50, 1 / 50
        for i in range(nS):
            tt = torch.full((G,), i * dt, device="cuda")
            x0 = net(z, tt, null_pool, cond_tokens=None)
            z = z + (x0 - z) * dt
        gen = (torch.sigmoid(z) > 0.5).float()
    gs = gen.view(G, -1).mean(1).cpu().numpy()
    print(f"gen solid_frac: mean={gs.mean():.3f} std={gs.std():.3f}  "
          f"(data mean={occs.mean():.3f})")
    # nearest-training IoU for each generated shape
    g = gen.cpu().numpy().reshape(G, -1)
    best = []
    for i in range(G):
        ious = (g[i, None, :] & occs.reshape(N, -1)).sum(-1) / np.maximum(
            (g[i, None, :] | occs.reshape(N, -1)).sum(-1), 1)
        best.append(float(ious.max()))
    print(f"nearest-training IoU: mean={np.mean(best):.3f} max={np.max(best):.3f}")
    os.makedirs("outputs/structgen/real", exist_ok=True)
    for i in range(G):
        np.save(f"outputs/structgen/real/gen_{i}.npy", gen[i, 0].cpu().numpy())
    print(f"saved {G} generated -> outputs/structgen/real/")


def _load(f):
    from structgen.data.nrrd import read_nrrd_occ
    return read_nrrd_occ(f, RES)


def _iou(a, b):
    inter = ((a > .5) & (b > .5)).sum()
    union = ((a > .5) | (b > .5)).sum().clamp_min(1)
    return (inter / union).item()


if __name__ == "__main__":
    main()
