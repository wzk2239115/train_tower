"""Real text→shape conditional generation.

Pairs ShapeNet NRRD occupancies with their captions (by modelId), trains the
voxel decoder CONDITIONED on text (proxy text encoder), then generates from a
held-out caption and measures IoU vs that caption's GT shape.
  - IoU high → conditioning routes to the right shape (not collapse)
  - this is the dev-machine proof; on the compute box swap text encoder → Stepfun
"""
import csv, glob, os, time
import numpy as np
import torch

from structgen.data.nrrd import read_nrrd_occ
from structgen.model.voxelnnet import VoxelVelocityNet
from structgen.model.backbone import ProxyBackbone

RES = 64
NRRD_DIR = "/tmp/opencode/shapenet_nrrd"
CAPTIONS = "captions.tablechair.csv"
BATCH = 4
STEPS = 900


def load_pairs():
    desc = {}
    with open(CAPTIONS) as f:
        for r in csv.DictReader(f):
            desc[r["modelId"]] = r["description"]
    pairs = []
    for n in sorted(glob.glob(NRRD_DIR + "/*.nrrd")):
        mid = os.path.basename(n)[:-5]
        if mid in desc:
            pairs.append((desc[mid], read_nrrd_occ(n, RES)))
    return pairs


def main():
    pairs = load_pairs()
    # small subset + heavy training: test whether conditional generation+sampling
    # works at all (memorization-scale). If it can't generate shapes it trained
    # on, it's a sampling bug, not a data-scale issue.
    pairs = pairs[:40]
    train, held = pairs, pairs[:4]
    print(f"pairs {len(pairs)}: train {len(train)} held(memorize-test) {len(held)}")
    caps = [p[0] for p in train]
    occs = torch.from_numpy(np.stack([p[1] for p in train]))[:, None].cuda()
    N = len(train)
    print(f"data solid_frac mean={occs.mean():.3f}")

    cond_dim = 128
    bb = ProxyBackbone(cond_dim=cond_dim, n_cond_tokens=8, image_size=RES).cuda()
    net = VoxelVelocityNet(field_channels=1, base_channels=40,
                           channel_mults=(1, 2, 4), num_blocks=2,
                           cond_dim=cond_dim, use_self_cond=False, cross_attn=True).cuda()
    opt = torch.optim.AdamW(list(bb.parameters()) + list(net.parameters()), lr=1e-4)
    import torch.nn.functional as F

    t0 = time.time()
    null_pool = torch.nn.Parameter(torch.randn(1, cond_dim, device="cuda") * 0.02)
    null_tok = torch.nn.Parameter(torch.randn(1, 1, cond_dim, device="cuda") * 0.02)
    opt = torch.optim.AdamW(list(bb.parameters()) + list(net.parameters())
                             + [null_pool, null_tok], lr=1e-4)
    import torch.nn.functional as F

    STEPS_R = 1200
    for it in range(STEPS_R):
        idx = np.random.randint(0, N, BATCH)
        gt = occs[idx]
        caps_b = [caps[i] for i in idx]
        cond = bb(caps_b)
        # CFG training: 15% of the batch uses the null (unconditional) embedding
        drop = torch.rand(BATCH) < 0.15
        pooled = cond.pooled.clone()
        toks = cond.tokens.clone()
        if drop.any():
            pooled[drop] = null_pool
            toks[drop] = null_tok.expand(-1, toks.shape[1], -1)
        t = torch.rand(BATCH, device="cuda").clamp(0.02, 0.98)
        noise = torch.randn_like(gt)
        z = (1 - t)[:, None, None, None, None] * noise + t[:, None, None, None, None] * gt
        x0 = net(z, t, pooled, cond_tokens=toks)
        bce = F.binary_cross_entropy_with_logits(x0, gt)
        mse = ((torch.sigmoid(x0) - gt) ** 2).mean()
        loss = bce + mse
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 200 == 0 or it == STEPS_R - 1:
            with torch.no_grad():
                iou = _iou((torch.sigmoid(x0) > 0.5), gt)
            print(f"it {it:4d} bce={bce.item():.4f} mse={mse.item():.4f} "
                  f"train_iou={iou:.3f} dt={(time.time()-t0)/(it+1):.2f}s")

    # GENERATE from held-out captions
    print("\n=== generate from held-out captions ===")
    net.eval(); bb.eval()
    h_caps = [p[0] for p in held]
    h_occ = torch.from_numpy(np.stack([p[1] for p in held])).cuda()
    with torch.no_grad():
        cond = bb(h_caps)
        G = len(h_caps)
        # diagnostic: what does the net predict at low vs high t (from noise)?
        zn = torch.randn(G, 1, RES, RES, RES, device="cuda")
        for tv in (0.05, 0.95):
            tt = torch.full((G,), tv, device="cuda")
            x0 = net(zn, tt, cond.pooled, cond_tokens=cond.tokens)
            print(f"  net@t={tv}: pred solid_frac={(torch.sigmoid(x0)>0.5).float().mean():.3f}")
        # sample with classifier-free guidance (amplify the condition)
        GUIDE = 4.0
        z = torch.randn(G, 1, RES, RES, RES, device="cuda")
        uncond_pool = null_pool.expand(G, -1)
        uncond_tok = null_tok.expand(G, -1, -1)
        nS, dt = 50, 1 / 50
        for i in range(nS):
            tt = torch.full((G,), i * dt, device="cuda")
            x0c = net(z, tt, cond.pooled, cond_tokens=cond.tokens)
            x0u = net(z, tt, uncond_pool, cond_tokens=uncond_tok)
            x0 = x0u + GUIDE * (x0c - x0u)     # CFG
            z = z + (x0 - z) * dt
        gen = (torch.sigmoid(x0) > 0.5).float()
    for i in range(min(6, G)):
        gi = _iou(gen[i:i+1], h_occ[i:i+1])
        nn_iou = max(_iou(gen[i:i+1], occs[j:j+1]) for j in range(N))
        print(f"  cap IoU={gi:.3f}  nearest-train IoU={nn_iou:.3f}  "
              f"gen_sf={gen[i].mean():.3f} gt_sf={h_occ[i].mean():.3f}")
        print(f"    cap: {h_caps[i][:80]}")
    os.makedirs("outputs/structgen/t2s", exist_ok=True)
    for i in range(G):
        np.save(f"outputs/structgen/t2s/gen_{i}.npy", gen[i, 0].cpu().numpy())
        np.save(f"outputs/structgen/t2s/gt_{i}.npy", h_occ[i].cpu().numpy())


def _iou(a, b):
    a = a > .5; b = b > .5
    return ((a & b).sum() / (a | b).sum().clamp_min(1)).item()


if __name__ == "__main__":
    main()
