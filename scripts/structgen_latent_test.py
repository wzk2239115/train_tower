"""Stage B test: flow matching IN the VAE latent (paper's approach).

Freeze the VAE, encode shapes → latents, train the flow decoder ON latents
(reusing geometry_decoder with field_channels=latent_ch, res=latent_res),
conditioned on captions + CFG. Generate latent from noise → decode → measure
IoU vs GT. If this does NOT collapse (unlike raw-voxel flow), the latent route
works — validating SparseFlex/TRELLIS for our setup.
"""
import csv, glob, os, time
import numpy as np
import torch
import torch.nn.functional as F

from structgen.data.nrrd import read_nrrd_occ
from structgen.model.vae import VoxelVAE
from structgen.model.voxelnnet import VoxelVelocityNet
from structgen.model.backbone import ProxyBackbone

RES, LAT_RES, LAT_CH = 64, 16, 32
NRRD_DIR = "/tmp/opencode/shapenet_nrrd"
CAPTIONS = "captions.tablechair.csv"
VAE_PT = "outputs/structgen/vae/vae.pt"
BATCH, STEPS = 8, 1200


def load_pairs():
    desc = {}
    with open(CAPTIONS) as f:
        for r in csv.DictReader(f):
            desc[r["modelId"]] = r["description"]
    out = []
    for n in sorted(glob.glob(NRRD_DIR + "/*.nrrd")):
        mid = os.path.basename(n)[:-5]
        if mid in desc:
            out.append((desc[mid], read_nrrd_occ(n, RES)))
    return out


def main():
    pairs = load_pairs()[:80]
    train = pairs
    held = pairs[:4]
    caps = [p[0] for p in train]
    occs = torch.from_numpy(np.stack([p[1] for p in train]))[:, None].cuda()
    N = len(train)
    print(f"shapes {N} solid_frac={occs.mean():.3f}")

    # --- frozen VAE: precompute latents ---
    vae = VoxelVAE(RES, LAT_RES, LAT_CH, base=24).cuda()
    vae.load_state_dict(torch.load(VAE_PT)); vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        latents = []
        for i in range(0, N, 32):
            mu, _ = vae.enc(occs[i:i+32])
            latents.append(mu)
        latents = torch.cat(latents, 0)   # (N, LAT_CH, L,L,L)
    print(f"latents {tuple(latents.shape)} mean={latents.mean():.3f} std={latents.std():.3f}")

    # --- latent flow decoder (reuse VoxelVelocityNet on the latent grid) ---
    cond_dim = 128
    bb = ProxyBackbone(cond_dim=cond_dim, n_cond_tokens=8, image_size=RES).cuda()
    net = VoxelVelocityNet(field_channels=LAT_CH, base_channels=192,
                           channel_mults=(1, 2), num_blocks=2,
                           cond_dim=cond_dim, use_self_cond=False, cross_attn=True).cuda()
    null_pool = torch.nn.Parameter(torch.randn(1, cond_dim, device="cuda") * 0.02)
    null_tok = torch.nn.Parameter(torch.randn(1, 1, cond_dim, device="cuda") * 0.02)
    opt = torch.optim.AdamW(list(bb.parameters()) + list(net.parameters())
                            + [null_pool, null_tok], lr=1e-4)
    t0 = time.time()
    for it in range(STEPS):
        idx = np.random.randint(0, N, BATCH)
        gt = latents[idx]               # latent target (continuous)
        cond = bb([caps[i] for i in idx])
        pooled, toks = cond.pooled.clone(), cond.tokens.clone()
        drop = torch.rand(BATCH) < 0.15
        if drop.any():
            pooled[drop] = null_pool
            toks[drop] = null_tok.expand(-1, toks.shape[1], -1)
        t = torch.rand(BATCH, device="cuda").clamp(0.02, 0.98)
        noise = torch.randn_like(gt)
        z = (1 - t)[:, None, None, None, None] * noise + t[:, None, None, None, None] * gt
        x0 = net(z, t, pooled, cond_tokens=toks)
        loss = F.mse_loss(x0, gt)       # x0-prediction MSE on latent (continuous)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 200 == 0 or it == STEPS - 1:
            print(f"it {it:4d} mse={loss.item():.4f} dt={(time.time()-t0)/(it+1):.2f}s")

    # --- generate: latent from noise (CFG) -> decode -> IoU ---
    print("\n=== generate from held captions (latent + CFG) ===")
    net.eval(); bb.eval()
    h_caps = [p[0] for p in held]
    h_occ = torch.from_numpy(np.stack([p[1] for p in held])).cuda()
    with torch.no_grad():
        cond = bb(h_caps)
        G = len(h_caps)
        GUIDE = 4.0
        z = torch.randn(G, LAT_CH, LAT_RES, LAT_RES, LAT_RES, device="cuda")
        nS, dt = 50, 1 / 50
        for i in range(nS):
            tt = torch.full((G,), i * dt, device="cuda")
            x0c = net(z, tt, cond.pooled, cond.tokens)
            x0u = net(z, tt, null_pool.expand(G, -1),
                      null_tok.expand(G, -1, -1).expand(-1, cond.tokens.shape[1], -1))
            x0 = x0u + GUIDE * (x0c - x0u)
            z = z + (x0 - z) * dt
        # decode latent -> occupancy
        recon = vae.dec(z)
        gen = (torch.sigmoid(recon) > 0.5)
    for i in range(G):
        g, r = gen[i], (h_occ[i] > .5)
        iou = ((g & r).sum() / (g | r).sum().clamp_min(1)).item()
        # nearest training shape
        gi = gen[i].cpu().numpy().reshape(-1)
        nn_iou = max((np.minimum(gi, o).sum()/np.maximum(gi, o).sum().clip(1))
                     for o in occs.cpu().numpy().reshape(N, -1))
        print(f"  cap IoU={iou:.3f} nearest-train={nn_iou:.3f} "
              f"gen_sf={gen[i].float().mean():.3f} gt_sf={h_occ[i].mean():.3f}")
        print(f"    {h_caps[i][:75]}")
    os.makedirs("outputs/structgen/latent", exist_ok=True)
    for i in range(G):
        np.save(f"outputs/structgen/latent/gen_{i}.npy", gen[i].cpu().numpy().astype(np.uint8))
        np.save(f"outputs/structgen/latent/gt_{i}.npy", h_occ[i, 0].cpu().numpy())


if __name__ == "__main__":
    main()
