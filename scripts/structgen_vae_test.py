"""Quick VAE reconstruction test on real ShapeNet occupancy.

If the VAE reconstructs shapes well (IoU high), the latent captures the shape
and flow-in-latent will work. This is fast (3D conv, not the expensive flow).
"""
import glob, os, time
import numpy as np
import torch

RES = 64
NRRD_DIR = "/tmp/opencode/shapenet_nrrd"
STEPS = 1500


def main():
    from structgen.data.nrrd import read_nrrd_occ
    from structgen.model.vae import VoxelVAE, vae_loss

    files = sorted(glob.glob(NRRD_DIR + "/*.nrrd"))
    occs = np.stack([read_nrrd_occ(f, RES) for f in files])
    occs_t = torch.from_numpy(occs)[:, None].cuda()
    N = len(occs)
    print(f"shapes {N} solid_frac={occs.mean():.3f}")

    vae = VoxelVAE(grid_res=64, latent_res=16, latent_ch=32, base=24).cuda()
    print(f"VAE params {sum(p.numel() for p in vae.parameters())/1e6:.2f}M, "
          f"latent {vae.latent_ch}x{vae.latent_res}^3 = "
          f"{vae.latent_ch * vae.latent_res**3} dims (vs {RES**3} voxels)")
    opt = torch.optim.AdamW(vae.parameters(), lr=2e-4)
    B = 16
    t0 = time.time()
    for it in range(STEPS):
        idx = np.random.randint(0, N, B)
        gt = occs_t[idx]
        recon, mu, logvar = vae(gt)
        loss, logs = vae_loss(recon, gt, mu, logvar, beta=1e-3)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 200 == 0 or it == STEPS - 1:
            with torch.no_grad():
                p = (torch.sigmoid(recon) > 0.5)
                g = (gt > .5)
                iou = ((p & g).sum() / (p | g).sum().clamp_min(1)).item()
            print(f"it {it:4d} bce={logs['loss/bce']:.4f} kl={logs['loss/kl']:.3f} "
                  f"rec_iou={iou:.3f} dt={(time.time()-t0)/(it+1):.2f}s")

    # held-out reconstruction check
    vae.eval()
    with torch.no_grad():
        recon, _, _ = vae(occs_t[:8])
        p = (torch.sigmoid(recon) > 0.5)
        g = (occs_t[:8] > .5)
        ious = [((p[i] & g[i]).sum() / (p[i] | g[i]).sum().clamp_min(1)).item()
                for i in range(8)]
    print(f"\nheld-out recon IoU: mean={np.mean(ious):.3f} min={np.min(ious):.3f} max={np.max(ious):.3f}")
    os.makedirs("outputs/structgen/vae", exist_ok=True)
    torch.save(vae.state_dict(), "outputs/structgen/vae/vae.pt")
    for i in range(4):
        np.save(f"outputs/structgen/vae/rec_{i}.npy", p[i, 0].cpu().numpy())
        np.save(f"outputs/structgen/vae/gt_{i}.npy", occs_t[i, 0].cpu().numpy())
    print("saved vae + 4 recon -> outputs/structgen/vae/")


if __name__ == "__main__":
    main()
