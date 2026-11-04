"""Fit one family's decoder, overfitting on purpose, then code and decode the family.

Recipe (one family):
1. Encode each material with the frozen StableMaterials encoder from maps downsampled by `code_scale` (coarse
   codes: 1/code_scale^2 the latent cells, with the decoder making the detail).
2. Narrow the decoder (`narrow` table, see vae.narrow_decoder) and fine-tune it on exactly these materials:
   MSE on every channel but AO, LPIPS on albedo, and a texture-statistics term, on random aligned crops of the
   codes and maps, with codes rounded at the training `step`.
3. Switch the decoder to circular padding (`wrap`), so decoded textures tile.
4. Quantize the codes at `code_step` and count their bytes: the ideal code length under one Laplace per latent
   channel fitted on the family (mean and scale stored once at fp16).

Writes into out/: decoder.safetensors, codes.npz (integer codes, names, channel means and scales), and
summary.json (bytes per material, decoder parameters and fp16 bytes, the break-even inputs).
"""

import json
import math
import time
import tomllib
from pathlib import Path

from latent_materials import data, vae as V

KEEP = [c for c in range(data.CHANNELS) if c != 6]  # every channel but AO, which the VAE does not decode


def latent_bits(q, scale, mean):
    """Code length in bits of integer latents q (N, C, h, w) under a per-channel Laplace(mean, scale)."""
    import torch

    b, x = scale.view(1, -1, 1, 1), q - mean.view(1, -1, 1, 1)

    def cdf(v):
        return 0.5 + 0.5 * torch.sign(v) * (1 - torch.exp(-v.abs() / b))

    return -torch.log2((cdf(x + 0.5) - cdf(x - 0.5)).clamp_min(1e-9)).sum((1, 2, 3))


def laplace(q):
    mean = q.mean((0, 2, 3))
    scale = (q - mean.view(1, -1, 1, 1)).abs().mean((0, 2, 3)).clamp_min(0.05)
    return mean, scale


def stat_loss(out, y, weight: float, scales=(1, 2)):
    """Per-channel mean/std and Gram matrix at full and half resolution, matched without spatial alignment."""
    import torch

    from latent_materials.cluster import texture_stats

    loss = 0.0
    for f in scales:
        a = out if f == 1 else torch.nn.functional.avg_pool2d(out, f)
        b = y if f == 1 else torch.nn.functional.avg_pool2d(y, f)
        s = b.pow(2).mean().sqrt().clamp_min(1e-6)
        for x, ref in zip(texture_stats(a, s), texture_stats(b, s)):
            loss = loss + (x - ref).abs().mean()
    return weight * loss


def finetune(vae, mu, mats, ft: dict, device: str, code_scale: int, log=print) -> None:
    import lpips
    import torch

    torch.manual_seed(ft.get("seed", 0))
    gen = torch.Generator().manual_seed(ft.get("seed", 0) + 1)
    q = torch.round(mu / ft.get("step", 1.0)) * ft.get("step", 1.0)
    n, scale = len(mats), mats.shape[-1] // q.shape[-1]
    crop = ft.get("crop") or mats.shape[-1]
    zc = crop // scale
    vae.requires_grad_(False)
    params = [*vae.decoder.parameters(), *vae.post_quant_conv.parameters()]
    for p in params:
        p.requires_grad_(True)
    vae.train()
    vae.enable_gradient_checkpointing()
    opt = torch.optim.Adam(params, lr=ft.get("lr", 5e-5))
    lp = lpips.LPIPS(net="alex", verbose=False).eval().to(device).requires_grad_(False)

    def sample(idx):
        zs, ys = [], []
        for i in idx:
            zy, zx = (int(torch.randint(0, d - zc + 1, (1,), generator=gen)) for d in q.shape[-2:])
            zs.append(q[i, :, zy : zy + zc, zx : zx + zc])
            ys.append(mats[i, :, zy * scale : zy * scale + crop, zx * scale : zx * scale + crop])
        return torch.stack(zs).to(device), torch.stack(ys).to(device).float() / 255

    steps, start = ft["steps"], time.monotonic()
    for s in range(steps):
        z, y = sample(torch.randperm(n, generator=gen)[: ft.get("batch", 2)])
        out = V.from_vae(V.decode(vae, z, code_scale))
        loss = ft.get("mse_weight", 1.0) * torch.nn.functional.mse_loss(out[:, KEEP], y[:, KEEP])
        loss = loss + ft.get("lpips_weight", 0.1) * lp(out[:, :3] * 2 - 1, y[:, :3] * 2 - 1).mean()
        if ft.get("stat_weight"):
            loss = loss + stat_loss(out[:, KEEP], y[:, KEEP], ft["stat_weight"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (s + 1) % max(1, steps // 50) == 0:
            log(f"step {s + 1}/{steps} loss {loss.item():.5f} ({time.monotonic() - start:.0f} s)")
    vae.requires_grad_(False)
    vae.eval()


def run(config: Path, root: Path, members: Path | None, out: Path, log=print) -> dict:
    """Fit a decoder to the materials listed in `members` (a family_<j>.txt), or the whole library."""
    import numpy as np
    import torch
    from safetensors.torch import save_file

    cfg = tomllib.loads(Path(config).read_text())
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    cs, wrap = cfg.get("code_scale", 2), cfg.get("wrap", True)
    dirs, names = data.library(root, Path(members).read_text().split() if members else None)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))  # decode rebuilds the same narrowed decoder
    vae = V.load_vae(device)
    if "narrow" in cfg:
        V.narrow_decoder(vae, cfg["narrow"])
    mats = data.load(dirs, uint8=True)
    with torch.no_grad():
        mu = torch.cat([vae.encode(V.to_vae(torch.nn.functional.interpolate(
            mats[i : i + 2].float() / 255, scale_factor=1 / cs, mode="area")).to(device)).latent_dist.mean.cpu()
            for i in range(0, len(mats), 2)])
    finetune(vae, mu, mats, cfg["finetune"], device, cs, log)
    if wrap:  # after the fine-tune, which sees crops that do not wrap
        V.wrap_decoder(vae)
    save_file({k: v.contiguous() for k, v in V.decoder_state(vae).items()}, out / "decoder.safetensors")
    step = cfg.get("code_step", 2.0)
    q = torch.round(mu / step)
    mean, scale = laplace(q)
    bits = latent_bits(q, scale, mean)
    shared = 2 * 2 * len(mean)  # a mean and a scale per channel at fp16
    np.savez_compressed(out / "codes.npz", q=q.to(torch.int16).numpy(), names=np.array(names), step=step,
                        mean=mean.numpy(), scale=scale.numpy(), code_scale=cs, wrap=wrap)
    params = V.decoder_params(vae)
    summary = {"materials": len(names), "code_step": step, "code_scale": cs, "latent_shape": list(q.shape[1:]),
               "bytes_per_material": (sum(math.ceil(b / 8) for b in bits.tolist()) + shared) / len(names),
               "decoder_params": params, "decoder_bytes_fp16": 2 * params}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"{len(names)} materials: {summary['bytes_per_material'] / 1024:.2f} KiB each, decoder {params / 1e6:.2f}M params")
    return summary


def decode(run: Path, out: Path, device: str | None = None, batch: int = 2, log=print) -> None:
    """Decode a fitted family's codes to four PNGs per material under out/<name>/."""
    import numpy as np
    import torch

    run = Path(run)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.loads((run / "config.json").read_text())
    c = np.load(run / "codes.npz")
    vae = V.load_vae(device)
    if "narrow" in cfg:
        V.narrow_decoder(vae, cfg["narrow"])
    V.load_decoder(vae, run / "decoder.safetensors")
    if bool(c["wrap"]):
        V.wrap_decoder(vae)
    names, step, cs = [str(n) for n in c["names"]], float(c["step"]), int(c["code_scale"])
    q = torch.from_numpy(c["q"]).float() * step
    for i in range(0, len(names), batch):
        with torch.no_grad():
            dec = V.from_vae(V.decode(vae, q[i : i + batch].to(device), cs, bool(c["wrap"]))).cpu()
        data.save(dec, names[i : i + batch], out)
        log(f"decoded {min(i + batch, len(names))}/{len(names)}")
