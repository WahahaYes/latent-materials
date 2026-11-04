"""Cluster a material library into look-alike families, one decoder per family.

Each material is described by texture statistics of the frozen StableMaterials encoder: channel means, spreads
and Gram matrices of its first three down blocks' features. Vectors are z-scored, projected on their first
`dims` principal components and clustered with k-means++ (best of `restarts`). Guards: every family holds at
least `min_size` materials and at most `max_share` of the library; an oversized family is split once with
2-means, and a partition that still breaks a guard raises DegeneratePartition rather than training on it.
"""

import csv
import json
from pathlib import Path

import numpy as np

from latent_materials import data, vae as V


class DegeneratePartition(Exception):
    pass


def texture_stats(f, scale):
    """Channel means, spreads and the Gram matrix of features f (B, C, H, W), divided by `scale`."""
    v = (f / scale).flatten(2)
    mu, sd = v.mean(-1), (v.var(-1, unbiased=False) + 1e-12).sqrt()
    gram = (v - mu[..., None]) @ (v - mu[..., None]).transpose(1, 2) / v.shape[-1]
    return mu, sd, gram


def encoder_features(vae, m):
    """The encoder's first three down blocks' outputs for materials m (N, 10, H, W) in [0, 1]."""
    feats = []
    hooks = [b.resnets[-1].register_forward_hook(lambda mod, i, o: feats.append(o)) for b in vae.encoder.down_blocks[:3]]
    try:
        vae.encoder(V.to_vae(m.clamp(0, 1)).to(vae.device))
    finally:
        for h in hooks:
            h.remove()
    return feats


def stat_vectors(vae, mats, batch: int = 8) -> np.ndarray:
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(mats), batch):
            parts = []
            for f in encoder_features(vae, mats[i : i + batch]):
                mu, sd, gram = texture_stats(f, f.pow(2).mean().sqrt().clamp_min(1e-6))
                tri = torch.triu_indices(gram.shape[1], gram.shape[2])
                parts += [mu, sd, gram[:, tri[0], tri[1]]]
            out.append(torch.cat(parts, 1).float().cpu().numpy())
    return np.concatenate(out)


def zscore_pca(F: np.ndarray, dims: int = 32) -> np.ndarray:
    sd = F.std(0)
    sd[sd == 0] = 1.0
    Z = (F - F.mean(0)) / sd
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    return Z @ Vt[: min(dims, Vt.shape[0])].T


def kmeans(X: np.ndarray, k: int, seed: int = 0, restarts: int = 20, iters: int = 300) -> np.ndarray:
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        C = [X[rng.integers(len(X))]]
        d2 = ((X - C[0]) ** 2).sum(1)
        for _ in range(1, k):
            C.append(X[rng.choice(len(X), p=d2 / max(d2.sum(), 1e-300))])
            d2 = np.minimum(d2, ((X - C[-1]) ** 2).sum(1))
        C, lab = np.array(C), np.full(len(X), -1)
        for _ in range(iters):
            new = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
            if (new == lab).all():
                break
            lab = new
            for j in range(k):
                if (lab == j).any():
                    C[j] = X[lab == j].mean(0)
        inertia = float(((X - C[lab]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, lab)
    return best[1]


def partition(X: np.ndarray, k: int, min_size: int, max_share: float, seed: int = 0) -> np.ndarray:
    lab = kmeans(X, k, seed)
    nxt = lab.max() + 1
    for j in sorted(set(lab.tolist())):  # split each oversized family once
        idx = np.where(lab == j)[0]
        if len(idx) / len(lab) > max_share and len(idx) >= 2:
            sub = kmeans(X[idx], 2, seed + j, restarts=10)
            lab[idx[sub == 1]] = nxt
            nxt += 1
    sizes = {j: int((lab == j).sum()) for j in sorted(set(lab.tolist()))}
    bad = [f"family {j} has {s} (< {min_size})" for j, s in sizes.items() if s < min_size]
    bad += [f"family {j} holds {s / len(lab):.0%} (> {max_share:.0%})" for j, s in sizes.items() if s / len(lab) > max_share]
    if bad:
        raise DegeneratePartition("; ".join(bad))
    return lab


def run(root: Path, out: Path, k: int = 3, resolution: int = 256, dims: int = 32, min_size: int = 4,
        max_share: float = 0.5, seed: int = 0, device: str = "cpu", log=print) -> dict[int, list[str]]:
    """Cluster the library under root; writes out/partition.csv, out/family_<j>.txt and out/guards.json."""
    dirs, names = data.library(root)
    vae = V.load_vae(device).requires_grad_(False)
    F = stat_vectors(vae, data.load(dirs, size=resolution))
    lab = partition(zscore_pca(F, dims), k, min_size, max_share, seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fams = {int(j): [n for n, l in zip(names, lab) if l == j] for j in sorted(set(lab.tolist()))}
    with (out / "partition.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["material", "family"])
        w.writerows(zip(names, lab.tolist()))
    for j, members in fams.items():
        (out / f"family_{j}.txt").write_text("\n".join(members) + "\n")
    (out / "guards.json").write_text(json.dumps({"k": k, "n": len(names), "sizes": {j: len(m) for j, m in fams.items()},
                                                 "min_size": min_size, "max_share": max_share, "seed": seed}, indent=2))
    log("family sizes: " + ", ".join(f"{j}: {len(m)}" for j, m in fams.items()))
    return fams
