"""MatSynth materials (Vecchio & Deschaintre 2024) as four maps per material: albedo, normal (OpenGL), ORM
(ambient occlusion, roughness, metallic) and height, all RGB PNGs at one square resolution.

`fetch` reads the dataset's parquet shards from Hugging Face at a pinned revision, keeps materials whose licence
is in `licenses` (CC0 by default), and writes them under <root>/<name>/ with order.txt and licenses.csv beside
them. It reads one row group of one map column at a time, so memory stays near one row group of images.
"""

import csv
import io
from pathlib import Path

import numpy as np

REPO = "gvecchio/MatSynth"
REVISION = "5d5c84bbaf19fbdf64c617073a719e97082f6202"
TEXTURES = (("albedo", 3), ("normal", 3), ("orm", 3), ("height", 1))
CHANNELS = sum(c for _, c in TEXTURES)
COLUMNS = {"albedo": "basecolor", "normal": "normal", "roughness": "roughness", "metallic": "metallic",
           "height": "height"}


def _shards(fs, split: str) -> list[str]:
    root = f"datasets/{REPO}@{REVISION}/data"
    return sorted(f for f in fs.ls(root, detail=False) if f.rsplit("/", 1)[1].startswith(f"{split}-"))


def _gray(img, size: int, default: float) -> np.ndarray:
    if img is None:
        return np.full((size, size), default, np.float32)
    return np.asarray(img.convert("L").resize((size, size), _lanczos()), np.float32) / 255


def _lanczos():
    from PIL import Image

    return Image.LANCZOS


def _u8(a: np.ndarray) -> np.ndarray:
    return (np.clip(a, 0, 1) * 255).round().astype(np.uint8)


def build(maps: dict, size: int, normal_dx: bool = False) -> dict[str, np.ndarray]:
    """PIL images per map -> the four textures as uint8 RGB arrays at size x size."""
    from PIL import Image

    albedo = np.asarray(maps["albedo"].convert("RGB").resize((size, size), _lanczos()))
    n = np.asarray(maps["normal"].convert("RGB"), np.float32) / 255 * 2 - 1
    if normal_dx:
        n[..., 1] = -n[..., 1]
    # resize as vectors, then renormalize, so filtering does not shorten them
    n = np.stack([np.asarray(Image.fromarray(n[..., c], "F").resize((size, size), _lanczos())) for c in range(3)], -1)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
    orm = _u8(np.stack([np.ones((size, size), np.float32), _gray(maps.get("roughness"), size, 0.5),
                        _gray(maps.get("metallic"), size, 0.0)], -1))
    height = _u8(np.repeat(_gray(maps.get("height"), size, 0.5)[..., None], 3, -1))
    return {"albedo": albedo, "normal": _u8(n * 0.5 + 0.5), "orm": orm, "height": height}


def fetch(root: Path, n: int, resolution: int = 1024, split: str = "train", licenses=("CC0",),
          per_category: int = 0, normal_dx: bool = False, log=print) -> list[str]:
    """Write the first `n` licensed materials (by name; round-robin over categories when per_category > 0 caps
    each category) under root. Returns their names. Existing material folders are kept."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    from PIL import Image

    fs, root = HfFileSystem(), Path(root)
    root.mkdir(parents=True, exist_ok=True)
    pool = []  # (name, category, licence, shard, row group, row in group)
    for shard in _shards(fs, split):
        with fs.open(shard, "rb") as f:
            pf = pq.ParquetFile(f)
            for g in range(pf.num_row_groups):
                for i, row in enumerate(pf.read_row_group(g, columns=["name", "metadata"]).to_pylist()):
                    md = row["metadata"] or {}
                    if md.get("license") in licenses:
                        pool.append((row["name"], md.get("category", ""), md["license"], shard, g, i))
    pool.sort()
    if per_category:
        by_cat = {}
        for p in pool:
            by_cat.setdefault(p[1], []).append(p)
        pool = [ps[i] for i in range(per_category) for ps in by_cat.values() if i < len(ps)]
    picks = pool[:n]
    if len(picks) < n:
        log(f"only {len(picks)} materials match licences {licenses}")
    todo = {}
    for p in picks:
        if not all((root / p[0] / f"{t}.png").is_file() for t, _ in TEXTURES):
            todo.setdefault((p[3], p[4]), []).append(p)
    for (shard, g), ps in sorted(todo.items()):
        maps = {p[0]: {} for p in ps}
        with fs.open(shard, "rb") as f:
            pf = pq.ParquetFile(f)
            for key, col in COLUMNS.items():
                rows = pf.read_row_group(g, columns=[col]).to_pylist()
                for p in ps:
                    cell = rows[p[5]][col]
                    maps[p[0]][key] = Image.open(io.BytesIO(cell["bytes"])) if cell else None
                del rows
        for p in ps:
            if maps[p[0]].get("albedo") is None or maps[p[0]].get("normal") is None:
                log(f"skip {p[0]}: no albedo or normal")
                continue
            d = root / p[0]
            d.mkdir(parents=True, exist_ok=True)
            for t, a in build(maps[p[0]], resolution, normal_dx).items():
                Image.fromarray(a).save(d / f"{t}.png")
            log(f"wrote {p[0]}")
    names = [p[0] for p in picks if (root / p[0] / "albedo.png").is_file()]
    (root / "order.txt").write_text("\n".join(names) + "\n")
    with (root / "licenses.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "category", "license", "source"])
        w.writerows([p[0], p[1], p[2], f"https://huggingface.co/datasets/{REPO}"] for p in picks if p[0] in names)
    return names


def library(root: Path, names: list[str] | None = None) -> tuple[list[Path], list[str]]:
    """Material folders under root: the named ones, or everything in order.txt."""
    root = Path(root)
    names = names or (root / "order.txt").read_text().split()
    return [root / m for m in names], names


def load(dirs: list[Path], uint8: bool = False, size: int | None = None):
    """Materials as one tensor (N, 10, H, W): float in [0, 1], or uint8. `size` resizes on load (area)."""
    import torch
    from PIL import Image

    def stack(d: Path) -> np.ndarray:
        maps = []
        for t, c in TEXTURES:
            im = Image.open(d / f"{t}.png").convert("RGB")
            if size and im.size != (size, size):
                im = im.resize((size, size), Image.BOX)
            maps.append(np.asarray(im)[..., :c])
        return np.concatenate(maps, -1)

    arr = torch.from_numpy(np.stack([stack(d) for d in dirs])).permute(0, 3, 1, 2)
    return arr if uint8 else arr.float() / 255


def save(decoded, names: list[str], out: Path) -> None:
    """Decoded material tensors (N, 10, H, W) in [0, 1] -> four PNGs per material under out/<name>/."""
    from PIL import Image

    img = (decoded.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    for name, m in zip(names, img):
        d = Path(out) / name
        d.mkdir(parents=True, exist_ok=True)
        at = 0
        for t, c in TEXTURES:
            Image.fromarray(np.repeat(m[..., at : at + c], 3 // c, -1)).save(d / f"{t}.png")
            at += c
