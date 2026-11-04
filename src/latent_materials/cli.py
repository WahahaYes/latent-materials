"""latent-materials: fetch MatSynth, cluster it into families, fit a decoder per family, decode, and do the
total-bytes accounting. Run `latent-materials <command> -h` for each command's options."""

import argparse
import json
from pathlib import Path


def cmd_fetch(a):
    from latent_materials import data

    names = data.fetch(a.root, a.n, a.resolution, a.split, tuple(a.licenses), a.per_category, a.normal_dx)
    print(f"{len(names)} materials under {a.root}")


def cmd_cluster(a):
    from latent_materials import cluster

    cluster.run(a.root, a.out, a.k, a.resolution, a.dims, a.min_size, a.max_share, a.seed, a.device)


def cmd_fit(a):
    from latent_materials import fit

    fit.run(a.config, a.root, a.family, a.out)


def cmd_decode(a):
    from latent_materials import fit

    fit.decode(a.run, a.out, a.device)


def cmd_breakeven(a):
    """Total bytes D + N*b against N*s, from fitted families' summary.json files and a per-material rival size."""
    runs = [json.loads((Path(r) / "summary.json").read_text()) for r in a.runs]
    n = sum(r["materials"] for r in runs)
    codes = sum(r["bytes_per_material"] * r["materials"] for r in runs)
    dec = sum(r["decoder_bytes_fp16"] // (2 if a.int8 else 1) for r in runs)
    rival = n * a.rival_bytes
    print(f"{len(runs)} families, {n} materials")
    print(f"codes {codes / 2**20:.2f} MiB + decoders {dec / 2**20:.2f} MiB ({'int8' if a.int8 else 'fp16'}) "
          f"= {(codes + dec) / 2**20:.2f} MiB, against {rival / 2**20:.2f} MiB for the rival")
    for r in runs:
        save = a.rival_bytes - r["bytes_per_material"]
        d = r["decoder_bytes_fp16"] // (2 if a.int8 else 1)
        print(f"  family of {r['materials']}: break-even N* = {d / save:.0f}" if save > 0 else
              f"  family of {r['materials']}: codes are not smaller than the rival; never breaks even")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="latent-materials", description=__doc__)
    sub = ap.add_subparsers(required=True)

    p = sub.add_parser("fetch", help="download a licence-filtered MatSynth subset as four maps per material")
    p.add_argument("--root", type=Path, default=Path("data/matsynth"))
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--split", default="train")
    p.add_argument("--licenses", nargs="+", default=["CC0"])
    p.add_argument("--per-category", type=int, default=0, help="cap per category, round-robin (0: no cap)")
    p.add_argument("--normal-dx", action="store_true", help="flip normal Y (sources in DirectX convention)")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("cluster", help="split the library into look-alike families")
    p.add_argument("--root", type=Path, default=Path("data/matsynth"))
    p.add_argument("--out", type=Path, default=Path("runs/families"))
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--resolution", type=int, default=256, help="size the encoder sees for clustering")
    p.add_argument("--dims", type=int, default=32)
    p.add_argument("--min-size", type=int, default=4)
    p.add_argument("--max-share", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=cmd_cluster)

    p = sub.add_parser("fit", help="fit one family's decoder and code its materials")
    p.add_argument("--config", type=Path, default=Path("configs/family.toml"))
    p.add_argument("--root", type=Path, default=Path("data/matsynth"))
    p.add_argument("--family", type=Path, help="a family_<j>.txt from cluster (default: the whole library)")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("decode", help="decode a fitted family to PNG maps")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device")
    p.set_defaults(func=cmd_decode)

    p = sub.add_parser("breakeven", help="total bytes with decoders counted, against a rival codec")
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--rival-bytes", type=float, required=True, help="rival bytes per material at matched quality")
    p.add_argument("--int8", action="store_true", help="count decoder weights at 8 bits instead of fp16")
    p.set_defaults(func=cmd_breakeven)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
