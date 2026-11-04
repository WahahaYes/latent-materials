# latent-materials

> **Work in progress.** This is the training recipe as it stands, published early. A formal methodology, an evaluation harness and results will be added later; until then, treat numbers you get from it as your own measurements.

A game ships a fixed, known set of textures. This project fits small neural decoders to that set on purpose: it splits a material library into look-alike families, overfits one small decoder to each family, and represents every material as a tiny latent code. A download carries the codes and one decoder per family; at install time each material is decoded once and written out in the GPU formats the engine already loads (BC7/BC5/BC4 with mips). Nothing changes at runtime.

It builds on StableMaterials' material autoencoder (Vecchio et al. 2024) and trains on MatSynth (Vecchio & Deschaintre 2024).

## How it works

A material is four maps: albedo, normal, roughness/metallic and height. The frozen StableMaterials encoder turns them into a grid of latent codes (18 channels at 1/8 resolution; encoded from 512 px for a 1024 px material, a 64 × 64 × 18 grid). A decoder turns the codes back into the maps.

The recipe:

1. **Fetch** a CC0 subset of MatSynth as four 1024 px maps per material.
2. **Cluster** the library into families using texture statistics of the frozen encoder's features (z-scored, PCA, k-means++), with guards against degenerate partitions.
3. **Fit** one decoder per family: narrow the pretrained decoder to about 1.2M parameters, then fine-tune it on exactly that family's materials (MSE, LPIPS and a texture-statistics loss on random crops), switch it to circular padding so textures tile, and quantize the codes.
4. **Decode** each family's codes back to PNG maps.
5. **Account** for total bytes, decoders included: a family pays for its decoder `D` once its `N` materials save more than `D` against a rival codec, `N* = D / (s − b)`.

## Usage

```sh
uv venv && uv pip install -e .        # pick the torch build for your GPU first if needed

latent-materials fetch --n 64                     # data/matsynth/<name>/{albedo,normal,orm,height}.png
latent-materials cluster --k 3                    # runs/families/family_<j>.txt
latent-materials fit --family runs/families/family_0.txt --out runs/fit/family_0
latent-materials decode --run runs/fit/family_0 --out decoded/family_0
latent-materials breakeven runs/fit/family_* --rival-bytes 29000
```

`configs/family.toml` holds the fit settings (narrowing, steps, losses, code step). The fit needs a CUDA GPU in practice: 32k steps at 256 px crops. MatSynth's source maps are 4K, so fetching downloads several GB per hundred materials.

## Status and limits

- The recipe was developed on another material library; this MatSynth version has not yet been run end to end. Expect rough edges.
- There is no evaluation here yet. The comparison we rely on (lit renders scored with DISTS and LPIPS against AVIF swept to matched quality) will come with the methodology.
- Decoders are not yet sized to their families; every family gets the same cut.
- Codes lose detail finer than the latent grid, and PSNR is several dB below a conventional codec at the same perceptual score.
- StableMaterials' encoder was itself trained on MatSynth, so MatSynth is an easier case than unseen game textures.

## Licences

- Code: [GNU AGPL-3.0](LICENSE). For other licensing, contact the author.
- StableMaterials weights are downloaded from Hugging Face at a pinned revision under their own licence (OpenRAIL); they are not included here.
- MatSynth materials are fetched under their own per-material licences; `fetch` keeps CC0 by default and writes `licenses.csv` beside the data.

Outside contributions are not being accepted yet.

## References

- G. Vecchio, R. Martin, A. Roullier, A. Kaiser, R. Rouffet, V. Deschaintre, T. Boubekeur. *StableMaterials: Enhancing Diversity in Material Generation via Semi-Supervised Learning.* 2024.
- G. Vecchio, V. Deschaintre. *MatSynth: A Modern PBR Materials Dataset.* CVPR 2024.
- K. Vaidyanathan et al. *Random-Access Neural Compression of Material Textures.* SIGGRAPH 2023.
