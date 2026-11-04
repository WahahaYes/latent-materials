"""StableMaterials' material VAE (Vecchio et al. 2024): loading, channel packing, decoder narrowing, decoding.

The VAE maps 9 channels in [-1, 1] (basecolor RGB, normal XY, height, roughness, metallic, one unused) to 18
latent channels at 1/8 resolution. Its weights are OpenRAIL-licensed and are downloaded from Hugging Face at a
pinned revision, never redistributed here.
"""

import json

REPO = "gvecchio/StableMaterials"
REVISION = "4d452731535bd5c74261d6645623a573326f6f36"


def load_vae(device: str):
    import torch
    from diffusers import AutoencoderKL
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    cfg = hf_hub_download(REPO, "vae/config.json", revision=REVISION)
    weights = hf_hub_download(REPO, "vae/diffusion_pytorch_model.safetensors", revision=REVISION)
    with open(cfg) as f:
        vae = AutoencoderKL.from_config(json.load(f))
    vae.load_state_dict(load_file(weights))
    return vae.eval().to(torch.device(device))


def to_vae(m):
    """Material channels (N, 10, H, W) in [0, 1] (albedo 3, normal 3, ORM 3, height 1) -> the VAE's 9 in [-1, 1]."""
    import torch

    x = torch.cat([m[:, 0:3], m[:, 3:5], m[:, 9:10], m[:, 7:8], m[:, 8:9], torch.zeros_like(m[:, :1])], 1)
    return x * 2 - 1


def from_vae(y):
    """The VAE's output -> material channels in [0, 1]: normal Z rebuilt from XY, ambient occlusion flat (1)."""
    import torch

    y = (y.clamp(-1, 1) + 1) / 2
    xy = y[:, 3:5] * 2 - 1
    z = (1 - (xy**2).sum(1, keepdim=True)).clamp_min(0).sqrt()
    normal = torch.cat([xy, z], 1)
    normal = normal / normal.norm(dim=1, keepdim=True).clamp_min(1e-6) * 0.5 + 0.5
    return torch.cat([y[:, 0:3], normal, torch.ones_like(y[:, :1]), y[:, 6:7], y[:, 7:8], y[:, 5:6]], 1)


def decode(vae, z, code_scale: int = 1, wrap: bool = False):
    """Decode latents. With code_scale > 1 the latents (encoded from maps downsampled by that factor) are
    upsampled bilinearly first, wrapping at the edges when `wrap` is set, and the decoder makes the detail."""
    import torch

    if code_scale > 1:
        f = torch.nn.functional
        if wrap:
            z = f.interpolate(f.pad(z, (1, 1, 1, 1), mode="circular"), scale_factor=code_scale, mode="bilinear",
                              align_corners=False)[..., code_scale:-code_scale, code_scale:-code_scale]
        else:
            z = f.interpolate(z, scale_factor=code_scale, mode="bilinear", align_corners=False)
    return vae.decode(z).sample


def wrap_decoder(vae) -> None:
    """Circular padding in every padded decoder convolution, so decoded textures tile without a seam."""
    import torch

    for m in vae.decoder.modules():
        if isinstance(m, torch.nn.Conv2d) and any(m.padding):
            m.padding_mode = "circular"


def narrow_decoder(vae, cut: dict) -> None:
    """Shrink the decoder in place before fitting it: keep the first `resnets` residual blocks of each up block,
    drop the mid block unless `mid`, keep a fraction `inner` of each residual block's inner channels (whole
    GroupNorm groups, ranked by weight L1 norm), and with `upsampler_1x1` turn each 3x3 upsampling convolution
    into a 1x1 initialised with the sum of its taps."""
    import torch

    class Skip(torch.nn.Module):
        def forward(self, x, *args, **kwargs):
            return x

    dec = vae.decoder
    if cut.get("resnets"):
        for b in dec.up_blocks:
            b.resnets = torch.nn.ModuleList(b.resnets[: cut["resnets"]])
    if not cut.get("mid", True):
        dec.mid_block = Skip()
    if cut.get("upsampler_1x1"):
        for b in dec.up_blocks:
            for u in b.upsamplers or []:
                c = torch.nn.Conv2d(u.conv.in_channels, u.conv.out_channels, 1).to(u.conv.weight)
                c.weight.data = u.conv.weight.data.sum((2, 3), keepdim=True).clone()
                c.bias.data = u.conv.bias.data.clone()
                u.conv = c
    keep = cut.get("inner", 1.0)
    if keep >= 1:
        return
    blocks = [r for b in dec.up_blocks for r in b.resnets]
    if not isinstance(dec.mid_block, Skip):
        blocks += list(dec.mid_block.resnets)
    for r in blocks:
        groups, width = r.norm2.num_groups, r.conv1.out_channels
        size = width // groups
        score = (r.conv1.weight.abs().sum((1, 2, 3)) + r.conv2.weight.abs().sum((0, 2, 3))).view(groups, size).sum(1)
        kept = max(1, round(groups * keep))
        idx = torch.cat([torch.arange(g * size, (g + 1) * size) for g in sorted(score.topk(kept).indices.tolist())])
        conv1 = torch.nn.Conv2d(r.conv1.in_channels, len(idx), 3, padding=1).to(r.conv1.weight)
        conv1.weight.data, conv1.bias.data = r.conv1.weight.data[idx].clone(), r.conv1.bias.data[idx].clone()
        norm2 = torch.nn.GroupNorm(kept, len(idx), eps=r.norm2.eps, affine=True).to(r.norm2.weight)
        norm2.weight.data, norm2.bias.data = r.norm2.weight.data[idx].clone(), r.norm2.bias.data[idx].clone()
        conv2 = torch.nn.Conv2d(len(idx), r.conv2.out_channels, 3, padding=1).to(r.conv2.weight)
        conv2.weight.data, conv2.bias.data = r.conv2.weight.data[:, idx].clone(), r.conv2.bias.data.clone()
        r.conv1, r.norm2, r.conv2 = conv1, norm2, conv2


def decoder_params(vae) -> int:
    return sum(p.numel() for m in (vae.decoder, vae.post_quant_conv) for p in m.parameters())


def decoder_state(vae) -> dict:
    """The decoding half (decoder and post-quant conv) as one state dict."""
    return {**{f"decoder.{k}": v for k, v in vae.decoder.state_dict().items()},
            **{f"post_quant_conv.{k}": v for k, v in vae.post_quant_conv.state_dict().items()}}


def load_decoder(vae, path) -> None:
    from safetensors.torch import load_file

    state = load_file(str(path), device=str(vae.device))
    vae.decoder.load_state_dict({k[8:]: v for k, v in state.items() if k.startswith("decoder.")})
    vae.post_quant_conv.load_state_dict({k[16:]: v for k, v in state.items() if k.startswith("post_quant_conv.")})
