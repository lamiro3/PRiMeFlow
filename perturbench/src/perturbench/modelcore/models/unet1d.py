"""
Guided-diffusion style 1D U-Net for parameterizing the velocity field
v_theta(x_t, cond, t) over gene-expression vectors.

This mirrors the design of OpenAI's guided-diffusion U-Net (Dhariwal & Nichol,
2021) that PRiMeFlow builds on, adapted from 2D images to 1D expression vectors:

  - the gene axis (length = n_genes) is treated as the "spatial" 1D axis
  - Conv1d replaces Conv2d
  - each residual block is modulated by a (time + condition) embedding via
    FiLM (feature-wise scale & shift) -- this is how guided-diffusion injects
    the timestep, and we fold the perturbation/covariate condition into the
    same embedding
  - down/up sampling halves/doubles the gene axis, with skip connections
  - optional multi-head self-attention at coarse resolutions

Shapes
------
Input  x_t : (B, n_genes)          gene-expression vector
       cond: (B, cond_dim)         compositional pert one-hot || covariate one-hot
       t   : (B,)                  time in [0, 1]
Output      : (B, n_genes)         predicted velocity

The channel dim is created internally: we lift the single expression value per
gene to `model_channels` feature channels, run the U-Net over the gene axis,
then project back to 1 channel.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# embeddings
# ----------------------------------------------------------------------
def timestep_embedding(t, dim, max_period=10000):
    """Sinusoidal embedding of a scalar t in [0,1] -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:  # zero-pad if odd
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


# ----------------------------------------------------------------------
# building blocks
# ----------------------------------------------------------------------
class ResBlock1D(nn.Module):
    """
    Residual block modulated by an embedding via FiLM (scale & shift).
    This is the guided-diffusion residual block, in 1D.
    """
    def __init__(self, in_ch, out_ch, emb_dim, dropout=0.0):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(min(32, in_ch), in_ch),
            nn.SiLU(),
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
        )
        # emb -> (scale, shift) : 2*out_ch
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 2 * out_ch),
        )
        self.out_norm = nn.GroupNorm(min(32, out_ch), out_ch)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
        )
        self.skip = (
            nn.Identity() if in_ch == out_ch
            else nn.Conv1d(in_ch, out_ch, kernel_size=1)
        )

    def forward(self, x, emb):
        h = self.in_layers(x)
        scale, shift = self.emb_layers(emb)[:, :, None].chunk(2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift  # FiLM modulation
        h = self.out_layers(h)
        return h + self.skip(x)


class AttentionBlock1D(nn.Module):
    """Multi-head self-attention over the gene axis."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, L = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)  # (B, 3C, L)
        q, k, v = qkv.reshape(B, self.num_heads, 3 * (C // self.num_heads), L).chunk(3, dim=2)
        scale = 1.0 / math.sqrt(q.shape[2])
        attn = torch.einsum("bhcl,bhcm->bhlm", q * scale, k)
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhlm,bhcm->bhcl", attn, v)
        out = out.reshape(B, C, L)
        return x + self.proj(out)


class Downsample1D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv1d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample1D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ----------------------------------------------------------------------
# the U-Net
# ----------------------------------------------------------------------
class GeneUNet1D(nn.Module):
    """
    Guided-diffusion style 1D U-Net velocity field.

    Args:
        n_genes: length of the gene axis
        cond_dim: dimension of the condition vector (pert || covariate)
        model_channels: base channel width
        channel_mult: channel multiplier per resolution level
        num_res_blocks: residual blocks per level
        attention_levels: which levels (indices into channel_mult) get attention
        time_dim: sinusoidal time embedding dim
        num_heads: attention heads
        dropout: dropout rate
    """
    def __init__(
        self,
        n_genes,
        cond_dim,
        model_channels=64,
        channel_mult=(1, 2, 4),
        num_res_blocks=2,
        attention_levels=(2,),
        time_dim=128,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        emb_dim = model_channels * 4

        # time + condition -> shared embedding used by every ResBlock (FiLM)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.time_dim = time_dim

        # lift 1 expression value/gene -> model_channels
        self.in_conv = nn.Conv1d(1, model_channels, kernel_size=3, padding=1)

        # ---- encoder ----
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        chs = [model_channels]
        ch = model_channels
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                block = nn.ModuleList([ResBlock1D(ch, out_ch, emb_dim, dropout)])
                if level in attention_levels:
                    block.append(AttentionBlock1D(out_ch, num_heads))
                self.down_blocks.append(block)
                ch = out_ch
                chs.append(ch)
            if level != len(channel_mult) - 1:
                self.downsamples.append(Downsample1D(ch))
                chs.append(ch)
            else:
                self.downsamples.append(None)

        # ---- bottleneck ----
        self.mid = nn.ModuleList([
            ResBlock1D(ch, ch, emb_dim, dropout),
            AttentionBlock1D(ch, num_heads),
            ResBlock1D(ch, ch, emb_dim, dropout),
        ])

        # ---- decoder ----
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_ch = model_channels * mult
            for i in range(num_res_blocks + 1):
                skip_ch = chs.pop()
                block = nn.ModuleList([ResBlock1D(ch + skip_ch, out_ch, emb_dim, dropout)])
                if level in attention_levels:
                    block.append(AttentionBlock1D(out_ch, num_heads))
                self.up_blocks.append(block)
                ch = out_ch
            if level != 0:
                self.upsamples.append(Upsample1D(ch))
            else:
                self.upsamples.append(None)

        # ---- output ----
        self.out = nn.Sequential(
            nn.GroupNorm(min(32, ch), ch),
            nn.SiLU(),
            nn.Conv1d(ch, 1, kernel_size=3, padding=1),
        )

        # pad gene axis up to a multiple of 2^(#downsamples) so up/down line up
        self.n_down = len(channel_mult) - 1
        self.mult = 2 ** self.n_down
        self.pad = (self.mult - n_genes % self.mult) % self.mult

    def forward(self, x_t, cond, t):
        # x_t: (B, n_genes) -> (B, 1, n_genes_padded)
        B = x_t.shape[0]
        h = x_t[:, None, :]
        if self.pad:
            h = F.pad(h, (0, self.pad))

        emb = self.time_mlp(timestep_embedding(t, self.time_dim)) + self.cond_mlp(cond)

        h = self.in_conv(h)

        # ---- encoder ----
        skips = [h]
        idx = 0
        for level in range(len(self.downsamples)):
            for _ in range(self.num_res_blocks):
                block = self.down_blocks[idx]; idx += 1
                h = block[0](h, emb)
                if len(block) > 1:
                    h = block[1](h)
                skips.append(h)
            ds = self.downsamples[level]
            if ds is not None:
                h = ds(h)
                skips.append(h)

        for m in self.mid:
            h = m(h, emb) if isinstance(m, ResBlock1D) else m(h)

        idx = 0
        for level in range(len(self.upsamples)):
            for _ in range(self.num_res_blocks + 1):
                block = self.up_blocks[idx]; idx += 1
                h = torch.cat([h, skips.pop()], dim=1)
                h = block[0](h, emb)
                if len(block) > 1:
                    h = block[1](h)
            us = self.upsamples[level]
            if us is not None:
                h = us(h)

        h = self.out(h)                      # (B, 1, n_genes_padded)
        h = h[:, 0, :]                        # (B, n_genes_padded)
        if self.pad:
            h = h[:, : self.n_genes]
        return h


if __name__ == "__main__":
    torch.manual_seed(0)
    B, n_genes, cond_dim = 4, 2000, 8
    net = GeneUNet1D(n_genes=n_genes, cond_dim=cond_dim,
                     model_channels=64, channel_mult=(1, 2, 4),
                     num_res_blocks=2, attention_levels=(2,))
    x = torch.randn(B, n_genes)
    cond = torch.randn(B, cond_dim)
    t = torch.rand(B)
    out = net(x, cond, t)
    n_params = sum(p.numel() for p in net.parameters())
    print("output shape:", tuple(out.shape), "(expected", (B, n_genes), ")")
    print("params:", f"{n_params/1e6:.2f}M")
    assert out.shape == (B, n_genes)
    # gradient check
    out.sum().backward()
    print("backward OK; grad on in_conv:", net.in_conv.weight.grad is not None)
