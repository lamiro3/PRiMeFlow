"""
unet1d.py — 유전자 발현(1D 벡터)용 U-Net velocity field
=========================================================

[ 이 파일이 하는 일 ]
  flow matching 의 속도장 v_θ(x_t, cond, t) 를 U-Net 으로 구현한다.
  원래 U-Net 은 이미지(2D)용이지만, 유전자 발현은 1D 벡터라서 Conv1d 로 옮긴다.
  (= 유전자 축을 "길이" 축으로 보고 1D convolution 을 굴린다)

[ 조건 주입 방식 : FiLM ]
  시간 t 와 condition(cond) 을 하나의 embedding 으로 합친 뒤,
  각 ResBlock 에서 scale/shift 로 feature 를 조절한다 (guided-diffusion 방식).

[ 유전자 임베딩 (gene_emb_dim) ]
  conv 는 모든 위치에 같은 가중치를 쓰므로, 입력이 노이즈뿐이면 "지금 몇 번째 유전자인지"
  알 방법이 없다 (유전자 순서는 임의라 이웃 유전자 간 의미도 없음).
  → 유전자별 기본 발현 수준을 못 배워 모든 유전자를 비슷한 값으로 예측하게 됨
    (noise source U-Net 진단: 예측 유전자별 평균의 std 0.13 vs 실제 0.57).
  그래서 유전자마다 학습 가능한 벡터(gene_emb_dim 차원)를 두고 발현값 채널 옆에
  채널로 붙여 in_conv 에 넣는다. level 0 skip connection 으로 출력 직전까지 전달됨.
  gene_emb_dim=0 이면 끔 (예전 checkpoint 호환).

[ 지난번 OOM 의 진짜 원인 → 이번 수정 ]
  - mid block 의 self-attention 이 유전자 길이 L 에 대해 O(L^2) 메모리를 먹어
    attention_levels=() 로 꺼도 mid attention 이 강제로 돌아 OOM 이 났다.
  - 이번엔 mid attention 을 use_mid_attention 플래그로 분리하고 기본 False.
    즉 기본 설정에서는 attention 이 전혀 없는 순수 conv U-Net → 메모리 안전.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────
# 시간 임베딩 : 스칼라 t (B,) -> (B, dim) sinusoidal
# ────────────────────────────────────────────────────────────────
def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:  # dim 이 홀수면 0 하나 패딩
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def _norm(ch):
    """GroupNorm. 그룹 수는 ch 를 나누어떨어지게 gcd(32, ch) — 32 배수 채널이면 32 그룹.
    (min(32, ch) 로 하면 ch=48 처럼 32 로 안 나눠지는 skip-concat 채널에서 에러)"""
    return nn.GroupNorm(math.gcd(32, ch), ch)


# ────────────────────────────────────────────────────────────────
# ResBlock : FiLM(시간+조건 embedding 으로 scale/shift) 을 적용한 residual block
# ────────────────────────────────────────────────────────────────
class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, dropout=0.0):
        super().__init__()
        self.in_layers = nn.Sequential(
            _norm(in_ch),
            nn.SiLU(),
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
        )
        # embedding -> (scale, shift) = 2*out_ch
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 2 * out_ch),
        )
        self.out_norm = _norm(out_ch)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
        )
        # 입력/출력 채널이 다르면 skip 도 1x1 conv 로 맞춰줌
        self.skip = (nn.Identity() if in_ch == out_ch
                     else nn.Conv1d(in_ch, out_ch, kernel_size=1))

    def forward(self, x, emb):
        h = self.in_layers(x)
        scale, shift = self.emb_layers(emb)[:, :, None].chunk(2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift   # FiLM 조절
        h = self.out_layers(h)
        return h + self.skip(x)


# ────────────────────────────────────────────────────────────────
# (선택) self-attention : 기본적으로 안 씀. 유전자 길이가 길면 메모리 폭발 주의.
# ────────────────────────────────────────────────────────────────
class AttentionBlock1D(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = _norm(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, L = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.reshape(B, self.num_heads, 3 * (C // self.num_heads), L).chunk(3, dim=2)
        scale = 1.0 / math.sqrt(q.shape[2])
        attn = torch.einsum("bhcl,bhcm->bhlm", q * scale, k).softmax(dim=-1)
        out = torch.einsum("bhlm,bhcm->bhcl", attn, v).reshape(B, C, L)
        return x + self.proj(out)


# ────────────────────────────────────────────────────────────────
# 다운/업 샘플 : 유전자 축 길이를 절반/두 배로
# ────────────────────────────────────────────────────────────────
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
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


# ────────────────────────────────────────────────────────────────
# 본체 : GeneUNet1D
# ────────────────────────────────────────────────────────────────
class GeneUNet1D(nn.Module):
    """
    입력  x_t : (B, n_genes)     발현 벡터
          cond: (B, cond_dim)    perturbation one-hot(합) || covariate one-hot
          t   : (B,)             시간 [0,1]
    출력       : (B, n_genes)     예측 velocity

    내부에서 (B, n_genes) -> (B, 1, n_genes) 로 채널축을 만들어 1D conv 를 굴리고
    마지막에 다시 (B, n_genes) 로 되돌린다.
    """
    def __init__(
        self,
        n_genes: int,
        cond_dim: int,
        model_channels: int = 64,
        channel_mult: tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        time_dim: int = 128,
        attention_levels: tuple = (),      # attention 넣을 down/up 레벨. 기본 없음.
        use_mid_attention: bool = False,   # bottleneck attention. 기본 False (OOM 방지)
        num_heads: int = 4,
        dropout: float = 0.1,
        gene_emb_dim: int = 0,             # 유전자 임베딩 채널 수. 0 이면 끔.
    ):
        super().__init__()
        self.n_genes = n_genes
        self.num_res_blocks = num_res_blocks
        self.time_dim = time_dim
        emb_dim = model_channels * 4

        # 시간/조건을 각각 emb_dim 으로 올린 뒤 더해서 하나의 embedding 으로
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))

        # 유전자 임베딩 : (gene_emb_dim, n_genes) — 유전자 위치마다 고유한 벡터
        self.gene_emb = (nn.Parameter(torch.randn(gene_emb_dim, n_genes))
                         if gene_emb_dim > 0 else None)

        # [발현값 1개 || 유전자 임베딩] /유전자 -> model_channels 채널로 올림
        self.in_conv = nn.Conv1d(1 + gene_emb_dim, model_channels, kernel_size=3, padding=1)

        # ---- 인코더 (down) ----
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

        # ---- bottleneck (mid) : attention 은 옵션 ----
        self.mid = nn.ModuleList([ResBlock1D(ch, ch, emb_dim, dropout)])
        if use_mid_attention:
            self.mid.append(AttentionBlock1D(ch, num_heads))
        self.mid.append(ResBlock1D(ch, ch, emb_dim, dropout))

        # ---- 디코더 (up) ----
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_ch = model_channels * mult
            for _ in range(num_res_blocks + 1):
                skip_ch = chs.pop()
                block = nn.ModuleList([ResBlock1D(ch + skip_ch, out_ch, emb_dim, dropout)])
                if level in attention_levels:
                    block.append(AttentionBlock1D(out_ch, num_heads))
                self.up_blocks.append(block)
                ch = out_ch
            self.upsamples.append(Upsample1D(ch) if level != 0 else None)

        # ---- 출력 : 채널 -> 1 ----
        self.out = nn.Sequential(
            _norm(ch),
            nn.SiLU(),
            nn.Conv1d(ch, 1, kernel_size=3, padding=1),
        )

        # 다운샘플 횟수만큼 길이가 2의 배수여야 하므로 부족하면 오른쪽에 패딩
        self.mult = 2 ** (len(channel_mult) - 1)
        self.pad = (self.mult - n_genes % self.mult) % self.mult

    def forward(self, x_t, cond, t):
        h = x_t[:, None, :]                       # (B, 1, n_genes)
        if self.gene_emb is not None:             # (B, 1 + gene_emb_dim, n_genes)
            h = torch.cat([h, self.gene_emb[None].expand(h.shape[0], -1, -1).to(h.dtype)], dim=1)
        if self.pad:
            h = F.pad(h, (0, self.pad))

        emb = self.time_mlp(timestep_embedding(t, self.time_dim)) + self.cond_mlp(cond)

        h = self.in_conv(h)
        skips = [h]

        # 인코더
        idx = 0
        for level in range(len(self.downsamples)):
            for _ in range(self.num_res_blocks):
                block = self.down_blocks[idx]; idx += 1
                h = block[0](h, emb)
                if len(block) > 1:
                    h = block[1](h)               # attention (있으면)
                skips.append(h)
            if self.downsamples[level] is not None:
                h = self.downsamples[level](h)
                skips.append(h)

        # bottleneck
        for m in self.mid:
            h = m(h, emb) if isinstance(m, ResBlock1D) else m(h)

        # 디코더
        idx = 0
        for level in range(len(self.upsamples)):
            for _ in range(self.num_res_blocks + 1):
                block = self.up_blocks[idx]; idx += 1
                h = torch.cat([h, skips.pop()], dim=1)   # skip connection
                h = block[0](h, emb)
                if len(block) > 1:
                    h = block[1](h)
            if self.upsamples[level] is not None:
                h = self.upsamples[level](h)

        h = self.out(h)[:, 0, :]                  # (B, n_genes_padded)
        if self.pad:
            h = h[:, :self.n_genes]
        return h