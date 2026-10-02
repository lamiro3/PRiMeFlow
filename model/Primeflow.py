"""
primeflow.py — Conditional Flow Matching 으로 perturbation 반응 예측
====================================================================

[ 큰 그림 : Input -> Output ]  (사수님 피드백대로 입출력 위주로 이해하기)
  Input  : 세포의 발현 x1 (HVG 2000차원) + 어떤 처리인지 condition
  Model  : 가우시안 노이즈 x0 에서 출발해, condition 을 보고, 실제 처리된 발현으로
           "흘려보내는" 속도장 v_θ 를 학습
  Output : 그 처리를 받았을 때의 예측 발현 (2000차원)

[ 파일 구성 — 위에서 아래로 ]
  1. __init__      : 하이퍼파라미터 + velocity field(MLP/U-Net) 구성
  2. utils         : time_embedding / encode_condition / velocity_forward
  3. 학습          : get_flow_loss (L_CFM + CFG condition drop) / training/validation_step
  4. 추론          : integrate (ODE 적분 + CFG) / predict (청킹 + clip)
  velocity field 의 U-Net 본체만 Unet1d.py 에 따로 둔다 (MLP 는 짧아서 여기 포함).

[ 논문(Algorithm 1) 대비 ]
  x0 ~ N(0,I) (noise 고정) / interpolant x_t=(1-t)x0+t·x1(+σε)
  target v = x1 - x0 / loss = MSE(v_θ, target) = L_CFM
  CFG : 학습 때 p_uncond 확률로 condition 을 비우고, 추론 때 cfg_weight 로 강조

[ 지난번 버그 → 이번 수정 ]
  - integrate 가 존재하지 않는 변수 x_t 를 참조 → x 로 수정
  - predict 가 세포 전체를 한 번에 integrate → 평가 때 OOM(30GiB) → chunk 로 분할
  - U-Net mid attention 강제 실행 OOM → unet1d 에서 기본 off
  - fp16 nan → 실행 시 trainer.precision=32 권장 (코드가 아니라 실행 플래그)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from omegaconf import DictConfig

from perturbench.data.types import Batch
from perturbench.data.transforms.base import Dispatch
from perturbench.modelcore.models.base import PerturbationModel
from .Unet1d import GeneUNet1D


class PrimeFlow(PerturbationModel):
    """유전자 발현 공간에서 직접 동작하는 conditional flow matching 모델."""

    # ==================================================================
    # 1. 생성자
    # ==================================================================
    def __init__(
        self,
        n_genes: int,               # 입력 유전자 수 (HVG 2000 이면 2000)
        n_perts: int,               # perturbation 종류 수 (control 제외)
        transform: Dispatch,        # 아래 3개는 train.py 가 datamodule 에서 자동 주입
        context: dict,
        evaluation: DictConfig,
        # --- flow matching ---
        time_dim: int = 128,        # 시간 임베딩 차원
        sigma: float = 0.0,         # interpolant 에 더하는 노이즈 크기 σ
        n_ode_steps: int = 100,     # 추론 시 Euler 적분 스텝 수
        # --- CFG ---
        p_uncond: float = 0.1,      # 학습 시 condition 을 비울 확률
        cfg_weight: float = 1.0,    # 추론 시 guidance 강도 (1이면 guidance 없음)
        # --- 아키텍처 ---
        architecture: str = "mlp",  # "mlp" | "unet" (먼저 mlp 로 검증 후 unet)
        hidden_dim: int = 1024,     # MLP 폭
        unet_channels: int = 32,
        unet_channel_mult: tuple = (1, 2, 4, 8),
        unet_num_res_blocks: int = 2,
        unet_gene_emb_dim: int = 0,     # U-Net 유전자 임베딩 채널 수 (0=끔, Unet1d.py 참고)
        # --- 기타 ---
        use_ode_predict: bool = True,   # False 면 predict 가 입력을 그대로 반환(배관 검증)
        inject_covariates: bool = True,
        predict_chunk: int = 512,       # 추론 청크 크기 (평가 OOM 방지)
        pred_clip_max: float = 12.0,    # 예측 clip 상한 (log1p 발현의 현실적 최대)
        # --- optimizer / scheduler (base 로 전달) ---
        lr: float | None = None,
        wd: float | None = None,
        lr_scheduler: DictConfig | None = None,
        **kwargs,                       # gene_names / embedding_width 등 train.py 가 넘기는 나머지
    ) -> None:
        super().__init__(
            n_genes=n_genes, n_perts=n_perts,
            transform=transform, context=context, evaluation=evaluation,
            lr=lr, wd=wd, lr_scheduler=lr_scheduler,
        )
        self.save_hyperparameters()

        self.n_genes = n_genes
        self.n_perts = n_perts
        self.time_dim = (time_dim // 2) * 2      # cos/sin 반씩 → 짝수 보장
        self.sigma = sigma
        self.n_ode_steps = n_ode_steps
        self.p_uncond = p_uncond
        self.cfg_weight = cfg_weight
        self.architecture = architecture
        self.use_ode_predict = use_ode_predict
        self.inject_covariates = inject_covariates
        self.predict_chunk = predict_chunk
        self.pred_clip_max = pred_clip_max

        # condition 차원 = perturbation one-hot(합) + covariate one-hot
        #   (n_total_covariates 는 base 가 context["covariate_uniques"] 로 계산)
        cond_dim = self.n_perts
        if inject_covariates:
            cond_dim += int(self.n_total_covariates)
        self.cond_dim = cond_dim

        # velocity field 만들기
        if architecture == "mlp":
            in_dim = self.n_genes + self.cond_dim + self.time_dim
            self.velocity = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, self.n_genes),
            )
        elif architecture == "unet":
            self.velocity = GeneUNet1D(
                n_genes=self.n_genes,
                cond_dim=self.cond_dim,
                model_channels=unet_channels,
                channel_mult=tuple(unet_channel_mult),
                num_res_blocks=unet_num_res_blocks,
                time_dim=self.time_dim,
                attention_levels=(),        # 기본: attention 없음 (메모리)
                use_mid_attention=False,
                gene_emb_dim=unet_gene_emb_dim,
            )
        else:
            raise ValueError(f"architecture 는 'mlp' 또는 'unet' — 받은 값: {architecture!r}")

        self._setup_logged = False

    # ==================================================================
    # 2. utils — 입력을 모델이 먹는 형태로
    # ==================================================================
    def time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """스칼라 t (B,) -> (B, time_dim) sinusoidal (MLP 경로에서 사용)."""
        half = self.time_dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=1)

    def encode_condition(self, perturbation: torch.Tensor,
                         covariates: dict | None) -> torch.Tensor:
        """[perturbation one-hot(합) || covariate one-hot] -> (B, cond_dim)."""
        cond = perturbation
        if self.inject_covariates and covariates is not None:
            covs = [v.to(self.device).reshape(perturbation.shape[0], -1)
                    for v in covariates.values()]
            cond = torch.cat([cond] + covs, dim=1)
        return cond

    def velocity_forward(self, x_t: torch.Tensor, cond: torch.Tensor,
                         t: torch.Tensor) -> torch.Tensor:
        """
        velocity field 호출. MLP 와 U-Net 입력 방식 차이를 여기서 흡수.
          MLP  : [발현 || condition || 시간임베딩] 이어붙여 입력
          U-Net: (발현, condition, t) 그대로 (내부에서 FiLM 처리)
        """
        if self.architecture == "mlp":
            h = torch.cat([x_t, cond, self.time_embedding(t)], dim=-1)
            return self.velocity(h)
        return self.velocity(x_t, cond, t)

    # ==================================================================
    # 3. 학습 — Conditional Flow Matching loss (L_CFM)
    # ==================================================================
    def get_flow_loss(self, batch: Batch) -> torch.Tensor:
        # squeeze() 는 세포 1개짜리 batch 에서 batch 축까지 지워버리므로 (B, -1) 로 고정
        x1 = batch.gene_expression.reshape(-1, self.n_genes)   # x1 = 실제 처리 발현
        perturbation = batch.perturbations.reshape(x1.shape[0], -1)
        covariates = batch.covariates

        if not self._setup_logged:
            print(f"[PrimeFlow] arch={self.architecture} | source=noise | "
                  f"n_genes={self.n_genes} | cond_dim={self.cond_dim} | "
                  f"p_uncond={self.p_uncond} | cfg_weight={self.cfg_weight}")
            self._setup_logged = True

        x0 = torch.randn_like(x1)                       # 출발점 = 표준 가우시안(논문 방식)
        cond = self.encode_condition(perturbation, covariates)

        # CFG: 학습 때 확률적으로 condition 을 비워서 unconditional 도 배우게 함
        if self.training and self.p_uncond > 0:
            drop = torch.rand(x1.shape[0], device=x1.device) < self.p_uncond
            cond = cond.clone()
            cond[drop] = 0.0                            # null condition = 영벡터

        t = torch.rand(x1.shape[0], device=x1.device)          # t ~ U(0,1)
        x_t = (1 - t)[:, None] * x0 + t[:, None] * x1          # 직선 경로
        if self.sigma > 0:
            x_t = x_t + self.sigma * torch.randn_like(x_t)     # noise schedule

        target = x1 - x0                                        # 목표 속도
        return F.mse_loss(self.velocity_forward(x_t, cond, t), target)

    def training_step(self, batch: Batch, batch_idx: int):
        loss = self.get_flow_loss(batch)
        self.log("train_loss", loss, prog_bar=True, logger=True,
                 batch_size=len(batch), sync_dist=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        loss = self.get_flow_loss(batch)
        self.log("val_loss", loss, on_step=True, prog_bar=True,
                 logger=True, batch_size=len(batch), sync_dist=True)
        return loss

    # ==================================================================
    # 4. 추론 — 노이즈에서 ODE 적분해 발현 생성
    # ==================================================================
    @torch.no_grad()
    def integrate(self, x0: torch.Tensor, cond: torch.Tensor,
                  cfg_weight: float = 1.0) -> torch.Tensor:
        """
        Euler 로 dx/dt = v(x, cond, t) 를 t:0->1 적분.
        cfg_weight != 1 이면 CFG: v = v_uncond + w (v_cond - v_uncond)
        """
        x = x0
        dt = 1.0 / self.n_ode_steps
        null_cond = torch.zeros_like(cond)
        for i in range(self.n_ode_steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device)
            v_cond = self.velocity_forward(x, cond, t)         # ← x (지난번 x_t 오타 수정)
            if cfg_weight != 1.0:
                v_uncond = self.velocity_forward(x, null_cond, t)
                v = v_uncond + cfg_weight * (v_cond - v_uncond)
            else:
                v = v_cond
            x = x + v * dt
        return x

    def predict(self, batch: Batch) -> torch.Tensor:
        control_expression = batch.gene_expression.reshape(-1, self.n_genes).to(self.device)
        perturbation = batch.perturbations.reshape(control_expression.shape[0], -1).to(self.device)
        covariates = (
            {k: v.to(self.device) for k, v in batch.covariates.items()}
            if batch.covariates is not None else None
        )

        if not self.use_ode_predict:
            return control_expression                          # 배관 검증용

        cond = self.encode_condition(perturbation, covariates)
        x0 = torch.randn_like(control_expression)              # 노이즈에서 출발

        # 청킹: 세포를 한 번에 다 넣으면 OOM → chunk 씩 잘라서 적분 후 이어붙임
        preds = []
        for i in range(0, x0.shape[0], self.predict_chunk):
            xb = x0[i:i + self.predict_chunk]
            cb = cond[i:i + self.predict_chunk]
            preds.append(self.integrate(xb, cb, self.cfg_weight))
        pred = torch.cat(preds, dim=0)

        # 발현은 음수 없음 + 비현실적으로 큰 값/inf 방지
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        pred = pred.clamp(min=0.0, max=self.pred_clip_max)
        return pred