import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from perturbench.data.types import Batch
from .base import PerturbationModel

from .unet1d import GeneUNet1D 


class FlowMatching(PerturbationModel):
    """
    Conditional flow matching model for perturbation response prediction
    """

    def __init__(
        self,
        n_genes: int,
        n_perts: int,
        hidden_dim: int = 1024,
        time_dim: int = 128,
        sigma: float = 0.0,
        n_ode_steps: int = 100,
        source: str = "control",  # "control" | "noise" -> 학습/추론 출발 분포 (반드시 일치해야 함)
        use_ode_predict: bool = True,  # False면 대조군을 그대로 반환 (pipeline 검증 용도)
        inject_covariates: bool = True,
        lr: float | None = None,
        wd: float | None = None,
        lr_scheduler_freq: int | None = None,
        lr_scheduler_interval: str | None = None,
        lr_scheduler_patience: int | None = None,
        lr_scheduler_factor: int | None = None,
        softplus_output: bool = False,  # 기본 False. True면 학습/추론 양쪽에 동일 적용됨
        datamodule: L.LightningDataModule | None = None,
        architecture: str = "mlp", # "mlp" | "unet" -> 속도장 모델 구조 선택
        unet_channels: int = 64,
        unet_channel_mult: tuple = (1, 2, 4, 8),
        unet_num_res_blocks: int = 2,
    ) -> None:
        """
        The constructor for the FlowMatching class.

        Args:
            n_genes (int): Number of genes in the dataset
            n_perts (int): Number of perturbations in the dataset (not including controls)
            hidden_dim (int): Width of the velocity field MLP
            time_dim (int): Dimension of the sinusoidal time embedding
            sigma (float): Std of extra noise added to the interpolant x_t
            n_ode_steps (int): Number of Euler steps used at inference
            source (str): Source distribution of the flow.
                "control" -> 대조군 발현에서 출발 (PerturBench 태스크와 자연스럽게 맞음)
                "noise"   -> 표준 가우시안에서 출발 (PRiMeFlow 방식)
            use_ode_predict (bool): False면 predict가 입력을 그대로 반환 (배관 검증용)
            inject_covariates: Whether to condition the velocity field on covariates
            softplus_output: Whether to apply a softplus activation to enforce
                non-negativity. 적용 시 학습 타깃도 동일하게 변환되어야 하므로
                학습/추론에 함께 반영됨.
            datamodule: The datamodule used to train the model
        """
        super(FlowMatching, self).__init__(
            datamodule=datamodule,
            lr=lr,
            wd=wd,
            lr_scheduler_interval=lr_scheduler_interval,
            lr_scheduler_freq=lr_scheduler_freq,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_scheduler_factor=lr_scheduler_factor,
        )
        self.save_hyperparameters(ignore=["datamodule"])

        if n_genes is not None:
            self.n_genes = n_genes
        if n_perts is not None:
            self.n_perts = n_perts

        if source not in ("control", "noise"):
            raise ValueError(f"source must be 'control' or 'noise', got {source!r}")
        self.source = source

        # time_dim이 홀수면 cos/sin을 붙였을 때 차원이 1 모자라므로 짝수로 내림
        self.time_dim = (time_dim // 2) * 2
        self.n_ode_steps = n_ode_steps
        self.sigma = sigma
        self.use_ode_predict = use_ode_predict
        self.inject_covariates = inject_covariates
        self.softplus_output = softplus_output
        self.architecture = architecture

        # 조건 차원 = perturbations + covariate(선택)
        cond_dim = self.n_perts

        if inject_covariates:
            if datamodule is None or datamodule.train_context is None:
                raise ValueError(
                    "If inject_covariates is True, datamodule must be provided"
                )
            cond_dim += int(np.sum([
                len(c) for c in datamodule.train_context["covariate_uniques"].values()
            ]))
        self.cond_dim = cond_dim
        
        #-----------------------------------------------------------------
        # velocity field ... 속도장 만들기 (MLP: 대조군, U-Net: PRiMeFlow 방식)
        #-----------------------------------------------------------------

        if architecture == "mlp":
            velo_in_dim = self.n_genes + self.cond_dim + self.time_dim
            velo_out_dim = self.n_genes

            # SiLU가 ReLU에 비해 좀 더 부드러운 곡선 형태를 띄고 있기 때문에 활성화 함수로 SiLU가 더 적합하다고 함
            # 그러나 이전에 SiLU만 넣고 학습을 돌렸을 때 결과가 매우 좋지 않았기 때문에
            # SiLU 입력층을 정규화시키게끔 layer를 추가해놨음.
            
            self.velocity = nn.Sequential(
                nn.Linear(velo_in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, velo_out_dim),
            )
            
        elif architecture == "unet":
            self.velocity = GeneUNet1D(
                n_genes=self.n_genes,
                cond_dim=self.cond_dim,
                model_channels=unet_channels,
                channel_mult=unet_channel_mult,
                num_res_blocks=unet_num_res_blocks,
                attention_levels=(), # unet1d.py에 적혀있는 대로 똑같이 적용한 거
                time_dim=self.time_dim
            )
        else:
            raise ValueError(f'Unknown architecture: {architecture}')
            
        

        self._setup_logged = False  # 한 번만 설정 요약을 출력하기 위한 플래그
        # 예측 clip 상한: log1p 정규화 발현의 현실적 최대치.
        # 데이터의 실제 최대 발현값 근처로 잡는 게 안전하다.
        self._pred_clip_max = 12.0

    # ------------------------------------------------------------------
    # 유틸
    # ------------------------------------------------------------------
    def time_embedding(self, t: torch.Tensor):
        """스칼라 t를 time_dim 차원으로 확장 (sinusoidal embedding)."""
        half = self.time_dim // 2
        freqs = torch.exp(
            -np.log(10000) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=1)

    # one-hot encoding for covariate; 논문 3.1 쪽 참고 바람
    def encode_condition(self, perturbation: torch.Tensor, covariates: dict | None):
        cond = perturbation

        if self.inject_covariates and covariates is not None:
            covs = [
                v.to(self.device).reshape(perturbation.shape[0], -1)
                for v in covariates.values()
            ]
            cond = torch.cat([cond] + covs, dim=1)
        return cond

    def _sample_source(self, x1: torch.Tensor, control: torch.Tensor | None):
        """학습 시 flow의 출발점 x0를 만든다."""
        if self.source == "noise":
            return torch.randn_like(x1)

        # source == "control"
        if control is None:
            raise ValueError(
                "source='control'인데 batch.controls가 없습니다. "
                "data config에서 control을 함께 로드하도록 설정하거나, "
                "source='noise'로 바꾸세요."
            )
        return control

    # 논문 표기랑 똑같이 작성한 거
    def forward(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor):
        if self.architecture == "mlp":
            # mlp의 경우 직접 자체적으로 합쳐줘야 하지만 unet은 이 과정이 필요 없음
            h = torch.cat([x_t, cond, self.time_embedding(t)], dim=-1)
            return self.velocity(h)
        
        elif self.architecture == "unet":
            return self.velocity(x_t, cond, t)
                

    # ------------------------------------------------------------------
    # Loss_fm (L_fm)
    # ------------------------------------------------------------------
    def get_flow_loss(self, batch: Batch):
        x1, control, perturbation, covariates, _ = self.unpack_batch(batch)

        if not self._setup_logged:
            print(
                f"[FlowMatching] source={self.source} | "
                f"controls_available={control is not None} | "
                f"use_ode_predict={self.use_ode_predict} | "
                f"n_ode_steps={self.n_ode_steps} | "
                f"softplus_output={self.softplus_output} | "
                f"n_genes={self.n_genes} | cond_dim={self.cond_dim}"
            )
            self._setup_logged = True

        x0 = self._sample_source(x1, control)

        # softplus를 쓸 경우, 모델 출력이 softplus를 거치므로
        # 학습 타깃도 동일한 공간에 있어야 한다 (inverse softplus로 보정)
        if self.softplus_output:
            x1 = self._inverse_softplus(x1)
            x0 = self._inverse_softplus(x0)

        cond = self.encode_condition(perturbation, covariates)
        t = torch.rand(x1.shape[0], device=x1.device)
        x_t = (1 - t)[:, None] * x0 + t[:, None] * x1

        if self.sigma > 0:  # sigma: noise schedule
            x_t = x_t + self.sigma * torch.randn_like(x_t)

        target = x1 - x0  # 목표 속도
        return F.mse_loss(self.forward(x_t, cond, t), target)

    @staticmethod
    def _inverse_softplus(y: torch.Tensor, eps: float = 1e-6):
        """softplus의 역함수. log(exp(y) - 1)을 수치적으로 안전하게 계산."""
        return y + torch.log(-torch.expm1(-y.clamp(min=eps)))

    def training_step(self, batch: Batch, batch_idx: int):
        loss = self.get_flow_loss(batch)
        self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=len(batch))
        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        val_loss = self.get_flow_loss(batch)
        self.log(
            "val_loss",
            val_loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
        )
        return val_loss

    # ------------------------------------------------------------------
    # 추론: ODE 적분
    # ------------------------------------------------------------------
    @torch.no_grad()
    def integrate(self, x0: torch.Tensor, cond: torch.Tensor):
        x = x0
        dt = 1.0 / self.n_ode_steps

        for i in range(self.n_ode_steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device)
            x = x + self.forward(x, cond, t) * dt
        return x

    def predict(self, batch: Batch):
        control_expression = batch.gene_expression.squeeze().to(self.device)
        perturbation = batch.perturbations.squeeze().to(self.device)
        covariates = (
            {k: v.to(self.device) for k, v in batch.covariates.items()}
            if batch.covariates is not None
            else None
        )

        if not self.use_ode_predict:
            return control_expression

        cond = self.encode_condition(perturbation, covariates)

        if self.source == "noise":
            x0 = torch.randn_like(control_expression)
        else:
            x0 = control_expression

        if self.softplus_output:
            x0 = self._inverse_softplus(x0)

        pred = self.integrate(x0, cond)

        if self.softplus_output:
            pred = F.softplus(pred)

        # ▼▼▼ 추가: 발현은 음수가 없고, log1p 공간에서 비현실적으로 큰 값 방지 ▼▼▼
        # 1) nan/inf를 0으로 치환  2) [0, max] 범위로 clip
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        pred = pred.clamp(min=0.0, max=self._pred_clip_max)
        # ▲▲▲

        return pred