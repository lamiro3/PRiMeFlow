"""
test_primeflow.py — 학습된 PrimeFlow checkpoint 로 추론 설정만 바꿔 test 하기
==========================================================================

cfg_weight / n_ode_steps 는 추론(test)에서만 쓰이는 값이라 재학습 없이 바꿔 볼 수 있다.
그런데 perturbench train.py 는 checkpoint 를 불러올 때 lr / evaluation 만 새 값으로 바꾸고
나머지는 checkpoint 에 저장된 hparams 를 그대로 쓴다 (명령줄의 model.cfg_weight=... 는 무시됨).
그래서 perturbench train() 을 그대로 호출하되, PrimeFlow.load_from_checkpoint 에
`+test_override.<이름>=<값>` 으로 준 값을 끼워 넣는다.

사용 (PRiMeFlow 루트에서):
  PYTHONPATH=$PWD python model/test_primeflow.py experiment=primeflow/norman19 \\
    "hydra.searchpath=[file://$PWD]" train=False \\
    ckpt_path=logs/train/runs/<run>/checkpoints/<epoch=...>.ckpt \\
    +test_override.cfg_weight=3.0 \\
    'hydra.run.dir=logs/train/runs/${now:%Y-%m-%d_%H-%M-%S}_<이름>'

  - 모델 구조 관련 값(architecture, unet_* 등)은 checkpoint 값이 쓰이므로 따로 안 줘도 됨.
  - test_override 에는 추론에만 쓰이는 값(cfg_weight, n_ode_steps, predict_chunk,
    pred_clip_max)만 넣을 것 — 구조 관련 값을 바꾸면 가중치 로드가 실패한다.
"""
import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # PRiMeFlow 루트

from perturbench.modelcore.train import train  # noqa: E402
from model.Primeflow import PrimeFlow  # noqa: E402

INFERENCE_ONLY = {'cfg_weight', 'n_ode_steps', 'predict_chunk', 'pred_clip_max'}


@hydra.main(version_base="1.3", config_path="pkg://perturbench.configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    overrides = OmegaConf.to_container(cfg.get('test_override') or {}, resolve=True)
    bad = set(overrides) - INFERENCE_ONLY
    if bad:
        raise ValueError(f"test_override 에는 추론 전용 값만 가능: {sorted(INFERENCE_ONLY)} (받은 값: {sorted(bad)})")
    if cfg.get('train') or not cfg.get('ckpt_path'):
        raise ValueError("test 전용 스크립트: train=False 와 ckpt_path=... 를 함께 주세요.")

    # load_from_checkpoint(**kwargs) 의 kwargs 는 저장된 hparams 를 덮어씀
    original = PrimeFlow.load_from_checkpoint
    PrimeFlow.load_from_checkpoint = staticmethod(
        lambda *args, **kwargs: original(*args, **{**kwargs, **overrides}))
    print(f"[test_primeflow] ckpt={cfg.ckpt_path} | override={overrides}")
    train(cfg)


if __name__ == '__main__':
    main()
