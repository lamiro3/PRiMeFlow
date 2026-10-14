import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

# ─────────────────────────────────────────────────────────────
# config
# ─────────────────────────────────────────────────────────────
# perturbench 가 csv logger 에 남기는 metric 이름 ({metric}_{aggregation}, rank 는 {metric}_rank_{aggregation})
metrics = [
    'rmse_average', 'rmse_rank_average',
    'cosine_pca_average', 'cosine_rank_pca_average',
    'cosine_logfc', 'cosine_rank_logfc',
    'r2_score_scores', 'top_k_recall_scores',
    'mmd_pca', 'mmd_rank_pca',
]

short = [
    'RMSE', 'RMSE\nrank',
    'cos\nPCA', 'cos rank\nPCA',
    'cos\nlogFC', 'cos rank\nlogFC',
    'R²', 'top-k\nrecall',
    'MMD\nPCA', 'MMD rank\nPCA',
]

# +1 = higher is better / -1 = lower is better
DIRECTION = [-1, -1, +1, -1, +1, -1, +1, +1, -1, -1]

# 평가 축: 성격이 다른 지표끼리는 한 숫자로 섞지 않고 축별로 따로 본다.
#   Accuracy     : 예측 평균 발현이 실제와 얼마나 가까운가
#   Specificity  : *_rank 지표. 예측이 다른 perturbation 이 아니라 자기
#                  perturbation 의 실제값에 가장 가까운가 (Accuracy 와 다른 신호라
#                  raw/rank 쌍으로 묶으면 서로 상쇄돼 정보가 사라짐)
#   DEG recall   : 실제 top-k DEG 를 맞히는가
#   Distribution : 단일세포 분포 수준에서 실제와 같은가
AXES = {
    'Accuracy':     [0, 2, 4, 6],
    'Specificity':  [1, 3, 5, 9],
    'DEG recall':   [7],
    'Distribution': [8],
}

# 정규화 = 지표별 모델 간 순위 (1 = best, 동점은 평균 순위).
# 예전 방식(일부 지표 robust scaling + 나머지 절대값 clip)은
#   - 지표마다 실질 변별력이 달라 의도치 않은 가중치가 생기고
#     (cos rank 는 1-x 하면 전부 0.9+ 로 붙어 사실상 상수)
#   - median/MAD 가 비교 집합에 의존해 run 하나 추가/제거만 해도 다른 모델 점수가 변했다.
# 순위는 스케일·이상치와 무관하고 모든 지표를 같은 무게로 다룬다. 차이의 '크기'는
# 버려지므로 히트맵 셀에 원값을 같이 적는다.
# 축 점수 = 축 안 지표들의 평균 순위. 축끼리 합친 overall 은 만들지 않는다
# (축 간 trade-off 를 숫자 하나로 가리게 되고, 축 가중치를 정할 근거도 없음).

# [Flow_Matching-MLP metric data]
# 1. SiLU [source: noise]
#   -> [np.inf, .4490, .6194, .5409, .1825, .4641, -.02963, .005778, .9169, .5525] // inf 나와서 망한 케이스
# 2. SiLU + LayerNorm(input 정규화) [source: noise]
#   -> [.313, .1879, .09682, .1535, .02615, .2692, -4.534, .08444, 4.342, .1793],
# 3. SiLU + LayerNorm(input 정규화) [source: control]
#   ->

# [Flow_Matching-Unet metric data]
# 1. source: noise
#   -> [.5454, .3652, .2612, .3807, .1479, .3034, .03775, .1074, 16.3, .429]
# 2. source: control
#   -> [.1117, .3904, .3361, .6192, .0807, .4616, -.9993, .0, 1.549, .6263]


# ─────────────────────────────────────────────────────────────
# 비교할 run: 표시 이름 -> run 폴더 (train 한 번 = logs/train/runs/<날짜_시간> 하나)
#   test metric 은 <run>/tensorboard/ 의 event 파일에서 자동으로 읽는다.
#   어떤 run 이 있는지는 `python metric_each_model_graph.py --list` 로 확인.
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # RUNS 의 상대경로 기준 (PRiMeFlow 루트)
RUN_ROOT = os.path.join(BASE_DIR, 'logs', 'train', 'runs')

RUNS = {
    # norman19, test split(final_test), seed 245 — 2026-09-29 재실행 결과
    'Linear_additive':            'logs/train/runs/2026-09-29_20-25-04_linear',
    'Latent_additive':            'logs/train/runs/2026-09-29_19-16-08_latent',
    'Decoder_only':               'logs/train/runs/2026-09-29_23-14-02_decoder',
    'CPA':                        'logs/train/runs/2026-09-29_19-16-02_cpa',
    'SAMS_VAE':                   'logs/train/runs/2026-09-29_19-16-06_sams',
    'Biolord':                    'logs/train/runs/2026-09-29_19-16-04_biolord',
    'PRiMeFlow(Unet, CFG=1.0)':   'logs/train/runs/2026-09-29_20-15-34_primeflow-unet',
    # 유전자 임베딩(unet_gene_emb_dim=16) 추가, 4-GPU DDP (유효 batch 512)
    'PRiMeFlow(Unet+gemb, CFG=1.0)': 'logs/train/runs/2026-09-30_01-04-58_primeflow-unet-gemb',
    # 위 gemb run 과 같은 checkpoint(epoch=40) 로 cfg_weight 만 바꿔 test (model/test_primeflow.py)
    'PRiMeFlow(Unet+gemb, CFG=3.0)': 'logs/train/runs/2026-09-30_10-34-58_primeflow-unet-gemb-cfg3',
    'PRiMeFlow(Unet+gemb, CFG=5.0)': 'logs/train/runs/2026-09-30_10-35-01_primeflow-unet-gemb-cfg5',
}

# figure 1 (메인 비교): baseline + 대표 PRiMeFlow 한 개.
#   순위는 비교 집합 안에서 매겨지므로 같은 모델의 변형을 여러 개 넣으면
#   baseline 들의 순위가 변형 개수만큼 밀려 비교가 왜곡된다.
MAIN_MODELS = [
    'Linear_additive', 'Latent_additive', 'Decoder_only',
    'CPA', 'SAMS_VAE', 'Biolord',
    'PRiMeFlow(Unet+gemb, CFG=1.0)',
]
# figure 2 (ablation): PRiMeFlow 변형끼리만 순위를 매긴다.
ABLATION_MODELS = [n for n in RUNS if n.startswith('PRiMeFlow')]

# 예전 perturbench 버전에서 뽑아 손으로 옮겨 적은 값 (지금 버전과 split/평가 코드가
# 달라 직접 비교 불가). 참고용으로만 남겨 둠. True 면 메인 비교에 같이 그림.
INCLUDE_LEGACY = False

LEGACY_MODELS = {
    'Linear_additive': [.05779, .009596, .8074, .01313, .622, .04545, -.02479, .002667, 1.334, .01364],
    'Latent_additive': [.0419, .01353, .8791, .02077, .7969, .008696, -.06922, .0, 3.247, .01304],
    'Decoder_only':    [.04305, .01111, .8817, .01111, .7588, .01061, .2202, .28, 3.326, .01313],
    'CPA':             [.0478, .02323, .8041, .03232, .7795, .009091, .07561, .02978, 2.218, .03788],
    'SAMS_VAE':        [.09383, .04697, -.07595, .3076, .4365, .03485, -.05133, .0004444, 2.302, .303],
    'Biolord':         [.09382, .0263, -.03116, .06717, .3933, .05051, -.07801, .0, 1.917, .07273],
    'Flow_Matching-MLP': [.08675, .4323, -.233, .3404, .4498, .3798, -.01807, .001333, 1.489, .4141],
    'Flow_Matching-Unet': [.0791, .1106, .262, .1465, .4012, .1899, .0145, .024, 1.115, .1157],
    'PRiMeFlow(CFG=1.0)': [.544, .378, .1411, .1101, .1808, .3449, .03581, .1071, 18.8, .447],
    'PRiMeFlow(CFG=3.0)': [.5455, .4051, .1574, .1131, .1911, .3662, .03466, .1058, 18.82, .4288],
    'PRiMeFlow(CFG=5.0)': [.5492, .4071, .1608, .1172, .1946, .3687, .03268, .1098, 19, .4591],
}


def read_test_metrics(run_dir):
    """run 의 test metric {이름: 값}. test 결과가 없으면 빈 dict.

    perturbench 는 test metric 을 첫 번째 logger(tensorboard)에만 기록한다
    (csv/metrics.csv 에는 학습 loss 만 있음). 그래서 tensorboard event 파일에서
    tag 별 마지막 값을 읽는다. cell-eval 이 성공한 run 은 그 metric 도 같이 들어온다.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    latest = {}   # tag -> (wall_time, value)
    for f in glob.glob(os.path.join(run_dir, 'tensorboard', '*', 'events.out.tfevents.*')):
        ea = EventAccumulator(f, size_guidance={'scalars': 0})
        ea.Reload()
        for tag in ea.Tags()['scalars']:
            ev = ea.Scalars(tag)[-1]
            if tag not in latest or ev.wall_time >= latest[tag][0]:
                latest[tag] = (ev.wall_time, ev.value)
    return {k: v for k, (_, v) in latest.items() if k in metrics}


def load_run_metrics(run_dir):
    """test metric 을 `metrics` 순서대로 반환."""
    found = read_test_metrics(run_dir)
    missing = [m for m in metrics if m not in found]
    if missing:
        raise ValueError(f"{run_dir}: test metric 이 없습니다 {missing} "
                         f"(test 가 끝나기 전이거나 test=False 로 돌린 run)")
    return [found[m] for m in metrics]


def list_runs():
    """logs/train/runs 아래 run 들의 override 와 test metric 유무를 출력."""
    for run_dir in sorted(glob.glob(os.path.join(RUN_ROOT, '*'))):
        ov_path = os.path.join(run_dir, '.hydra', 'overrides.yaml')
        overrides = ([l.strip('- \n') for l in open(ov_path) if l.strip()]
                     if os.path.exists(ov_path) else [])
        has_test = len(read_test_metrics(run_dir)) == len(metrics)
        print(f"{'[test O]' if has_test else '[test X]'} {os.path.relpath(run_dir, BASE_DIR)}")
        for o in overrides:
            print(f"           {o}")


# ─────────────────────────────────────────────────────────────
# normalize: 지표별 순위 (1 = best)
# ─────────────────────────────────────────────────────────────
def rank_matrix(raw):
    """(모델, 지표) 원값 -> 지표별 순위. 방향을 맞춘 뒤 큰 값이 1위, 동점은 평균 순위."""
    return np.stack([rankdata(-DIRECTION[i] * raw[:, i], method='average')
                     for i in range(raw.shape[1])], axis=1)


def axis_scores(rk):
    """지표 순위 -> 축별 평균 순위 (모델, 축)."""
    return np.stack([rk[:, idxs].mean(axis=1) for idxs in AXES.values()], axis=1)
# ─────────────────────────────────────────────────────────────
# plotting
# ─────────────────────────────────────────────────────────────
INK, MUTED, GRID = '#222222', '#666666', '#E3E3E3'
HIGHLIGHT, OTHER = '#C8102E', '#9A9A9A'


def _is_primeflow(name):
    return name.startswith('PRiMeFlow')


def _model_order(names):
    """행 순서: PRiMeFlow 를 맨 위로, 나머지는 RUNS 에 적은 순서 그대로 (순위로 재정렬하지 않음)."""
    return sorted(range(len(names)), key=lambda i: (not _is_primeflow(names[i]), i))


def plot_axis_summary(names, raw, title, path, highlight=True):
    """발표용 요약: 축마다 패널 하나, 모델별 평균 순위를 점으로 (오른쪽일수록 좋음).
    overall 점수는 일부러 만들지 않는다 (축 간 trade-off 를 숫자 하나로 가리지 않기 위해)."""
    n = len(names)
    per_axis = axis_scores(rank_matrix(raw))
    order = _model_order(names)
    y = np.arange(n)

    fig, axes = plt.subplots(1, len(AXES), figsize=(3.4 * len(AXES) + 2.2, 1.0 + 0.48 * n),
                             sharey=True, layout='constrained')
    for j, (ax, (axis_name, idxs)) in enumerate(zip(axes, AXES.items())):
        for row, m in enumerate(order):
            v = per_axis[m, j]
            hi = highlight and _is_primeflow(names[m])
            c = HIGHLIGHT if (hi or not highlight) else OTHER
            ax.hlines(row, n, v, color=c, lw=2, alpha=0.35, zorder=1)
            ax.scatter(v, row, s=90 if hi else 60, color=c, zorder=3,
                       edgecolor='white', linewidth=1.5)
            ax.text(v - 0.22, row, f"{v:g}" if v == int(v) else f"{v:.1f}",
                    va='center', ha='left', fontsize=10, color=INK,
                    fontweight='bold' if hi else 'normal')

        ax.set_xlim(n + 0.4, 0.4)        # 1위가 오른쪽
        ax.set_xticks(range(n, 0, -1))
        ax.set_ylim(n - 0.5, -0.5)
        ax.grid(axis='x', color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, labelsize=9.5, labelcolor=MUTED)
        for sp in ['top', 'right', 'left']:
            ax.spines[sp].set_visible(False)
        ax.spines['bottom'].set_color(GRID)
        sub = ', '.join(short[i].replace('\n', ' ') for i in idxs)
        ax.set_title(f"{axis_name}\n", fontsize=13, fontweight='bold', color=INK, loc='left')
        ax.text(0, 1.0, sub, transform=ax.transAxes, fontsize=9, color=MUTED, va='bottom')

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([names[m] for m in order], fontsize=11, color=INK)
    if highlight:
        for lbl, m in zip(axes[0].get_yticklabels(), order):
            if _is_primeflow(names[m]):
                lbl.set_color(HIGHLIGHT)
                lbl.set_fontweight('bold')

    fig.suptitle(title, fontsize=15, fontweight='bold', color=INK)
    fig.supxlabel(f"mean rank among {n} models  (1 = best, right = better)",
                  fontsize=10, color=MUTED)
    fig.savefig(path, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_metric_table(names, raw, title, path):
    """부록용 상세: 색 = 지표별 순위(진할수록 좋음), 숫자 = 원값."""
    n = len(names)
    rk = rank_matrix(raw)
    order = _model_order(names)
    cols = [i for idxs in AXES.values() for i in idxs]
    M = (n - rk[order][:, cols]) / (n - 1)      # 1위 = 1, 꼴찌 = 0
    R = raw[order][:, cols]

    fig, ax = plt.subplots(figsize=(13, 1.6 + 0.5 * n))
    ax.imshow(M, cmap=plt.cm.Blues, vmin=-0.1, vmax=1.05, aspect='auto')
    for r in range(n):
        for c in range(len(cols)):
            ax.text(c, r, f"{R[r, c]:.3g}", ha='center', va='center', fontsize=10,
                    color='white' if M[r, c] > 0.55 else INK)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{short[i]} {'↓' if DIRECTION[i] == -1 else '↑'}" for i in cols],
                       fontsize=9.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([names[m] for m in order], fontsize=11)
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which='minor', color='white', lw=2)
    ax.tick_params(which='both', length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    start = 0
    for axis_name, idxs in AXES.items():   # 축 묶음 헤더 + 구분선
        w = len(idxs)
        ax.text(start + (w - 1) / 2, -0.7, axis_name.replace(' ', '\n') if w == 1 else axis_name,
                ha='center', va='bottom', fontsize=10 if w == 1 else 11.5,
                fontweight='bold', color=INK)
        if start:
            ax.axvline(start - 0.5, color='white', lw=6)
        start += w

    ax.set_title(title, fontsize=14, fontweight='bold', color=INK, pad=62)
    ax.annotate(f"Number = raw value.  Colour = rank among {n} models on that metric (darker = better).",
                xy=(0.5, 1.0), xycoords='axes fraction', xytext=(0, 40), textcoords='offset points',
                ha='center', va='bottom', fontsize=9.5, color=MUTED)
    fig.savefig(path, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# export: 발표자료 제작용 데이터 (python metric_each_model_graph.py --export)
# ─────────────────────────────────────────────────────────────
EXPORT_DIR = os.path.join(BASE_DIR, 'presentation_data')

METRIC_INFO = {   # metric -> 설명 (발표자료에서 지표를 소개할 때 쓰는 문구)
    'rmse_average':            'Root mean squared error between predicted and observed mean expression per perturbation',
    'rmse_rank_average':       'Rank of the true perturbation among all perturbations when ranked by RMSE to the prediction (normalized, 0 = perfect)',
    'cosine_pca_average':      'Cosine similarity of predicted vs observed mean expression in PCA space',
    'cosine_rank_pca_average': 'Rank version of cosine (PCA): is the prediction closest to its own perturbation? (normalized, 0 = perfect)',
    'cosine_logfc':            'Cosine similarity of predicted vs observed log fold-change (vs control)',
    'cosine_rank_logfc':       'Rank version of cosine (logFC) (normalized, 0 = perfect)',
    'r2_score_scores':         'R² of predicted vs observed expression',
    'top_k_recall_scores':     'Recall of the observed top-k differentially expressed genes among the predicted top-k',
    'mmd_pca':                 'Maximum mean discrepancy between predicted and observed single-cell distributions in PCA space',
    'mmd_rank_pca':            'Rank version of MMD (PCA) (normalized, 0 = perfect)',
}

MODEL_INFO = {    # 표시 이름 -> (group, 설명)
    'Linear_additive': ('baseline', 'Adds per-gene perturbation effects linearly to control expression'),
    'Latent_additive': ('baseline', 'Autoencoder; adds perturbation effects in a learned latent space'),
    'Decoder_only':    ('baseline', 'Decoder from perturbation/covariate embeddings to expression, no encoder'),
    'CPA':             ('baseline', 'Compositional Perturbation Autoencoder (Lotfollahi et al.)'),
    'SAMS_VAE':        ('baseline', 'Sparse additive mechanism shift VAE'),
    'Biolord':         ('baseline', 'Disentangled representation model (Piran et al.)'),
    'PRiMeFlow(Unet, CFG=1.0)':
        ('PRiMeFlow', 'Conditional flow matching in gene space, 1D U-Net velocity field (16 ch), no gene embedding'),
    'PRiMeFlow(Unet+gemb, CFG=1.0)':
        ('PRiMeFlow', 'Same U-Net + 16-dim learned gene embedding channel; 4-GPU DDP (effective batch 512); main PRiMeFlow model'),
    'PRiMeFlow(Unet+gemb, CFG=3.0)':
        ('PRiMeFlow', 'Same checkpoint as Unet+gemb (epoch 40), evaluated with classifier-free guidance weight 3.0'),
    'PRiMeFlow(Unet+gemb, CFG=5.0)':
        ('PRiMeFlow', 'Same checkpoint as Unet+gemb (epoch 40), evaluated with classifier-free guidance weight 5.0'),
}


def export_data(models, main_names, out_dir=EXPORT_DIR):
    """원값 / 지표별 순위 / 축별 평균 순위를 CSV 로, 그림을 figures/ 로 내보낸다."""
    import shutil
    import pandas as pd

    os.makedirs(os.path.join(out_dir, 'figures'), exist_ok=True)
    axis_of = {i: a for a, idxs in AXES.items() for i in idxs}
    label = [s.replace('\n', ' ') for s in short]

    pd.DataFrame({
        'metric': metrics, 'label': label, 'axis': [axis_of[i] for i in range(len(metrics))],
        'direction': ['lower_is_better' if d == -1 else 'higher_is_better' for d in DIRECTION],
        'description': [METRIC_INFO[m] for m in metrics],
    }).to_csv(os.path.join(out_dir, 'metrics_info.csv'), index=False)

    pd.DataFrame([{
        'model': n, 'group': MODEL_INFO.get(n, ('', ''))[0],
        'in_main_comparison': n in main_names, 'in_ablation': n in ABLATION_MODELS,
        'run_dir': RUNS.get(n, ''), 'description': MODEL_INFO.get(n, ('', ''))[1],
    } for n in models]).to_csv(os.path.join(out_dir, 'models_info.csv'), index=False)

    raw_all = pd.DataFrame([models[n] for n in models], index=list(models), columns=label)
    raw_all.rename_axis('model').to_csv(os.path.join(out_dir, 'raw_metrics.csv'), float_format='%.4g')

    long_rows = []
    for tag, names in (('main', main_names), ('ablation', ABLATION_MODELS)):
        raw = np.array([models[n] for n in names], dtype=float)
        rk = rank_matrix(raw)
        per_axis = axis_scores(rk)
        pd.DataFrame(rk, index=names, columns=label).rename_axis('model') \
            .to_csv(os.path.join(out_dir, f'{tag}_metric_ranks.csv'))
        pd.DataFrame(per_axis.round(3), index=names, columns=list(AXES)).rename_axis('model') \
            .to_csv(os.path.join(out_dir, f'{tag}_axis_ranks.csv'))
        for r, n in enumerate(names):
            for i in range(len(metrics)):
                long_rows.append(dict(comparison=tag, model=n, metric=label[i], axis=axis_of[i],
                                      direction=DIRECTION[i], value=raw[r, i], rank=rk[r, i],
                                      n_models=len(names)))
    pd.DataFrame(long_rows).to_csv(os.path.join(out_dir, 'metrics_long.csv'), index=False, float_format='%.4g')

    for f in ('metric_summary.png', 'metric_ablation.png', 'metric_heatmap.png'):
        if os.path.exists(os.path.join(BASE_DIR, f)):
            shutil.copy(os.path.join(BASE_DIR, f), os.path.join(out_dir, 'figures', f))
    print(f"exported -> {os.path.relpath(out_dir, BASE_DIR)}/")


if __name__ == '__main__':
    if '--list' in sys.argv:
        list_runs()
        sys.exit(0)

    models = {name: load_run_metrics(os.path.join(BASE_DIR, d)) for name, d in RUNS.items()}
    main_names = list(MAIN_MODELS)
    if INCLUDE_LEGACY:
        for k, v in LEGACY_MODELS.items():
            models[f'{k} (legacy)'] = v
            main_names.append(f'{k} (legacy)')

    figures = [
        (plot_axis_summary, main_names,
         'PRiMeFlow vs. baselines (Norman19, test split)', 'metric_summary.png', {}),
        (plot_metric_table, main_names,
         'All metrics: PRiMeFlow vs. baselines', 'metric_heatmap.png', {}),
        (plot_axis_summary, ABLATION_MODELS,
         'PRiMeFlow ablation: gene embedding & CFG weight', 'metric_ablation.png',
         dict(highlight=False)),
    ]
    for fn, names, title, path, kw in figures:
        if len(names) < 2:
            print(f"skip {path}: 비교할 모델이 2개 이상 필요합니다.")
            continue
        raw = np.array([models[n] for n in names], dtype=float)
        fn(names, raw, title, path, **kw)
        print(f"saved {path}")

    if '--export' in sys.argv:
        export_data(models, main_names)
