import re

import matplotlib
matplotlib.use('Agg')

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────────────────────
# config
# ─────────────────────────────────────────────────────────────
# rank 지표는 값이 0 근처에 몰려 있어 뒤집으면 전부 1에 붙는다.
#   False = 절대 스케일 유지 (정직하지만 변별력 낮음)
#   True  = 7개 모델 내 상대 순위로 재정규화 (변별력 높음, 의미는 '상대적')
RELATIVE_RANK = False

metrics = [
    'rmse_avg', 'rmse_rank_avg',
    'cos_pca_avg', 'cos_rank_pca_avg',
    'cos_logfc', 'cos_rank_logfc',
    'r2_score', 'top_k_recall_score',
    'mmd_pca', 'mmd_rank_pca',
]

short = [
    'RMSE', 'RMSE\nrank',
    'cos\nPCA', 'cos rank\nPCA',
    'cos\nlogFC', 'cos rank\nlogFC',
    'R²', 'top-k\nrecall',
    'MMD\nPCA', 'MMD rank\nPCA',
]

# +1 = higher is better / -1 = lower is better (스케일링 후 뒤집음)
DIRECTION = [-1, -1, +1, -1, +1, -1, +1, +1, -1, -1]

UNBOUNDED = [0, 1, 8, 9]     # 이론적 상한 없음 -> robust(median/MAD) scaling
RANK_IDX = [3, 5]            # 0~1 이지만 값이 0 근처에 몰린 rank 지표
R2_IDX = 6

# overall score 집계용 지표군: 같은 신호의 raw/rank 쌍(RMSE, cos PCA, cos logFC,
# MMD)을 하나로 묶어 군 내 평균부터 낸 뒤 군끼리 평균한다.
# 그냥 10개 지표를 통으로 평균 내면 raw/rank 쌍이 6/10을 차지해서 사실상
# 같은 신호가 3번 중복 반영되고, R²·top-k recall 같은 단일 지표는
# 1/10로 묻힌다.
METRIC_GROUPS = {
    'RMSE': [0, 1],
    'cos_PCA': [2, 3],
    'cos_logFC': [4, 5],
    'R2': [6],
    'top_k_recall': [7],
    'MMD_PCA': [8, 9],
}

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


models = {
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

palette = {
    'Linear_additive': "#2097F9", 'Latent_additive': "#1DBF4A",
    'Decoder_only': "#F99F20",    'CPA': "#F92720",
    'SAMS_VAE': "#F92097",        'Biolord': "#A220F9",
    'Flow_Matching-MLP': "#E1D830", 'Flow_Matching-Unet': "#23e4b7"
}

# 팔레트에 없는 모델은 이름 접두어(괄호 앞부분)로 "family"를 묶어서
# 한 색상 계열의 농도 그라데이션을 자동 배정한다.
# 예: PRiMeFlow(CFG=1.0/3.0/5.0) -> Reds 계열로 CFG가 커질수록 진해짐.
# 새 variant가 더 늘어나도 팔레트를 손으로 안 건드려도 되게 하기 위함.
FAMILY_CMAPS = {
    'PRiMeFlow': cm.Reds,
}
DEFAULT_CMAP = cm.tab10


def _family_of(name):
    return re.split(r'[\(\[]', name)[0].strip()


_families = {}
for _n in models:
    if _n not in palette:
        _families.setdefault(_family_of(_n), []).append(_n)

for _fam, _members in _families.items():
    _cmap = FAMILY_CMAPS.get(_fam, DEFAULT_CMAP)
    _members = sorted(_members)
    for _i, _n in enumerate(_members):
        _t = 0.45 + 0.45 * (_i / max(len(_members) - 1, 1))
        palette[_n] = mcolors.to_hex(_cmap(_t))

# ─────────────────────────────────────────────────────────────
# normalize: 모든 지표를 0~1, higher = better 로 통일
# ─────────────────────────────────────────────────────────────
names = list(models.keys())
raw = np.array([models[n] for n in names], dtype=float)
norm = raw.copy()

def robust_scale(col, clip=3.0):
    """median/MAD 기반 0~1 정규화.
    min-max는 값 하나가 극단적으로 튀면(outlier) 그 하나가 스케일의
    양 끝을 정의해버려서 나머지 모델들이 좁은 구간으로 뭉개진다.
    (예: MMD_PCA에서 PRiMeFlow만 ~19, 나머지는 1~3 -> min-max로는
    나머지 모델들 간 실제 차이가 거의 안 보이게 됨)
    median/MAD 기준으로 표준화한 뒤 ±clip 표준편차에서 잘라내면,
    이상치는 스케일 끝단으로 clip되고 나머지 값들은 자기들끼리의
    상대적 차이를 그대로 유지한다."""
    med = np.median(col)
    mad = np.median(np.abs(col - med)) * 1.4826   # 정규분포 하에서 std와 동일 스케일
    if mad == 0:
        lo, hi = col.min(), col.max()
        return (col - lo) / (hi - lo) if hi > lo else np.full_like(col, 0.5)
    z = np.clip((col - med) / mad, -clip, clip)
    return (z + clip) / (2 * clip)


minmax_cols = list(UNBOUNDED) + (RANK_IDX if RELATIVE_RANK else [])

for i in range(len(metrics)):
    if i in minmax_cols:
        norm[:, i] = robust_scale(raw[:, i])
    else:
        norm[:, i] = np.clip(raw[:, i], 0, 1)   # r2 포함: 음수는 0, 상한 1

for i, d in enumerate(DIRECTION):
    if d == -1:
        norm[:, i] = 1.0 - norm[:, i]

group_score = np.stack(
    [norm[:, idxs].mean(axis=1) for idxs in METRIC_GROUPS.values()], axis=1)
score = group_score.mean(axis=1)                 # 지표군 평균의 평균 = overall score
order = np.argsort(-score)                      # 종합 점수 내림차순

labels = [f"{s}\n{'↓' if d == -1 else '↑'}" for s, d in zip(short, DIRECTION)]

# ─────────────────────────────────────────────────────────────
# figure 1: 모델별 개별 레이더 (small multiples)
# ─────────────────────────────────────────────────────────────
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

best = norm.max(axis=0).tolist()                # 지표별 최고 성능 = 기준선
best += best[:1]

n_models = len(names)
ncols = 3
nrows = int(np.ceil((n_models + 1) / ncols))   # +1 = 범례 칸
cell = 5.7   # 모델 수가 늘어도 레이더 패널 하나의 크기는 고정
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * cell, nrows * cell),
                         subplot_kw=dict(polar=True), layout='constrained')
fig.get_layout_engine().set(hspace=0.14, wspace=0.10)   # 타이틀 충돌 방지
axes = axes.ravel()

for ax_i, m_i in enumerate(order):
    ax = axes[ax_i]
    n = names[m_i]
    v = norm[m_i].tolist()
    v += v[:1]

    ax.fill(angles, best, color="#BBBBBB", alpha=0.22, zorder=1)
    ax.plot(angles, best, color="#999999", lw=0.8, ls='--', zorder=2)

    ax.plot(angles, v, color=palette[n], lw=2, zorder=3)
    ax.fill(angles, v, color=palette[n], alpha=0.35, zorder=3)

    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.tick_params(axis='x', pad=6)
    ax.grid(color="#DDDDDD", lw=0.7)
    ax.spines['polar'].set_color("#CCCCCC")
    ax.set_title(f"{ax_i+1}. {n}\noverall {score[m_i]:.3f}",
                 fontsize=13, color=palette[n], fontweight='bold', pad=22)

legend_ax = axes[n_models]        # ← axes[-1] 금지 (그게 8번째 모델을 덮어썼던 원인)
legend_ax.axis('off')
legend_ax.set_title("How to read", fontsize=13, fontweight='bold', pad=18)
legend_ax.text(0.5, 0.45,
               "- Farther out = better\n"
               "- Dashed grey = best value\n"
               "   achieved on each metric\n"
               "- (down arrow) = lower-is-better\n"
               "   metric, axis flipped\n"
               "- Panels sorted by overall score\n"
               "- Overall = mean of 6 metric-group\n"
               "   means (RMSE, cos PCA, cos logFC,\n"
               "   MMD, R², top-k recall), each\n"
               "   robust-scaled (median/MAD)",
               transform=legend_ax.transAxes, ha='center', va='center',
               fontsize=11.5, linespacing=1.9)

for j in range(n_models + 1, len(axes)):   # 9칸 중 남는 칸 숨김
    axes[j].axis('off')

fig.suptitle("Per-model performance profile - wider is better", fontsize=19, fontweight='bold', y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig('metric_radar.png', dpi=140, bbox_inches='tight', facecolor='white')

# ─────────────────────────────────────────────────────────────
# figure 2: 히트맵 (색 = 정규화 점수, 숫자 = 원본 값) + 종합 점수 막대
# ─────────────────────────────────────────────────────────────
heatmap_h = max(6, 1.4 + 0.62 * n_models)   # 모델(행) 수에 맞춰 세로로 늘어남
fig2, (axh, axb) = plt.subplots(
    1, 2, figsize=(16, heatmap_h), gridspec_kw=dict(width_ratios=[5, 1], wspace=0.05))

M = norm[order]
R = raw[order]
row_names = [names[i] for i in order]

axh.imshow(M, cmap='YlGnBu', vmin=0, vmax=1, aspect='auto')

axh.set_xticks(range(len(metrics)))
axh.set_xticklabels([f"{s}\n{'↓' if d == -1 else '↑'}" for s, d in zip(short, DIRECTION)],
                    fontsize=9)
axh.set_yticks(range(len(row_names)))
axh.set_yticklabels(row_names, fontsize=11)
axh.set_xticks(np.arange(-.5, len(metrics), 1), minor=True)
axh.set_yticks(np.arange(-.5, len(row_names), 1), minor=True)
axh.grid(which='minor', color='white', lw=2)
axh.tick_params(which='minor', length=0)

for r in range(M.shape[0]):
    for c in range(M.shape[1]):
        axh.text(c, r, f"{R[r, c]:.3g}", ha='center', va='center', fontsize=8.5,
                 color='white' if M[r, c] > 0.6 else '#222222')

axh.set_title("Raw metric values  (colour = normalized score, darker = better)",
              fontsize=14, fontweight='bold', pad=14)

s_sorted = score[order]
bars = axb.barh(range(len(row_names)), s_sorted,
                color=[palette[n] for n in row_names], alpha=0.85)
axb.set_ylim(len(row_names) - 0.5, -0.5)
axb.set_xlim(0, 1)
axb.set_yticks([])
axb.set_xlabel("overall score", fontsize=11)
axb.set_title("Overall\n(6-group mean)", fontsize=14, fontweight='bold', pad=14)
for b, s in zip(bars, s_sorted):
    axb.text(min(s + 0.03, 0.97), b.get_y() + b.get_height() / 2,
             f"{s:.3f}", va='center', fontsize=10, fontweight='bold')
for sp in ['top', 'right']:
    axb.spines[sp].set_visible(False)

fig2.savefig('metric_heatmap.png', dpi=140, bbox_inches='tight', facecolor='white')
print("saved")