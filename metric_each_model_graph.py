import matplotlib
matplotlib.use('Agg')

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

UNBOUNDED = [0, 1, 8, 9]     # 이론적 상한 없음 -> min-max
RANK_IDX = [3, 5]            # 0~1 이지만 값이 0 근처에 몰린 rank 지표
R2_IDX = 6

# [Flow_Matching-MLP metric data]
# 1. SiLU [source: noise]
#   -> [np.inf, .4490, .6194, .5409, .1825, .4641, -.02963, .005778, .9169, .5525] // inf 나와서 망한 케이스
# 2. SiLU + LayerNorm(input 정규화) [source: noise]
#   -> [.313, .1879, .09682, .1535, .02615, .2692, -4.534, .08444, 4.342, .1793],
# 3. SiLU + LayerNorm(input 정규화) [source: control]
#   -> 


models = {
    'Linear_additive': [.05779, .009596, .8074, .01313, .622, .04545, -.02479, .002667, 1.334, .01364],
    'Latent_additive': [.0419, .01353, .8791, .02077, .7969, .008696, -.06922, .0, 3.247, .01304],
    'Decoder_only':    [.04305, .01111, .8817, .01111, .7588, .01061, .2202, .28, 3.326, .01313],
    'CPA':             [.0478, .02323, .8041, .03232, .7795, .009091, .07561, .02978, 2.218, .03788],
    'SAMS_VAE':        [.09383, .04697, -.07595, .3076, .4365, .03485, -.05133, .0004444, 2.302, .303],
    'Biolord':         [.09382, .0263, -.03116, .06717, .3933, .05051, -.07801, .0, 1.917, .07273],
    'Flow_Matching-MLP': [.08675, .4323, -.233, .3404, .4498, .3798, -.01807, .001333, 1.489, .4141],
    'Flow_Matching-Unet': [.5454, .3652, .2612, .3807, .1479, .3034, .03775, .1074, 16.3, .429]
}

palette = {
    'Linear_additive': "#2097F9", 'Latent_additive': "#1DBF4A",
    'Decoder_only': "#F99F20",    'CPA': "#F92720",
    'SAMS_VAE': "#F92097",        'Biolord': "#A220F9",
    'Flow_Matching-MLP': "#E1D830", 'Flow_Matching-Unet': "#23e4b7"
}

# ─────────────────────────────────────────────────────────────
# normalize: 모든 지표를 0~1, higher = better 로 통일
# ─────────────────────────────────────────────────────────────
names = list(models.keys())
raw = np.array([models[n] for n in names], dtype=float)
norm = raw.copy()

minmax_cols = list(UNBOUNDED) + (RANK_IDX if RELATIVE_RANK else [])

for i in range(len(metrics)):
    if i in minmax_cols:
        lo, hi = raw[:, i].min(), raw[:, i].max()
        norm[:, i] = (raw[:, i] - lo) / (hi - lo) if hi > lo else 0.5
    else:
        norm[:, i] = np.clip(raw[:, i], 0, 1)   # r2 포함: 음수는 0, 상한 1

for i, d in enumerate(DIRECTION):
    if d == -1:
        norm[:, i] = 1.0 - norm[:, i]

score = norm.mean(axis=1)
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
fig, axes = plt.subplots(nrows, ncols, figsize=(17, 17),
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
               "- Panels sorted by overall score",
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
fig2, (axh, axb) = plt.subplots(
    1, 2, figsize=(16, 6), gridspec_kw=dict(width_ratios=[5, 1], wspace=0.05))

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
axb.set_title("Mean", fontsize=14, fontweight='bold', pad=14)
for b, s in zip(bars, s_sorted):
    axb.text(min(s + 0.03, 0.97), b.get_y() + b.get_height() / 2,
             f"{s:.3f}", va='center', fontsize=10, fontweight='bold')
for sp in ['top', 'right']:
    axb.spines[sp].set_visible(False)

fig2.savefig('metric_heatmap.png', dpi=140, bbox_inches='tight', facecolor='white')
print("saved")