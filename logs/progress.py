"""
progress.py — logs/train/runs 아래 학습 run 들의 진행 현황판

사용 (PRiMeFlow 루트에서, perturbench env):
    python logs/progress.py            # 한 번 출력
    python logs/progress.py -w 10      # 10초마다 갱신 (Ctrl+C 로 종료)
    python logs/progress.py -w 10 -a   # 오래된 run 까지 전부 표시 (기본: 최근 24시간)

상태 판정
    학습중   : metrics.csv 가 최근 3분 안에 갱신됨
    test 1/2 : 예측 저장 후 기본 metric(rmse/cosine/r2/top-k/mmd) 계산 중
    test 2/2 : cell-eval 실행 중 (evaluation/cell_eval/ 생성됨, 기본 timeout 15분)
    완료     : evaluation/summary_by_cov.csv 저장 + cell-eval 종료(성공/실패) 기록
    에러     : baseline 큐(status.txt)에 exit 가 찍혔는데 test 결과가 없음 → logs/baselines/<run>.out 확인
    멈춤?    : 3분 넘게 갱신 없음 (에러로 죽었을 수 있음 → .log 확인)

test 행 아래 └ 줄: test 시작 후 경과, 현재 단계, 그 단계에 들어간 시각,
                   완료된 test 들의 평균 소요시간 기준 남은 시간(추정)
"""
import argparse
import csv
import datetime as dt
import glob
import os
import re
import subprocess
import time
import unicodedata

import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.path.join(BASE_DIR, 'logs', 'train', 'runs')
STALE_SEC = 180


def _run_name(run_dir, overrides):
    """폴더명 끝의 _<모델> 또는 override 의 experiment/model 로 표시 이름을 만든다."""
    base = os.path.basename(run_dir)
    m = re.match(r'\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d_(.+)', base)
    if m:
        return m.group(1)
    exp = next((o.split('=', 1)[1] for o in overrides if o.startswith('experiment=')), '?')
    name = exp.split('/')[-1].replace('_best_params_norman19', '')
    extra = [o.split('=', 1)[1] for o in overrides if o.startswith('model.architecture=')]
    return name + (f'-{extra[0]}' if extra else '')


def _start_time(run_dir):
    m = re.search(r'(\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d)', os.path.basename(run_dir))
    return dt.datetime.strptime(m.group(1), '%Y-%m-%d_%H-%M-%S') if m else None


def _w(s):
    """터미널 표시 폭 (한글 등 전각 문자는 2칸)."""
    return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in s)


def _l(s, n):
    return s + ' ' * max(n - _w(s), 0)


def _r(s, n):
    return ' ' * max(n - _w(s), 0) + s


def _fmt_dur(minutes):
    if minutes is None:
        return '-'
    return f'{minutes / 60:.1f}h' if minutes >= 60 else f'{minutes:.0f}m'


def _log_time(line):
    """train.log 한 줄 앞의 [YYYY-mm-dd HH:MM:SS,ms] 타임스탬프."""
    m = re.match(r'\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)', line)
    return dt.datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S') if m else None


def _mtime(path):
    return dt.datetime.fromtimestamp(os.path.getmtime(path))


def read_exit_codes():
    """baseline 큐 status.txt 의 '<시각> <name> (gpu N) exit=K' → {name: [(시각, K), ...]}."""
    path = os.path.join(BASE_DIR, 'logs', 'baselines', 'status.txt')
    codes = {}
    if os.path.exists(path):
        for line in open(path):
            m = re.match(r'(\S+ \S+) (\S+) \(gpu \d+\) exit=(\d+)', line)
            if m:
                when = dt.datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
                codes.setdefault(m.group(2), []).append((when, int(m.group(3))))
    return codes


def read_test(run_dir):
    """test 단계 정보. test 를 시작하지 않았으면 None.

    test metric 은 csv logger 가 아니라 tensorboard 와 evaluation/summary_by_cov.csv 에만
    저장되므로(perturbench 가 첫 번째 logger 에만 기록), 완료 여부는 evaluation/ 파일과
    train.log 의 cell-eval 기록으로 판단한다.
    """
    log_path = os.path.join(run_dir, 'train.log')
    if not os.path.exists(log_path):
        return None
    start = launch = None
    n_launch = n_ce_fail = 0
    for line in open(log_path, errors='ignore'):
        if 'Starting testing' in line:
            start = _log_time(line)
        elif 'Launching cell-eval' in line:
            n_launch += 1
            launch = launch or _log_time(line)
        elif re.search(r'cell-eval subprocess (failed|timed out|succeeded but)', line):
            n_ce_fail += 1
    if start is None:
        return None

    ev_dir = os.path.join(run_dir, 'evaluation')
    summary = os.path.join(ev_dir, 'summary_by_cov.csv')
    ce_dirs = [d for d in glob.glob(os.path.join(ev_dir, 'cell_eval', '*')) if os.path.isdir(d)]
    ce_ok = [d for d in ce_dirs if os.path.exists(os.path.join(d, 'agg_results.csv'))]
    preds = glob.glob(os.path.join(ev_dir, 'predicted_anndata', '*.h5ad'))
    t = dict(start=start, end=None, stage=None, since=None, core_done=os.path.exists(summary),
             ce_note='')

    if t['core_done'] and n_launch and len(ce_ok) + n_ce_fail >= n_launch:
        t['end'] = max([_mtime(summary)] + [_mtime(os.path.join(d, 'agg_results.csv')) for d in ce_ok])
        t['ce_note'] = f' (cell-eval 실패 {n_ce_fail}건 → 기본 metric 만 있음)' if n_ce_fail else ''
    elif n_launch or ce_dirs:
        t['stage'] = f'2/2 cell-eval 실행 중 ({len(ce_ok) + n_ce_fail}/{max(n_launch, len(ce_dirs), 1)} covariate 끝남, 최대 15m)'
        t['since'] = launch or min(_mtime(d) for d in ce_dirs)
    elif t['core_done']:
        t['stage'] = '2/2 cell-eval 준비 중 (기본 metric 저장 완료)'
        t['since'] = _mtime(summary)
    elif preds:
        t['stage'] = f'1/2 기본 metric 계산 중 (예측 파일 {len(preds)}개 저장 완료)'
        t['since'] = max(_mtime(f) for f in preds)
    else:
        t['stage'] = '0/2 test 예측 중'
        t['since'] = start
    return t


def _exit_after(exit_codes, name, run_dir):
    """같은 이름으로 다시 돌린 run 이 있으므로, 이 run 시작 이후에 찍힌 exit 만 이 run 의 것."""
    start_t = _start_time(run_dir)
    return next((c for when, c in (exit_codes or {}).get(name, [])
                 if start_t and when >= start_t), None)


def read_run(run_dir, exit_codes=None):
    hydra_dir = os.path.join(run_dir, '.hydra')
    ov_path = os.path.join(hydra_dir, 'overrides.yaml')
    overrides = yaml.safe_load(open(ov_path)) if os.path.exists(ov_path) else []
    cfg_path = os.path.join(hydra_dir, 'config.yaml')
    cfg = yaml.safe_load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    max_epochs = (cfg.get('trainer') or {}).get('max_epochs')
    patience = ((cfg.get('callbacks') or {}).get('early_stopping') or {}).get('patience')

    info = dict(name=_run_name(run_dir, overrides or []), dir=os.path.basename(run_dir),
                epoch=None, max_epochs=max_epochs, speed=None, eta_max=None, eta_es=None,
                val=None, best_ep=None, status='대기', test=None)

    files = glob.glob(os.path.join(run_dir, 'csv', 'version_*', 'metrics.csv'))
    path = max(files, key=lambda f: int(re.search(r'version_(\d+)', f).group(1))) if files else None
    rows = []
    if path:
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
    if not rows:
        # 학습 기록이 없는 run: 아직 시작 전이거나, checkpoint 로 test 만 돌리는 run (train=False)
        test = read_test(run_dir)
        info['test'] = test
        if test is None:
            info['status'] = '준비중'
        elif test['end'] or (test['core_done'] and _exit_after(exit_codes, info['name'], run_dir) == 0):
            if not test['end']:
                test['end'] = _mtime(os.path.join(run_dir, 'evaluation', 'summary_by_cov.csv'))
            info['status'] = '완료'
        else:
            info['status'] = 'test ' + test['stage'][:3]
        return info

    epochs = [int(r['epoch']) for r in rows if r.get('epoch')]
    val_key = 'val_loss_epoch' if 'val_loss_epoch' in rows[0] else 'val_loss'
    vals = [(int(r['epoch']), float(r[val_key])) for r in rows if r.get(val_key) and r.get('epoch')]

    ep = max(epochs) if epochs else 0
    info['epoch'] = ep
    if vals:
        best_ep, best_v = min(vals, key=lambda x: x[1])
        info['best_ep'], info['val'] = best_ep, vals[-1][1]

    start = _start_time(run_dir)
    mtime = os.path.getmtime(path)
    if start and ep > 0:
        per_ep = (dt.datetime.fromtimestamp(mtime) - start).total_seconds() / 60 / (ep + 1)
        info['speed'] = per_ep
        if max_epochs:
            info['eta_max'] = max(max_epochs - 1 - ep, 0) * per_ep
        if patience is not None and info['best_ep'] is not None:
            info['eta_es'] = max(info['best_ep'] + patience - ep, 0) * per_ep

    test = read_test(run_dir)
    info['test'] = test
    exit_code = _exit_after(exit_codes, info['name'], run_dir)
    if test and (test['end'] or (exit_code == 0 and test['core_done'])):
        if not test['end']:              # cell-eval 을 안 돌린 run: 기본 metric 저장 시각이 끝
            test['end'] = _mtime(os.path.join(run_dir, 'evaluation', 'summary_by_cov.csv'))
        info['status'] = '완료'
    elif exit_code is not None:
        info['status'] = f'에러({exit_code})' if exit_code else '종료'
    elif test is not None:
        info['status'] = 'test ' + test['stage'][:3]
    elif time.time() - mtime < STALE_SEC:
        info['status'] = '학습중'
    else:
        info['status'] = '멈춤?'
    return info


def gpu_line():
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return ''
    parts = []
    for line in out.strip().splitlines():
        i, u, used, tot = [x.strip() for x in line.split(',')]
        parts.append(f'GPU{i} {u:>3}% {int(used) / 1024:4.1f}/{int(tot) / 1024:.0f}G')
    return ' | '.join(parts)


def render(show_all):
    runs = sorted(glob.glob(os.path.join(RUN_ROOT, '*')))
    if not show_all:
        cutoff = dt.datetime.now() - dt.timedelta(hours=24)
        runs = [r for r in runs if (_start_time(r) or dt.datetime.min) >= cutoff]
    exit_codes = read_exit_codes()
    infos = [read_run(r, exit_codes) for r in runs]
    now = dt.datetime.now()
    # 끝난 test 들의 소요시간 → 진행 중 test 의 남은 시간 추정 기준
    test_durs = [(x['test']['end'] - x['test']['start']).total_seconds() / 60
                 for x in infos if x['test'] and x['test']['end']]
    ref_dur = sum(test_durs) / len(test_durs) if test_durs else None

    lines = [f"[{dt.datetime.now():%H:%M:%S}] 학습 진행 현황  ({RUN_ROOT})", gpu_line(), '']
    cols = [('run', 30, _l), ('상태', 8, _l), ('epoch', 11, _r), ('속도/ep', 9, _r),
            ('val_loss', 11, _r), ('best_ep', 9, _r), ('남은(최대)', 12, _r), ('조기종료시', 12, _r)]
    hdr = ''.join(f(t, n) for t, n, f in cols)
    lines += [hdr, '-' * _w(hdr)]
    for x in infos:
        ep = '-' if x['epoch'] is None else f"{x['epoch']}/{x['max_epochs'] or '?'}"
        speed = '-' if x['speed'] is None else f"{x['speed'] * 60:.0f}s"
        val = '-' if x['val'] is None else f"{x['val']:.4g}"
        best = '-' if x['best_ep'] is None else str(x['best_ep'])
        training = x['status'] in ('학습중', '멈춤?')     # 남은 시간 열은 학습 단계에서만 의미 있음
        eta_max = _fmt_dur(x['eta_max']) if training else '-'
        eta_es = _fmt_dur(x['eta_es']) if training else '-'
        vals_row = [x['name'][:29], x['status'], ep, speed, val, best, eta_max, eta_es]
        lines.append(''.join(f(v, n) for v, (_, n, f) in zip(vals_row, cols)))
        t = x['test']
        if t and t['end']:
            dur = (t['end'] - t['start']).total_seconds() / 60
            lines.append(f"  └ test {t['start']:%H:%M} → {t['end']:%H:%M} 완료 (소요 {_fmt_dur(dur)}){t['ce_note']}")
        elif t and x['status'].startswith('test'):
            el = (now - t['start']).total_seconds() / 60
            st = (now - t['since']).total_seconds() / 60
            eta = ''
            if ref_dur is not None:
                eta = (f" | 남은 ~{_fmt_dur(ref_dur - el)} (완료 test 평균 {_fmt_dur(ref_dur)} 기준)"
                       if ref_dur > el else f" | 완료 test 평균 {_fmt_dur(ref_dur)} 초과")
            lines.append(f"  └ test {t['start']:%H:%M} 시작, {_fmt_dur(el)} 경과 | "
                         f"{t['stage']}, 이 단계 {_fmt_dur(st)}째{eta}")
    if not infos:
        lines.append('(run 없음)')

    status_txt = os.path.join(BASE_DIR, 'logs', 'baselines', 'status.txt')
    if os.path.exists(status_txt):
        lines += ['', '[baseline 큐 status.txt 최근 기록]']
        lines += ['  ' + l.rstrip() for l in open(status_txt).readlines()[-6:]]
    lines += ['', "남은(최대)=max_epochs 까지 / 조기종료시=지금부터 val_loss 개선이 없으면 early stopping 까지 (+ test 평가 시간 별도)",
              "val_loss 는 모델마다 loss 종류가 달라 서로 비교 불가 (decoder/linear=MSE ~0.1, latent/cpa/sams/biolord=NLL ~수백, PrimeFlow=CFM ~0.4)"]
    return '\n'.join(lines)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-w', '--watch', type=float, default=0, help='N초마다 갱신')
    ap.add_argument('-a', '--all', action='store_true', help='24시간 이전 run 도 표시')
    args = ap.parse_args()
    if args.watch <= 0:
        print(render(args.all))
    else:
        try:
            while True:
                text = render(args.all)
                print('\033[2J\033[H' + text, flush=True)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            pass
