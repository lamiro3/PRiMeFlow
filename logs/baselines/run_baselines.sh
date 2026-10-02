#!/bin/bash
# norman19 baseline 6종을 GPU 4장에 큐로 나눠 실행.
# 공정 비교를 위해 모두 test split(final_test) 평가 + seed 245 로 통일.
cd /data/project/taehyeon/PRiMeFlow
export PATH=/data/project/taehyeon/Anaconda3/envs/perturbench/bin:$PATH
JOBS=(cpa biolord sams latent linear decoder)
N_GPU=4
run() {  # $1=model $2=gpu
  CUDA_VISIBLE_DEVICES=$2 train experiment=neurips2025/norman19/$1_best_params_norman19 \
    data/evaluation=final_test seed=245 paths.data_dir=$PWD/data \
    "hydra.run.dir=logs/train/runs/\${now:%Y-%m-%d_%H-%M-%S}_$1" \
    > logs/baselines/$1.out 2>&1
  local code=$?   # $(date) 보다 먼저 저장해야 train 의 exit code 가 남음
  echo "$(date '+%F %T') $1 (gpu $2) exit=$code" >> logs/baselines/status.txt
}
worker() {  # GPU 하나가 큐에서 자기 몫(i, i+N_GPU, ...)을 순서대로 처리
  for ((i=$1; i<${#JOBS[@]}; i+=N_GPU)); do
    echo "$(date '+%F %T') ${JOBS[$i]} (gpu $1) start" >> logs/baselines/status.txt
    run ${JOBS[$i]} $1
  done
}
for g in $(seq 0 $((N_GPU-1))); do worker $g & sleep 2; done
wait
echo "$(date '+%F %T') ALL DONE" >> logs/baselines/status.txt
