#!/usr/bin/env bash
#
# Experiment 1.1 – Byzantine robustness without server-side attacks.
# This sweep launches (dataset × attack × defense) jobs covering:
#   • Datasets: MNIST+LeNet, CIFAR-10+ResNet-18
#   • FL setup: 10 servers, 100 clients, Dirichlet α=1.0 (non-IID), 1 local epoch, 100 rounds
#   • Client attacks (m_c = 0.2): noise, signflip, alie, adaptive, ipm
#   • Defenses: Local, FedAvg, Krum, Median, FLTrust, DnC, ClippedClustering, SignGuard, Ours
# Each job runs in its own screen session, logs to ./logs/exp1_1_*, and uses the `abrfl` conda env.
# Workloads are distributed round-robin across two GPUs (IDs 0 and 1).
# 为了减轻单次负载，可指定参数：./run_exp1_1_sweep.sh <dataset|all> <attack|all>

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="${PROJECT_ROOT}/logs/exp1_1_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_ROOT}"

if [[ -z "${CONDA_EXE:-}" ]]; then
  echo "[ERROR] Conda is not available in the current shell." >&2
  exit 1
fi
CONDA_BASE="$(dirname "$(dirname "${CONDA_EXE}")")"
CONDA_SETUP="source \"${CONDA_BASE}/etc/profile.d/conda.sh\" && conda activate abrfl"

# Optional CLI filters: ./run_exp1_1_sweep.sh [dataset] [attack]
TARGET_DATASET="${1:-all}"
TARGET_ATTACK="${2:-all}"

# dataset:model:learning_rate:optimizer:local_epochs
declare -a DATASETS=(
  "mnist:lenet:0.001:adam:1"
  "cifar10:resnet18:0.001:adam:1"
)

declare -a ATTACKS=("noise" "signflip" "alie" "adaptive" "ipm")
declare -a DEFENSES=("fedavg" "krum" "median" "fltrust" "dnc" "clipped" "signguard" "ours")

GPUS=(0 1)
gpu_idx=0

# ========== Resource Management Configuration ==========
# For 128 CPU cores, 2 GPUs, targeting 30-40 concurrent experiments
MAX_CONCURRENT=50       # Maximum concurrent experiments
BATCH_SIZE=6             # Launch experiments in batches to avoid scheduler overload
BATCH_DELAY=60           # Seconds to wait between batches (let processes stabilize)
STARTUP_DELAY=10          # Seconds between individual experiment launches

echo "Logs will be stored under ${LOG_ROOT}"
echo "Resource config: MAX_CONCURRENT=${MAX_CONCURRENT}, BATCH_SIZE=${BATCH_SIZE}"
DATA_LOADER_WORKERS=2
echo "Using num_workers=${DATA_LOADER_WORKERS} per experiment (optimized for high concurrency)"
echo ""

defense_extra_args() {
  local defense="$1"
  case "${defense}" in
    krum)
      echo "--krum-byzantine-ratio 0.2"
      ;;
    fltrust)
      echo "--fltrust-root-percent 0.01"
      ;;
    dnc)
      echo "--dnc-num-clusters 2"
      ;;
    clipped)
      echo "--clipped-num-clusters 2 --clipped-threshold auto"
      ;;
    *)
      echo ""
      ;;
  esac
}

jobs_launched=0
batch_count=0

# Helper function to get current running job count
get_running_count() {
  local count
  count=$(screen -ls 2>&1 | grep -c "exp1_1_") || count=0
  echo "${count}"
}

echo "Starting experiment sweep..."
echo ""

for dataset_cfg in "${DATASETS[@]}"; do
  IFS=":" read -r dataset_name model_name lr optimizer local_epochs <<<"${dataset_cfg}"

  if [[ "${TARGET_DATASET}" != "all" && "${dataset_name}" != "${TARGET_DATASET}" ]]; then
    continue
  fi

  for attack in "${ATTACKS[@]}"; do
    if [[ "${TARGET_ATTACK}" != "all" && "${attack}" != "${TARGET_ATTACK}" ]]; then
      continue
    fi

    for defense in "${DEFENSES[@]}"; do
      # ========== Concurrency Control ==========
      # Wait if we've reached the maximum concurrent experiments
      while [ "$(get_running_count)" -ge "${MAX_CONCURRENT}" ]; do
        running="$(get_running_count)"
        echo "[$(date +%H:%M:%S)] At capacity (${running}/${MAX_CONCURRENT}), waiting 30s..."
        sleep 30
      done

      # ========== Batch Delay ==========
      # After completing a batch, wait for processes to stabilize
      if [ $((batch_count % BATCH_SIZE)) -eq 0 ] && [ "${batch_count}" -gt 0 ]; then
        running="$(get_running_count)"
        echo ""
        echo "=========================================="
        echo "Batch $((batch_count / BATCH_SIZE)) launched"
        echo "Currently running: ${running} experiments"
        echo "Waiting ${BATCH_DELAY}s for batch to stabilize..."
        echo "=========================================="
        echo ""
        sleep ${BATCH_DELAY}
      fi

      gpu_id="${GPUS[$((gpu_idx % ${#GPUS[@]}))]}"
      gpu_idx=$((gpu_idx + 1))

      session="exp1_1_${dataset_name}_${attack}_${defense}"
      log_file="${LOG_ROOT}/${session}.log"
      extra_args="$(defense_extra_args "${defense}")"

      cmd=$(cat <<EOF
cd "${PROJECT_ROOT}" && \
${CONDA_SETUP} && \
CUDA_VISIBLE_DEVICES=${gpu_id} python scripts/run_example.py \
  --defense ${defense} \
  --dataset ${dataset_name} \
  --model ${model_name} \
  --num-clients 100 \
  --num-servers 10 \
  --rounds 100 \
  --local-epochs ${local_epochs} \
  --batch-size 64 \
  --alpha 1.0 \
  --lr ${lr} \
  --optimizer ${optimizer} \
  --num-workers ${DATA_LOADER_WORKERS} \
  --client-attack ${attack} \
  --client-attack-params '{}' \
  --server-attack none \
  --server-attack-params '{}' \
  --malicious-client-ratio 0.2 \
  --malicious-server-ratio 0.0 \
  ${extra_args} \
  --result-root "./runs" \
  --run-name "exp1_1_${dataset_name}_${attack}_${defense}"
EOF
)

      running="$(get_running_count)"
      echo "[$(date +%H:%M:%S)] [${jobs_launched}] Launching ${session} on GPU ${gpu_id} (running: ${running}/${MAX_CONCURRENT})"
      screen -S "${session}" -dm bash -lc "${cmd} 2>&1 | tee \"${log_file}\""

      jobs_launched=$((jobs_launched + 1))
      batch_count=$((batch_count + 1))

      # Small delay between individual launches to avoid overwhelming the scheduler
      sleep ${STARTUP_DELAY}
    done
  done
done

if [[ "${jobs_launched}" -eq 0 ]]; then
  echo "No jobs matched dataset='${TARGET_DATASET}' attack='${TARGET_ATTACK}'."
else
  echo ""
  echo "=========================================="
  echo "✅ Dispatched ${jobs_launched} Experiment 1.1 jobs"
  echo "=========================================="
  echo ""
  echo "Monitoring commands:"
  echo "  • List sessions:     screen -ls | grep exp1_1"
  echo "  • Attach to session: screen -r exp1_1_<name>"
  echo "  • Monitor GPU:       watch -n 2 nvidia-smi"
  echo "  • Check progress:    ls -lh runs/exp1_1_* | wc -l"
  echo "  • View logs:         tail -f ${LOG_ROOT}/*.log"
  echo ""
  echo "Resource allocation:"
  echo "  • Max concurrent: ${MAX_CONCURRENT} experiments"
  echo "  • CPU workers: ${MAX_CONCURRENT} × 2 = $((MAX_CONCURRENT * 2)) cores (~$((MAX_CONCURRENT * 2 * 100 / 128))% of 128 cores)"
  echo "  • GPUs: 2 (round-robin assignment)"
  echo ""
fi
