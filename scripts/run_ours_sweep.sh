#!/usr/bin/env bash
#
# Launch a sweep of "ours" experiments under different attack settings.
# Each experiment runs inside its own screen session with logs captured to disk.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="${PROJECT_ROOT}/logs/ours_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_ROOT}"

# Activate Conda environment helper
if [[ -z "${CONDA_EXE:-}" ]]; then
  echo "[ERROR] Conda is not available in the current shell." >&2
  exit 1
fi
CONDA_BASE="$(dirname "$(dirname "${CONDA_EXE}")")"
CONDA_SETUP="source \"${CONDA_BASE}/etc/profile.d/conda.sh\" && conda activate abrfl"

# Format: client_attack:server_attack:mal_client_ratio:mal_server_ratio
declare -a JOB_MATRIX=(
  "signflip:none:0.2:0.0"
  "adaptive:none:0.2:0.0"
  "alie:none:0.2:0.0"
  "minsum:none:0.2:0.0"
  "minmax_binary:none:0.2:0.0"
  "agr_tailored:none:0.2:0.0"
  "label_flip:none:0.2:0.0"
  "signflip:noise:0.2:0.3"
  "adaptive:delayed:0.2:0.3"
)

GPUS=(0 1)
gpu_idx=0

echo "Logs will be stored in ${LOG_ROOT}"

for job in "${JOB_MATRIX[@]}"; do
  IFS=":" read -r client_attack server_attack mc_ratio ms_ratio <<<"${job}"

  session="ours_${client_attack}_${server_attack}"
  log_file="${LOG_ROOT}/${session}.log"
  gpu_id="${GPUS[$((gpu_idx % ${#GPUS[@]}))]}"
  gpu_idx=$((gpu_idx + 1))

  cmd=$(cat <<EOF
cd "${PROJECT_ROOT}" && \
${CONDA_SETUP} && \
CUDA_VISIBLE_DEVICES=${gpu_id} python scripts/run_example.py \
  --defense ours \
  --dataset mnist \
  --model lenet \
  --num-clients 100 \
  --num-servers 10 \
  --rounds 100 \
  --local-epochs 1 \
  --batch-size 64 \
  --lr 0.001 \
  --optimizer adam \
  --alpha 0.5 \
  --client-attack ${client_attack} \
  --client-attack-params '{}' \
  --server-attack ${server_attack} \
  --server-attack-params '{}' \
  --malicious-client-ratio ${mc_ratio} \
  --malicious-server-ratio ${ms_ratio} \
  --result-root "./runs" \
  --log \
  --run-name "mnist_lenet_${session}"
EOF
)

  echo "Launching ${session} on GPU ${gpu_id}. Log: ${log_file}"
  screen -S "${session}" -dm bash -lc "${cmd} 2>&1 | tee \"${log_file}\""
done

echo "All jobs dispatched. Use 'screen -ls' to monitor sessions."
