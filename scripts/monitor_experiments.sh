#!/usr/bin/env bash
#
# Monitor running experiments in real-time
# Usage: ./monitor_experiments.sh
#

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

while true; do
  clear
  echo "=========================================="
  echo "Experiment 1.1 Monitor - $(date)"
  echo "=========================================="
  echo ""

  # Running experiments
  running=$(screen -ls 2>/dev/null | grep -c "exp1_1_" || echo "0")
  echo -e "${GREEN}Running experiments: ${running}${NC}"
  echo ""

  # Completed experiments
  completed=$(ls -d runs/exp1_1_* 2>/dev/null | wc -l)
  total=45  # MNIST: 5 attacks × 9 defenses
  echo -e "${YELLOW}Completed: ${completed}/${total} ($((completed * 100 / total))%)${NC}"
  echo ""

  # GPU usage
  echo "=== GPU Usage ==="
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{printf "GPU %s: %3d%% utilization, %5d/%5d MB\n", $1, $2, $3, $4}'
  echo ""

  # CPU usage
  echo "=== CPU Usage ==="
  cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
  echo "CPU: ${cpu_usage}% (128 cores available)"
  echo ""

  # Memory usage
  echo "=== Memory Usage ==="
  free -h | grep Mem | awk '{printf "Memory: %s / %s used (%s)\n", $3, $2, $3/$2*100"%"}'
  echo ""

  # Check for stuck experiments (still at round 0 after 5 minutes)
  echo "=== Checking for stuck experiments ==="
  stuck_count=0
  for log in logs/exp1_1_*/exp1_1_*.log 2>/dev/null; do
    if [ -f "$log" ]; then
      # Get last round from log
      last_round=$(grep -oP 'Round \K\d+' "$log" 2>/dev/null | tail -1)
      file_age=$(($(date +%s) - $(stat -c %Y "$log" 2>/dev/null || echo $(date +%s))))

      # If still at round 0 after 5 minutes, it's stuck
      if [ -z "$last_round" ] || ([ "$last_round" = "0" ] && [ $file_age -gt 300 ]); then
        stuck_count=$((stuck_count + 1))
        exp_name=$(basename "$log" .log)
        echo -e "${RED}  STUCK: ${exp_name} (${file_age}s, round ${last_round:-0})${NC}"
      fi
    fi
  done
  if [ $stuck_count -eq 0 ]; then
    echo -e "${GREEN}  No stuck experiments detected ✓${NC}"
  else
    echo -e "${RED}  Found ${stuck_count} stuck experiments!${NC}"
  fi
  echo ""

  # Recent activity
  echo "=== Recent Log Activity ==="
  for log in logs/exp1_1_*/exp1_1_*.log 2>/dev/null; do
    if [ -f "$log" ]; then
      last_modified=$(($(date +%s) - $(stat -c %Y "$log")))
      if [ $last_modified -lt 60 ]; then
        exp_name=$(basename "$log" .log)
        last_line=$(tail -1 "$log" 2>/dev/null | cut -c1-60)
        echo "  ${exp_name}: ${last_line}..."
      fi
    fi
  done | head -5
  echo ""

  echo "Press Ctrl+C to exit. Refreshing in 10s..."
  sleep 10
done
