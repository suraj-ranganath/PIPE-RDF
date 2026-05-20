#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-pipe-rdf-arr}"
MAX_CLIENT_JOBS="${MAX_CLIENT_JOBS:-4}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
MAX_GPU_MEMORY_USED_MB="${MAX_GPU_MEMORY_USED_MB:-1024}"
MAX_GPU_UTILIZATION="${MAX_GPU_UTILIZATION:-20}"
RUN_SCHEMA_PROFILE="${RUN_SCHEMA_PROFILE:-1}"
RUN_FULL_RUNS="${RUN_FULL_RUNS:-1}"
RUN_AUDIT_PACKET="${RUN_AUDIT_PACKET:-1}"
RUN_CROSS_MODEL_PROBES="${RUN_CROSS_MODEL_PROBES:-1}"
RUN_UTILITY_EVAL="${RUN_UTILITY_EVAL:-1}"
KEEP_VLLM_SERVERS="${KEEP_VLLM_SERVERS:-0}"

mapfile -t AUTO_GPUS < <(
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F, -v max_mem="$MAX_GPU_MEMORY_USED_MB" -v max_util="$MAX_GPU_UTILIZATION" '
        {
          gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3);
          if ($2 <= max_mem && $3 <= max_util) print $2 " " $3 " " $1;
        }' \
    | sort -n \
    | awk '{print $3}'
)
if [ "${#AUTO_GPUS[@]}" -lt 2 ]; then
  echo "Need at least two idle GPUs for parallel full runs; found ${#AUTO_GPUS[@]}." >&2
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
  exit 1
fi
GPU_SCHEMA_C="${GPU_SCHEMA_C:-${AUTO_GPUS[0]:-0}}"
GPU_SPB_FULL="${GPU_SPB_FULL:-${AUTO_GPUS[1]:-1}}"
GPU_PROBE_2B="${GPU_PROBE_2B:-$GPU_SCHEMA_C}"
GPU_PROBE_9B="${GPU_PROBE_9B:-$GPU_SPB_FULL}"

TP_SCHEMA_C="${TP_SCHEMA_C:-1}"
TP_SPB_FULL="${TP_SPB_FULL:-1}"
TP_PROBE_2B="${TP_PROBE_2B:-1}"
TP_PROBE_9B="${TP_PROBE_9B:-1}"

MODEL_2B="${MODEL_2B:-Qwen/Qwen3.5-2B}"
MODEL_4B="${MODEL_4B:-Qwen/Qwen3.5-4B}"
MODEL_9B="${MODEL_9B:-Qwen/Qwen3.5-9B}"

PORT_SCHEMA_C="${PORT_SCHEMA_C:-8001}"
PORT_SPB_FULL="${PORT_SPB_FULL:-8002}"
PORT_PROBE_2B="${PORT_PROBE_2B:-8011}"
PORT_PROBE_9B="${PORT_PROBE_9B:-8012}"

if [ -f "$HOME/miniforge3/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$HOME/miniforge3/bin/activate"
fi
conda activate "$ENV_NAME"

mkdir -p artifacts/arr_jobs/logs artifacts/arr_jobs/configs
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv \
  | tee "artifacts/arr_jobs/logs/gpu_status_start.csv"

if [ -x scripts/ds_serv6_graphdb.sh ]; then
  scripts/ds_serv6_graphdb.sh status
fi

VLLM_PIDS=()

cleanup() {
  if [ "$KEEP_VLLM_SERVERS" = "1" ]; then
    return
  fi
  if [ "${#VLLM_PIDS[@]}" -gt 0 ]; then
    echo "Stopping vLLM servers started by this script"
    kill "${VLLM_PIDS[@]}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

start_vllm() {
  local gpu="$1"
  local port="$2"
  local model="$3"
  local log_name="$4"
  local tensor_parallel_size="$5"
  echo "Starting vLLM: gpu=$gpu port=$port model=$model tensor_parallel_size=$tensor_parallel_size"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m vllm.entrypoints.openai.api_server \
    --host "$VLLM_HOST" \
    --port "$port" \
    --model "$model" \
    --served-model-name "$model" \
    --tensor-parallel-size "$tensor_parallel_size" \
    --max-model-len 16384 \
    --language-model-only \
    --reasoning-parser qwen3 \
    --gpu-memory-utilization 0.85 \
    $VLLM_EXTRA_ARGS \
    > "artifacts/arr_jobs/logs/${log_name}.vllm.log" 2>&1 &
  local pid=$!
  VLLM_PIDS+=("$pid")
  echo "$pid" > "artifacts/arr_jobs/logs/${log_name}.vllm.pid"
}

wait_for_server() {
  local port="$1"
  local model="$2"
  for _ in $(seq 1 180); do
    if curl -fsS "http://${VLLM_HOST}:${port}/v1/models" >/dev/null 2>&1; then
      python scripts/vllm_smoke.py --base-url "http://${VLLM_HOST}:${port}/v1" --model "$model"
      return 0
    fi
    sleep 5
  done
  echo "Timed out waiting for vLLM on port $port" >&2
  return 1
}

render_config() {
  local base="$1"
  local model="$2"
  local port="$3"
  local out="$4"
  python - "$base" "$model" "$port" "$out" <<'PY'
import sys
import yaml
base, model, port, out = sys.argv[1:5]
with open(base, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
cfg["openai_base_url"] = f"http://127.0.0.1:{port}/v1"
cfg.setdefault("models", {})["chat"] = model
with open(out, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
}

run_pipeline_job() {
  local cfg="$1"
  local name="$2"
  echo "Launching pipeline: $name"
  python scripts/preflight_check.py --config "$cfg"
  python scripts/run_pipeline_ollama.py --config "$cfg" --run-name "$name"
}

latest_run_dir() {
  local run_name="$1"
  find artifacts/runs -maxdepth 1 -type d -name "*_${run_name}" -print \
    | sort \
    | tail -n 1
}

if [ "$RUN_SCHEMA_PROFILE" = "1" ]; then
  python scripts/profile_schema.py --config configs/arr_schema_c_200.yaml \
    --output-dir artifacts/schema_profiles/arr_schema_c_200
  python scripts/profile_schema.py --config configs/arr_spb_full_200.yaml \
    --output-dir artifacts/schema_profiles/arr_spb_full_200
fi

SCHEMA_C_RUN_DIR="${SCHEMA_C_RUN_DIR:-}"
SPB_FULL_RUN_DIR="${SPB_FULL_RUN_DIR:-}"

if [ "$RUN_FULL_RUNS" = "1" ]; then
  start_vllm "$GPU_SCHEMA_C" "$PORT_SCHEMA_C" "$MODEL_4B" "schema_c_4b" "$TP_SCHEMA_C"
  start_vllm "$GPU_SPB_FULL" "$PORT_SPB_FULL" "$MODEL_4B" "spb_full_4b" "$TP_SPB_FULL"
  wait_for_server "$PORT_SCHEMA_C" "$MODEL_4B"
  wait_for_server "$PORT_SPB_FULL" "$MODEL_4B"

  run_pipeline_job configs/arr_schema_c_200.yaml arr_schema_c_qwen35_4b_200 &
  run_pipeline_job configs/arr_spb_full_200.yaml arr_spb_full_qwen35_4b_200 &
  wait

  SCHEMA_C_RUN_DIR="$(latest_run_dir arr_schema_c_qwen35_4b_200)"
  SPB_FULL_RUN_DIR="$(latest_run_dir arr_spb_full_qwen35_4b_200)"
fi

if [ -z "$SCHEMA_C_RUN_DIR" ]; then
  SCHEMA_C_RUN_DIR="$(latest_run_dir arr_schema_c_qwen35_4b_200)"
fi
if [ -z "$SPB_FULL_RUN_DIR" ]; then
  SPB_FULL_RUN_DIR="$(latest_run_dir arr_spb_full_qwen35_4b_200)"
fi
if { [ "$RUN_AUDIT_PACKET" = "1" ] || [ "$RUN_UTILITY_EVAL" = "1" ]; } \
  && { [ -z "$SCHEMA_C_RUN_DIR" ] || [ -z "$SPB_FULL_RUN_DIR" ]; }; then
  echo "Could not locate full run artifact directories." >&2
  exit 1
fi
echo "Full run dirs: schema_c=$SCHEMA_C_RUN_DIR spb_full=$SPB_FULL_RUN_DIR"

if [ "$RUN_AUDIT_PACKET" = "1" ]; then
  python scripts/sample_semantic_audit.py \
    --input "schema_c=${SCHEMA_C_RUN_DIR}/data/benchmark_phase3_balanced.jsonl" \
    --input "spb_full=${SPB_FULL_RUN_DIR}/data/benchmark_phase3_balanced.jsonl" \
    --per-category 12 \
    --output-dir artifacts/audits/arr_semantic_audit
fi

if [ "$RUN_CROSS_MODEL_PROBES" = "1" ]; then
  start_vllm "$GPU_PROBE_2B" "$PORT_PROBE_2B" "$MODEL_2B" "probe_2b" "$TP_PROBE_2B"
  start_vllm "$GPU_PROBE_9B" "$PORT_PROBE_9B" "$MODEL_9B" "probe_9b" "$TP_PROBE_9B"
  wait_for_server "$PORT_PROBE_2B" "$MODEL_2B"
  wait_for_server "$PORT_PROBE_9B" "$MODEL_9B"

  for schema_cfg in configs/arr_cross_model_schema_c_50.yaml configs/arr_cross_model_spb_full_50.yaml; do
    schema_name="$(basename "$schema_cfg" .yaml)"
    render_config "$schema_cfg" "$MODEL_2B" "$PORT_PROBE_2B" "artifacts/arr_jobs/configs/${schema_name}_2b.yaml"
    render_config "$schema_cfg" "$MODEL_4B" "$PORT_SCHEMA_C" "artifacts/arr_jobs/configs/${schema_name}_4b.yaml"
    render_config "$schema_cfg" "$MODEL_9B" "$PORT_PROBE_9B" "artifacts/arr_jobs/configs/${schema_name}_9b.yaml"
  done

  job_count=0
  for cfg in artifacts/arr_jobs/configs/*_2b.yaml artifacts/arr_jobs/configs/*_4b.yaml artifacts/arr_jobs/configs/*_9b.yaml; do
    name="$(basename "$cfg" .yaml)"
    run_pipeline_job "$cfg" "$name" &
    job_count=$((job_count + 1))
    if [ "$job_count" -ge "$MAX_CLIENT_JOBS" ]; then
      wait -n
      job_count=$((job_count - 1))
    fi
  done
  wait
fi

if [ "$RUN_UTILITY_EVAL" = "1" ]; then
  render_config configs/arr_schema_c_200.yaml "$MODEL_2B" "$PORT_PROBE_2B" \
    artifacts/arr_jobs/configs/utility_schema_c_2b.yaml
  render_config configs/arr_schema_c_200.yaml "$MODEL_4B" "$PORT_SCHEMA_C" \
    artifacts/arr_jobs/configs/utility_schema_c_4b.yaml
  render_config configs/arr_spb_full_200.yaml "$MODEL_2B" "$PORT_PROBE_2B" \
    artifacts/arr_jobs/configs/utility_spb_full_2b.yaml
  render_config configs/arr_spb_full_200.yaml "$MODEL_4B" "$PORT_SPB_FULL" \
    artifacts/arr_jobs/configs/utility_spb_full_4b.yaml

  python scripts/evaluate_downstream_utility.py \
    --config artifacts/arr_jobs/configs/utility_schema_c_2b.yaml \
    --input-jsonl "${SCHEMA_C_RUN_DIR}/data/benchmark_phase3_balanced.jsonl" \
    --schema-name schema_c_qwen35_2b \
    --output-dir artifacts/downstream_utility/schema_c_qwen35_2b
  python scripts/evaluate_downstream_utility.py \
    --config artifacts/arr_jobs/configs/utility_schema_c_4b.yaml \
    --input-jsonl "${SCHEMA_C_RUN_DIR}/data/benchmark_phase3_balanced.jsonl" \
    --schema-name schema_c_qwen35_4b \
    --output-dir artifacts/downstream_utility/schema_c_qwen35_4b
  python scripts/evaluate_downstream_utility.py \
    --config artifacts/arr_jobs/configs/utility_spb_full_2b.yaml \
    --input-jsonl "${SPB_FULL_RUN_DIR}/data/benchmark_phase3_balanced.jsonl" \
    --schema-name spb_full_qwen35_2b \
    --output-dir artifacts/downstream_utility/spb_full_qwen35_2b
  python scripts/evaluate_downstream_utility.py \
    --config artifacts/arr_jobs/configs/utility_spb_full_4b.yaml \
    --input-jsonl "${SPB_FULL_RUN_DIR}/data/benchmark_phase3_balanced.jsonl" \
    --schema-name spb_full_qwen35_4b \
    --output-dir artifacts/downstream_utility/spb_full_qwen35_4b
fi

echo "ARR generation runs complete. Inspect artifacts/runs and artifacts/arr_jobs/logs."
