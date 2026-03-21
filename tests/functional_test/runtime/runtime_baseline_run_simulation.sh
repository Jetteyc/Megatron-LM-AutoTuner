#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "megatron-lm-autotuner" ]]; then
    echo "Warning: current conda env is '${CONDA_DEFAULT_ENV:-<unset>}'"
    echo "Recommended: conda activate megatron-lm-autotuner"
fi

if [ -f .secrets/env.sh ]; then
    source .secrets/env.sh
fi

CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/runtime_baseline_config.json}"
if [[ $# -ge 1 ]] && [[ "$1" != -* ]]; then
    CONFIG_FILE="$1"
    shift
fi

echo "Running runtime baseline with simulator metrics from config"
echo "  config: $CONFIG_FILE"
if [[ -n "${MODEL_FILTER:-}" ]]; then
    echo "  model_filter: ${MODEL_FILTER}"
fi
echo "  python: $PYTHON_BIN"

bash "${SCRIPT_DIR}/runtime_baseline_run.sh" "$CONFIG_FILE" "$@"

LATEST_SUMMARY=$(
    "$PYTHON_BIN" - "$CONFIG_FILE" "${MODEL_FILTER:-}" <<'PY'
import contextlib
import importlib.util
import io
import sys
from pathlib import Path

config_path = Path(sys.argv[1]).expanduser().resolve()
model_filter = sys.argv[2]
repo_root = Path.cwd()
module_path = repo_root / "tests" / "functional_test" / "runtime" / "runtime_baseline_run_from_config.py"
spec = importlib.util.spec_from_file_location(
    "runtime_baseline_run_from_config", module_path
)
assert spec is not None
assert spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

with contextlib.redirect_stdout(io.StringIO()):
    config = module.load_config(config_path)
    selected = module.select_model_entries(config, model_filter)
    distributed = {
        "master_addr": "localhost",
        "master_port": 6010,
        "num_nodes": 1,
        "node_rank": 0,
    }
    output_dirs = []
    for merged_cfg in selected:
        spec_dict = module.build_run_spec(repo_root, merged_cfg, distributed)
        output_dirs.append(spec_dict["output_dir"])

latest = None
for output_dir in output_dirs:
    for candidate in output_dir.glob("*/**/runtime_baseline/runtime_summary.json"):
        if latest is None or candidate.stat().st_mtime > latest.stat().st_mtime:
            latest = candidate

print("" if latest is None else str(latest))
PY
)

if [[ -z "${LATEST_SUMMARY:-}" ]]; then
    echo "No runtime_summary.json found for selected config entries"
    exit 1
fi

echo
echo "Latest summary: $LATEST_SUMMARY"
"$PYTHON_BIN" - "$LATEST_SUMMARY" <<'PY'
import json
import sys

summary_path = sys.argv[1]
with open(summary_path, "r") as fp:
    data = json.load(fp)

for item in data:
    print(
        "test_case_idx={idx} measured_time_s={measured:.4f} simulated_time_s={sim:.4f} "
        "simulated_pp_s={sim_pp:.4f} simulated_dp_s={sim_dp:.4f} "
        "measured_toks={measured_tps:.2f} simulated_toks={sim_tps:.2f}".format(
            idx=item["test_case_idx"],
            measured=item["time_s"],
            sim=item["simulated_time_s"],
            sim_pp=item["simulated_pp_compute_time_s"],
            sim_dp=item["simulated_dp_allreduce_time_s"],
            measured_tps=item["throughput_tokens_s"],
            sim_tps=item["simulated_throughput_tokens_s"],
        )
    )
PY
