# Runtime Functional Test

This directory provides a minimal runtime check for the baseline GPTModel runtime.

## Usage

1. (Optional) source runtime environment from `.secrets/env.sh`.
2. Activate the recommended conda env and repo-local Python path:

```bash
conda activate megatron-lm-autotuner
export PYTHONPATH=verl:Megatron-LM
```

3. Edit `tests/functional_test/runtime/runtime_baseline_config.json`:
   - JSON root must be an array
   - each entry must be `{ "model_name": "...", "configs": { ... } }`
   - keep one entry by default; add multiple entries only when you intentionally want a batch run
   - `configs` can include `case` / `parallel` / `runtime` / `paths` / `env`
   - `parallel.vpp_size` is optional; omit it or set it to `null` / `"None"` to keep `virtual_pipeline_model_parallel_size=None`
   - `runtime.use_fused_kernels` is optional; set it in JSON instead of using a separate shell config
   - example with full fields: `Qwen/Qwen2.5-0.5B` in `runtime_baseline_config.json`
   - optional sample file: `tests/functional_test/runtime/runtime_baseline_config_sample.json`
   - distributed launch args are from shell env vars: `MASTER_ADDR`, `MASTER_PORT`, `NUM_NODES`, `NODE_RANK`
4. Run:

```bash
bash tests/functional_test/runtime/runtime_baseline_run.sh
```

Custom config path:

```bash
bash tests/functional_test/runtime/runtime_baseline_run.sh /path/to/config.json
```

Dry run (print command only):

```bash
bash tests/functional_test/runtime/runtime_baseline_run.sh \
  tests/functional_test/runtime/runtime_baseline_config.json \
  --dry-run
```

Filter models by substring:

```bash
MODEL_FILTER=Qwen3-8B bash tests/functional_test/runtime/runtime_baseline_run.sh
```

`runtime_baseline_run.sh` is the canonical entrypoint. It calls `runtime_baseline_run_from_config.py`, which generates test case JSON files and launches `torchrun -m AutoTuner.runtime.baseline.main` per selected model entry.

`runtime_baseline_run_qwen_longctx.sh` remains only as a compatibility alias to `runtime_baseline_run.sh`.

## Simulation Example

For a single, copy-pasteable baseline run that also prints simulated PP/DP time from the output summary:

```bash
bash tests/functional_test/runtime/runtime_baseline_run_simulation.sh
```

Use the same config file for simulation runs. The simulation script reuses `runtime_baseline_run.sh` and then prints the latest `runtime_summary.json` for the selected config entries.

```bash
bash tests/functional_test/runtime/runtime_baseline_run_simulation.sh
```

If you need fused-kernel behavior, set `runtime.use_fused_kernels` in `runtime_baseline_config.json`.

Set DP all-reduce simulator values in `AutoTuner/testbench/profile/configs/local/ddp_simulate_config.json`:

```json
{
  "dp_allreduce_bandwidth_gbps": 50,
  "dp_allreduce_latency_us": 30
}
```
