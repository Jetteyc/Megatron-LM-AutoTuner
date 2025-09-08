# AutoTuner for Megatron + TransformerEngine

## Scenarios

This is a practical auto-tuner on Megatron targeted at post-training frameworks like [verl](https://github.com/volcengine/verl) project.

Our performance tuning target is MFU in MCore training process on both forward-only models and forward-backward-update models, which leads to high performance on training side in RLHF.

Speaking of tuning dimensions, currently include:

|   |   |   |   |   |   |
|---|---|---|---|---|---|
|Dense Layer|TP|CP|DP|PP|VPP|
|MoE Parallel Folding|ETP|EP|EDP|||
|Pipeline layout|||||
|Seqlen|max_token_len|||||
|Recompute|recompute_granularity|recompute_method|recompute_num_layers|recompute_modules||

Target shapes:

- thd (mainly)
- bshd

## Methods

This auto-tuner works based on Profiling-Planning method.

### Dense Models

To achieve high MFU on dense models:

- GEMM and TP communication shall be overlapped
- Attention calculation and ring P2P shall be overlapped
- The overlap shall not affect too much computation efficiency

And based on above, 


## MeE Models

## Challenges

