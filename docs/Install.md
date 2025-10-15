# Install

## Clone Project

Make sure include all submodules:

```sh
git clone git@github.com:ETOgaosion/Megatron-LM-AutoTuner.git --recurse-submodules
```

Or you have already cloned:

```sh
git submodule update --init --recursive
```

## Environment

- CUDA 12.8: To use RTX-5090
- torch 2.8.0+cu128
- flash_attn 2.8.1 (local), 2.7.4 (Image)
    - directly use wheels
- Megatron: keep the latest, as a submodule (dev)
- TransformerEngine: keep the latest, as a submodule (dev)
- verl: keep the latest, as a submodule (dev)

## Image

Use `verlai/verl:app-verl0.6-transformers4.56.1-sglang0.5.2-mcore0.13.0-te2.2`, too slow

## Local Development

- CUDA 12.8: Use [runfile installer](https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=runfile_local)
- Download torch, torchvision, torchaudio 2.8.0
- Install apex
- Install flash attention wheel
    - check CXXABI: `ABI_FLAG=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")`
    - download: https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abi${ABI_FLAG}-cp312-cp312-linux_x86_64.whl
    - install
- Install TransformerEngine in current repo: `pip install --no-deps -e .`
- Install Megatron in current repo
- Install verl in current repo