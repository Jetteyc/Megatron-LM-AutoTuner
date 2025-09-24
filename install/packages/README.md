# Install Packages from wheels

## Flash Attention

- Version: 2.8.3
- Download Link: [https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl](https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl)
- To check your cxx11abi, execute: `python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')"`