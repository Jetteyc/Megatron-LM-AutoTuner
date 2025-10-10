# Convinient testbench for observation

Through the testbench of AutoTuner, we can make straight-forward observation of the operators performance.

There are two ways of testbench launching methods: get_data launch and nsys profile launch. The first way means we just use our timers and memory sensors to detect an operators performance and return the data for future use. The latter means we use nsys profile

The design of testbench follows OOP rules.

The structures of testbench:

```sh
AutoTuner
├── testbench
│   ├── functional      # test base capability like CPU-GPU bandwidth
│   ├── __init__.py
│   ├── ops             # override of Megatron operators, mainly for nvtx insertion
│   ├── ops_test        # wrapper of operator testing
│   └── profile         # profile main entrance
│       ├── cases                       # test cases (models and externel configs)
│       ├── configs                     # configs to override huggingface model config and transformer config
│       ├── __init__.py
│       ├── launcher                    # different ways to launch model runner
│       ├── main.py                     # main entrence of get_data launch way
│       ├── nsys_main.py                # main entrence of nsys profile launch way
│       └── op_mapping.py
└── utils               # assistence function, to avoid repetition
```

