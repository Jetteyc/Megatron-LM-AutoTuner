from dataclasses import dataclass


@dataclass
class ProfileConfig:
    profile_mode: bool = False
    warmup_iters: int = 2  # warmup_iters operator executions
