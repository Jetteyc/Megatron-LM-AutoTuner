from dataclasses import dataclass

@dataclass
class ProfileConfig:
    profile_mode: bool = False
    warmup: int = 2 # warmup operator executions
    
    