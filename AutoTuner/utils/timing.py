import time
from typing import Any, Callable, Dict, Optional, Tuple

import torch


class TimerContext:
    def __init__(self, name: str = "", cuda_sync: bool = False):
        self.name: str = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.cuda_sync: bool = cuda_sync

    def __enter__(self) -> "TimerContext":
        if self.cuda_sync:
            torch.cuda.synchronize()
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.cuda_sync:
            torch.cuda.synchronize()
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        self.result = str(self.elapsed_time) + " seconds"
        print(f"{self.name} took {self.elapsed_time:.6f} seconds")
    
    def get_result(self) -> str:
        return self.result


class Timer:
    @staticmethod
    def time_function(
        func: Callable, *args: Any, cuda_sync: bool = False, **kwargs: Any
    ) -> Tuple[Any, Dict[str, float]]:
        with TimerContext(name=func.__name__, cuda_sync=cuda_sync) as timer:
            result = func(*args, **kwargs)
        return result, {func.__name__: timer.end_time - timer.start_time}

    @staticmethod
    def time_decorator(
        preffix: Optional[str] = None, cuda_sync: bool = False
    ) -> Callable:
        def decorator(func: Callable) -> Callable:
            def wrapper(*args: Any, **kwargs: Any) -> Tuple[Any, Dict[str, float]]:
                with TimerContext(
                    name=f"{preffix} {func.__name__}", cuda_sync=cuda_sync
                ) as timer:
                    result = func(*args, **kwargs)
                return result, {
                    f"{preffix} {func.__name__}": timer.end_time - timer.start_time
                }

            return wrapper

        return decorator
