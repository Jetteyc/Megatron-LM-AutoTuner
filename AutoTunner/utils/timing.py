import time
import torch

class TimerContext:
    def __init__(self, name="", cuda_sync=False):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.cuda_sync = cuda_sync

    def __enter__(self):
        if self.cuda_sync:
            torch.cuda.synchronize()
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.cuda_sync:
            torch.cuda.synchronize()
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        self.result = self.elapsed_time
        print(f"{self.name} took {self.elapsed_time:.6f} seconds")


class Timer:
    @staticmethod
    def time_function(func, *args, cuda_sync=False, **kwargs):
        with TimerContext(name=func.__name__, cuda_sync=cuda_sync) as timer:
            result = func(*args, **kwargs)
        return result, {func.__name__: timer.end_time - timer.start_time}


    @staticmethod
    def time_decorator(preffix=None, cuda_sync=False):
        def decorator(func):
            def wrapper(*args, **kwargs):
                with TimerContext(name=f"{preffix} {func.__name__}", cuda_sync=cuda_sync) as timer:
                    result = func(*args, **kwargs)
                return result, {f"{preffix} {func.__name__}": timer.end_time - timer.start_time}
            return wrapper
        return decorator