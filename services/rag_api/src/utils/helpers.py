import functools
import time
import logging
import inspect 

def measure_performance(logger=None, unit="s"):
    time_factors = {"s": 1, "ms": 1000, "us": 1_000_000}
    factor = time_factors.get(unit, 1)

    if logger is None:
        logger = logging.getLogger("PerformanceLogger")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s",
                "%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

    def decorator(func):
        func_name = func.__qualname__

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            elapsed = (end_time - start_time) * factor
            logger.info(f"[PERF] {func_name} took {elapsed:.2f} {unit}")
            return result

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            
            start_time = time.perf_counter()
            result = await func(*args, **kwargs)
            end_time = time.perf_counter()
            elapsed = (end_time - start_time) * factor
            logger.info(f"[PERF] {func_name} took {elapsed:.2f} {unit}")
            return result
        
        @functools.wraps(func)
        def sync_generator_wrapper(*args, **kwargs):
            
            start_time = time.perf_counter()
            try:
                for item in func(*args, **kwargs):
                    yield item
            finally:
                end_time = time.perf_counter()
                elapsed = (end_time - start_time) * factor
                logger.info(f"[PERF-STREAM] {func_name} took {elapsed:.2f} {unit}")

        @functools.wraps(func)
        async def async_generator_wrapper(*args, **kwargs):
            
            start_time = time.perf_counter()
            try:
                async for item in func(*args, **kwargs):
                    yield item
            finally:
                end_time = time.perf_counter()
                elapsed = (end_time - start_time) * factor
                logger.info(f"[PERF-STREAM] {func_name} took {elapsed:.2f} {unit}")

        if inspect.isasyncgenfunction(func):
            return async_generator_wrapper
        elif inspect.iscoroutinefunction(func):
            return async_wrapper
        elif inspect.isgeneratorfunction(func):
            return sync_generator_wrapper
        else:
            return sync_wrapper

    return decorator