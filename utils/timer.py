import time
import logging
from collections import defaultdict
from functools import wraps

logger = logging.getLogger(__name__)

_timing_registry: dict[str, list[float]] = defaultdict(list)


def record_timing(name: str, elapsed: float) -> None:
    _timing_registry[name].append(elapsed)


def get_timings() -> dict[str, float]:
    return {k: sum(v) for k, v in _timing_registry.items()}


def get_timing_details() -> dict[str, list[float]]:
    return dict(_timing_registry)


def clear_timings() -> None:
    _timing_registry.clear()


def timer(func=None, *, name=None):
    def decorator(f):
        label = name or f.__qualname__

        @wraps(f)
        def wrapper(*args, **kwargs):
            _logger = logging.getLogger(f.__module__)
            start = time.perf_counter()
            try:
                result = f(*args, **kwargs)
                elapsed = time.perf_counter() - start
                record_timing(label, elapsed)
                _logger.info("Finished %s in %.2fs", label, elapsed)
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                _logger.error("%s raised %s after %.2fs", label, e, elapsed)
                raise

        return wrapper

    if func is None:
        return decorator
    return decorator(func)
