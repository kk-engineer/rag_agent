import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        _logger = logging.getLogger(func.__module__)
        start = time.perf_counter()
        _logger.info("Started %s ...", func.__qualname__)
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            _logger.info("Finished %s in %.2fs", func.__qualname__, elapsed)
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start
            _logger.error("%s raised %s after %.2fs", func.__qualname__, e, elapsed)
            raise

    return wrapper
