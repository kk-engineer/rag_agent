from .timer import timer, get_timings, clear_timings
import logging
from rich.console import Console
from rich.logging import RichHandler


def setup_logging(level=logging.INFO, console=None):
    if console is None:
        console = Console(stderr=True)
    handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=False,
    )
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level)


def suppress_noisy_loggers():
    for name in [
        "httpx",
        "httpcore",
        "urllib3",
        "openai",
        "datasets",
        "ragas",
        "langchain",
        "langchain_community",
        "langchain_core",
        "langchain_experimental",
        "PIL",
    ]:
        logging.getLogger(name).setLevel(logging.WARNING)
    # Chromadb telemetry/internal ERRORs are never actionable
    for name in ["chromadb", "chromadb.telemetry", "chromadb.rate_limiting"]:
        logging.getLogger(name).setLevel(logging.CRITICAL)
