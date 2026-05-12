from .timer import timer
import logging


def setup_logging(level=logging.INFO):
    handler = logging.StreamHandler()
    handler.setFormatter(CustomFormatter())
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(level)


class CustomFormatter(logging.Formatter):
    grey = "\033[38;20m"
    cyan = "\033[36m"
    green = "\033[32m"
    yellow = "\033[33m"
    red = "\033[31m"
    reset = "\033[0m"

    fmt = "%(asctime)s  %(levelname)-8s %(message)s"
    datefmt = "%H:%M:%S"

    FORMATS = {
        logging.DEBUG: grey + fmt + reset,
        logging.INFO: cyan + fmt + reset,
        logging.WARNING: yellow + fmt + reset,
        logging.ERROR: red + fmt + reset,
        logging.CRITICAL: red + fmt + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.DEBUG])
        formatter = logging.Formatter(log_fmt, datefmt=self.datefmt)
        return formatter.format(record)
