import logging
import sys

def setup(use_stdout=True, filename=None, log_level=logging.DEBUG):
    """setup some basic logging"""

    log = logging.getLogger('')
    log.setLevel(log_level)
    fmt = logging.Formatter("%(asctime)s [%(name)-15s] %(message)s", datefmt="%y-%m-%d %H:%M:%S")

    if use_stdout:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(log_level)
        ch.setFormatter(fmt)
        log.addHandler(ch)

    if not filename is None:
        fh = logging.FileHandler(filename)
        fh.setLevel(log_level)
        fh.setFormatter(fmt)
        log.addHandler(fh)
