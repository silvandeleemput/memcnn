import os
import json
import logging
import sys
import time


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

    if filename is not None:
        fh = logging.FileHandler(filename)
        fh.setLevel(log_level)
        fh.setFormatter(fmt)
        log.addHandler(fh)


class SummaryWriter(object):
    def __init__(self, log_dir):
        self._log_dir = log_dir
        self._log_file = os.path.join(log_dir, "scalars.json")
        self._summary = {}
        self._load_if_exists()

    def _load_if_exists(self):
        if os.path.exists(self._log_file):
            with open(self._log_file, "r") as f:
                self._summary = json.load(f)

    def add_scalar(self, name, value, iteration):
        if name not in self._summary:
            self._summary[name] = []
        self._summary[name].append([time.time(), int(iteration), float(value)])

    def flush(self):
        with open(self._log_file, "w") as f:
            json.dump(self._summary, f)

    def close(self):
        self.flush()
