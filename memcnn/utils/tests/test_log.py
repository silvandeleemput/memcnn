import logging
from memcnn.utils.log import setup


def test_setup(tmp_path):
    logfile = str(tmp_path / 'testlog.log')
    setup(use_stdout=True, filename=logfile, log_level=logging.DEBUG)
