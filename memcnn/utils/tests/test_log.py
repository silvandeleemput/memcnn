import logging
from memcnn.utils.log import setup, SummaryWriter


def test_setup(tmp_path):
    logfile = str(tmp_path / 'testlog.log')
    setup(use_stdout=True, filename=logfile, log_level=logging.DEBUG)


def test_summary_writer(tmp_path):
    logfile = tmp_path / 'scalars.json'

    assert not logfile.exists()
    writer = SummaryWriter(log_dir=str(tmp_path))
    writer.add_scalar("test_value", 0.5, 1)
    writer.add_scalar("test_value", 2.5, 2)
    writer.add_scalar("test_value2", 123, 1)
    writer.flush()
    assert logfile.exists()

    writer = SummaryWriter(log_dir=str(tmp_path))

    assert "test_value" in writer._summary
    assert "test_value2" in writer._summary
    assert len(writer._summary["test_value"]) == 2

    writer.add_scalar("test_value", 123.4, 3)
    writer.close()

    writer = SummaryWriter(log_dir=str(tmp_path))

    assert "test_value" in writer._summary
    assert "test_value2" in writer._summary
    assert len(writer._summary["test_value"]) == 3
