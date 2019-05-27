import pytest
import os
import memcnn.experiment.factory
from memcnn.config import Config


def test_get_attr_from_module():
    a = memcnn.experiment.factory.get_attr_from_module('memcnn.experiment.factory.get_attr_from_module')
    assert a is memcnn.experiment.factory.get_attr_from_module


def test_load_experiment_config():
    cfg_fname = os.path.join(Config.get_dir(), 'experiments.json')
    memcnn.experiment.factory.load_experiment_config(cfg_fname, ['cifar10', 'resnet110'])


@pytest.mark.skip(reason="Covered more efficiently by test_train.test_run_experiment")
def test_experiment_config_parser(tmp_path):
    tmp_data_dir = tmp_path / "tmpdata"
    cfg_fname = os.path.join(Config.get_dir(), 'experiments.json')
    cfg = memcnn.experiment.factory.load_experiment_config(cfg_fname, ['cifar10', 'resnet110'])
    memcnn.experiment.factory.experiment_config_parser(cfg, str(tmp_data_dir), workers=None)


def test_circular_dependency(tmp_path):
    p = str(tmp_path / "circular.json")
    content = u'{ "circ": { "base": "circ" } }'
    with open(p, 'w') as fh:
        fh.write(content)
    with open(p, 'r') as fh:
        assert fh.read() == content
    with pytest.raises(RuntimeError):
        memcnn.experiment.factory.load_experiment_config(p, ['circ'])
