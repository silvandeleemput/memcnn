import pytest
import os
import sys
import torch
from memcnn.train import run_experiment, main
try:
    from pathlib2 import Path
except ImportError:
    from pathlib import Path


def test_main(tmp_path):
    sys.argv = ['train.py', 'cifar10', 'resnet34', '--fresh', '--no-cuda', '--workers=0']
    data_dir = str(tmp_path / "tmpdata")
    results_dir = str(tmp_path / "resdir")
    os.makedirs(data_dir)
    os.makedirs(results_dir)
    with pytest.raises(KeyError):
        main(data_dir=data_dir, results_dir=results_dir)


def dummy_dataloaders(*args, **kwargs):
    return None, None


def dummy_trainer(manager, *args, **kwargs):
    manager.save_train_state(2)


class DummyDataset(object):
    def __init__(self, *args, **kwargs):
        pass


class DummyModel(torch.nn.Module):
    def __init__(self, block):
        super(DummyModel, self).__init__()
        self.block = block
        self.conv = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv(x)


def test_run_experiment(tmp_path):
    exptags = ['testsetup']
    exp_file = str(Path(__file__).parent / "resources" / "experiments.json")
    data_dir = str(tmp_path / "tmpdata")
    results_dir = str(tmp_path / "resdir")
    run_params = dict(
        experiment_tags=exptags, data_dir=data_dir, results_dir=results_dir,
        start_fresh=True, use_cuda=False, workers=None, experiments_file=exp_file
    )
    with pytest.raises(RuntimeError):
        run_experiment(**run_params)
    os.makedirs(data_dir)
    with pytest.raises(RuntimeError):
        run_experiment(**run_params)
    os.makedirs(results_dir)
    run_experiment(**run_params)
    run_params["start_fresh"] = False
    run_experiment(**run_params)


def test_train_revnet164_with_memory_saving(tmp_path):
    exptags = ['cifar10', 'revnet164', 'epoch1']
    exp_file = str(Path(__file__).parent / "resources" / "experiments.json")
    data_dir = str(tmp_path / "tmpdata")
    results_dir = str(tmp_path / "resdir")
    os.makedirs(data_dir)
    os.makedirs(results_dir)
    run_experiment(experiment_tags=exptags, data_dir=data_dir, results_dir=results_dir,
                   start_fresh=True, use_cuda=False, workers=None, experiments_file=exp_file)
