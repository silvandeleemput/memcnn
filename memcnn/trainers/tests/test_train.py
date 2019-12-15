import json

import pytest
import os
import sys
import torch

from memcnn.experiment.manager import ExperimentManager
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


@pytest.mark.parametrize("network", [
    pytest.param(network,
                 marks=pytest.mark.skipif(
                     condition=("FULL_NETWORK_TESTS" not in os.environ) and ("revnet38" != network),
                     reason="Too memory intensive for CI so these tests are disabled by default. "
                            "Set FULL_NETWORK_TESTS environment variable to enable the tests.")
                 )
    for network in ["resnet32", "resnet110", "resnet164", "revnet38", "revnet110", "revnet164"]
])
@pytest.mark.parametrize("use_cuda", [
    False,
    pytest.param(True, marks=pytest.mark.skipif(condition=not torch.cuda.is_available(), reason="No GPU available"))
])
def test_train_networks(tmp_path, network, use_cuda):
    exptags = ["cifar10", network, "epoch5"]
    exp_file = str(Path(__file__).parent / "resources" / "experiments.json")
    data_dir = str(tmp_path / "tmpdata")
    results_dir = str(tmp_path / "resdir")
    os.makedirs(data_dir)
    os.makedirs(results_dir)
    run_experiment(experiment_tags=exptags, data_dir=data_dir, results_dir=results_dir,
                   start_fresh=True, use_cuda=use_cuda, workers=None, experiments_file=exp_file,
                   disp_iter=1,
                   save_iter=5,
                   valid_iter=5,)
    experiment_dir = os.path.join(results_dir, '_'.join(exptags))
    assert os.path.exists(experiment_dir)
    manager = ExperimentManager(experiment_dir)
    scalars_file = os.path.join(manager.log_dir, "scalars.json")
    assert os.path.exists(scalars_file)
    with open(scalars_file, "r") as f:
        results = json.load(f)
    # no results should hold any NaN values
    assert not any([val != val for t, i, val in results["train_loss"]])
