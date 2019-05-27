import copy
from memcnn.experiment.manager import ExperimentManager
import torch.nn


def test_experiment_manager(tmp_path):
    exp_dir = tmp_path / "test_exp_dir"
    man = ExperimentManager(str(exp_dir))
    assert man.model is None
    assert man.optimizer is None

    man.make_dirs()
    assert exp_dir.exists()
    assert (exp_dir / "log").exists()
    assert (exp_dir / "state" / "model").exists()
    assert (exp_dir / "state" / "optimizer").exists()
    assert man.all_dirs_exists()
    assert man.any_dir_exists()

    man.delete_dirs()
    assert not exp_dir.exists()
    assert not (exp_dir / "log").exists()
    assert not (exp_dir / "state" / "model").exists()
    assert not (exp_dir / "state" / "optimizer").exists()
    assert not man.all_dirs_exists()
    assert not man.any_dir_exists()

    man.make_dirs()

    man.model = torch.nn.Conv2d(2, 1, 3)
    w = man.model.weight.clone()
    man.save_model_state(0)
    with torch.no_grad():
        man.model.weight.zero_()
    man.save_model_state(100)
    assert not man.model.weight.equal(w)
    assert man.get_last_model_iteration() == 100

    man.load_model_state(0)
    assert man.model.weight.equal(w)

    optimizer = torch.optim.SGD(man.model.parameters(), lr=0.01)
    man.optimizer = optimizer

    man.save_train_state(100)

    w = man.model.weight.clone()
    sd = copy.deepcopy(man.optimizer.state_dict())

    man.model.train()

    x = torch.ones(5, 2, 5, 5)
    x.requires_grad = True
    y = torch.ones(5, 1, 3, 3)
    y.requires_grad = False

    ypred = man.model(x)
    loss = torch.nn.MSELoss()(y, ypred)
    man.optimizer.zero_grad()
    loss.backward()
    man.optimizer.step()

    man.save_train_state(101)
    assert not man.model.weight.equal(w)
    # assert sd != man.optimizer.state_dict()
    w2 = man.model.weight.clone()
    # sd2 = copy.deepcopy(man.optimizer.state_dict())

    man.load_train_state(100)
    assert man.model.weight.equal(w)
    # assert sd == man.optimizer.state_dict()

    man.load_last_train_state() # should be 101
    assert not man.model.weight.equal(w)
    # assert sd != man.optimizer.state_dict()
    assert man.model.weight.equal(w2)
    # assert sd2 == man.optimizer.state_dict()
    # TODO fix state dict optimizer parameters asserts
