import os
import glob
import torch
import logging
import shutil
import numpy as np

class ExperimentManager(object):

    def __init__(self, experiment_dir, model=None, optimizer=None):
        self.logger = logging.getLogger(type(self).__name__)
        self.experiment_dir = experiment_dir
        self.model = model
        self.optimizer = optimizer
        self.model_dir = os.path.join(self.experiment_dir, "state", "model")
        self.optim_dir = os.path.join(self.experiment_dir, "state", "optimizer")
        self.log_dir = os.path.join(self.experiment_dir, "log")
        self.dirs = (self.experiment_dir, self.model_dir, self.log_dir, self.optim_dir)

    def make_dirs(self):
        for d in self.dirs:
            if not os.path.exists(d):
                os.makedirs(d)
        assert(self.all_dirs_exists())

    def delete_dirs(self):
        for d in self.dirs:
            if os.path.exists(d):
                shutil.rmtree(d)
        assert(not self.any_dir_exists())

    def any_dir_exists(self):
        return any([os.path.exists(d) for d in self.dirs])

    def all_dirs_exists(self):
        return all([os.path.exists(d) for d in self.dirs])

    def save_model_state(self, epoch):
        model_fname = os.path.join(self.model_dir, "{}.pt".format(epoch))
        self.logger.info("Saving model state to: {}".format(model_fname))
        torch.save(self.model.state_dict(), model_fname)

    def load_model_state(self, epoch):
        model_fname = os.path.join(self.model_dir, "{}.pt".format(epoch))
        self.logger.info("Loading model state from: {}".format(model_fname))
        self.model.load_state_dict(torch.load(model_fname))

    def save_optimizer_state(self, epoch):
        optim_fname = os.path.join(self.optim_dir, "{}.pt".format(epoch))
        self.logger.info("Saving optimizer state to: {}".format(optim_fname))
        torch.save(self.optimizer.state_dict(), optim_fname)

    def load_optimizer_state(self, epoch):
        optim_fname = os.path.join(self.optim_dir, "{}.pt".format(epoch))
        self.logger.info("Loading optimizer state from {}".format(optim_fname))
        self.optimizer.load_state_dict(torch.load(optim_fname))

    def save_train_state(self, epoch):
        self.save_model_state(epoch)
        self.save_optimizer_state(epoch)

    def load_train_state(self, epoch):
        self.load_model_state(epoch)
        self.load_optimizer_state(epoch)

    def get_last_model_iteration(self):
        return np.array([0] + [int(os.path.basename(e).split(".")[0]) for e in glob.glob(os.path.join(self.model_dir, "*.pt"))]).max()

    def load_last_train_state(self):
        self.load_train_state(self.get_last_model_iteration())
