import json
import os


class Config(dict):
    def __init__(self, dic=None, verbose=False):
        super(Config, self).__init__()
        if dic is None:
            fname = self.get_filename()
            if verbose:
                print("loading default {0}".format(fname))
            with open(fname, "r") as f:
                dic = json.load(f)
        self.update(dic)

    @staticmethod
    def get_filename():
        return os.path.join(Config.get_dir(), "config.json")

    @staticmethod
    def get_dir():
        return os.path.dirname(__file__)
