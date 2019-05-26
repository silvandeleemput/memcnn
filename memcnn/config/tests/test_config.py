import unittest
import json
import os
from memcnn.experiment.factory import load_experiment_config, experiment_config_parser
from memcnn.config import Config
import memcnn.config


class ConfigTestCase(unittest.TestCase):

    class ConfigTest(Config):
        @staticmethod
        def get_filename():
            return os.path.join(Config.get_dir(), "config.json.example")

    def setUp(self):
        self.config = ConfigTestCase.ConfigTest()

        self.config_fname = os.path.join(os.path.dirname(__file__), "..", "config.json.example")
        self.experiments_fname = os.path.join(os.path.dirname(__file__), "..", "experiments.json")

        def load_json_file(fname):
            with open(fname, 'r') as f:
                data = json.load(f)
            return data

        self.load_json_file = load_json_file

    def test_loading_main_config(self):
        self.assertTrue(os.path.exists(self.config.get_filename()))
        data = self.config
        self.assertTrue(isinstance(data, dict))
        self.assertTrue("data_dir" in data)
        self.assertTrue("results_dir" in data)

    def test_loading_experiments_config(self):
        self.assertTrue(os.path.exists(self.experiments_fname))
        data = self.load_json_file(self.experiments_fname)
        self.assertTrue(isinstance(data, dict))

    def test_experiment_configs(self):
        data = self.load_json_file(self.experiments_fname)
        config = self.config
        keys = data.keys()
        for key in keys:
            result = load_experiment_config(self.experiments_fname, [key])
            self.assertTrue(isinstance(result, dict))
            if "dataset" in result:
                experiment_config_parser(result, config['data_dir'])

    def test_config_get_filename(self):
        self.assertEqual(Config.get_filename(), os.path.join(os.path.dirname(memcnn.config.__file__), "config.json"))

    def test_config_get_dir(self):
        self.assertEqual(Config.get_dir(), os.path.dirname(memcnn.config.__file__))

    def test_verbose(self):
        ConfigTestCase.ConfigTest(verbose=True)


if __name__ == '__main__':
    unittest.main()
