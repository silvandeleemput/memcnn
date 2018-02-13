import unittest
import json
import os
from memcnn.experiment.factory import load_experiment_config, experiment_config_parser

class ConfigTestCase(unittest.TestCase):

    def setUp(self):
        self.config_fname = os.path.join(os.path.dirname(__file__),
                        "..", "config.json")
        self.experiments_fname = os.path.join(os.path.dirname(__file__),
                        "..", "experiments.json")

        def load_json_file(fname):
            with open(fname, 'r') as f:
                data = json.load(f)
            return data

        self.load_json_file = load_json_file

    def test_loading_main_config(self):
        self.assertTrue(os.path.exists(self.config_fname))
        data = self.load_json_file(self.config_fname)
        self.assertTrue(isinstance(data, dict))
        self.assertTrue("data_dir" in data)
        self.assertTrue("results_dir" in data)

    def test_loading_experiments_config(self):
        self.assertTrue(os.path.exists(self.experiments_fname))
        data = self.load_json_file(self.experiments_fname)
        self.assertTrue(isinstance(data, dict))

    def test_experiment_configs(self):
        data = self.load_json_file(self.experiments_fname)
        config = self.load_json_file(self.config_fname)
        keys = data.keys()
        for key in keys:
            result = load_experiment_config(self.experiments_fname, [key])
            self.assertTrue(isinstance(result, dict))
            if "dataset" in result:
                results = experiment_config_parser(result, config['data_dir'])


if __name__ == '__main__':
    unittest.main()