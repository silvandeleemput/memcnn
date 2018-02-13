import unittest
from memcnn.config.tests.test_config import ConfigTestCase
from memcnn.models.tests.test_revop import ReversibleOperationsTestCase

def collect_tests():
    suites = [ConfigTestCase, ReversibleOperationsTestCase]
    alltests = unittest.TestSuite([unittest.TestLoader().loadTestsFromTestCase(suite) for suite in suites])
    return alltests

if __name__ == '__main__':
   suite = collect_tests()
   runner=unittest.TextTestRunner()
   runner.run(suite)
