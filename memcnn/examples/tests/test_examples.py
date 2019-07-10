import os
import sys
import memcnn


def test_example_minimal():
    example_fname = os.path.join(os.path.dirname(memcnn.__file__), 'examples', 'minimal.py')
    ret_code = os.system(sys.executable + ' ' + example_fname)
    assert(ret_code == 0)
