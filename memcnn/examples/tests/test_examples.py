import os
import subprocess as sp
import sys
import memcnn


def test_example_minimal():
    example_fname = os.path.join(os.path.abspath(os.path.dirname(memcnn.__file__)), 'examples', 'minimal.py')
    p = sp.Popen([os.path.abspath(sys.executable), example_fname], stdout=sp.DEVNULL, stdin=sp.DEVNULL)
    p.communicate()
    assert(p.returncode == 0)
