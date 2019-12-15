import torch
import sys


def test_minimal():
    import minimal
    # Input and inversed output should be approximately the same
    assert torch.allclose(minimal.X, minimal.X2, atol=1e-06)

    # Output of the wrapped invertible module is unlikely to match the normal output of F
    assert not torch.allclose(minimal.Y2, minimal.Y)

    # Cleanup minimal module and variables
    del minimal.X
    del minimal.Y
    del minimal.Y2
    del minimal.X2
    del minimal
    del sys.modules['minimal']
