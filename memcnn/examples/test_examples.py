import torch


def test_minimal():
    import minimal
    # Input and inversed output should be approximately the same
    assert torch.allclose(minimal.X, minimal.X2, atol=1e-06)

    # Output of the reversible block is unlikely to match the normal output of F
    assert not torch.allclose(minimal.Y2, minimal.Y)
