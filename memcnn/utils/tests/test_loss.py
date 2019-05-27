import torch
from memcnn.utils.loss import _assert_no_grad, CrossEntropyLossTF


def test_assert_no_grad():
    data = torch.ones(3, 3, 3)
    data.requires_grad = False
    _assert_no_grad(data)


def test_crossentropy_tf():
    batch_size = 5
    shape = (batch_size, 2)
    loss = CrossEntropyLossTF()
    ypred = torch.ones(*shape)
    ypred.requires_grad = True
    y = torch.ones(batch_size, dtype=torch.int64)
    y.requires_grad = False
    w = torch.ones(*shape)
    w.requires_grad = False
    w2 = torch.zeros(*shape)
    w2.requires_grad = False

    out1 = loss(ypred, y)
    assert len(out1.shape) == 0

    out2 = loss(ypred, y, w)
    assert len(out2.shape) == 0

    out3 = loss(ypred, y, w2)
    assert out3 == 0
    assert len(out3.shape) == 0
