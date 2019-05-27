import pytest
import torch
from memcnn.utils.stats import AverageMeter, accuracy


@pytest.mark.parametrize('val,n', [(1, 1), (14, 10), (10, 14), (5, 1), (1, 5), (0, 10)])
def test_average_meter(val, n):
    meter = AverageMeter()
    assert meter.val == 0
    assert meter.avg == 0
    assert meter.sum == 0
    assert meter.count == 0
    meter.update(val, n=n)
    assert meter.val == val
    assert meter.avg == val
    assert meter.sum == val * n
    assert meter.count == n


@pytest.mark.parametrize('topk,klass', [((1,), 4), ((1, 3,), 2), ((5,), 1)])
def test_accuracy(topk, klass, num_klasses=5):  # output, target,
    batch_size = 5
    target = torch.ones(batch_size, dtype=torch.long) * klass
    output = torch.zeros(batch_size, num_klasses)
    output[:, klass] = 1
    res = accuracy(output, target, topk)
    assert len(res) == len(topk)
    assert all([e == 100.0 for e in res])
