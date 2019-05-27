import pytest
import torch
from memcnn.models.resnet import ResNet, BasicBlock, Bottleneck, RevBasicBlock, RevBottleneck


@pytest.mark.parametrize('block,batch_norm_fix', [(BasicBlock, True), (Bottleneck, False), (RevBasicBlock, False), (RevBottleneck, True)])
def test_resnet(block, batch_norm_fix):
    model = ResNet(block, [2, 2, 2, 2], num_classes=2, channels_per_layer=None,
                   init_max_pool=True, batch_norm_fix=batch_norm_fix, strides=None)
    model.eval()
    with torch.no_grad():
        x = torch.ones(2, 3, 32, 32)
        model.forward(x)
