import pytest
import torch
from torch import nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

import torchvision
from torch.utils.checkpoint import checkpoint
from torchvision.models.resnet import resnet18, BasicBlock
import torchvision.transforms as transforms

import memcnn


class InvertibleBlock(nn.Module):
    def __init__(self, block, keep_input, enabled=True):
        """The input block should already be split across channels
        """
        super().__init__()
        # self.invertible_module = memcnn.AdditiveCoupling(block)
        self.invertible_block = memcnn.InvertibleModuleWrapper(fn=memcnn.AdditiveCoupling(block),
                                                               keep_input=keep_input,
                                                               keep_input_inverse=keep_input, disable=not enabled)

    def forward(self, x, inverse=False):
        # return checkpoint(self.invertible_module.forward, x)
        if inverse:
            return self.invertible_block.inverse(x)
        else:
            return self.invertible_block(x)


@pytest.mark.parametrize("inv_enabled", (True,))
@pytest.mark.parametrize("amp_enabled", (False, True))
def test_cuda_amp(tmp_path, inv_enabled, amp_enabled):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = resnet18(num_classes=10)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root=tmp_path, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    # Replace with invertible blocks
    model.layer1 = nn.Sequential(
        InvertibleBlock(BasicBlock(32, 32), keep_input=False, enabled=inv_enabled),
        InvertibleBlock(BasicBlock(32, 32), keep_input=False, enabled=inv_enabled)
    )

    model.to(device)
    # if amp_enabled:
    #     model.half()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scaler = GradScaler(enabled=amp_enabled)

    running_loss = 0.0

    for i, data in enumerate(trainloader):

        # -------- AMP ----------
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast(enabled=amp_enabled):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        # loss.backward()
        # optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # -----------------------

        running_loss += loss.item()

        break
