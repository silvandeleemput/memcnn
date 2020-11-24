import pytest
import torch
from torch import nn
import torch.optim as optim

import torchvision
from torch.utils.checkpoint import checkpoint
from torchvision.models.resnet import resnet18, BasicBlock
import torchvision.transforms as transforms

import memcnn

try:
    from torch.cuda.amp import autocast, GradScaler
except ModuleNotFoundError:
    pass


class InvertibleBlock(nn.Module):
    def __init__(self, block, keep_input, enabled=True):
        super().__init__()
        self.invertible_block = memcnn.InvertibleModuleWrapper(
            fn=memcnn.AdditiveCoupling(block),
            keep_input=keep_input,
            keep_input_inverse=keep_input,
            disable=not enabled,
        )

    def forward(self, x, inverse=False):
        if inverse:
            return self.invertible_block.inverse(x)
        else:
            return self.invertible_block(x)


class CheckPointBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.invertible_module = memcnn.AdditiveCoupling(block)

    def forward(self, x, inverse=False):
        return checkpoint(self.invertible_module.forward, x)


@pytest.mark.skipif(
    condition="autocast" not in locals(),
    reason="torch.cuda.amp could not be found. torch version is < 1.6.",
)
@pytest.mark.parametrize(
    "use_checkpointing, inv_enabled", ((True, False), (False, True,), (False, False))
)
@pytest.mark.parametrize("amp_enabled", (False, True))
def test_cuda_amp(tmp_path, inv_enabled, amp_enabled, use_checkpointing):
    if not torch.cuda.is_available() and amp_enabled:
        pytest.skip("This test requires a GPU to be available")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = resnet18(num_classes=10)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = torchvision.datasets.CIFAR10(
        root=tmp_path, train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=2
    )

    # Replace layer1
    if not use_checkpointing:
        model.layer1 = nn.Sequential(
            InvertibleBlock(BasicBlock(32, 32), keep_input=False, enabled=inv_enabled),
            InvertibleBlock(BasicBlock(32, 32), keep_input=False, enabled=inv_enabled),
        )
    else:
        model.layer1 = nn.Sequential(
            CheckPointBlock(BasicBlock(32, 32)), CheckPointBlock(BasicBlock(32, 32))
        )

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scaler = GradScaler(enabled=amp_enabled)

    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast(enabled=amp_enabled):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        break
