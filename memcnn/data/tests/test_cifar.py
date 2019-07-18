import pytest
from memcnn.data.cifar import get_cifar_data_loaders, random_crop_transform
import torch.utils.data as data
import numpy as np
from PIL import Image


@pytest.mark.parametrize('crop_size,img_size', [(4, (32, 32)), (0, (32, 32))])
def test_random_crop_transform(crop_size, img_size):
    np.random.seed(42)
    img = np.random.random((img_size[0], img_size[1], 3))
    imgres = random_crop_transform(img, crop_size, img_size)
    assert imgres.shape == img.shape
    assert imgres.dtype == img.dtype
    if crop_size == 0:
        assert np.array_equal(img, imgres)


@pytest.mark.parametrize('max_epoch,batch_size', [(10, 2), (20, 4), (1, 1)])
def test_cifar_data_loaders(max_epoch, batch_size):
    np.random.seed(42)

    class TestDataset(data.Dataset):
        def __init__(self, train=True, *args, **kwargs):
            self.train = train
            self.args = args
            self.kwargs = kwargs
            if self.train:
                self.train_data = (np.random.random_sample((20, 32, 32, 3)) * 255).astype(np.uint8)
            else:
                self.test_data = (np.random.random_sample((10, 32, 32, 3)) * 255).astype(np.uint8)
            self.transform = lambda val: val

        def __getitem__(self, idx):
            img = self.train_data[idx] if self.train else self.test_data[idx]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, np.array(idx)

        def __len__(self):
            return len(self.train_data) if self.train else len(self.test_data)

    max_epoch = 10
    batch_size = 2
    workers = 0
    train_loader, val_loader = get_cifar_data_loaders(TestDataset, '', max_epoch, batch_size, workers=workers)

    xsize = (batch_size, 3, 32, 32)
    ysize = (batch_size, )
    count = 0
    for x, y in train_loader:
        count += 1
        assert x.shape == xsize
        assert y.shape == ysize

    assert count == max_epoch
    assert count == len(train_loader)

    count = 0
    for x, y in val_loader:
        count += 1
        assert x.shape == xsize
        assert y.shape == ysize

    assert count == len(val_loader.dataset) // batch_size
    assert count == len(val_loader)
