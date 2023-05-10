import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from memcnn.data.sampling import NSamplesRandomSampler
import functools


def random_crop_transform(x, crop_size=3, img_size=(32, 32)):
    cz = (crop_size + 1) // 2
    x_pad = np.pad(x, ((cz, cz), (cz, cz), (0, 0)), mode='constant')
    sx, sy = np.random.randint(crop_size + 1), np.random.randint(crop_size + 1)
    return x_pad[sx:sx + img_size[0], sy:sy + img_size[1], :]


def tonumpy_fn(x):
    return np.array(x.getdata()).reshape(x.size[1], x.size[0], 3)


def random_lr_flip_fn(x):
    return np.copy(x[:, ::-1, :]) if np.random.random() >= 0.5 else x


def mean_subtract_fn(x, mean=0):
    return x.astype(np.float32) - mean


def reformat_fn(x):
    return x.transpose(2, 0, 1).astype(np.float32)


def get_cifar_data_loaders(dataset, data_dir, max_epoch, batch_size, workers):

    train_set = dataset(root=data_dir, train=True, download=True)
    valid_set = dataset(root=data_dir, train=False, download=True)

    # calculate mean subtraction img with backwards compatibility for torchvision < 0.2.2
    tdata = train_set.train_data if hasattr(train_set, 'train_data') else train_set.data
    vdata = valid_set.test_data if hasattr(valid_set, 'test_data') else valid_set.data
    mean_img = np.concatenate((tdata, vdata), axis=0).mean(axis=0)

    mean_subtract_partial_fn = functools.partial(mean_subtract_fn, mean=mean_img)

    # define transforms
    randomcroplambda = transforms.Lambda(random_crop_transform)
    tonumpy = transforms.Lambda(tonumpy_fn)
    randomlrflip = transforms.Lambda(random_lr_flip_fn)
    meansubtraction = transforms.Lambda(mean_subtract_partial_fn)
    reformat = transforms.Lambda(reformat_fn)
    totensor = transforms.Lambda(torch.from_numpy)
    tfs = transforms.Compose([
        tonumpy,
        meansubtraction,
        randomcroplambda,
        randomlrflip,
        reformat,
        totensor
    ])

    train_set.transform = tfs
    valid_set.transform = tfs
    sampler = NSamplesRandomSampler(train_set, max_epoch * batch_size)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size, shuffle=False,
                              sampler=sampler, num_workers=workers,
                              pin_memory=True)

    val_loader = DataLoader(valid_set,
                            batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=True)

    return train_loader, val_loader
