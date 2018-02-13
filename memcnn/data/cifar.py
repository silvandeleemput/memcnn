import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from memcnn.data.sampling import NSamplesRandomSampler

def get_cifar_data_loaders(dataset, data_dir, max_epoch, batch_size, workers):

    train_set = dataset(root=data_dir, train=True, download=True)
    valid_set = dataset(root=data_dir, train=False, download=True)

    # calculate mean subtraction img...
    mean_img = np.concatenate((train_set.train_data, valid_set.test_data), axis=0).mean(axis=0)

    # define transforms
    def random_crop_sub(x, crop_size=4, img_size=(32, 32)):
        cz = crop_size / 2
        x_pad = np.pad(x, ((cz, cz), (cz, cz), (0, 0)), mode='constant')
        sx, sy = np.random.randint(crop_size), np.random.randint(crop_size)
        return x_pad[sx:sx+img_size[0], sy:sy+img_size[1], :]

    randomcrop = transforms.Lambda(lambda x: random_crop_sub(x))
    tonumpy = transforms.Lambda(lambda x: np.array(x.getdata()).reshape(x.size[1], x.size[0], 3))
    randomlrflip = transforms.Lambda(lambda x: np.copy(x[:,::-1,:]) if np.random.random() >= 0.5 else x)
    meansubtraction = transforms.Lambda(lambda x: x.astype(np.float) - mean_img)
    totensor = transforms.Lambda(lambda x: torch.from_numpy(x.transpose(2, 0, 1).astype(np.float32)))
    tfs = transforms.Compose([
                    tonumpy,
                    meansubtraction,
                    randomcrop,
                    randomlrflip,
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

