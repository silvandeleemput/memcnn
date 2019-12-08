import torch
from torch.utils.data.sampler import Sampler


class NSamplesRandomSampler(Sampler):
    """Samples elements randomly, with replacement,
    always in blocks all elements of the dataset.
    Only the remainder will be sampled with less elements.

    Arguments:
        data_source (Dataset): dataset to sample from
        nsamples (int): number of total samples. Note: will always be cast to int
    """

    @property
    def nsamples(self):
        return self._nsamples

    @nsamples.setter
    def nsamples(self, value):
        self._nsamples = int(value)

    def __init__(self, data_source, nsamples):
        self.data_source = data_source
        self.nsamples = nsamples

    def __iter__(self):
        samples = torch.LongTensor()
        len_data_source = len(self.data_source)
        for _ in range(self.nsamples // len_data_source):
            samples = torch.cat((samples, torch.randperm(len_data_source).long()))
        if self.nsamples % len_data_source > 0:
            samples = torch.cat((samples, torch.randperm(self.nsamples % len_data_source).long()))
        return iter(samples)

    def __len__(self):
        return self.nsamples
