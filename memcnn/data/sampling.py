import torch
from torch.utils.data.sampler import Sampler

class NSamplesRandomSampler(Sampler):
    """Samples elements randomly, with replacement,
    always in blocks all elements of the dataset.
    Only the remainder will be sampled with less elements.

    Arguments:
        data_source (Dataset): dataset to sample from
        nsamples (int): number of total samples
    """

    def __init__(self, data_source, nsamples):
        self.data_source = data_source
        self.nsamples = nsamples

    def __iter__(self):
        samples = torch.LongTensor()
        ldsource = len(self.data_source)
        for _ in range(self.nsamples / ldsource):
            samples = torch.cat((samples, torch.randperm(len(self.data_source)).long()))
        if self.nsamples % ldsource > 0:
            samples = torch.cat((samples, torch.randperm(self.nsamples % ldsource).long()))
        return iter(samples)

    def __len__(self):
        return self.nsamples
