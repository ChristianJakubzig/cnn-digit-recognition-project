import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

kwargs = {'num_workers' : 1, 'pin_memory' : True}

'''

'''
train = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1037,), (0.3081))],
        ),
    ), batch_size=64, shuffle=True, **kwargs
)

test = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1037,), (0.3081))],
        ),
    ), batch_size=64, shuffle=True, **kwargs
)
