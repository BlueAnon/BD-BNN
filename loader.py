import os
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def dataloader_cifar10(split='train', batch_size=128):
    if split == 'train':
        data_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_flag = True
    else:
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_flag = False

    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_root = os.path.join(dir_path, 'Data', 'cifar10')

    dataset = datasets.CIFAR10(data_root, download=True, train=train_flag, transform=data_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader


def dataloader_cifar100(split='train', batch_size=128):
    if split == 'train':
        data_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_flag = True
    else:
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_flag = False
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_root = os.path.join(dir_path, 'Data', 'cifar100')
    dataset = datasets.CIFAR100(data_root, train=train_flag, transform=data_transform, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    return loader

def dataloader_imagenet(split='train', batch_size=128, data_path='./', distributed=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if split == 'train':
        traindir = os.path.join(data_path, 'train')
        dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        train_flag = True
        if distributed:
            train_sampler = DataLoader.distributed.DistributedSampler(dataset)
        else:
            train_sampler = None
    else:
        train_sampler = None
        valdir = os.path.join(data_path, 'val')
        dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
        train_flag = False

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train_flag, num_workers=16,pin_memory=True, sampler=train_sampler)

    return loader