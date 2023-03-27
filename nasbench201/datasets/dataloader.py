from copy import deepcopy
import os, json

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms

from PIL import Image
import random
from typing import Callable, Optional

from .DownsampledImageNet import ImageNet16, ImageNet16Rebuild


class TensorDataset(Dataset):
    def __init__(self, images, labels):  # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


class CIFAR10Rebuild(dset.CIFAR10):
    def __init__(self, root: str, train: bool = True, data_running_index=None, debug=False,
                 transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, ):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform)
        if debug:
            self.data = self.data[:10, :, :, :]
        self.initial_indices = list(range(self.__len__()))
        self.initial_classes = list(range(10))
        self.initial_prob = np.ones(self.__len__()) / self.__len__()
        self.initial_class_prob = np.ones(10) / 10

        self.update_running_indices()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        index = self.data_running_index[index]
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)

    def update_running_indices(self, running_indices=None):
        if running_indices is None:
            self.data_running_index = self.initial_indices
        else:
            self.data_running_index = running_indices


class CIFAR100Rebuild(dset.CIFAR100):
    def __init__(self, root: str, train: bool = True, data_running_index=None, debug=False,
                 transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, ):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform)
        if debug:
            self.data = self.data[:10, :, :, :]
        self.initial_indices = list(range(self.__len__()))
        # self.initial_classes = list(range(100))
        self.initial_prob = np.ones(self.__len__()) / self.__len__()
        # self.initial_class_prob = np.ones(100) / 10

        self.update_running_indices()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        index = self.data_running_index[index]
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)

    def update_running_indices(self, running_indices=None):
        if running_indices is None:
            self.data_running_index = self.initial_indices
        else:
            self.data_running_index = running_indices


class CIFAR10SpecificIndices(dset.CIFAR10):
    def __init__(self, large_batch: int, root: str, train: bool = True, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, ):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform)
        self.indices = list(range(len(self.data)))
        self.large_batch = large_batch
        self.running_indices = None

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # self.update_class_training()
        actual_index = self.running_indices[index]
        img, target = self.data[actual_index], self.targets[actual_index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.large_batch


def convert_param(original_lists):
    assert isinstance(original_lists, list), "The type is not right : {:}".format(
        original_lists
    )
    ctype, value = original_lists[0], original_lists[1]
    is_list = isinstance(value, list)
    if not is_list:
        value = [value]
    outs = []
    for x in value:
        if ctype == "int":
            x = int(x)
        elif ctype == "str":
            x = str(x)
        elif ctype == "bool":
            x = bool(int(x))
        elif ctype == "float":
            x = float(x)
        elif ctype == "none":
            if x.lower() != "none":
                raise ValueError(
                    "For the none type, the value must be none instead of {:}".format(x)
                )
            x = None
        else:
            raise TypeError("Does not know this type : {:}".format(ctype))
        outs.append(x)
    if not is_list:
        outs = outs[0]
    return outs


def load_config(path):
    path = str(path)
    assert os.path.exists(path), "Can not find {:}".format(path)
    # Reading data back
    with open(path, "r") as f:
        data = json.load(f)
    content = {k: convert_param(v) for k, v in data.items()}

    return content


def get_dataloader(args, model, dataset, train_idx=None):
    if dataset == 'cifar10':
        # CIFAR-10 Dataset
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        # Dataloader
        data_path = os.path.join(args.data_root, args.dataset)
        train_data = dset.CIFAR10(root=data_path, train=True, download=False, transform=train_transform)
        valid_data = dset.CIFAR10(root=data_path, train=False, download=False, transform=valid_transform)
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.train_batch_size, shuffle=True,
            pin_memory=True, num_workers=0 if args.debug else 8)
        valid_loader = torch.utils.data.DataLoader(
            valid_data, batch_size=args.valid_batch_size, shuffle=False,
            pin_memory=True, num_workers=0 if args.debug else 16)
        return train_loader, valid_loader

    elif dataset == 'cifar100':
        CIFAR_MEAN = [x / 255 for x in [125.3, 123.0, 113.9]]
        CIFAR_STD = [x / 255 for x in [63.0, 62.1, 66.7]]
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        # Dataloader
        data_path = os.path.join(args.data_root, args.dataset)
        train_data = dset.CIFAR100(root=data_path, train=True, download=False, transform=train_transform)
        valid_data = dset.CIFAR100(root=data_path, train=False, download=False, transform=valid_transform)
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.train_batch_size, shuffle=True,
            pin_memory=True, num_workers=0 if args.debug else 8)
        valid_loader = torch.utils.data.DataLoader(
            valid_data, batch_size=args.valid_batch_size, shuffle=False,
            pin_memory=True, num_workers=0 if args.debug else 16)
        return train_loader, valid_loader

    elif dataset == 'imagenet16':
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std = [x / 255 for x in [63.22, 61.26, 65.09]]
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(16, padding=2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        data_path = os.path.join(args.data_root, args.dataset)
        train_data = ImageNet16(data_path, True, train_transform, 120)
        valid_data = ImageNet16(data_path, False, valid_transform, 120)
        assert len(train_data) == 151700 and len(valid_data) == 6000
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.train_batch_size, shuffle=True,
            pin_memory=True, num_workers=0 if args.debug else 8)
        valid_loader = torch.utils.data.DataLoader(
            valid_data, batch_size=args.valid_batch_size, shuffle=False,
            pin_memory=True, num_workers=0 if args.debug else 16)
        return train_loader, valid_loader

    elif dataset == 'cifar10_half':
        # CIFAR-10 Dataset
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        valid_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        cifar_split = load_config("./dataset/cifar-split.txt")
        train_split, valid_split = cifar_split['train'], cifar_split['valid']

        # Dataloader
        data_path = os.path.join('./dataset/cifar10')
        train_data = dset.CIFAR10(root=data_path, train=True, download=False, transform=train_transform)
        test_data = dset.CIFAR10(root=data_path, train=False, download=False, transform=test_transform)
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.train_batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(train_split),
            num_workers=0 if args.debug else 16, pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.valid_batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_split),
            num_workers=0 if args.debug else 16, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=args.valid_batch_size, shuffle=False,
            pin_memory=True, num_workers=0 if args.debug else 16)
        return train_loader, valid_loader, test_loader

    elif dataset == 'cifar10_rebuild_loader':
        # CIFAR-10 Dataset
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        # Shuffle by dataloader
        train_data = CIFAR10Rebuild('./dataset/cifar10', data_running_index=train_idx, debug=args.debug,
                                    train=True, transform=train_transform)
        valid_data = dset.CIFAR10(root='./dataset/cifar10', train=False, download=False, transform=valid_transform)
        if args.debug:
            valid_data.data = valid_data.data[:10, :, :, :]

        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.train_batch_size, shuffle=True,
            pin_memory=True, num_workers=0 if args.debug else 8)
        valid_loader = torch.utils.data.DataLoader(
            valid_data, batch_size=args.valid_batch_size, shuffle=False,
            pin_memory=True, num_workers=0 if args.debug else 16)

        return train_data, valid_data, train_loader, valid_loader

    elif dataset == 'cifar100_rebuild_loader':
        CIFAR_MEAN = [x / 255 for x in [125.3, 123.0, 113.9]]
        CIFAR_STD = [x / 255 for x in [63.0, 62.1, 66.7]]
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        # Shuffle by dataloader
        train_data = CIFAR100Rebuild('./dataset/cifar100', data_running_index=train_idx, debug=args.debug,
                                    train=True, transform=train_transform)
        valid_data = dset.CIFAR100(root='./dataset/cifar100', train=False, download=False, transform=valid_transform)
        if args.debug:
            valid_data.data = valid_data.data[:10, :, :, :]

        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.train_batch_size, shuffle=True,
            pin_memory=True, num_workers=0 if args.debug else 8)
        valid_loader = torch.utils.data.DataLoader(
            valid_data, batch_size=args.valid_batch_size, shuffle=False,
            pin_memory=True, num_workers=0 if args.debug else 16)

        return train_data, valid_data, train_loader, valid_loader

    elif dataset == 'imagenet16_rebuild_loader':
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std = [x / 255 for x in [63.22, 61.26, 65.09]]
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(16, padding=2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        # Shuffle by dataloader
        data_path = './dataset/imagenet16'
        train_data = ImageNet16Rebuild(data_path, True, train_transform, 120)
        valid_data = ImageNet16(data_path, False, valid_transform, 120)
        assert len(train_data) == 151700 and len(valid_data) == 6000
        if args.debug:
            valid_data.data = valid_data.data[:10, :, :, :]

        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.train_batch_size, shuffle=True,
            pin_memory=True, num_workers=0 if args.debug else 8)
        valid_loader = torch.utils.data.DataLoader(
            valid_data, batch_size=args.valid_batch_size, shuffle=False,
            pin_memory=True, num_workers=0 if args.debug else 16)

        return train_data, valid_data, train_loader, valid_loader
    else:
        raise ValueError('Wrong dataset: %s' % args.dataset)
