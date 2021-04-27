# encoding=utf-8

import collections
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

from data_loader import KittiLoader, MyKittiLoader
from models_resnet import Resnet18_md, Resnet50_md, ResnetModel
from transforms import image_transforms


def find_free_gpu():
    """
    :return:
    """
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > tmp.py')
    memory_left_gpu = [int(x.split()[2]) for x in open('tmp.py', 'r').readlines()]

    most_free_gpu_idx = np.argmax(memory_left_gpu)
    # print(str(most_free_gpu_idx))
    return int(most_free_gpu_idx)


def select_device(device='', apex=False, batch_size=None):
    """
    :param device:
    :param apex:
    :param batch_size:
    :return:
    """
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = 'Using CUDA ' + ('Apex ' if apex else '')  # apex for mixed precision https://github.com/NVIDIA/apex
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (s, i, x[i].name, x[i].total_memory / c))
    else:
        print('Using CPU')

    print('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')


def to_device(input, device):
    """
    :param input:
    :param device:
    :return:
    """
    if torch.is_tensor(input):
        return input.to(device=device)
    elif isinstance(input, str):
        return input
    elif isinstance(input, collections.Mapping):
        return {k: to_device(sample, device=device) for k, sample in input.items()}
    elif isinstance(input, collections.Sequence):
        return [to_device(sample, device=device) for sample in input]
    else:
        raise TypeError(f"Input must contain tensor, dict or list, found {type(input)}")


def get_model(model, input_channels=3, pretrained=False):
    """
    :param model:
    :param input_channels:
    :param pretrained:
    :return:
    """
    if model == 'resnet50_md':
        out_model = Resnet50_md(input_channels)
    elif model == 'resnet18_md':
        out_model = Resnet18_md(input_channels)
    else:
        out_model = ResnetModel(input_channels, encoder=model, pretrained=pretrained)

    return out_model


def prepare_dataloader(root_dir,
                       mode,
                       augment_parameters,
                       do_augmentation,
                       batch_size,
                       size,
                       num_workers):
    """
    :param root_dir:
    :param mode:
    :param augment_parameters:
    :param do_augmentation:
    :param batch_size:
    :param size:
    :param num_workers:
    :return:
    """
    data_dirs = os.listdir(root_dir)

    # Define data augmentation
    data_transform = image_transforms(mode=mode,
                                      augment_parameters=augment_parameters,
                                      do_augmentation=do_augmentation,
                                      size=size)
    datasets = [MyKittiLoader(os.path.join(root_dir, dir), mode, transform=data_transform)
                for dir in data_dirs if os.path.isdir(root_dir + '/' + dir) and dir != 'val']
    # datasets = [KittiLoader(data_directory, mode, transform=data_transform)
    #             for data_dir in data_dirs]
    dataset = ConcatDataset(datasets)
    n_imgs = len(dataset)

    print('Use a dataset with', n_imgs, 'images')
    if mode == 'train':
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True)
        print('Using number of workers for training: {:d}.'.format(num_workers))
    else:
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True)

    return n_imgs, loader
