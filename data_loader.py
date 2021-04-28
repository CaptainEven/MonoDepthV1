import os

from PIL import Image
from torch.utils.data import Dataset


class MyKittiLoader(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        """
        :param root_dir:
        :param mode:
        :param transform:
        """
        # root_dir = root_dir.replace('image_02', '')
        # @even: process each sub dir
        sub_dirs = [root_dir + '/' + x for x in os.listdir(root_dir)
                    if os.path.isdir(root_dir + '/' + x)]
        sub_dirs.sort()

        self.left_paths = []
        for sub_dir in sub_dirs:
            left_dir = os.path.join(sub_dir, 'image_02/data/')
            left_img_paths = sorted([left_dir + x for x in os.listdir(left_dir)])
            self.left_paths = self.left_paths + left_img_paths

        if mode == 'train':
            right_dir = os.path.join(root_dir, 'image_03/data/')
            self.right_paths = [x.replace('image_02', 'image_03') for x in self.left_paths
                                if os.path.isfile(x) and os.path.isfile(x.replace('image_02', 'image_03'))]
            assert len(self.right_paths) == len(self.left_paths)

        self.transform = transform
        self.mode = mode

    def __len__(self):
        """
        :return:
        """
        return len(self.left_paths)

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        """
        left_image = Image.open(self.left_paths[idx])
        if self.mode == 'train':  # train mode
            right_image = Image.open(self.right_paths[idx])
            sample = {
                'left_image': left_image,
                'right_image': right_image,
            }

            if self.transform:
                sample = self.transform(sample)  # 原来transform可以接收一个字典...
            return sample, self.left_paths[idx]  # left_image path

        else:  # test mode: only left image is needed
            if self.transform:
                left_image = self.transform(left_image)

            return left_image, self.left_paths[idx]  # left image path


class KittiLoader(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        """
        :param root_dir:
        :param mode:
        :param transform:
        """
        # root_dir = root_dir.replace('image_02', '')
        left_dir = os.path.join(root_dir, 'image_02/data/')
        self.left_paths = sorted([os.path.join(left_dir, f_name) for f_name in os.listdir(left_dir)])

        if mode == 'train':
            right_dir = os.path.join(root_dir, 'image_03/data/')
            self.right_paths = sorted([os.path.join(right_dir, f_name) for f_name in os.listdir(right_dir)])
            assert len(self.right_paths) == len(self.left_paths)

        self.transform = transform
        self.mode = mode

    def __len__(self):
        """
        :return:
        """
        return len(self.left_paths)

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        """
        left_image = Image.open(self.left_paths[idx])
        if self.mode == 'train':
            right_image = Image.open(self.right_paths[idx])
            sample = {'left_image': left_image, 'right_image': right_image}

            if self.transform:
                sample = self.transform(sample)
                return sample
            else:
                return sample
        else:
            if self.transform:
                left_image = self.transform(left_image)

            return left_image
