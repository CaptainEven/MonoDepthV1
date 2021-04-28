# encoding=utf-8

import os

import matplotlib.pyplot as plt
import skimage.transform
from easydict import EasyDict as edict
from tqdm import tqdm

from main import Manager


def test_img_pairs():
    """
    :return:
    """
    dict_parameters_test = edict(
        {
            'val_data_dir': '/mnt/diskd/public/kitti_val',
            'model_path': './checkpoints/monodepth_resnet18_001.pth',
            'output_directory': '/mnt/diskd/public/kitti_output',
            'input_height': 256,
            'input_width': 512,
            'model': 'resnet18_md',
            'pretrained': False,
            'mode': 'test',  # test mode here
            'augment_parameters': None,
            'device': 'cuda:0',  # 'cuda:0'
            'input_channels': 3,
            'num_workers': 0,  # 4
            'batch_size': 1,
            'use_multiple_gpu': False
        }
    )
    model_test = Manager(dict_parameters_test)
    disps, disps_pp = model_test.test()

    # # Save a color image
    # disp_to_img = skimage.transform.resize(disps[10].squeeze(), [375, 1242], mode='constant')
    # plt.imshow(disp_to_img, cmap='plasma')
    # plt.imsave(os.path.join(dict_parameters_test.output_directory,
    #                         dict_parameters_test.model_path.split('/')[-1][:-4]+'_test_output.png'),
    #            disp_to_img,
    #            cmap='plasma')

    # Save all color images
    for i in tqdm(range(disps.shape[0])):
        disp_to_img = skimage.transform.resize(disps[i].squeeze(), [375, 1242], mode='constant')
        plt.imsave(os.path.join(dict_parameters_test.output_directory,
                                'pred_' + str(i) + '.png'),
                   disp_to_img,
                   cmap='plasma')
    print('Save color disparity images done.')

def test_disps():
    """
    :return:
    """


if __name__ == '__main__':
    test_img_pairs()