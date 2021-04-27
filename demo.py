# encoding=utf-8

import os
import torch
import numpy as np
import skimage.transform
import matplotlib.pyplot as plt
from easydict import EasyDict as edict

from main import Manager
from tqdm import tqdm

dict_parameters_test = edict(
    {
        'data_dir': '/mnt/diskd/public/kitti_val',
        'model_path': './checkpoints/monodepth_resnet18_001_cpt.pth',
        'output_directory': '/mnt/diskd/public/kitti_output',
        'input_height': 256,
        'input_width': 512,
        'model': 'resnet18',
        'pretrained': False,
        'mode': 'test',
        'device': 'cuda:0',  # 'cuda:0'
        'input_channels': 3,
        'num_workers': 4,
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
               'pred_'+str(i)+'.png'), disp_to_img, cmap='plasma')


