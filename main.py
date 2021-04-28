# encoding=utf-8

import argparse
import os
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as pil

import torch
import torch.optim as optim
from tqdm import tqdm

from loss import MonodepthLoss
from utils import find_free_gpu, select_device
from utils import get_model, to_device, prepare_dataloader

# custom modules
# plot params
mpl.rcParams['figure.figsize'] = (15, 10)


def return_arguments():
    """
    :return:
    """
    parser = argparse.ArgumentParser(description='PyTorch Monodepth')

    parser.add_argument('--mode',
                        type=str,
                        default='train',  # train or test
                        help='mode: train or test (default: train)')

    parser.add_argument('--train_data_dir',
                        type=str,
                        default='/mnt/diskd/public/kitti_data/',
                        help='path to the dataset folder. \
                        It should contain subfolders with following structure: \
                        "image_02/data" for left images and \
                        "image_03/data" for right images')

    parser.add_argument('--val_data_dir',
                        type=str,
                        default='/mnt/diskd/public/kitti_val/',
                        help='path to the validation dataset folder. \
                            It should contain subfolders with following structure:\
                            "image_02/data" for left images and \
                            "image_03/data" for right images')

    parser.add_argument('--model_path',
                        default='./checkpoints/monodepth_resnet18.pth',
                        help='path to the trained model')
    parser.add_argument('--ckpt_path',
                        type=str,
                        default='./checkpoints/monodepth.pt')

    parser.add_argument('--is_resume',
                        type=bool,
                        default=False,  # True
                        help='Whether to resume from pre-trained checkpoint.')

    parser.add_argument('--output_directory',
                        default='/mnt/diskd/public/kitti_output',
                        help='where save dispairities\
                        for tested images')

    parser.add_argument('--input_height',
                        type=int,
                        default=320,  # 256, 320
                        help='input height')

    parser.add_argument('--input_width',
                        type=int,
                        default=1024,  # 512, 1024
                        help='input width')

    parser.add_argument('--model',
                        type=str,
                        default='resnet18',  # 'resnet18_md'
                        help='encoder architecture: ' +
                             'resnet18_md or resnet50_md ' + '(default: resnet18)'
                             + 'or torchvision version of any resnet model')

    parser.add_argument('--pretrained',
                        type=bool,
                        default=True,  # False
                        help='Use weights of pretrained model')

    parser.add_argument('--epochs',
                        type=int,
                        default=50,
                        help='number of total epochs to run')

    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-4,
                        help='initial learning rate (default: 1e-4)')

    parser.add_argument('--batch_size',
                        type=int,
                        default=12,  # 12, 24, 32, 256
                        help='mini-batch size (default: 256)')

    parser.add_argument('--adjust_lr',
                        type=bool,
                        default=True,
                        help='apply learning rate decay or not\
                        (default: True)')

    parser.add_argument('--device',
                        default='7',  # cuda:0
                        help='choose cpu or cuda:0 device')

    parser.add_argument('--do_augmentation',
                        type=bool,
                        default=True,
                        help='do augmentation of images or not')

    parser.add_argument('--augment_parameters',
                        type=list,
                        default=[0.8,
                                 1.2,
                                 0.5,
                                 2.0,
                                 0.8,
                                 1.2],
                        help='lowest and highest values for gamma, \
                        brightness and color respectively')

    parser.add_argument('--print_images',
                        type=bool,
                        default=False,
                        help='print disparity and image \
                        generated from disparity on every iteration')

    parser.add_argument('--print_weights',
                        type=bool,
                        default=False,
                        help='print weights of every layer')

    parser.add_argument('--input_channels',
                        type=int,
                        default=3,
                        help='Number of channels in input tensor')

    parser.add_argument('--num_workers',
                        type=int,
                        default=0,  # 4
                        help='Number of workers in dataloader')

    parser.add_argument('--use_multiple_gpu',
                        type=bool,
                        default=False)

    args = parser.parse_args()
    print(args)
    return args


def adjust_learning_rate(optimizer, epoch, learning_rate):
    """
    Sets the learning rate to the initial LR\
        decayed by 2 every 10 epochs after 30 epoches
    :param optimizer:
    :param epoch:
    :param learning_rate:
    :return:
    """
    if epoch >= 30 and epoch < 40:
        lr = learning_rate / 2
    elif epoch >= 40:
        lr = learning_rate / 4
    else:
        lr = learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def post_process_disparity(disp):
    """
    :param disp:
    :return:
    """
    (_, h, w) = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    (l, _) = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)

    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


class Manager:
    def __init__(self, args):
        """
        :param args:
        """
        self.args = args

        # Set up device
        device = str(find_free_gpu())
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        print('Using gpu: {:s}'.format(device))
        device = select_device(device='cpu' if not torch.cuda.is_available() else device)
        self.device = device

        # Set up network
        self.model = get_model(args.model,
                               input_channels=args.input_channels,
                               pretrained=args.pretrained)
        self.model = self.model.to(self.device)
        if args.use_multiple_gpu:
            self.model = torch.nn.DataParallel(self.model)

        if args.mode == 'train':
            # Define loss function
            self.loss_function = MonodepthLoss(n=4,
                                               SSIM_w=0.85,
                                               disp_gradient_w=0.1, lr_w=1).to(self.device)

            # Define optimizer
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
            # self.n_val_imgs, self.val_loader = prepare_dataloader(args.val_data_dir,
            #                                                       args.mode,
            #                                                       args.augment_parameters,
            #                                                       False,
            #                                                       args.batch_size,
            #                                                       (args.input_height, args.input_width),
            #                                                       args.num_workers)

            # Set up checkpoint
            if args.is_resume:
                if os.path.isfile(args.model_path):
                    self.model.load_state_dict(torch.load(args.model_path))
                    print('Resumed from {:s}.'.format(args.model_path))
                else:
                    print('Checkpoint file {:s} do not exists.'.format(args.model_path))
            else:
                print('Network is not resumed from a check point.')

        else:
            self.n_val_imgs, self.val_loader = prepare_dataloader(args.val_data_dir,
                                                                  args.mode,
                                                                  args.augment_parameters,
                                                                  False,
                                                                  args.batch_size,
                                                                  (args.input_height, args.input_width),
                                                                  args.num_workers)
            # Set up checkpoint
            if not os.path.isfile(args.model_path):
                print('[Err]: invalid model checkpoint file.')
                return

            self.model.load_state_dict(torch.load(args.model_path))
            print('Resumed from {:s}.'.format(args.model_path))

            # Set up test mode parameters
            args.augment_parameters = None
            args.do_augmentation = False
            args.batch_size = 1

        # Load data
        self.output_directory = args.output_directory
        if not os.path.isdir(self.output_directory):
            print('[Err]: invalid output directoty.')
            return

        self.input_height = args.input_height
        self.input_width = args.input_width

        if self.args.mode == 'train':
            self.n_train_imgs, self.loader = prepare_dataloader(args.train_data_dir,
                                                                args.mode,
                                                                args.augment_parameters,
                                                                args.do_augmentation,
                                                                args.batch_size,
                                                                (args.input_height, args.input_width),
                                                                args.num_workers)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def train(self, loss_freq=10, save_freq=500):
        """
        :return:
        """
        losses = []
        val_losses = []
        best_loss = float('Inf')
        best_val_loss = float('Inf')

        ## evaluation mode
        # self.model.eval()

        # self.running_val_loss = 0.0
        # for data in self.val_loader:
        #     data = to_device(data, self.device)
        #     left = data['left_image']
        #     right = data['right_image']
        #     disps = self.model.forward(left)
        #     loss = self.loss_function(disps, [left, right])
        #     val_losses.append(loss.item())
        #     self.running_val_loss += loss.item()
        # self.running_val_loss /= self.n_val_imgs / self.args.batch_size
        # print('Val_loss:', self.running_val_loss)

        self.avg_loss_list = []
        for epoch in range(self.args.epochs):
            if self.args.adjust_lr:
                adjust_learning_rate(self.optimizer, epoch, self.args.learning_rate)

            c_time = time.time()

            ## train mode
            self.model.train()

            self.running_epoch_loss = 0.0
            self.running_batch_loss = 0.0
            self.avg_loss = 0.0
            for batch_idx, (data, left_img_path) in tqdm(enumerate(self.loader)):
                # Load data
                data = to_device(data, self.device)
                left = data['left_image']
                right = data['right_image']

                # One optimization iteration
                self.optimizer.zero_grad()
                disps = self.model.forward(left)
                loss = self.loss_function(disps, [left, right])
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

                # Print statistics
                if self.args.print_weights:
                    j = 1
                    for (name, parameter) in self.model.named_parameters():
                        if name.split(sep='.')[-1] == 'weight':
                            plt.subplot(5, 9, j)
                            plt.hist(parameter.data.view(-1))
                            plt.xlim([-1, 1])
                            plt.title(name.split(sep='.')[0])
                            j += 1
                    plt.show()

                ## TODO: save intermidiate results: the synthesized images...
                if self.args.print_images:
                    print('disp_left_est[0]')
                    disp_left_est = self.loss_function.disp_left_est[0][0, :, :, :].cpu().detach().numpy()
                    plt.imshow(np.squeeze(np.transpose(disp_left_est, (1, 2, 0))))
                    plt.show()

                    print('left_est[0]')
                    left_est = self.loss_function.left_est[0][0, :, :, :].cpu().detach().numpy()
                    plt.imshow(np.transpose(left_est, (1, 2, 0)))
                    plt.show()

                    print('disp_right_est[0]')
                    disp_right_est = self.loss_function.disp_right_est[0][0, :, :, :].cpu().detach().numpy()
                    plt.imshow(np.squeeze(np.transpose(disp_right_est, (1, 2, 0))))
                    plt.show()

                    print('right_est[0]')
                    right_est = self.loss_function.right_est[0][0, :, :, :].cpu().detach().numpy()
                    plt.imshow(np.transpose(right_est, (1, 2, 0)))
                    plt.show()

                # @even: loss
                self.running_batch_loss += loss.item()
                self.running_epoch_loss += loss.item()

                if batch_idx != 0 and (batch_idx + 1) % loss_freq == 0:
                    print('Epoch {:03d} | Batch {:05d}/{:05d} | Average loss of the last {:d} batches: {:5.3f}.'
                          .format(epoch,
                                  batch_idx + 1,
                                  len(self.loader),
                                  loss_freq,
                                  self.running_batch_loss / loss_freq))
                    self.running_batch_loss = 0.0

                if batch_idx != 0 and (batch_idx + 1) % save_freq == 0:
                    # Save checkpoint
                    save_ckpt_path = self.args.model_path[:-4] + '_epoch{:d}_cpt.pth'.format(epoch)
                    self.save(save_ckpt_path)
                    print('{:s} saved.'.format(save_ckpt_path))

            # Epoch avg loss for a batch
            self.avg_loss = self.running_epoch_loss / float(len(self.loader))
            print('Average loss of epoch{:d}: {:.3f}.'.format(epoch, self.avg_loss))
            self.avg_loss_list.append(self.avg_loss)
            print('Average loss list:\n', self.avg_loss_list)

            # Reset loss statistics
            self.running_epoch_loss = 0.0
            self.avg_loss = 0.0

            ## evaluation mode
            self.model.eval()

            # self.running_val_loss = 0.0
            # for data in self.val_loader:
            #     data = to_device(data, self.device)
            #     left = data['left_image']
            #     right = data['right_image']
            #     disps = self.model(left)
            #     loss = self.loss_function(disps, [left, right])
            #     val_losses.append(loss.item())
            #     self.running_val_loss += loss.item()

            # Estimate loss per image
            # self.running_loss /= self.n_train_imgs / self.args.batch_size
            # self.running_val_loss /= self.n_val_imgs / self.args.batch_size

            print('Epoch:', epoch + 1,
                  'train_loss:', self.avg_loss,
                  # 'val_loss:', self.running_val_loss,
                  'time:', round(time.time() - c_time, 3), 's')

            self.save(self.args.model_path[:-4] + '_last.pth')

            # if self.running_val_loss < best_val_loss:
            #     self.save(self.args.model_path[:-4] + '_cpt.pth')
            #     best_val_loss = self.running_val_loss
            #     print('Model checkpoint saved')

            self.save(self.args.model_path[:-4] + '_epoch{:d}_cpt.pth'.format(epoch))
            # best_val_loss = self.running_val_loss
            print('Model checkpoint saved')

        self.avg_loss_list = []
        print('Finished Training. Best loss:', best_loss)
        self.save(self.args.model_path)

    def test(self):
        """
        :return:
        """
        print('Net input width: {:d}.'.format(self.args.input_width))

        # Set evaluation mode
        self.model.eval()

        disparities = np.zeros((self.n_val_imgs, self.input_height, self.input_width),
                               dtype=np.float32)
        disparities_pp = np.zeros((self.n_val_imgs, self.input_height, self.input_width),
                                  dtype=np.float32)

        with torch.no_grad():
            for i, (data, left_img_path) in tqdm(enumerate(self.val_loader)):
                # Get the inputs
                data = to_device(data, self.device)
                left = data.squeeze()

                # Do a forward pass
                disps = self.model.forward(left)
                disp = disps[0][:, 0, :, :].unsqueeze(1)
                disparities[i] = disp[0].squeeze().cpu().numpy()  # only need the left disparity

                # Get disparities after post processing
                disparities_pp[i] = post_process_disparity(disps[0][:, 0, :, :].cpu().numpy())

                # @even: Scale back from [0, 1] to metric disparity values
                disp = disparities[i]
                disp_pp = disparities_pp[i]
                disp *= float(self.args.input_width)
                disp_pp *= float(self.args.input_width)
                disparities[i] = disp
                disparities_pp[i] = disp_pp

                # @even: save each stereo pair's disparity independantly
                img_name = os.path.split(left_img_path[0])[-1].replace('.jpg', '')
                disp_npy_path = self.output_directory + '/{:s}_disp.npy'.format(img_name)
                disp_pp_npy_path = self.output_directory + '/{:s}_disp_pp.npy'.format(img_name)
                np.save(disp_npy_path, disparities[i])
                np.save(disp_pp_npy_path, disparities[i])
                print('{:s} saved.'.format(disp_npy_path))
                print('{:s} saved.'.format(disp_pp_npy_path))

                # @even: save each stereo pair's disparity image independantly
                disp_img_path = self.output_directory + '/{:s}_disp.png'.format(img_name)
                disp_pp_img_path = self.output_directory + '/{:s}_disp_pp.png'.format(img_name)
                disp_img = pil.fromarray(disp.astype('uint16'))
                disp_pp_img = pil.fromarray(disp_pp.astype('uint16'))
                disp_img.save(disp_img_path)
                disp_pp_img.save(disp_pp_img_path)
                print('{:s} saved.'.format(disp_img_path))
                print('{:s} saved.'.format(disp_pp_img_path))

        disps_f_path = self.output_directory + '/disparities.npy'
        disps_pp_f_path = self.output_directory + '/disparities_pp.npy'
        np.save(disps_f_path, disparities)
        np.save(disps_pp_f_path, disparities_pp)
        print('{:s} saved.'.format(disps_f_path))
        print('{:s} saved.'.format(disps_pp_f_path))
        print('Finished Testing.')

        return disparities, disparities_pp

    def save(self, path):
        """
        :param path:
        :return:
        """
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """
        :param path:
        :return:
        """
        self.model.load_state_dict(torch.load(path))


def run():
    """
    :return:
    """
    args = return_arguments()
    manager = Manager(args)

    if args.mode == 'train':
        manager.train()
    elif args.mode == 'test':
        manager.test()


def test_disparity():
    """
    :return:
    """
    output_dir = '/mnt/diskd/public/kitti_data/output/'
    disp_f_path = '/mnt/diskd/public/kitti_data/output/disparities_pp.npy'
    if not os.path.isfile(disp_f_path):
        print('[Err]: invalid disparity file.')
        return

    disps = np.load(disp_f_path)
    for i in tqdm(range(disps.shape[0])):
        disp = disps[i].squeeze()
        disp_npy_f_path = output_dir + '/{:010d}.npy'.format(i)
        np.save(disp_npy_f_path, disp)
        # disp.tofile(disp_npy_f_path)

        # disp_to_img = skimage.transform.resize(disp, [375, 1242], mode='constant')
        # plt.imsave(os.path.join(output_dir, 'pred_' + str(i) + '.png'), disp_to_img, cmap='plasma')


if __name__ == '__main__':
    run()
    test_disparity()
