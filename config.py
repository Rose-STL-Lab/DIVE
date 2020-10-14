import argparse
import os
import numpy as np
import os
import random
import torch

from utils.logger import Logger
from utils.visualizer import Visualizer



is_train, split = None, None
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# hardware
parser.add_argument('--n_workers', type=int, default=8, help='number of threads')
parser.add_argument('--gpus', type=str, default='1', help='visible GPU ids, separated by comma')

# data
parser.add_argument('--dset_dir', type=str,  default='./')
parser.add_argument('--dset_name', type=str, default='moving_mnist')

# Input and Output length
parser.add_argument('--n_frames_input', type=int, default=10)
parser.add_argument('--n_frames_output', type=int, default=10)

# Components
parser.add_argument('--num_objects', type=int, nargs='+', default=[2], help='Max number of digits in Moving MNIST videos.')
parser.add_argument('--n_components', type=int, default=2)

# Dimensionality hyperparameters
parser.add_argument('--model', type=str, default='dive', help='Model name')
parser.add_argument('--image_latent_size', type=int, default=256,
                         help='Output size of image encoder')
parser.add_argument('--appearance_latent_size', type=int, default=128,
                         help='Size of appearance mixed vector and half of the size of the static appearance vector.')
parser.add_argument('--pose_latent_size', type=int, default=3,
                         help='Size of pose vector')
parser.add_argument('--hidden_size', type=int, default=64,
                         help='Size of the main hidden variables')
parser.add_argument('--var_app_size', type=int, default=48, help='Size of the varying appearance hidden representation')
parser.add_argument('--ngf', type=int, default=8,
                         help='number of channels in encoder and decoder')
parser.add_argument('--stn_scale_prior', type=float, default=3,
                         help='The scale of the spatial transformer prior.')

# ckpt and logging
parser.add_argument('--ckpt_dir', type=str,
                         default=os.path.join('./tensorboard', 'ckpt'),
                         help='directory for checkpoints and logs')
parser.add_argument('--ckpt_name', type=str, default='', help='checkpoint name')
parser.add_argument('--log_every', type=int, default=50, help='log every x steps')
parser.add_argument('--save_every', type=int, default=50, help='save every x epochs')
parser.add_argument('--evaluate_every', type=int, default=50, help='evaluate on val set every x epochs')

# Variations to moving MNIST
parser.add_argument('--image_size', type=int, default=[64, 64])
parser.add_argument('--crop_size', type=int, default=[64, 64], help='Visible size on the bottom right side of the frame')
parser.add_argument('--use_crop_size', type=bool, default=False, help='save every x epochs')
parser.add_argument('--num_missing', type=int, default=1, help='Number of timesteps with missing object per component')
parser.add_argument('--ini_et_alpha', type=int, default=100, help='Initial value alpha for the elastic transformation')

# Flags
parser.add_argument('--with_imputation', type=bool, default=True, help='Whether we use imputation')
parser.add_argument('--with_var_appearance', type=bool, default=True, help='Whether we use varying appearance')
parser.add_argument('--soft_labels', type=bool, default=False, help='Whether we use Soft-labels (True) or Hard-labels (False)')


# Training helpers
parser.add_argument('--gamma_p_app', type=float, default=0.3,
                    help='Probability of generating static-only appearance')
parser.add_argument('--gamma_p_imp', type=float, default=0.25,
                    help='Probability of imputing independently from the missing labels')
parser.add_argument('--gamma_switch_step', type=int, default=3e3,
                    help='How many iterations before reducing gamma of appearance.')

def parse(is_train):

    if is_train:

        parser.add_argument('--batch_size', type=int, default=64, help='batch size per gpu')
        parser.add_argument('--n_epochs', type=int, default=1000, help='total # of epochs')
        parser.add_argument('--n_iters', type=int, default=0, help='total # of iterations')
        parser.add_argument('--start_epoch', type=int, default=0, help='starting epoch')
        parser.add_argument('--lr_init', type=float, default=1e-3, help='initial learning rate')
        parser.add_argument('--lr_decay', type=int, default=1, choices=[0, 1], help='whether to decay learning rate')
        parser.add_argument('--load_ckpt_dir', type=str, default='', help='load checkpoint dir placeholder')
        parser.add_argument('--load_ckpt_epoch', type=int, default=0, help='epoch to load checkpoint')

    elif not is_train:

        # hyperparameters
        parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        parser.add_argument('--which_epochs', type=int, nargs='+', default=[-1],
                        help='which epochs to evaluate, -1 to load latest checkpoint')
        parser.add_argument('--save_visuals', type=int, default=0, help='Save results to tensorboard')
        parser.add_argument('--save_all_results', type=int, default=0, help='Save results to tensorboard')


    opt = parser.parse_args()

    if opt.dset_name == 'moving_mnist':
        opt.n_channels = 1
        opt.stn_scale_prior = 3
        opt.num_objects = [2]
        opt.n_components = 2
        if opt.crop_size[0] < opt.image_size[0] or opt.crop_size[1] < opt.image_size[1]:
            opt.use_crop_size = True

    else:
        # TODO: implement MOTSChallenge pedestrian
        raise NotImplementedError

    assert opt.n_frames_input > 0 and opt.n_frames_output > 0


    opt.dset_path = os.path.join(opt.dset_dir, opt.dset_name)

    if is_train:
        opt.is_train = True
        opt.split = 'train'


    elif not is_train:
        opt.is_train = False
        opt.split = 'val'

    ckpt_name = 'dive_2'
    opt.ckpt_name = ckpt_name
    opt.ckpt_path = os.path.join(opt.ckpt_dir, opt.dset_name, ckpt_name)

    # Logging
    log = ['Arguments: ']
    for k, v in sorted(vars(opt).items()):
        log.append('{}: {}'.format(k, v))

    return opt, log


def build(is_train, tb_dir=None):
  opt, log = parse(is_train)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus
  os.makedirs(opt.ckpt_path, exist_ok=True)

  # Set seed
  torch.manual_seed(666)
  torch.cuda.manual_seed_all(666)
  np.random.seed(666)
  random.seed(666)

  logger = Logger(opt.ckpt_path, opt.split)

  if tb_dir is not None:
    tb_path = os.path.join(opt.ckpt_path, tb_dir)
    vis = Visualizer(tb_path)
  else:
    vis = None

  logger.print(log)

  return opt, logger, vis
