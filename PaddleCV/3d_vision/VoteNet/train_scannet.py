#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import time
import shutil
import argparse
import ast
import logging
import numpy as np

import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.framework as framework

from models import *
from data.scannet_reader import ScannetReader
from utils import *

logging.root.handlers = []
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("PointNet++ semantic segmentation train script")
    parser.add_argument(
        '--model',
        type=str,
        default='MSG',
        help='SSG or MSG model to train, default MSG')
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='training batch size, default 16')
    parser.add_argument(
        '--num_points',
        type=int,
        default=8192,
        help='number of points in a sample, default: 8192')
    parser.add_argument(
        '--num_target',
        type=int,
        default=256,
        help='Proposal number [default: 256]')
    parser.add_argument(
        '--vote_factor',
        type=int,
        default=1,
        help='Vote factor [default: 1]')
    parser.add_argument(
        '--cluster_sampling',
        type=str,
        default='vote_fps',
        help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
    parser.add_argument(
        '--ap_iou_thresh',
        type=float,
        default=0.25,
        help='AP IoU threshold [default: 0.25]')
    parser.add_argument(
        '--num_classes',
        type=int,
        default=18,
        help='number of classes in dataset, default: 18')
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='initial learning rate, default 0.001')
    parser.add_argument(
        '--lr_decay',
        type=float,
        default=0.5,
        help='learning rate decay gamma, default 0.5')
    parser.add_argument(
        '--lr_decay_steps',
        type=str,
        default='80,120,160',
        help='When to decay the learning rate (in epochs) [default: 80,120,160]')
    parser.add_argument(
        '--lr_decay_rates',
        type=str,
        default='0.1,0.1,0.1',
        help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
    parser.add_argument(
        '--no_height',
        action='store_true',
        help='Do NOT use height signal in input.')
    parser.add_argument(
        '--use_color',
        action='store_true',
        help='Use RGB color in input.')
    parser.add_argument(
        '--bn_momentum',
        type=float,
        default=0.99,
        help='initial batch norm momentum, default 0.99')
    parser.add_argument(
        '--decay_steps',
        type=int,
        default=20,
        help='learning rate and batch norm momentum decay steps, default 20')
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.,
        help='L2 regularization weight decay coeff, default 0.')
    parser.add_argument(
        '--dump_results',
        action='store_true',
        help='Dump results.')
    parser.add_argument(
        '--epoch',
        type=int,
        default=201,
        help='epoch number. default 201.')
    parser.add_argument(
        '--data_dir',
        type=str,
        # default='dataset/Indoor3DSemSeg/indoor3d_sem_seg_hdf5_data',
        default='data',
        help='dataset directory')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='checkpoints_seg',
        help='directory name to save train snapshoot')
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='path to resume training based on previous checkpoints. '
             'None for not resuming any checkpoints.')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='mini-batch interval for logging.')
    parser.add_argument(
        '--enable_ce',
        action='store_true',
        help='The flag indicating whether to run the task '
             'for continuous evaluation.')
    args = parser.parse_args()
    return args

# parser = argparse.ArgumentParser()
# parser.add_argument('--model', default='votenet', help='Model file name [default: votenet]')
# parser.add_argument('--dataset', default='scannet', help='Dataset name. [default: scannet]')
# parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
# parser.add_argument('--log_dir', default='log', help='Dump dir to save model checkpoint [default: log]')
# parser.add_argument('--dump_dir', default=None, help='Dump dir to save sample outputs [default: None]')
# parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
# parser.add_argument('--num_target', type=int, default=256, help='Proposal number [default: 256]')
# parser.add_argument('--vote_factor', type=int, default=1, help='Vote factor [default: 1]')
# parser.add_argument('--cluster_sampling', default='vote_fps', help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
# parser.add_argument('--ap_iou_thresh', type=float, default=0.25, help='AP IoU threshold [default: 0.25]')
# parser.add_argument('--max_epoch', type=int, default=180, help='Epoch to run [default: 180]')
# parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
# parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
# parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
# parser.add_argument('--bn_decay_step', type=int, default=20, help='Period of BN decay (in epochs) [default: 20]')
# parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
# parser.add_argument('--lr_decay_steps', default='80,120,160', help='When to decay the learning rate (in epochs) [default: 80,120,160]')
# parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
# parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
# parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
# parser.add_argument('--use_sunrgbd_v2', action='store_true', help='Use V2 box labels for SUN RGB-D dataset')
# parser.add_argument('--overwrite', action='store_true', help='Overwrite existing log and dump folders.')
# parser.add_argument('--dump_results', action='store_true', help='Dump results.')
# FLAGS = parser.parse_args()

# ----------- GLOBAL CONFIG -------------
# BATCH_SIZE = FLAGS.batch_size
# NUM_POINT = FLAGS.num_point
# MAX_EPOCH = FLAGS.max_epoch
# BASE_LEARNING_RATE = FLAGS.learning_rate
# BN_DECAY_STEP = FLAGS.bn_decay_step
# BN_DECAY_RATE = FLAGS.bn_decay_rate
# LR_DECAY_STEPS = [int(x) for x in FLAGS.lr_decay_steps.split(',')]
# LR_DECAY_RATES = [float(x) for x in FLAGS.lr_decay_rates.split(',')]
# assert(len(LR_DECAY_STEPS)==len(LR_DECAY_RATES))
# LOG_DIR = FLAGS.log_dir
# DEFAULT_DUMP_DIR = os.path.join(BASE_DIR, os.path.basename(LOG_DIR))
# DUMP_DIR = FLAGS.dump_dir if FLAGS.dump_dir is not None else DEFAULT_DUMP_DIR
# DEFAULT_CHECKPOINT_PATH = os.path.join(LOG_DIR, 'checkpoint.tar')
# CHECKPOINT_PATH = FLAGS.checkpoint_path if FLAGS.checkpoint_path is not None \
#     else DEFAULT_CHECKPOINT_PATH
# FLAGS.DUMP_DIR = DUMP_DIR


def train():
    args = parse_args()
    print_arguments(args)
    # check whether the installed paddle is compiled with GPU
    check_gpu(args.use_gpu)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    assert args.model in ['MSG', 'SSG'], "--model can only be 'MSG' or 'SSG'"

    # build model
    if args.enable_ce:
        SEED = 102
        fluid.default_main_program().random_seed = SEED
        framework.default_startup_program().random_seed = SEED

    startup = fluid.program()
    train_prog = fluid.program()

    with fluid.program_guard(train_prog, startup):
        with fluid.unique_name.guard():
            train_model = VoteNet(
                num_class=args.num_classes
            )


def get_cards():
    num = 0
    cards = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cards != '':
        num = len(cards.split(","))
    return num

if __name__ == '__main__':
    train()