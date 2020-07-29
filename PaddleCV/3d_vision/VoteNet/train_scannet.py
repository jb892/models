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
from data.scannet_reader import ScannetDetectionReader
from utils import *

logging.root.handlers = []
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("VoteNet 3D Object Detection train script")
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
        default=20000,
        help='number of points in a sample, default: 20000')
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
        '--num_heading_bin',
        type=int,
        default=1,
        help='number of heading bin, default: 1')
    parser.add_argument(
        '--num_size_cluster',
        type=int,
        default=18,
        help='number of size of cluster, default: 18')
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
        '--dump_dir',
        type=str,
        default=None,
        help='Dump dir to save sample outputs [default: None]')
    parser.add_argument(
        '--epoch',
        type=int,
        default=181,
        help='epoch number. default 181.')
    parser.add_argument(
        '--data_dir',
        type=str,
        # default='dataset/Indoor3DSemSeg/indoor3d_sem_seg_hdf5_data',
        default='data',
        help='dataset directory')
    parser.add_argument(
        '--mean_size_arr_path',
        type=str,
        default='../dataset/scannet/meta_data/scannet_means.npz',
        help='ScanNet mean npy file path')
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

def train():
    args = parse_args()
    print_arguments(args)
    # check whether the installed paddle is compiled with GPU
    check_gpu(args.use_gpu)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    # build model
    if args.enable_ce:
        SEED = 102
        fluid.default_main_program().random_seed = SEED
        framework.default_startup_program().random_seed = SEED

    # Build train model
    startup = fluid.program()
    train_prog = fluid.program()

    NUM_INPUT_FEATURE_CHANNEL = int(args.use_color) * 3 + int(not args.no_height) * 1

    with fluid.program_guard(train_prog, startup):
        with fluid.unique_name.guard():
            train_model = VoteNet(
                num_class=args.num_classes,
                num_points=args.num_points,
                num_heading_bin=args.num_heading_bin,
                num_size_cluster=args.num_size_cluster,
                num_proposal=args.num_target,
                input_feature_dim=NUM_INPUT_FEATURE_CHANNEL,
                vote_factor=args.vote_factor,
                sampling=args.cluster_sampling
            )

            train_model.build()
            train_feeds = train_model.get_feeds()
            train_outputs = train_model.get_outputs()
            train_loader = train_model.get_loader()
            train_loss = train_outputs['loss']
            lr = layers.exponential_decay(
                learning_rate=args.lr,
                decay_steps=args.decay_steps,
                decay_rate=args.lr_decay,
                staircase=True
            )
            lr = layers.clip(lr, 1e-5, args.lr)
            params = []
            for var in train_prog.list_vars():
                if fluid.io.is_parameter(var):
                    params.append(var.name)
            optimizer = fluid.optimizer.Adam(
                learning_rate=lr,
                regularization=fluid.regularizer.L2Decay(args.weight_decay)
            )
            optimizer.minimize(train_loss, parameter_list=params)
    train_keys, train_values = parse_outputs(train_outputs)

    # Build test model
    test_prog = fluid.Program()
    with fluid.program_guard(test_prog, startup):
        with fluid.unique_name.guard():
            test_model = VoteNet(
                num_class=args.num_classes,
                num_points=args.num_points,
                num_heading_bin=args.num_heading_bin,
                num_size_cluster=args.num_size_cluster,
                num_proposal=args.num_target,
                input_feature_dim=NUM_INPUT_FEATURE_CHANNEL,
                vote_factor=args.vote_factor,
                sampling=args.cluster_sampling
            )
            test_model.build()
            test_feeds = test_model.get_feeds()
            test_outputs = test_model.get_outputs()
            test_loader = test_model.get_loader()
    test_prog = test_prog.clone(True)
    test_keys, test_values = parse_outputs(test_outputs)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup)

    if args.resume:
        if not os.path.isdir(args.resume):
            assert os.path.exists("{}.pdparams".format(args.resume)), \
                "Given resume weight {}.pdparams not exist.".format(args.resume)
            assert os.path.exists("{}.pdopt".format(args.resume)), \
                "Given resume optimizer state {}.pdopt not exist.".format(args.resume)
        fluid.load(train_prog, args.resume, exe)

    build_strategy = fluid.BuildStrategy()
    build_strategy.memory_optimize = False
    build_strategy.enable_inplace = False
    build_strategy.fuse_all_optimizer_ops = False
    train_compile_prog = fluid.compiler.CompiledProgram(
        train_prog).with_data_parallel(loss_name=train_loss.name,
                                       build_strategy=build_strategy)
    test_compile_prog = fluid.compiler.CompiledProgram(test_prog)

    def save_model(exe, prog, path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        logger.info("Save model to {}".format(path))
        fluid.save(prog, path)

    # Get reader
    scannet_reader_tr = ScannetDetectionReader(args.data_dir, args.mean_size_arr_path, mode='train')
    scannet_reader_te = ScannetDetectionReader(args.data_dir, args.mean_size_arr_path, mode='test')
    train_reader = scannet_reader_tr.get_reader(args.batch_size, args.num_points)
    test_reader = scannet_reader_te.get_reader(args.batch_size, args.num_points)
    train_loader.set_sample_list_generator(train_reader, place)
    test_loader.set_sample_list_generator(test_reader, place)

    train_stat = Stat()
    test_stat = Stat()

    ce_time = 0
    ce_loss = []

    for epoch_id in range(args.epoch):
        try:
            train_loader.start()
            train_iter = 0
            train_periods = []

            while True:
                cur_time = time.time()
                train_outs = exe.run(train_compile_prog, fetch_list=train_values + [lr.name])
                period = time.time() - cur_time
                train_periods.append(period)
                train_stat.update(train_keys, train_outs[:-1])
                if train_iter % args.log_interval == 0:
                    log_str = ""
                    for name, values in zip(train_keys + ['learning_rate'], train_outs):
                        log_str += "{}: {:.5f}, ".format(name, np.mean(values))
                        if name == 'loss':
                            ce_loss.append(np.mean(values))
                    logger.info(
                        "[TRAIN] Epoch {}, batch {}: {}time: {:.2f}".format(epoch_id, train_iter, log_str, period))
                train_iter += 1

        except fluid.core.EOFException:
            logger.info("[TRAIN] Epoch {} finished, {}average time: {:.2f}".format(epoch_id, train_stat.get_mean_log(),
                                                                                   np.mean(train_periods[1:])))
            ce_time = np.mean(train_periods[1:])
            save_model(exe, train_prog, os.path.join(args.save_dir, str(epoch_id), "votenet_det"))

            # evaluation
            if not args.enable_ce:
                try:
                    test_loader.start()
                    test_iter = 0
                    test_periods = []
                    while True:
                        cur_time = time.time()
                        test_outs = exe.run(test_compile_prog, fetch_list=test_values)
                        period = time.time() - cur_time
                        test_periods.append(period)
                        test_stat.update(test_keys, test_outs)
                        if test_iter % args.log_interval == 0:
                            log_str = ""
                            for name, value in zip(test_keys, test_outs):
                                log_str += "{}: {:.4f}, ".format(name, np.mean(value))
                            logger.info("[TEST] Epoch {}, batch {}: {}time: {:.2f}".format(epoch_id, test_iter, log_str,
                                                                                           period))
                        test_iter += 1
                except fluid.core.EOFException:
                    logger.info(
                        "[TEST] Epoch {} finished, {}average time: {:.2f}".format(epoch_id, test_stat.get_mean_log(),
                                                                                  np.mean(test_periods[1:])))
                finally:
                    test_loader.reset()
                    test_stat.reset()
                    test_periods = []

        finally:
            train_loader.reset()
            train_stat.reset()
            train_periods = []

    # only for ce
    if args.enable_ce:
        card_num = get_cards()
        _loss = 0
        _time = 0
        try:
            _time = ce_time
            _loss = np.mean(ce_loss[1:])
        except:
            print("ce info error")
        print("kpis\ttrain_seg_%s_duration_card%s\t%s" % (args.model, card_num, _time))
        print("kpis\ttrain_seg_%s_loss_card%s\t%f" % (args.model, card_num, _loss))

def get_cards():
    num = 0
    cards = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cards != '':
        num = len(cards.split(","))
    return num

if __name__ == '__main__':
    train()