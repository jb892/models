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

np.random.seed(1024)

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
        default=8,
        help='training batch size, default 8')
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
        '--use_height',
        type=bool,
        default=True,
        help='Use height signal in input.')
    parser.add_argument(
        '--use_color',
        type=bool,
        default=True,
        help='Use RGB color in input.')
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
        '--data_dir',
        type=str,
        default='data',
        help='dataset directory')
    parser.add_argument(
        '--mean_size_arr_path',
        type=str,
        default='../dataset/scannet/meta_data/scannet_means.npz',
        help='ScanNet mean npy file path')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=100,
        help='mini-batch interval for logging.')
    parser.add_argument(
        '--weights',
        type=str,
        default='checkpoints_seg_2020_Aug_4/96/votenet_det',
        help='directory name to save train snapshoot')
    args = parser.parse_args()
    return args

def eval():
    args = parse_args()
    print_arguments(args)
    # check whether the installed paddle is compiled with GPU
    check_gpu(args.use_gpu)

    # Build model
    startup = fluid.Program()
    eval_prog = fluid.Program()

    NUM_INPUT_FEATURE_CHANNEL = int(args.use_color) * 3 + int(args.use_height) * 1

    with fluid.program_guard(eval_prog, startup):
        with fluid.unique_name.guard():
            eval_model = VoteNet(num_class=args.num_classes,
                                num_points=args.num_points,
                                num_heading_bin=args.num_heading_bin,
                                num_size_cluster=args.num_size_cluster,
                                num_proposal=args.num_target,
                                input_feature_dim=NUM_INPUT_FEATURE_CHANNEL,
                                vote_factor=args.vote_factor,
                                sampling=args.cluster_sampling,
                                batch_size=args.batch_size)
            eval_model.build_input()
            eval_model.build()
            eval_feeds = eval_model.get_feeds()
            eval_outputs = eval_model.get_outputs()
            eval_loader = eval_model.get_loader()
    eval_prog = eval_prog.clone(True)
    eval_keys, eval_values = parse_outputs(eval_outputs)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup)

    if not os.path.isdir(args.weights):
        assert os.path.exists("{}.pdparams".format(args.weights)), \
            "Given resume weight {}.pdparams not exist.".format(args.weights)
    fluid.load(eval_prog, args.weights, exe)

    eval_compile_prog = fluid.compiler.CompiledProgram(eval_prog)

    # get reader
    scannet_reader = ScannetDetectionReader(num_points=args.num_points,
                                           use_color=args.use_color,
                                           use_height=args.use_height,
                                           augment=True,
                                           mode='test')
    eval_reader = scannet_reader.get_reader(args.batch_size)
    eval_loader.set_sample_list_generator(eval_reader, place)

    eval_stat = Stat()

    try:
        eval_loader.start()
        eval_iter = 0
        eval_periods = []
        eval_point_set = []
        eval_pred = []

        while True:
            cur_time = time.time()
            eval_outs = exe.run(eval_compile_prog, fetch_list=eval_values)
            period = time.time() - cur_time
            eval_periods.append(period)
            eval_stat.update(eval_keys, eval_outs)
            if eval_iter % args.log_interval == 0:
                log_str = ""
                for name, value in zip(eval_keys, eval_outs):
                    if name == 'pred':
                        eval_pred.append(value)
                        continue
                    elif name == 'xyz':
                        eval_point_set.append(value)
                        continue

                    log_str += "{}: {:.4f}, ".format(name, np.mean(value))
                logger.info("[EVAL] batch {}: {}time: {:.2f}".format(eval_iter, log_str, period))
            eval_iter += 1

    except fluid.core.EOFException:
        logger.info("[EVAL] Eval finished, {}average time: {:.2f}".format(eval_stat.get_mean_log(),
                                                                          np.mean(eval_periods[0:])))
    finally:
        eval_loader.reset()

    # Save resulting scene in color ply
    if args.dump_results:
        # Calculate the extra zero-valued batchs
        extra_num_points = scannet_reader.extra_zero_batch_lens * args.num_points

        # Convert list to np.array and reshape to shape=[tot_num_point, 3]. Then only take preds that's not in extra-batch
        eval_pred = np.array(eval_pred).reshape((-1, 1))[:-extra_num_points]
        eval_point_set = np.array(eval_point_set).reshape((-1, 3))[:-extra_num_points]

        logger.info('eval_pred.shape: {}'.format(eval_pred.shape))
        logger.info('eval_point_set.shape: {}'.format(eval_point_set.shape))

        # Export Scene in ply file.
        color_label_list = convert_pred_to_color(eval_pred)
        export_eval_scene_ply(eval_point_set, scene_name, color_label_list)


if __name__ == "__main__":
    eval()