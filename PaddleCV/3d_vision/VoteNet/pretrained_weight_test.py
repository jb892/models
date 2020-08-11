#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
"""
load pytorch pretrained weight and inference using paddle model
"""

import os
import sys
import time
import shutil
import argparse
import ast
import logging
import numpy as np
import h5py

import paddle.fluid as fluid
import paddle.fluid.layers as layers

from models import *
from utils import *
from data.scannet_reader import ScannetDetectionReader
from models.ap_helper import APCalculator, parse_groundtruths, parse_predictions

logging.root.handlers = []
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("VoteNet Load pytorch trained weight and inference script")
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='training batch size, default 1')
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
        '--weight_file',
        type=str,
        default='votenet_weight.h5',
        help='model param weight train by pytorch')
    parser.add_argument(
        '--use_height',
        type=bool,
        default=True,
        help='Use height signal in input.')
    parser.add_argument(
        '--use_color',
        type=bool,
        default=False,
        help='Use RGB color in input.')
    parser.add_argument(
        '--mode',
        type=str,
        default='infer',
        help='Set demo script mode to eval or infer, default: infer')
    args = parser.parse_args()
    return args

# ----------------------------------------
# Point Cloud Sampling
# ----------------------------------------

def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if replace is None: replace = (pc.shape[0]<num_sample)
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]

def preprocess_point_cloud(point_cloud, num_points):
    ''' Prepare the numpy point cloud (N,3) for forward pass '''
    point_cloud = point_cloud[:, 0:3] # do not use color for now
    floor_height = np.percentile(point_cloud[:, 2], 0.99)
    height = point_cloud[:, 2] - floor_height
    point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1)  # (N,4) or (N,7)
    point_cloud = random_sampling(point_cloud, num_points)
    pc = np.expand_dims(point_cloud.astype(np.float32), 0)  # (1,20000,4)
    return pc

def eval_one_epoch():
    # TODO: complete this eval function
    pass

def demo():

    # ==== Parse arguments ====
    args = parse_args()
    print_arguments(args)

    # check whether the installed paddle is compiled with GPU
    check_gpu(args.use_gpu)

    # Used for AP calculation
    DATASET_CONFIG = ScannetDatasetConfig()

    CONFIG_DICT = {
        'remove_empty_box': False,
        'use_3d_nms': True,
        'nms_iou': 0.25,
        'use_old_type_nms': False,
        'cls_nms': True,
        'per_class_proposal': True,
        'conf_thresh': 0.05,
        'batch_size': args.batch_size,
        'dataset_config': DATASET_CONFIG
    }

    logger.info('==== Start building inference model ====')
    startup = fluid.Program()
    infer_prog = fluid.Program()

    INPUT_FEATURE_DIM = int(args.use_color) * 3 + int(args.use_height) * 1

    with fluid.program_guard(infer_prog, startup):
        with fluid.unique_name.guard():
            infer_model = VoteNet(
                num_class=args.num_classes,
                num_points=args.num_points,
                num_heading_bin=args.num_heading_bin,
                num_size_cluster=args.num_size_cluster,
                num_proposal=args.num_target,
                input_feature_dim=INPUT_FEATURE_DIM,
                vote_factor=args.vote_factor,
                sampling=args.cluster_sampling,
                batch_size=args.batch_size,
            )

            infer_model.build_input()
            infer_model.build()
            infer_outputs = infer_model.get_outputs('test')
            infer_loader = infer_model.get_loader()

    logger.info('==== Model built ====')

    infer_prog = infer_prog.clone(True)
    infer_keys, infer_values = parse_outputs(infer_outputs)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup)

    logger.info('==== Load weight params from {} ===='.format(args.weight_file))

    weight_dict = {}
    with h5py.File(args.weight_file, 'r') as f:
        for key in f.keys():
            # print(key)
            weight_dict[key] = np.array(f[key], dtype=np.float32)
        f.close()

    for block in infer_prog.blocks:
        for param in block.all_parameters():
            if param.name not in weight_dict:
                logger.info('{} is not in weight dict!'.format(param.name))
                continue

            pd_var = fluid.global_scope().find_var(param.name)
            pd_param = pd_var.get_tensor()
            logger.info("load: {}, shape: {}".format(param.name, param.shape))
            logger.info("Before setting the numpy array value: {}".format(np.array(pd_param).ravel()[:5]))
            # logger.info("Setting numpy array value: {}".format(weight_dict[param.name].ravel()[:5]))
            pd_param.set(weight_dict[param.name], place)
            logger.info("After setting the numpy array value: {}".format(np.array(pd_param).ravel()[:5]))

    logger.info('==== Weight restored ====')

    # ==== TODO: Load indoor scene ply model ====

    # ==== TODO: Preprocessing ====

    # ==== TODO: Inference ====

    # ==== TODO: Dump result ====


if __name__ == '__main__':
    demo()