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
    args = parser.parse_args()
    return args

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

    # ==== Build train model ====
    startup = fluid.Program()
    infer_prog = fluid.Program()

    with fluid.program_guard(infer_prog, startup):
        with fluid.unique_name.guard():
            infer_model = VoteNet(
                num_class=args.num_classes,
                num_points=args.num_points,
                num_heading_bin=args.num_heading_bin,
                num_size_cluster=args.num_size_cluster,
                num_proposal=args.num_target,
                input_feature_dim=1,
                vote_factor=args.vote_factor,
                sampling=args.cluster_sampling,
                batch_size=args.batch_size,
            )

            infer_model.build_input()
            infer_model.build()
            infer_outputs = infer_model.get_outputs('test')
            infer_loader = infer_model.get_loader()
    infer_prog = infer_prog.clone(True)
    infer_keys, infer_values = parse_outputs(infer_outputs)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup)

    # ==== Load weight param and restore ====
    # weight_dict = {}
    # with h5py.File(args.weight_file, 'r') as f:
    #     for key in f.keys():
    #         print(key)
    #         weight_dict[key] = np.array(f[key], dtype=np.float32)
    #     f.close()

    for block in infer_prog.blocks:
        for param in block.all_parameters():
            print(param.name)
            pd_var = fluid.global_scope().find_var(param.name)
            pd_param = pd_var.get_tensor()
            print(np.array(pd_param))
            break

            # pd_var = fluid.global_scope().find_var(param.name)
            # pd_param = pd_var.get_tensor()
            # print("load: {}, shape: {}".format(param.name, param.shape))
            # print("Before setting the numpy array value: {}".format(np.array(pd_param).ravel()[:5]))
            # pd_param.set(np.ones(param.shape), place)
            # print("After setting the numpy array value: {}".format(np.array(pd_param).ravel()[:5]))




if __name__ == '__main__':
    demo()