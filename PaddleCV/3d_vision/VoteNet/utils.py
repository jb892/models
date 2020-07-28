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
Contains paddle utility functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import six
import logging
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.param_attr import ParamAttr
import numpy as np
from plyfile import PlyElement, PlyData

__all__ = ["check_gpu", "print_arguments", "parse_outputs", "Stat",
           "export_eval_scene_ply", "convert_pred_to_color", "get_class_map"]

logger = logging.getLogger(__name__)


def convert_pred_to_color(pred, mode='wall_only'):
    """
    :param pred: shape=[total_point_size, 1], type=int64
    :return:
    """
    if mode == 'wall_only':
        label_rgb_list = []
        for label in pred:
            if label > 2 or label == 0:
                label_rgb_list.append([192, 192, 192]) # grey
            elif label == 1:
                label_rgb_list.append([0, 191, 255]) # deep sky blue
            elif label == 2:
                label_rgb_list.append([255, 215, 0])
            else:
                logger.error('The prediction figure is negative!')
                exit(-1)
        return label_rgb_list

    else:
        logger.error("Other mode is Not implemented!")
        exit(-1)

def export_eval_scene_ply(xyz, name, color=None):

    if color is not None:
        # export color
        assert len(color) == len(xyz)
        pts = np.array([(p[0], p[1], p[2], c[0], c[1], c[2]) for p, c in zip(xyz, color)],
                       dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    else:
        pts = np.array([(p[0], p[1], p[2]) for p in xyz], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    el = PlyElement.describe(pts, 'vertex')
    PlyData([el]).write('eval_scene_{}_pred.ply'.format(name))

def test_export_eval_scene_ply():
    npy_path = 'test/0_xyz.npy'
    assert(os.path.exists(npy_path))

    xyz = np.load(npy_path)
    export_eval_scene_ply(xyz)

def get_class_map():
    type2class = {
        'cabinet': 0,
        'bed': 1,
        'chair': 2,
        'sofa': 3,
        'table': 4,
        'door': 5,
        'window': 6,
        'bookshelf': 7,
        'picture': 8,
        'counter': 9,
        'desk': 10,
        'curtain': 11,
        'refrigerator': 12,
        'showercurtain': 13,
        'toilet': 14,
        'sink': 15,
        'bathtub': 16,
        'garbagebin': 17
    }

    class2color = {
        0: (160, 82, 45),
        1: (138, 43, 226),
        2: (0, 128, 0),
        3: (255, 215, 0),
        4: (0, 191, 255),
        5: (255, 0, 0),
        6: (255, 165, 0),
        7: (128, 128, 128),
        8: (128, 128, 128),
        9: (128, 128, 128),
        10: (128, 128, 128),
        11: (128, 128, 128),
        12: (128, 128, 128),
        13: (128, 128, 128),
        14: (128, 128, 128),
        15: (128, 128, 128),
        16: (128, 128, 128),
        17: (128, 128, 128)
    }

    class2type = {type2class[t]: t for t in type2class}

    nyu40ids = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])

    nyu40id2class = {nyu40id: i for i, nyu40id in enumerate(list(nyu40ids))}

    return type2class, class2color

def check_gpu(use_gpu):
    """
    Log error and exit when set use_gpu=True in paddlepaddle
    cpu version.
    """
    err = "Config use_gpu cannot be set as True while you are " \
          "using paddlepaddle cpu version ! \nPlease try: \n" \
          "\t1. Install paddlepaddle-gpu to run model on GPU \n" \
          "\t2. Set --use_gpu=False to run model on CPU"

    try:
        if use_gpu and not fluid.is_compiled_with_cuda():
            logger.error(err)
            sys.exit(1)
    except Exception as e:
        pass


def print_arguments(args):
    """Print argparse's arguments.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        parser.add_argument("name", default="Jonh", type=str, help="User name.")
        args = parser.parse_args()
        print_arguments(args)

    :param args: Input argparse.Namespace for printing.
    :type args: argparse.Namespace
    """
    logger.info("-----------  Configuration Arguments -----------")
    for arg, value in sorted(six.iteritems(vars(args))):
        logger.info("%s: %s" % (arg, value))
    logger.info("------------------------------------------------")


def parse_outputs(outputs):
    keys, values = [], []
    for k, v in outputs.items():
        keys.append(k)
        v.persistable = True
        values.append(v.name)
    return keys, values


class Stat(object):
    def __init__(self):
        self.stats = {}

    def update(self, keys, values):
        for k, v in zip(keys, values):
            if k not in self.stats:
                self.stats[k] = []
            self.stats[k].append(v)

    def reset(self):
        self.stats = {}

    def get_mean_log(self):
        log = ""
        for k, v in self.stats.items():
            log += "avg_{}: {:.4f}, ".format(k, np.mean(v))
        return log

