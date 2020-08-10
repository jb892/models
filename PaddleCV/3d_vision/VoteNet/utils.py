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

ROOT_DIR = os.getcwd()

__all__ = ["check_gpu", "print_arguments", "parse_outputs", "Stat",
           "export_eval_scene_ply", "convert_pred_to_color", "get_class_map", "ScannetDatasetConfig"]

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

class ScannetDatasetConfig(object):
    def __init__(self):
        self.num_class = 18
        self.num_heading_bin = 1
        self.num_size_cluster = 18

        self.type2class = {'cabinet': 0, 'bed': 1, 'chair': 2, 'sofa': 3, 'table': 4, 'door': 5,
                           'window': 6, 'bookshelf': 7, 'picture': 8, 'counter': 9, 'desk': 10, 'curtain': 11,
                           'refrigerator': 12, 'showercurtrain': 13, 'toilet': 14, 'sink': 15, 'bathtub': 16,
                           'garbagebin': 17}
        self.class2color = {
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
        self.class2type = {self.type2class[t]: t for t in self.type2class}
        self.nyu40ids = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
        self.nyu40id2class = {nyu40id: i for i, nyu40id in enumerate(list(self.nyu40ids))}
        self.mean_size_arr = np.load(os.path.join(ROOT_DIR, 'dataset/scannet/meta_data/scannet_means.npz'))['arr_0']
        self.type_mean_size = {}
        for i in range(self.num_size_cluster):
            self.type_mean_size[self.class2type[i]] = self.mean_size_arr[i, :]

    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class
            [optinal] also small regression number from
            class center angle to current angle.

            angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
            return is class of int32 of 0,1,...,N-1 and a number such that
                class*(2pi/N) + number = angle

            NOT USED.
        '''
        assert (False)

    def class2angle(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class.

        As ScanNet only has axis-alined boxes so angles are always 0. '''
        return 0

    def size2class(self, size, type_name):
        ''' Convert 3D box size (l,w,h) to size class and size residual '''
        size_class = self.type2class[type_name]
        size_residual = size - self.type_mean_size[type_name]
        return size_class, size_residual

    def class2size(self, pred_cls, residual):
        ''' Inverse function to size2class '''
        return self.mean_size_arr[pred_cls, :] + residual

    def param2obb(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle(heading_class, heading_residual)
        box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle * -1
        return obb


def rotate_aligned_boxes(input_boxes, rot_mat):
    centers, lengths = input_boxes[:, 0:3], input_boxes[:, 3:6]
    new_centers = np.dot(centers, np.transpose(rot_mat))

    dx, dy = lengths[:, 0] / 2.0, lengths[:, 1] / 2.0
    new_x = np.zeros((dx.shape[0], 4))
    new_y = np.zeros((dx.shape[0], 4))

    for i, crnr in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
        crnrs = np.zeros((dx.shape[0], 3))
        crnrs[:, 0] = crnr[0] * dx
        crnrs[:, 1] = crnr[1] * dy
        crnrs = np.dot(crnrs, np.transpose(rot_mat))
        new_x[:, i] = crnrs[:, 0]
        new_y[:, i] = crnrs[:, 1]

    new_dx = 2.0 * np.max(new_x, 1)
    new_dy = 2.0 * np.max(new_y, 1)
    new_lengths = np.stack((new_dx, new_dy, lengths[:, 2]), axis=1)

    return np.concatenate([new_centers, new_lengths], axis=1)


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

