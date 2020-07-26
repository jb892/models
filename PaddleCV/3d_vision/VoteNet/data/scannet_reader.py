# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import os.path as osp
import signal
import numpy as np
import h5py
import random
import pickle
import logging
import glob
from plyfile import PlyData, PlyElement

import paddle.fluid as fluid
import paddle.fluid.framework as framework

__all__ = ["ScannetReader"]

logger = logging.getLogger(__name__)

class ScannetReader(object):
    def __init__(self, data_dir, mode='train'):
        self.data_dir = data_dir
        self.mode = mode
        # self.data_filename = os.path.join(self.data_dir, 'scannet_%s.pickle' % (mode))
        # self.load_data()
        # assert os.path.isdir('data/train/')
        # assert os.path.isdir('data/test/')

        self.scene_xyz_name_list = glob.glob(os.path.join(data_dir, '{}/*_xyz.npy'.format(mode)))
        print(self.scene_xyz_name_list[:10])
        self.scene_label_name_list = glob.glob(os.path.join(data_dir, '{}/*_lab.npy'.format(mode)))


    def _read_data_file(self, fname):
        assert osp.isfile(fname), \
            "{} is not a file".format(fname)
        with open(fname) as f:
            return [line.strip() for line in f]

    # TODO: Load scannet data with .pickle extension
    def load_data(self):
        logger.info('Loading ScannetV2 dataset from {} ...'.format(self.data_dir))

        # Read pickle file
        with open(self.data_filename, 'rb') as fp:
            self.scene_points_list = pickle.load(fp, encoding='bytes')
            self.semantic_labels_list = pickle.load(fp, encoding='bytes')
        # change to [unanotated, wall, floor]
        for idx, item in enumerate(self.semantic_labels_list):
            item[item > 2] = 0
            self.semantic_labels_list[idx] = item

        if self.mode == 'train':
            labelweights = np.zeros(3)
            for seg in self.semantic_labels_list:
                tmp, _ = np.histogram(seg, range(4))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights / np.sum(labelweights)
            self.labelweights = 1 / np.log(1.2 + labelweights)
        elif self.mode == 'test':
            self.labelweights = np.ones(3)

        # # Write npy file.
        # os.system('mkdir {}/'.format(self.mode))
        #
        # for idx, scene in enumerate(self.scene_points_list):
        #     np.save('{}/{}_xyz.npy'.format(self.mode, idx), scene)
        #     np.save('{}/{}_lab.npy'.format(self.mode, idx), self.semantic_labels_list[idx])

        logger.info("Load data finished")

    def load_scene(self, idx, mode):
        scene = np.load('{}/{}_xyz.npy'.format(mode, idx))
        label = np.load('{}/{}_lab.npy'.format(mode, idx))
        return scene, label

    def get_reader(self, batch_size, num_points, shuffle=True):
        scene_size = len(self.scene_xyz_name_list)

        num_batches = int(np.ceil(scene_size/batch_size))

        logger.info('scene_size: {}'.format(scene_size))
        logger.info('Mode: ' + self.mode)
        # self.npoints = num_points

        # return point_set, semantic_seg, sample_weight
        def reader():
            batch_out = []

            for idx, xyz_path in enumerate(self.scene_xyz_name_list):

                point_set = np.load(xyz_path)
                label_path = xyz_path[:-7] + 'lab.npy'
                semantic_seg = np.load(label_path).astype(np.int32)

                # logger.info('len: {}, {}'.format(len(point_set), len(semantic_seg)))

            # for i in range(scene_size):
                # index = np.random.random_integers(0, scene_size, 1)[0]
                # point_set = self.scene_points_list[index]
                # semantic_seg = self.semantic_labels_list[index].astype(np.int32)

                # point_set = np.load('data/{}/{}_xyz.npy'.format(mode, i))
                # semantic_seg = np.load('data/{}/{}_lab.npy'.format(mode, i)).astype(np.int32)

                coordmax = np.max(point_set, axis=0)
                coordmin = np.min(point_set, axis=0)
                smpmin = np.maximum(coordmax - [1.5, 1.5, 3.0], coordmin)
                smpmin[2] = coordmin[2]
                smpsz = np.minimum(coordmax - smpmin, [1.5, 1.5, 3.0])
                smpsz[2] = coordmax[2] - coordmin[2]
                isvalid = False
                for i in range(10):
                    curcenter = point_set[np.random.choice(len(semantic_seg), 1)[0], :]
                    curmin = curcenter - [0.75, 0.75, 1.5]
                    curmax = curcenter + [0.75, 0.75, 1.5]
                    curmin[2] = coordmin[2]
                    curmax[2] = coordmax[2]
                    curchoice = np.sum((point_set >= (curmin - 0.2)) * (point_set <= (curmax + 0.2)), axis=1) == 3
                    cur_point_set = point_set[curchoice, :]
                    cur_semantic_seg = semantic_seg[curchoice]
                    if len(cur_semantic_seg) == 0:
                        continue
                    mask = np.sum((cur_point_set >= (curmin - 0.01)) * (cur_point_set <= (curmax + 0.01)), axis=1) == 3
                    vidx = np.ceil((cur_point_set[mask, :] - curmin) / (curmax - curmin) * [31.0, 31.0, 62.0])
                    vidx = np.unique(vidx[:, 0] * 31.0 * 62.0 + vidx[:, 1] * 62.0 + vidx[:, 2])
                    isvalid = np.sum(cur_semantic_seg > 0) / len(cur_semantic_seg) >= 0.7 and len(
                        vidx) / 31.0 / 31.0 / 62.0 >= 0.02
                    if isvalid:
                        break
                choice = np.random.choice(len(cur_semantic_seg), num_points, replace=True)
                point_set = cur_point_set[choice, :]
                semantic_seg = cur_semantic_seg[choice]
                mask = mask[choice]
                # sample_weight = self.labelweights[semantic_seg]
                # sample_weight *= mask
                feature = np.zeros((num_points, 6)).astype(np.float32)

                batch_out.append((point_set, feature, semantic_seg))

                if len(batch_out) == batch_size:
                    yield batch_out
                    batch_out = []

        return reader

def export_eval_scene_ply():
    output_path = 'eval/'
    xyz = np.load('test/0_xyz.npy')
    pts = np.array([(p[0], p[1], p[2]) for p in xyz], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(pts, 'vertex')
    PlyData([el]).write('eval/evalScene.ply')

def export_eval_whole_scene(batch_size, num_points):
    point_set_ini = np.load('test/0_xyz.npy')
    semantic_seg_ini = np.load('test/0_xyz.npy')

    coordmax = np.max(point_set_ini, axis=0)
    coordmin = np.min(point_set_ini, axis=0)
    nsubvolume_x = np.ceil((coordmax[0] - coordmin[0]) / 1.5).astype(np.int32)
    nsubvolume_y = np.ceil((coordmax[1] - coordmin[1]) / 1.5).astype(np.int32)

    point_sets = list()
    semantic_segs = list()
    isvalid = False
    for i in range(nsubvolume_x):
        for j in range(nsubvolume_y):
            curmin = coordmin + [i * 1.5, j * 1.5, 0]
            curmax = coordmin + [(i + 1) * 1.5, (j + 1) * 1.5, coordmax[2] - coordmin[2]]
            curchoice = np.sum((point_set_ini >= (curmin - 0.2)) * (point_set_ini <= (curmax + 0.2)), axis=1) == 3
            cur_point_set = point_set_ini[curchoice, :]
            cur_semantic_seg = semantic_seg_ini[curchoice]
            if len(cur_semantic_seg) == 0:
                continue
            mask = np.sum((cur_point_set >= (curmin - 0.001)) * (cur_point_set <= (curmax + 0.001)), axis=1) == 3
            choice = np.random.choice(len(cur_semantic_seg), num_points, replace=True)
            point_set = cur_point_set[choice, :]  # Nx3
            semantic_seg = cur_semantic_seg[choice]  # N
            mask = mask[choice]
            if sum(mask) / float(len(mask)) < 0.01:
                continue
            point_sets.append(np.expand_dims(point_set, 0))  # 1xNx3
            semantic_segs.append(np.expand_dims(semantic_seg, 0))  # 1xN
    point_sets = np.concatenate(tuple(point_sets), axis=0)
    semantic_segs = np.concatenate(tuple(semantic_segs), axis=0)

    print(len(point_sets))

    np.save('eval/whole_scene_xyz.npy', point_sets)
    np.save('eval/whole_scene_lab.npy', semantic_segs)

    return point_sets, semantic_segs

class ScannetReaderEval(object):
    # load whole scene data
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.load_data()

    def load_data(self):
        logger.info('Loading ScannetV2 dataset from {} ...'.format(self.data_dir))


        logger.info("Load data finished")

# TODO: finish this function below
# def save_ply_color(point_set, semantic_seg):
#


# def _term_reader(signum, frame):
#     logger.info('pid {} terminated, terminate reader process '
#                 'group {}...'.format(os.getpid(), os.getpgrp()))
#     os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)
#
# signal.signal(signal.SIGINT, _term_reader)
# signal.signal(signal.SIGTERM, _term_reader)

if __name__ == '__main__':
    export_eval_whole_scene(16, 8192)
    # r = ScannetReader('/home/jake/Documents/paddle/pointnet2_paddle/data', 'train')
    # m_r = r.get_reader(8, 8192)
