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

__all__ = ["ScannetReader", "ScannetWholeSceneReader"]

logger = logging.getLogger(__name__)

class ScannetReader(object):
    def __init__(self, data_dir, mode='train'):
        self.data_dir = data_dir
        self.mode = mode

        self.scene_xyz_name_list = glob.glob(os.path.join(data_dir, '{}/*_xyz.npy'.format(mode)))
        # print(self.scene_xyz_name_list[:10])
        self.scene_label_name_list = glob.glob(os.path.join(data_dir, '{}/*_lab.npy'.format(mode)))


    def _read_data_file(self, fname):
        assert osp.isfile(fname), \
            "{} is not a file".format(fname)
        with open(fname) as f:
            return [line.strip() for line in f]

    # # TODO: Load scannet data with .pickle extension
    # def load_data(self):
    #     logger.info('Loading ScannetV2 dataset from {} ...'.format(self.data_dir))
    #
    #     # Read pickle file
    #     with open(self.data_filename, 'rb') as fp:
    #         self.scene_points_list = pickle.load(fp, encoding='bytes')
    #         self.semantic_labels_list = pickle.load(fp, encoding='bytes')
    #     # # change to [unanotated, wall, floor]
    #     # for idx, item in enumerate(self.semantic_labels_list):
    #     #     item[item > 2] = 0
    #     #     self.semantic_labels_list[idx] = item
    #
    #     if self.mode == 'train':
    #         labelweights = np.zeros(3)
    #         for seg in self.semantic_labels_list:
    #             tmp, _ = np.histogram(seg, range(4))
    #             labelweights += tmp
    #         labelweights = labelweights.astype(np.float32)
    #         labelweights = labelweights / np.sum(labelweights)
    #         self.labelweights = 1 / np.log(1.2 + labelweights)
    #     elif self.mode == 'test':
    #         self.labelweights = np.ones(3)
    #
    #     # # Write npy file.
    #     # os.system('mkdir {}/'.format(self.mode))
    #     #
    #     # for idx, scene in enumerate(self.scene_points_list):
    #     #     np.save('{}/{}_xyz.npy'.format(self.mode, idx), scene)
    #     #     np.save('{}/{}_lab.npy'.format(self.mode, idx), self.semantic_labels_list[idx])
    #
    #     logger.info("Load data finished")

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

        def reader():
            batch_out = []

            for idx, xyz_path in enumerate(self.scene_xyz_name_list):

                point_set = np.load(xyz_path)
                label_path = xyz_path[:-7] + 'lab.npy'
                semantic_seg = np.load(label_path).astype(np.int32)

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

                feature = np.zeros((num_points, 6)).astype(np.float32)

                batch_out.append((point_set, feature, semantic_seg))

                if len(batch_out) == batch_size:
                    yield batch_out
                    batch_out = []

        return reader

class ScannetWholeSceneReader(object):
    def __init__(self, data_dir, scene_name, batch_size, num_points):
        self.data_dir = data_dir
        self.scene_name = scene_name
        self.batch_size = batch_size
        self.num_points = num_points
        self.extra_zero_batch_lens = 0

    def load_and_preprocess(self, save_npy=False):
        # Load whole scene npy file
        whole_scene_xyz_path = os.path.join(self.data_dir, '{}_xyz.npy'.format(self.scene_name))
        whole_scene_lab_path = os.path.join(self.data_dir, '{}_lab.npy'.format(self.scene_name))
        logger.info('Loading scene {} ...'.format(whole_scene_xyz_path))

        whole_scene_xyz = np.load(whole_scene_xyz_path)
        whole_scene_lab = np.load(whole_scene_lab_path)

        logger.info('Finished scene loading.')

        # Start scene preprocessing
        logger.info('Whole scene preprocessing...')

        coordmax = np.max(whole_scene_xyz, axis=0)
        coordmin = np.min(whole_scene_xyz, axis=0)
        nsubvolume_x = np.ceil((coordmax[0] - coordmin[0]) / 1.5).astype(np.int32)
        nsubvolume_y = np.ceil((coordmax[1] - coordmin[1]) / 1.5).astype(np.int32)

        logger.info('Scene division: nsubvolume_x={}, nsubvolume_y={}'.format(nsubvolume_x, nsubvolume_y))

        point_sets = list()
        semantic_segs = list()
        isvalid = False
        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):
                curmin = coordmin + [i * 1.5, j * 1.5, 0]
                curmax = coordmin + [(i + 1) * 1.5, (j + 1) * 1.5, coordmax[2] - coordmin[2]]
                curchoice = np.sum((whole_scene_xyz >= (curmin - 0.2)) * (whole_scene_xyz <= (curmax + 0.2)), axis=1) == 3
                cur_point_set = whole_scene_xyz[curchoice, :]
                cur_semantic_seg = whole_scene_lab[curchoice]
                if len(cur_semantic_seg) == 0:
                    continue
                mask = np.sum((cur_point_set >= (curmin - 0.001)) * (cur_point_set <= (curmax + 0.001)), axis=1) == 3
                choice = np.random.choice(len(cur_semantic_seg), self.num_points, replace=True)
                point_set = cur_point_set[choice, :]  # Nx3
                semantic_seg = cur_semantic_seg[choice]  # N
                mask = mask[choice]
                if sum(mask) / float(len(mask)) < 0.01:
                    continue
                point_sets.append(point_set)  # Nx3
                semantic_segs.append(semantic_seg)  # N
        # point_sets = np.concatenate(tuple(point_sets), axis=0)
        # semantic_segs = np.concatenate(tuple(semantic_segs), axis=0)

        logger.info('Whole scene subvolume in total: {}'.format(len(point_sets)))

        self.point_sets = np.array(point_sets)
        self.semantic_segs = np.array(semantic_segs).reshape((-1, self.num_points, 1))

        logger.info("Point set list shape: {}".format(self.point_sets.shape))
        logger.info("Semantic label list shape: {}".format(self.semantic_segs.shape))

        if save_npy:
            # Saving whole scene npy file
            logger.info('Saving whole scene npy file...')
            np.save('eval/{}_whole_xyz.npy'.format(self.scene_name), point_sets)
            np.save('eval/{}_whole_lab.npy'.format(self.scene_name), semantic_segs)
            logger.info('Finished saving whole scene npy.')

        logger.info('Finished preprocessing.')

    def get_reader(self, shuffle=False):
        total_sample = len(self.point_sets)

        total_num_iter = int(np.ceil(total_sample/self.batch_size))

        logger.info('Total_num_batches: {}'.format(total_num_iter))

        if shuffle:
            choice = np.random.choice(len(self.point_sets))
            self.point_sets = self.point_sets[choice, :, :]
            self.semantic_segs = self.semantic_segs[choice, :, :]

        batch_size = self.batch_size
        self.extra_zero_batch_lens = batch_size * total_num_iter - total_sample

        if self.extra_zero_batch_lens > 0:
            extra_zero_batch_xyz = []
            extra_zero_batch_lab = []
            # Fill-in extra all-zero batches
            for i in range(self.extra_zero_batch_lens):
                extra_zero_batch_xyz.append(np.zeros((self.num_points, 3)).astype(np.float32))
                extra_zero_batch_lab.append(np.zeros((self.num_points, 1)).astype(np.int64))
            extra_zero_batch_xyz = np.array(extra_zero_batch_xyz)
            extra_zero_batch_lab = np.array(extra_zero_batch_lab)

            logger.info('Point_sets shape: {}'.format(self.point_sets.shape))

            # Concatenate
            self.point_sets = np.concatenate((self.point_sets, extra_zero_batch_xyz), axis=0)
            self.semantic_segs = np.concatenate((self.semantic_segs, extra_zero_batch_lab), axis=0)

        logger.info('Point_sets shape: {}'.format(self.point_sets.shape))

        num_points = self.num_points
        point_sets = self.point_sets
        semantic_segs = self.semantic_segs

        def reader():
            batch_out = []
            for idx, xyz in enumerate(point_sets):
                # Currently, we set feature to none
                # TODO: to test feeding in color, normal or height feature.
                feature = np.zeros((num_points, 6)).astype(np.float32)
                batch_out.append((xyz, feature, semantic_segs[idx]))

                if len(batch_out) == batch_size:
                    yield batch_out
                    batch_out = []

        return reader

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

def load_pickle_and_save_npy(data_dir, mode='train'):
    data_filename = os.path.join(data_dir, 'scannet_%s.pickle' % (mode))
    logger.info('Loading ScannetV2 dataset from {} ...'.format(data_filename))
    # Read pickle file
    with open(data_filename, 'rb') as fp:
        scene_points_list = pickle.load(fp, encoding='bytes')
        semantic_labels_list = pickle.load(fp, encoding='bytes')

    # Write npy file.
    os.system('mkdir {}/'.format(mode))

    for idx, scene in enumerate(scene_points_list):
        np.save('{}/{}_xyz.npy'.format(mode, idx), scene)
        np.save('{}/{}_lab.npy'.format(mode, idx), semantic_labels_list[idx])

def test_pickle_and_save_npy():
    pickle_datapath = '/media/jake/DATA/Documents/pointnet2/data/scannet_data_pointnet2'
    mode = 'test'
    load_pickle_and_save_npy(pickle_datapath, mode)


def _term_reader(signum, frame):
    logger.info('pid {} terminated, terminate reader process '
                'group {}...'.format(os.getpid(), os.getpgrp()))
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)

signal.signal(signal.SIGINT, _term_reader)
signal.signal(signal.SIGTERM, _term_reader)

if __name__ == '__main__':
    # export_eval_whole_scene(16, 8192)
    # r = ScannetReader('/home/jake/Documents/paddle/pointnet2_paddle/data', 'train')
    # m_r = r.get_reader(8, 8192)

    test_pickle_and_save_npy()