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
import pickle
import logging
import glob
from plyfile import PlyData, PlyElement
# Mesh IO
import trimesh
import matplotlib.pyplot as pyplot

import paddle.fluid as fluid
import paddle.fluid.framework as framework

MAX_NUM_OBJ = 64
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
nyu40ids = np.array([3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39])
nyu40id2class = {nyu40id: i for i,nyu40id in enumerate(list(nyu40ids))}
SCANNET_DIR = 'dataset/scannet'
MEAN_SIZE_ARR_PATH = os.path.join(SCANNET_DIR, 'meta_data/scannet_means.npz')

__all__ = ["ScannetDetectionReader", "ScannetReader", "ScannetWholeSceneReader"]

logger = logging.getLogger(__name__)

class ScannetDetectionReader(object):
    def __init__(self, num_points=20000, use_color=False, use_height=True, augment=False, mode='train'):
        self.num_points = num_points
        self.use_color = use_color
        self.use_height = use_height
        self.augment = augment
        self.mode = mode
        print(os.getcwd())
        self.data_path = os.path.join(SCANNET_DIR, 'scannet_train_detection_data')
        all_scan_names = list(set([os.path.basename(x)[0:12] for x in os.listdir(self.data_path) if x.startswith('scene')]))

        if mode=='all':
            self.scan_names = all_scan_names
        elif mode in ['train', 'val', 'test']:
            split_filenames = os.path.join(SCANNET_DIR, 'meta_data/scannetv2_{}.txt'.format(mode))
            with open(split_filenames, 'r') as f:
                self.scan_names = f.read().splitlines()
            # remove unavailiable scans
            num_scans = len(self.scan_names)
            self.scan_names = [sname for sname in self.scan_names if sname in all_scan_names]
            logger.info('kept {} scans out of {}'.format(len(self.scan_names), num_scans))
        else:
            logger.error('illegal mode name')
            exit(-1)

    def get_reader(self, batch_size):
        scan_names = self.scan_names
        data_path = self.data_path
        use_color = self.use_color
        use_height = self.use_height
        num_points = self.num_points
        augment = self.augment
        mean_size_arr = np.load(MEAN_SIZE_ARR_PATH)['arr_0']

        # logger.info("batch_size: {}".format(batch_size))
        # logger.info("num_points: {}".format(num_points))
        # logger.info("scan_names: {}".format(scan_names))
        # logger.info("data_path: {}".format(data_path))
        # logger.info("use_color: {}".format(use_color))
        # logger.info("use_height: {}".format(use_height))
        # logger.info("augment: {}".format(augment))

        def reader():
            """
            Returns a dict with following keys:
                point_clouds: (N,3)
                features: (N, C) RGB+Height
                center_label: (MAX_NUM_OBJ,3) for GT box center XYZ
                sem_cls_label: (MAX_NUM_OBJ,) semantic class index
                angle_class_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
                angle_residual_label: (MAX_NUM_OBJ,)
                size_classe_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
                size_residual_label: (MAX_NUM_OBJ,3)
                box_label_mask: (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
                point_votes: (N,3) with votes XYZ
                point_votes_mask: (N,) with 0/1 with 1 indicating the point is in one of the object's OBB.
                scan_idx: int scan index in scan_names list
                pcl_color: unused
            """
            batch_out = []

            for scan_name in scan_names:
                mesh_vertices = np.load(os.path.join(data_path, scan_name) + '_vert.npy')
                instance_labels = np.load(os.path.join(data_path, scan_name) + '_ins_label.npy')
                semantic_labels = np.load(os.path.join(data_path, scan_name) + '_sem_label.npy')
                instance_bboxes = np.load(os.path.join(data_path, scan_name) + '_bbox.npy')

                point_cloud = mesh_vertices[:, :3]
                if use_color:
                    pcl_color = (mesh_vertices[:, 3:6] - MEAN_COLOR_RGB) / 256.0

                if use_height:
                    floor_height = np.percentile(point_cloud[:, 2], 0.99)
                    height = np.expand_dims(point_cloud[:, 2] - floor_height, axis=-1)

                if use_color and use_height:
                    features = np.concatenate([pcl_color, height], axis=-1)
                elif use_color:
                    features = pcl_color
                elif use_height:
                    features = height
                else:
                    features = np.empty()

                # ------------------------------- LABELS ------------------------------
                target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
                target_bboxes_mask = np.zeros((MAX_NUM_OBJ))
                angle_classes = np.zeros((MAX_NUM_OBJ,))
                angle_residuals = np.zeros((MAX_NUM_OBJ,))
                size_classes = np.zeros((MAX_NUM_OBJ,))
                size_residuals = np.zeros((MAX_NUM_OBJ, 3))

                point_cloud, choices = random_sampling(point_cloud, num_points, return_choices=True)
                instance_labels = instance_labels[choices]
                semantic_labels = semantic_labels[choices]

                features = features[choices]

                target_bboxes_mask[0:instance_bboxes.shape[0]] = 1
                target_bboxes[0:instance_bboxes.shape[0], :] = instance_bboxes[:, 0:6]

                # ------------------------------- DATA AUGMENTATION ------------------------------
                if augment:
                    if np.random.random() > 0.5:
                        # Flipping along the YZ plane
                        point_cloud[:, 0] = -1 * point_cloud[:, 0]
                        target_bboxes[:, 0] = -1 * target_bboxes[:, 0]

                    if np.random.random() > 0.5:
                        # Flipping along the XZ plane
                        point_cloud[:, 1] = -1 * point_cloud[:, 1]
                        target_bboxes[:, 1] = -1 * target_bboxes[:, 1]

                        # Rotation along up-axis/Z-axis
                    rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
                    rot_mat = rotz(rot_angle)
                    point_cloud = np.dot(point_cloud, np.transpose(rot_mat))
                    target_bboxes = rotate_aligned_boxes(target_bboxes, rot_mat)

                # compute votes *AFTER* augmentation
                # generate votes
                # Note: since there's no map between bbox instance labels and
                # pc instance_labels (it had been filtered
                # in the data preparation step) we'll compute the instance bbox
                # from the points sharing the same instance label.
                point_votes = np.zeros([num_points, 3])
                point_votes_mask = np.zeros(num_points)
                for i_instance in np.unique(instance_labels):
                    # find all points belong to that instance
                    ind = np.where(instance_labels == i_instance)[0]
                    # find the semantic label
                    if semantic_labels[ind[0]] in nyu40ids:
                        x = point_cloud[ind, :]
                        center = 0.5 * (x.min(0) + x.max(0))
                        point_votes[ind, :] = center - x
                        point_votes_mask[ind] = 1.0
                point_votes = np.tile(point_votes, (1, 3))  # make 3 votes identical

                class_ind = [np.where(nyu40ids == x)[0][0] for x in instance_bboxes[:, -1]]
                # NOTE: set size class as semantic class. Consider use size2class.
                size_classes[0:instance_bboxes.shape[0]] = class_ind
                size_residuals[0:instance_bboxes.shape[0], :] = \
                    target_bboxes[0:instance_bboxes.shape[0], 3:6] - mean_size_arr[class_ind, :]

                # ret_dict = {}
                point_cloud = point_cloud.astype(np.float32)
                center_label = target_bboxes.astype(np.float32)[:, 0:3]
                heading_class_label = angle_classes.astype(np.int64)
                heading_residual_label = angle_residuals.astype(np.float32)
                size_class_label = size_classes.astype(np.int64)
                size_residual_label = size_residuals.astype(np.float32)
                target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
                target_bboxes_semcls[0:instance_bboxes.shape[0]] = \
                    [nyu40id2class[x] for x in instance_bboxes[:, -1][0:instance_bboxes.shape[0]]]
                sem_cls_label = target_bboxes_semcls.astype(np.int64)
                box_label_mask = target_bboxes_mask.astype(np.float32)
                vote_label = point_votes.astype(np.float32)
                vote_label_mask = point_votes_mask.astype(np.int64)

                # print('point_cloud.shape: ', point_cloud.shape)
                # print('features.shape: ', features.shape)
                # print('center_label.shape: ', center_label.shape)
                # print('heading_class_label.shape: ', heading_class_label.shape)
                # print('heading_residual_label.shape: ', heading_residual_label.shape)
                # print('size_class_label.shape: ', size_class_label.shape)
                # print('size_residual_label.shape: ', size_residual_label.shape)
                # print('sem_cls_label.shape: ', sem_cls_label.shape)
                # print('box_label_mask.shape: ', box_label_mask.shape)
                # print('vote_label.shape: ', vote_label.shape)
                # print('vote_label_mask.shape: ', vote_label_mask.shape)

                batch_out.append((point_cloud,
                                  features,
                                  center_label,
                                  heading_class_label,
                                  heading_residual_label,
                                  size_class_label,
                                  size_residual_label,
                                  sem_cls_label,
                                  box_label_mask,
                                  vote_label,
                                  vote_label_mask))

                if len(batch_out) == batch_size:
                    yield batch_out
                    batch_out = []
        return reader

class ScannetReader(object):
    def __init__(self, data_dir, mean_size_arr_path, mode='train'):
        self.data_dir = data_dir
        self.mode = mode

        self.scene_xyz_name_list = glob.glob(os.path.join(data_dir, '{}/*_xyz.npy'.format(mode)))
        # print(self.scene_xyz_name_list[:10])
        self.scene_label_name_list = glob.glob(os.path.join(data_dir, '{}/*_lab.npy'.format(mode)))
        self.mean_size_arr = np.load(mean_size_arr_path)['arr_0']

    def _read_data_file(self, fname):
        assert osp.isfile(fname), \
            "{} is not a file".format(fname)
        with open(fname) as f:
            return [line.strip() for line in f]

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
        mean_size_arr = self.mean_size_arr
        scene_xyz_name_list = self.scene_xyz_name_list

        def reader():
            batch_out = []

            for idx, xyz_path in enumerate(scene_xyz_name_list):

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

                batch_out.append((point_set, feature, semantic_seg, mean_size_arr))

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


############# Visualizaion ########

def viz_votes(pc, point_votes, point_votes_mask, name=''):
    """ Visualize point votes and point votes mask labels
    pc: (N,3 or 6), point_votes: (N,9), point_votes_mask: (N,)
    """
    inds = (point_votes_mask == 1)
    pc_obj = pc[inds, 0:3]
    pc_obj_voted1 = pc_obj + point_votes[inds, 0:3]
    write_ply(pc_obj, 'pc_obj{}.ply'.format(name))
    write_ply(pc_obj_voted1, 'pc_obj_voted1{}.ply'.format(name))


def viz_obb(pc, label, mask, angle_classes, angle_residuals,
            size_classes, size_residuals, name=''):
    """ Visualize oriented bounding box ground truth
    pc: (N,3)
    label: (K,3)  K == MAX_NUM_OBJ
    mask: (K,)
    angle_classes: (K,)
    angle_residuals: (K,)
    size_classes: (K,)
    size_residuals: (K,3)
    """
    oriented_boxes = []
    K = label.shape[0]
    for i in range(K):
        if mask[i] == 0: continue
        obb = np.zeros(7)
        obb[0:3] = label[i, 0:3]
        heading_angle = 0  # hard code to 0
        mean_size_arr = np.load(MEAN_SIZE_ARR_PATH)['arr_0']
        box_size = mean_size_arr[size_classes[i], :] + size_residuals[i, :]
        obb[3:6] = box_size
        obb[6] = -1 * heading_angle
        print(obb)
        oriented_boxes.append(obb)
    write_oriented_bbox(oriented_boxes, 'gt_obbs{}.ply'.format(name))
    write_ply(label[mask == 1, :], 'gt_centroids{}.ply'.format(name))

def write_oriented_bbox(scene_bbox, out_filename):
    """Export oriented (around Z axis) scene bbox to meshes
    Args:
        scene_bbox: (N x 7 numpy array): xyz pos of center and 3 lengths (dx,dy,dz)
            and heading angle around Z axis.
            Y forward, X right, Z upward. heading angle of positive X is 0,
            heading angle of positive Y is 90 degrees.
        out_filename: (string) filename
    """

    def heading2rotmat(heading_angle):
        pass
        rotmat = np.zeros((3, 3))
        rotmat[2, 2] = 1
        cosval = np.cos(heading_angle)
        sinval = np.sin(heading_angle)
        rotmat[0:2, 0:2] = np.array([[cosval, -sinval], [sinval, cosval]])
        return rotmat

    def convert_oriented_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:6]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3, 3] = 1.0
        trns[0:3, 0:3] = heading2rotmat(box[6])
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box))

    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to ply file
    trimesh.io.export.export_mesh(mesh_list, out_filename, file_type='ply')

    return

def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)


def write_ply_color(points, labels, filename, num_classes=None, colormap=pyplot.cm.jet):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    labels = labels.astype(int)
    N = points.shape[0]
    if num_classes is None:
        num_classes = np.max(labels) + 1
    else:
        assert (num_classes > np.max(labels))

    vertex = []
    # colors = [pyplot.cm.jet(i/float(num_classes)) for i in range(num_classes)]
    colors = [colormap(i / float(num_classes)) for i in range(num_classes)]
    for i in range(N):
        c = colors[labels[i]]
        c = [int(x * 255) for x in c]
        vertex.append((points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
    vertex = np.array(vertex,
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=True).write(filename)

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

def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


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

    # test_pickle_and_save_npy()
    reader = ScannetDetectionReader()

    pass