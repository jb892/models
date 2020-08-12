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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid
import paddle.fluid.layers as layers
import logging

from backbone_module import Pointnet2Backbone
from voting_module import VotingModule
from proposal_module import ProposalModule
from loss_helper import get_loss

__all__ = ['VoteNet']

logger = logging.getLogger(__name__)

MAX_NUM_OBJ = 64

class VoteNet(object):
    r"""
        A deep neural network for 3D object detection with end-to-end optimizable hough voting.

        Parameters
        ----------
        num_class: int
            Number of semantics classes to predict over -- size of softmax classifier
        num_heading_bin: int
        num_size_cluster: int
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
        vote_factor: (default: 1)
            Number of votes generated from each seed point.
    """
    def __init__(self, num_class, num_points, num_heading_bin, num_size_cluster,
                 input_feature_dim=4, num_proposal=128, vote_factor=1, sampling='vote_fps', batch_size=1, bn_momentum=0.9):
        self.num_class = num_class
        self.num_points = num_points
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.batch_size = batch_size
        self.bn_momentum=bn_momentum

        # init Pointnet2Backbone
        self.backbone_net = Pointnet2Backbone(input_feature_dim=input_feature_dim, bn_momentum=bn_momentum, name='backbone_net')

        # init VotingModule
        self.vgen = VotingModule(self.vote_factor, 256, batch_size=batch_size, bn_momentum=bn_momentum, name='vgen')

        # init proposal module
        self.pnet = ProposalModule(num_class, num_heading_bin, num_size_cluster, num_proposal, sampling, bn_momentum=bn_momentum, name='pnet')

    def build_input(self):
        self.xyz = fluid.data(name='xyz',
                              shape=[self.batch_size, self.num_points, 3],
                              dtype='float32',
                              lod_level=0)
        self.feature = fluid.data(name='feature',
                                  shape=[self.batch_size, self.num_points, self.input_feature_dim],
                                  dtype='float32',
                                  lod_level=0)
        self.center_label = fluid.data(name='center_label',
                                shape=[self.batch_size, MAX_NUM_OBJ, 3],
                                dtype='float32',
                                lod_level=0)
        self.heading_class_label = fluid.data(name='heading_class_label',
                                              shape=[self.batch_size, MAX_NUM_OBJ],
                                              dtype='int64',
                                              lod_level=0)
        self.heading_residual_label = fluid.data(name='heading_residual_label',
                                              shape=[self.batch_size, MAX_NUM_OBJ],
                                              dtype='float32',
                                              lod_level=0)
        self.size_class_label = fluid.data(name='size_class_label',
                                                 shape=[self.batch_size, MAX_NUM_OBJ],
                                                 dtype='int64',
                                                 lod_level=0)
        self.size_residual_label = fluid.data(name='size_residual_label',
                                           shape=[self.batch_size, MAX_NUM_OBJ, 3],
                                           dtype='float32',
                                           lod_level=0)
        self.sem_cls_label = fluid.data(name='sem_cls_label',
                                              shape=[self.batch_size, MAX_NUM_OBJ],
                                              dtype='int64',
                                              lod_level=0)
        self.box_label_mask = fluid.data(name='box_label_mask',
                                              shape=[self.batch_size, MAX_NUM_OBJ],
                                              dtype='float32',
                                              lod_level=0)
        self.vote_label = fluid.data(name='vote_label',
                                        shape=[self.batch_size, self.num_points, 9],
                                        dtype='float32',
                                        lod_level=0)
        self.vote_label_mask = fluid.data(name='vote_label_mask',
                                        shape=[self.batch_size, self.num_points],
                                        dtype='int64',
                                        lod_level=0)
        self.loader = fluid.io.DataLoader.from_generator(
            feed_list=[self.xyz, self.feature, self.center_label, self.heading_class_label, self.heading_residual_label,
                       self.size_class_label, self.size_residual_label, self.sem_cls_label, self.box_label_mask, self.vote_label,
                       self.vote_label_mask],
            capacity=64,
            use_double_buffer=True,
            iterable=False)
        self.feed_vars = [self.xyz, self.feature, self.center_label, self.heading_class_label, self.heading_residual_label,
                          self.size_class_label, self.size_residual_label, self.sem_cls_label, self.box_label_mask, self.vote_label,
                          self.vote_label_mask]

    def build(self):
        end_points = {}

        # Backbone point feature learning
        end_points = self.backbone_net.build(self.xyz, self.feature, end_points)

        # =========== Hough voting ==========
        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features
        end_points['seed_inds'] = end_points['fp2_inds']

        # Convert features tensor from 'NWC' to 'NCW' format
        features = layers.transpose(features, perm=[0, 2, 1])
        vote_xyz, vote_features = self.vgen.build(xyz, features) # vote_features.shape = [B, out_dim, num_vote]
        features_norm = layers.sqrt(layers.reduce_sum(layers.pow(vote_features, factor=2.0), dim=1, keep_dim=False))
        vote_features = vote_features / layers.unsqueeze(features_norm, 1) # features_norm.shape = [B, 1, num_vote]

        # Convert features tensor from 'NCW' to 'NWC'
        # vote_features = layers.transpose(vote_features, perm=[0, 2, 1])

        end_points['vote_xyz'] = vote_xyz
        end_points['vote_features'] = vote_features

        # Vote aggregation and detection
        end_points = self.pnet.build(vote_xyz, vote_features, end_points, batch_size=self.batch_size)#, self.mean_size_arr)

        config = {
            'num_heading_bin': self.num_heading_bin,
            'num_size_cluster': self.num_size_cluster,
            'num_class': self.num_class,
            # 'mean_size_arr': self.mean_size_arr,
            'batch_size': self.batch_size
        }

        # Put labels into the end_points dict
        end_points['center_label'] = self.center_label
        end_points['heading_class_label'] = self.heading_class_label
        end_points['heading_residual_label'] = self.heading_residual_label
        end_points['size_class_label'] = self.size_class_label
        end_points['size_residual_label'] = self.size_residual_label
        end_points['sem_cls_label'] = self.sem_cls_label
        end_points['box_label_mask'] = self.box_label_mask
        end_points['vote_label'] = self.vote_label
        end_points['vote_label_mask'] = self.vote_label_mask

        # Calculate loss
        self.loss, self.end_points = get_loss(end_points, config)

    def get_feeds(self):
        return self.feed_vars

    def get_loss(self):
        return self.loss

    def get_loader(self):
        return self.loader

    def get_outputs(self, mode='test'):
        if mode == 'train':
            return {'loss': self.loss,
                    'obj_acc': self.end_points['obj_acc'],
                    'box_loss': self.end_points['box_loss'],
                    'center_loss': self.end_points['center_loss'],
                    'heading_cls_loss': self.end_points['heading_cls_loss'],
                    'heading_reg_loss': self.end_points['heading_reg_loss'],
                    'neg_ratio': self.end_points['neg_ratio'],
                    'objectness_loss': self.end_points['objectness_loss'],
                    'pos_ratio': self.end_points['pos_ratio'],
                    'sem_cls_loss': self.end_points['sem_cls_loss'],
                    'size_cls_loss': self.end_points['size_cls_loss'],
                    'size_reg_loss': self.end_points['size_reg_loss'],
                    'vote_loss': self.end_points['vote_loss']}
        elif mode == 'test':
            return {'loss': self.loss,
                    'obj_acc': self.end_points['obj_acc'],
                    'box_loss': self.end_points['box_loss'],
                    'center_loss': self.end_points['center_loss'],
                    'heading_cls_loss': self.end_points['heading_cls_loss'],
                    'heading_reg_loss': self.end_points['heading_reg_loss'],
                    'neg_ratio': self.end_points['neg_ratio'],
                    'objectness_loss': self.end_points['objectness_loss'],
                    'pos_ratio': self.end_points['pos_ratio'],
                    'sem_cls_loss': self.end_points['sem_cls_loss'],
                    'size_cls_loss': self.end_points['size_cls_loss'],
                    'size_reg_loss': self.end_points['size_reg_loss'],
                    'vote_loss': self.end_points['vote_loss'],
                    'point_clouds': self.xyz,
                    'center': self.end_points['center'],
                    'heading_scores': self.end_points['heading_scores'],
                    'heading_residuals': self.end_points['heading_residuals'],
                    'size_scores': self.end_points['size_scores'],
                    'size_residuals': self.end_points['size_residuals'],
                    'sem_cls_scores': self.end_points['sem_cls_scores'],
                    'objectness_scores': self.end_points['objectness_scores'],
                    'center_label': self.end_points['center_label'],
                    'heading_class_label': self.end_points['heading_class_label'],
                    'heading_residual_label': self.end_points['heading_residual_label'],
                    'size_class_label': self.end_points['size_class_label'],
                    'size_residual_label': self.end_points['size_residual_label'],
                    'sem_cls_label': self.end_points['sem_cls_label'],
                    'box_label_mask': self.end_points['box_label_mask']}
        else:
            logger.error('mode param must equal to test or train.')
            exit(-1)