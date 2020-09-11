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
Contains PointNet++ backbone.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid.layers as layers

from .pointnet2_utils import PointnetSAModuleVotes, PointnetFPModule

__all__ = ['Pointnet2Backbone']

class Pointnet2Backbone(object):

    def __init__(self,  input_feature_dim=0, batch_size=1, bn_momentum=0.9, name=None):
        self.input_feature_dim = input_feature_dim
        self.batch_size = batch_size
        self.bn_momentum = bn_momentum
        self.name = name

    def build(self, xyz, features=None, end_points=None):
        """
        Backbone module for VoteNet.

        :param xyz: contains a list of points in xyz coord, shape=[None, num_points, 3]
        :param features: contains a list of feature with C channels for each point. shape=[None, num_points, C]
        :param end_points: save intermediate output
        :return: end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        if features is not None:
            features = layers.transpose(features, perm=[0, 2, 1])

        # --------- 4 SET ABSTRACTION LAYERS ---------
        l1_xyz, l1_feature, l1_inds = PointnetSAModuleVotes(
            xyz=xyz,
            features=features,
            npoint=2048,
            radius=0.2,
            nsample=64,
            mlps=[64, 64, 128],
            use_xyz=True,
            normalize_xyz=True,
            name=self.name+'.sa1',
            end_points=end_points,
            bn_momentum=self.bn_momentum
        )

        # Save SA1 layer output result
        end_points['sa1_inds'] = l1_inds
        end_points['sa1_xyz'] = l1_xyz
        end_points['sa1_features'] = l1_feature

        # print('sa1_inds.shape', l1_inds.shape)
        # print('sa1_xyz.shape', l1_xyz.shape)
        # print('sa1_features.shape', l1_feature.shape)

        l2_xyz, l2_feature, l2_inds = PointnetSAModuleVotes(
            xyz=l1_xyz,
            features=l1_feature,
            npoint=1024,
            radius=0.4,
            nsample=32,
            mlps=[128, 128, 256],
            use_xyz=True,
            normalize_xyz=True,
            name=self.name+'.sa2',
            bn_momentum=self.bn_momentum
        )

        # print('sa2_inds.shape', l2_inds.shape)
        # print('sa2_xyz.shape', l2_xyz.shape)
        # print('sa2_features.shape', l2_feature.shape)

        # Save SA2 layer output result
        end_points['sa2_inds'] = l2_inds
        end_points['sa2_xyz'] = l2_xyz
        end_points['sa2_features'] = l2_feature

        l3_xyz, l3_feature, l3_inds = PointnetSAModuleVotes(
            xyz=l2_xyz,
            features=l2_feature,
            npoint=512,
            radius=0.8,
            nsample=16,
            mlps=[128, 128, 256],
            use_xyz=True,
            normalize_xyz=True,
            name=self.name+'.sa3',
            bn_momentum=self.bn_momentum
        )

        # Save SA3 layer output result
        end_points['sa3_xyz'] = l3_xyz
        end_points['sa3_features'] = l3_feature

        # print('sa3_xyz.shape', l3_xyz.shape)
        # print('sa3_features.shape', l3_feature.shape)

        l4_xyz, l4_feature, l4_inds = PointnetSAModuleVotes(
            xyz=l3_xyz,
            features=l3_feature,
            npoint=256,
            radius=1.2,
            nsample=16,
            mlps=[128, 128, 256],
            use_xyz=True,
            normalize_xyz=True,
            name=self.name+'.sa4',
            bn_momentum=self.bn_momentum
        )

        # Save SA4 layer output result
        end_points['sa4_xyz'] = xyz
        end_points['sa4_features'] = features

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        fp1_feature = PointnetFPModule(unknown=l3_xyz, known=l4_xyz, unknown_feats=l3_feature, known_feats=l4_feature,
                                      mlps=[256, 256], name=self.name+'.fp1', batch_size=self.batch_size, end_points=end_points,
                                       bn_momentum=self.bn_momentum)
        fp2_feature = PointnetFPModule(unknown=l2_xyz, known=l3_xyz, unknown_feats=l2_feature, known_feats=fp1_feature,
                                       mlps=[256, 256], name=self.name+'.fp2', batch_size=self.batch_size, bn_momentum=self.bn_momentum)

        # Save fp layer output
        end_points['fp2_features'] = fp2_feature
        end_points['fp2_xyz'] = end_points['sa2_xyz']

        num_of_seeds = end_points['fp2_xyz'].shape[1]

        end_points['fp2_inds'] = end_points['sa1_inds'][:, 0:num_of_seeds] # indices among the entire input point clouds
        return end_points