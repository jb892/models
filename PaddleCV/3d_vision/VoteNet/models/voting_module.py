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
Contains Voting module.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid
import paddle.fluid.layers as layers
from utils import conv1d

__all__ = ['VotingModule']

class VotingModule(object):
    def __init__(self, vote_factor, seed_feature_dim, batch_size=1, name=None, bn_momentum=0.99):
        """ Votes generation from seed point features.

        Args:
            vote_facotr: int
                number of votes generated from each seed point
            seed_feature_dim: int
                number of channels of seed point features
            vote_feature_dim: int
                number of channels of vote features
        """
        self.vote_factor = vote_factor
        self.in_dim = seed_feature_dim
        self.out_dim = self.in_dim  # due to residual feature, in_dim has to be == out_dim
        self.name = name
        self.batch_size = batch_size
        self.bn_momentum = bn_momentum

    def build(self, seed_xyz, seed_features, end_points=None):
        """
        Arguments:
            seed_xyz: (batch_size, num_seed, 3) Pytorch tensor - NWC
            seed_features: (batch_size, feature_dim, num_seed) - NCW
        Returns:
            vote_xyz: (batch_size, num_seed*vote_factor, 3)
            vote_features: (batch_size, vote_feature_dim, num_seed*vote_factor)
        """
        batch_size = self.batch_size
        num_seed = seed_xyz.shape[1]
        num_vote = num_seed * self.vote_factor

        net = conv1d(seed_features, self.in_dim, filter_size=1, conv_name=self.name+'.conv1', bn_name=self.name+'.bn1', bn_momentum=self.bn_momentum)
        net = conv1d(net, self.in_dim, filter_size=1, conv_name=self.name+'.conv2', bn_name=self.name+'.bn2', bn_momentum=self.bn_momentum)
        net = conv1d(net,
                     num_filters=(3+self.out_dim)*self.vote_factor,
                     filter_size=1,
                     bn=False,
                     act=None,
                     conv_name=self.name+'.conv3')  # (batch_size, (3+out_dim)*vote_factor, num_seed)

        net = layers.transpose(net, perm=[0, 2, 1])
        net = layers.reshape(net, shape=[batch_size, num_seed, self.vote_factor, 3+self.out_dim])
        offset = net[:, :, :, 0:3]
        vote_xyz = layers.unsqueeze(seed_xyz, 2) + offset
        vote_xyz = layers.reshape(vote_xyz, shape=[batch_size, num_vote, 3])  # vote_xyz.shape = [B, num_vote, 3]

        residual_features = net[:, :, :, 3:]  # (batch_size, num_seed, vote_factor, out_dim)

        vote_features = layers.unsqueeze(layers.transpose(seed_features, perm=[0, 2, 1]), 2) + residual_features
        vote_features = layers.reshape(vote_features, shape=[batch_size, num_vote, self.out_dim])
        vote_features = layers.transpose(vote_features, perm=[0, 2, 1])
        # vote_features.shape = [B, out_dim, num_vote]

        # print('vote_features.shape = ', vote_features.shape)
        # if end_points is not None:class VotingModule(object):
    def __init__(self, vote_factor, seed_feature_dim):
        """ Votes generation from seed point features.

        Args:
            vote_facotr: int
                number of votes generated from each seed point
            seed_feature_dim: int
                number of channels of seed point features
            vote_feature_dim: int
                number of channels of vote features
        """
        self.vote_factor = vote_factor
        self.in_dim = seed_feature_dim
        self.out_dim = self.in_dim  # due to residual feature, in_dim has to be == out_dim

    def build(self, seed_xyz, seed_features):
        """
        Arguments:
            seed_xyz: (batch_size, num_seed, 3) Pytorch tensor
            seed_features: (batch_size, feature_dim, num_seed) Pytorch tensor
        Returns:
            vote_xyz: (batch_size, num_seed*vote_factor, 3)
            vote_features: (batch_size, vote_feature_dim, num_seed*vote_factor)
        """
        batch_size = seed_xyz.shape()[0]
        num_seed = seed_xyz.shape()[1]
        num_vote = num_seed * self.vote_factor

        net = conv1d(seed_features, self.out_dim, filter_size=1)
        net = conv1d(net, self.out_dim, filter_size=1)
        net = conv1d(net,
                     num_filters=(3+self.out_dim)*self.vote_factor,
                     filter_size=1,
                     bn=False,
                     act=None)  # (batch_size, (3+out_dim)*vote_factor, num_seed)

        net = fluid.layers.transpose(net, perm=[0, 2, 1])
        net = fluid.layers.reshape(net, shape=[batch_size, num_seed, self.vote_factor, 3+self.out_dim])
        offset = net[:, :, :, 0:3]
        vote_xyz = fluid.layers.unsqueeze(seed_xyz, 2) + offset
        vote_xyz = fluid.layers.reshape(vote_xyz, shape=[batch_size, num_vote, 3])

        residual_features = net[:, :, :, 3:] # (batch_size, num_seed, vote_factor, out_dim)
        vote_features = fluid.layers.unsqueeze(fluid.layers.transpose(seed_features, perm=[0, 2, 1]), 2) + residual_features
        vote_features = fluid.layers.reshape(vote_features, shape=[batch_size, num_vote, self.out_dim])
        vote_features = fluid.layers.transpose(vote_features, perm=[0, 2, 1])

        return vote_xyz, vote_features
        #     end_points['voting_features'] = vote_features

        return vote_xyz, vote_features