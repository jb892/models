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
Contains Proposal module.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers

from utils import conv1d
from .pointnet2_utils import PointnetSAModuleVotes
from ext_op import *

__all__ = ['ProposalModule']

MEAN_SIZE_ARR_PATH = os.path.join('dataset/scannet/meta_data/scannet_means.npz')
MEAN_SIZE_ARR = np.load(MEAN_SIZE_ARR_PATH)['arr_0']

def decode_scores(net, end_points, num_class, num_heading_bin, num_size_cluster, mean_size_arr=None, batch_size=1):
    """
    :param net: (batch_size, 1024, ..)
    :param end_points:
    :param num_class:
    :param num_heading_bin:
    :param num_size_cluster:
    :param mean_size_arr:
    :param batch_size:
    :return:
    """
    net = layers.transpose(net, perm=[0, 2, 1])

    # num_proposal should be 256
    num_proposal = net.shape[1]

    objectness_scores = net[:, :, 0:2]
    end_points['objectness_scores'] = objectness_scores

    base_xyz = end_points['aggregated_vote_xyz']  # (batch_size, num_proposal, 3)
    center = base_xyz + net[:, :, 2:5]  # (batch_size, num_proposal, 3)
    end_points['center'] = center

    heading_scores = net[:, :, 5:5 + num_heading_bin]
    heading_residuals_normalized = net[:, :, 5 + num_heading_bin:5 + num_heading_bin * 2]
    end_points['heading_scores'] = heading_scores  # Bxnum_proposalxnum_heading_bin
    end_points[
        'heading_residuals_normalized'] = heading_residuals_normalized  # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
    end_points['heading_residuals'] = heading_residuals_normalized * (
                np.pi / num_heading_bin)  # Bxnum_proposalxnum_heading_bin

    size_scores = net[:, :, 5 + num_heading_bin * 2:5 + num_heading_bin * 2 + num_size_cluster]
    size_residuals_normalized = layers.reshape(net[:, :,
                                5 + num_heading_bin * 2 + num_size_cluster:5 + num_heading_bin * 2 + num_size_cluster * 4],
                                shape=[batch_size, num_proposal, num_size_cluster, 3])  # Bxnum_proposalxnum_size_clusterx3
    end_points['size_scores'] = size_scores
    end_points['size_residuals_normalized'] = size_residuals_normalized
    if mean_size_arr is None:
        mean_size_arr = layers.assign(np.array(MEAN_SIZE_ARR, dtype='float32'))
    else:
        mean_size_arr = layers.assign(np.array(mean_size_arr, dtype='float32'))
    end_points['size_residuals'] = size_residuals_normalized * layers.unsqueeze(layers.unsqueeze(mean_size_arr, 0), 0)

    sem_cls_scores = net[:, :, 5 + num_heading_bin * 2 + num_size_cluster * 4:]  # Bxnum_proposalx10
    end_points['sem_cls_scores'] = sem_cls_scores
    return end_points

class ProposalModule(object):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, num_proposal, sampling, seed_feat_dim=256,
                 name=None, bn_momentum=0.9):
        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        # self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim
        self.name = name
        self.bn_momentum = bn_momentum

    def build(self, xyz, features, end_points, batch_size=1, mean_size_arr=None):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4)
        """
        if self.sampling == 'vote_fps':
            # Farthest point sampling (FPS) on votes
            xyz, features, fps_inds = PointnetSAModuleVotes(
                xyz=xyz,
                features=features,
                npoint=self.num_proposal,
                radius=0.3,
                nsample=16,
                mlps=[128, 128, 128],
                use_xyz=True,
                normalize_xyz=True,
                name=self.name+'.vote_aggregation',
                bn_momentum=self.bn_momentum
            )
            sample_inds = fps_inds
        elif self.sampling == 'seed_fps':
            # FPS on seed and choose the votes corresponding to the seeds
            # This gets us a slightly better coverage of *object* votes than vote_fps (which tends to get more cluster votes)
            sample_inds = farthest_point_sampling(end_points['seed_xyz'], self.num_proposal)
            xyz, features, _ = PointnetSAModuleVotes(
                xyz=xyz,
                features=features,
                inds=sample_inds,
                npoint=self.num_proposal,
                radius=0.3,
                nsample=16,
                mlps=[128, 128, 128],
                use_xyz=True,
                normalize_xyz=True,
                name=self.name+'.vote_aggregation',
                bn_momentum=self.bn_momentum
            )
        elif self.sampling == 'random':
            # Random sampling from the votes
            num_seed = end_points['seed_xyz'].shape[1]
            batch_size = batch_size
            sample_inds = fluid.layers.randint(0, num_seed, shape=(batch_size, self.num_proposal), dtype='int32')
            xyz, features, _ = PointnetSAModuleVotes(
                xyz=xyz,
                features=features,
                inds=sample_inds,
                npoint=self.num_proposal,
                radius=0.3,
                nsample=16,
                mlps=[128, 128, 128],
                use_xyz=True,
                normalize_xyz=True,
                name=self.name+'.vote_aggregation',
                bn_momentum=self.bn_momentum
            )
        else:
            print('Unknown sampling strategy: %s. Exiting!'%(self.sampling))
            exit()
        end_points['aggregated_vote_xyz'] = xyz # (batch_size, num_proposal, 3)
        end_points['aggregated_vote_inds'] = sample_inds # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal
        end_points['aggregated_vote_features'] = features


        # --------- PROPOSAL GENERATION ---------
        net = conv1d(input=features, num_filters=128, filter_size=1, bn=True,
                     conv_name=self.name+'.conv1', bn_name=self.name+'.bn1', bn_momentum=self.bn_momentum)
        net = conv1d(input=net, num_filters=128, filter_size=1, bn=True,
                     conv_name=self.name+'.conv2', bn_name=self.name+'.bn2', bn_momentum=self.bn_momentum)
        net = conv1d(input=net,
                     num_filters=2+3+self.num_heading_bin*2+self.num_size_cluster*4+self.num_class,
                     filter_size=1,
                     bn=False,
                     act=None,
                     conv_name=self.name+'.conv3',
                     bn_momentum=self.bn_momentum) #(batch_size, 2+3+num_heading_bin*2+num_size_cluster*4, num_proposal)
        end_points['pnet_conv1d'] = net

        end_points = decode_scores(net, end_points, self.num_class, self.num_heading_bin, self.num_size_cluster,
                                   mean_size_arr, batch_size)
        return end_points
