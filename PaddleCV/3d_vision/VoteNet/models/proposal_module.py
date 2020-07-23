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
import sys
import numpy as np
import paddle.fluid as fluid
from utils import conv1d
from pointnet2_modules import PointnetSAModuleVotes
from ext_op import *

__all__ = ['ProposalModule']

def decode_scores(net, end_points, num_class, num_heading_bin, num_size_cluster, mean_size_arr):
    net_transposed = fluid.layers.transpose(net, perm=[0, 2, 1])  # (batch_size, 1024, ..)
    batch_size = net_transposed.shape[0]
    num_proposal = net_transposed.shape[1]

    objectness_scores = net_transposed[:, :, 0:2]
    end_points['objectness_scores'] = objectness_scores

    base_xyz = end_points['aggregated_vote_xyz']  # (batch_size, num_proposal, 3)
    center = base_xyz + net_transposed[:, :, 2:5]  # (batch_size, num_proposal, 3)
    end_points['center'] = center

    heading_scores = net_transposed[:, :, 5:5 + num_heading_bin]
    heading_residuals_normalized = net_transposed[:, :, 5 + num_heading_bin:5 + num_heading_bin * 2]
    end_points['heading_scores'] = heading_scores # Bxnum_proposalxnum_heading_bin
    end_points['heading_residuals_normalized'] = heading_residuals_normalized # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
    end_points['heading_residuals'] = heading_residuals_normalized * (np.pi / num_heading_bin) # Bxnum_proposalxnum_heading_bin

    size_scores = net_transposed[:, :, 5 + num_heading_bin * 2:5 + num_heading_bin * 2 + num_size_cluster]
    size_residuals_normalized = net_transposed[:, :,
                                5 + num_heading_bin * 2 + num_size_cluster:5 + num_heading_bin * 2 + num_size_cluster * 4]

    size_residuals_normalized = fluid.layers.reshape(size_residuals_normalized, shape=[batch_size, num_proposal, num_size_cluster, 3]) # Bxnum_proposalxnum_size_clusterx3

    end_points['size_scores'] = size_scores
    end_points['size_residuals_normalized'] = size_residuals_normalized

    mean_size_arr_t = fluid.Tensor()
    mean_size_arr_t.set(mean_size_arr.astype(np.float32), fluid.CUDAPlace(0))
    mean_size_arr_t = fluid.layers.unsqueeze(fluid.layers.unsqueeze(mean_size_arr_t, axes=0), axes=0)

    end_points['size_residuals'] = size_residuals_normalized * mean_size_arr_t

    sem_cls_scores = net_transposed[:, :, 5 + num_heading_bin * 2 + num_size_cluster * 4:]  # Bxnum_proposalx10
    end_points['sem_cls_scores'] = sem_cls_scores
    return end_points

class ProposalModule(object):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling, seed_feat_dim=256):
        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim

    def build(self, xyz, features, end_points):
        if self.sampling == 'vote_fps':
            # Farthest point sampling (FPS) on votes
            xyz, features, fps_inds = PointnetSAModuleVotes(
                xyz=xyz,
                features=features,
                npoint=self.num_proposal,
                radius=0.3,
                nsample=16,
                mlps=[self.seed_feat_dim, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True
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
                mlps=[self.seed_feat_dim, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True
            )
        elif self.sampling == 'random':
            # Random sampling from the votes
            num_seed = end_points['seed_xyz'].shape()[1]
            batch_size = end_points['seed_xyz'].shape()[0]
            sample_inds = fluid.layers.randint(0, num_seed, shape=(batch_size, self.num_proposal), dtype='int32')
            xyz, features, _ = PointnetSAModuleVotes(
                xyz=xyz,
                features=features,
                inds=sample_inds,
                npoint=self.num_proposal,
                radius=0.3,
                nsample=16,
                mlps=[self.seed_feat_dim, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True
            )
        else:
            print('Unknown sampling strategy: %s. Exiting!'%(self.sampling))
            exit()
        end_points['aggregated_vote_xyz'] = xyz # (batch_size, num_proposal, 3)
        end_points['aggregated_vote_inds'] = sample_inds # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        # --------- PROPOSAL GENERATION ---------
        net = conv1d(input=features, num_filters=128, filter_size=1, bn=True, act='relu')
        net = conv1d(input=net, num_filters=128, filter_size=1, bn=True, act='relu')
        net = conv1d(input=net, num_filters=2+3+self.num_heading_bin*2+self.num_size_cluster*4+self.num_class,
                     filter_size=1, bn=False, act=None)

        end_points = decode_scores(net, end_points, self.num_class, self.num_heading_bin, self.num_size_cluster,
                                   self.mean_size_arr)
        return end_points

if __name__=='__main__':
    # TODO: test proposalModule

    pass