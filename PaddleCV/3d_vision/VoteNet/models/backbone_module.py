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

import sys
import os
import paddle.fluid as fluid

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))

from pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule

__all__ = ['Pointnet2Backbone']

class Pointnet2Backbone(object):
    def __init__(self,  input_feature_dim=0):
        self.input_feature_dim = input_feature_dim

    def build(self, xyz, feature=None, use_xyz=True,):
        featuresi = fluid.layers.transpose(feature, perm=[0, 2, 1])

        # --------- 4 SET ABSTRACTION LAYERS ---------
        l1_xyz, l1_feature, l1_inds = PointnetSAModuleVotes(
            xyz=xyz,
            features=featuresi,
            npoint=2048,
            radius=0.2,
            nsample=64,
            mlps=[self.input_feature_dim, 64, 64, 128],
            use_xyz=use_xyz,
            normalize_xyz=True,
            name='layer1'
        )

        l2_xyz, l2_feature, l2_inds = PointnetSAModuleVotes(
            xyz=l1_xyz,
            features=l1_feature,
            npoint=1024,
            radius=0.4,
            nsample=32,
            mlps=[128, 128, 128, 256],
            use_xyz=use_xyz,
            normalize_xyz=True,
            name='layer2'
        )

        l3_xyz, l3_feature, l3_inds = PointnetSAModuleVotes(
            xyz=l2_xyz,
            features=l2_feature,
            npoint=512,
            radius=0.8,
            nsample=16,
            mlps=[256, 128, 128, 256],
            use_xyz=use_xyz,
            normalize_xyz=True,
            name='layer3'
        )

        l4_xyz, l4_feature, l4_inds = PointnetSAModuleVotes(
            xyz=l3_xyz,
            features=l3_feature,
            npoint=256,
            radius=1.2,
            nsample=16,
            mlps=[256, 128, 128, 256],
            use_xyz=use_xyz,
            normalize_xyz=True,
            name='layer4'
        )

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        l3_feature = PointnetFPModule(l3_xyz, l4_xyz, l3_feature, l4_feature, mlps=[256+256, 256, 256], name='fa_layer1')
        fp2_feature = PointnetFPModule(l2_xyz, l3_xyz, l2_feature, l3_feature, mlps=[256+256, 256, 256], name='fa_layer2')

        num_of_seeds = l2_xyz.shape()[1]
        fp2_inds = l1_inds[:, 0:num_of_seeds] # indices among the entire input point clouds
        return l2_xyz, fp2_feature, fp2_inds

if __name__ == '__main__':
    pass