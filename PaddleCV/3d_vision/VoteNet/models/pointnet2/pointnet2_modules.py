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
Contains PointNet++  utility functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant, Normal
from ext_op import *
from typing import List

__all__ = ["PointnetSAModuleVotes", "PointnetFPModule"]

def query_and_group(xyz, new_xyz, radius, nsample, features=None, use_xyz=True, \
                    ret_grouped_xyz=False, normalize_xyz=False, sample_uniformly=False, \
                    ret_unique_cnt=False):
    """
    Perform query_ball and group_points

    Args:
        xyz (Variable): xyz coordiantes features with shape [B, N, 3]
        new_xyz (Variable): centriods features with shape [B, npoint, 3]
        radius (float32): radius of ball
        nsample (int32): maximum number of gather features
        features (Variable): features with shape [B, N, C]
        use_xyz (bool): whether use xyz coordiantes features

    Returns:
        out (Variable): features with shape [B, npoint, nsample, C + 3]
    """
    idx = query_ball(xyz, new_xyz, radius, nsample)
    idx.stop_gradient = True

    # Added sample uniform option
    if sample_uniformly:
        unique_cnt = fluid.layers.zeros((idx.shape[0], idx.shape[1]))
        for i_batch in range(idx.shape[0]):
            for i_region in range(idx.shape[1]):
                _, unique_ind = fluid.layers.unique(idx[i_batch, i_region, :])
                num_unique = unique_ind.shape[0]
                unique_cnt[i_batch, i_region] = num_unique
                sample_ind = fluid.layers.randint(0, num_unique, shape=(nsample - num_unique,), dtype=np.int64)
                all_ind = fluid.layers.concat([unique_ind, unique_ind[sample_ind]])
                idx[i_batch, i_region, :] = all_ind

    xyz = fluid.layers.transpose(xyz, perm=[0, 2, 1]) # convert memory layout from NHC to NCH
    grouped_xyz = group_points(xyz, idx) # (B, 3, npoint, nsample)
    expand_new_xyz = fluid.layers.unsqueeze(fluid.layers.transpose(new_xyz, perm=[0, 2, 1]), axes=[-1])
    expand_new_xyz = fluid.layers.expand(expand_new_xyz, [1, 1, 1, grouped_xyz.shape[3]])
    grouped_xyz -= expand_new_xyz

    if normalize_xyz:
        grouped_xyz /= radius

    if features is not None:
        grouped_features = group_points(features, idx)
        new_features = fluid.layers.concat([grouped_xyz, grouped_features], axis=1) \
                if use_xyz else grouped_features
    else:
        assert use_xyz, "use_xyz should be True when features is None"
        new_features = grouped_xyz

    ret = [new_features]
    if ret_grouped_xyz:
        ret.append(grouped_xyz)
    if ret_unique_cnt:
        ret.append(unique_cnt)
    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)

def group_all(xyz, features=None, use_xyz=True, ret_grouped_xyz=False):
    """
    Group all xyz and features when npoint is None
    See query_and_group
    """
    xyz = fluid.layers.transpose(xyz,perm=[0, 2, 1])
    grouped_xyz = fluid.layers.unsqueeze(xyz, axes=[2])
    if features is not None:
        grouped_features = fluid.layers.unsqueeze(features, axes=[2])
        new_features = fluid.layers.concat([grouped_xyz, grouped_features], axis=1) if use_xyz else grouped_features
    else:
        new_features = grouped_xyz

    if ret_grouped_xyz:
        return new_features, grouped_xyz
    else:
        return new_features

def conv_bn(input, out_channels, bn=True, bn_momentum=0.95, act='relu', name=None):
    def _get_kaiming_init():
        fan_in = input.shape[1]
        std = (1.0 / fan_in / 3.0) ** 0.5
        return Normal(0., std, 0.)

    param_attr = ParamAttr(name='{}_conv_weight'.format(name),
                           initializer=_get_kaiming_init())
    bias_attr = ParamAttr(name='{}_conv_bias'.format(name)) \
                                  if not bn else False
    out = fluid.layers.conv2d(input,
                              num_filters=out_channels,
                              filter_size=1,
                              stride=1,
                              padding=0,
                              dilation=1,
                              param_attr=param_attr,
                              bias_attr=bias_attr,
			                  act=act if not bn else None)
    if bn:
        bn_name = name + "_bn"
        out = fluid.layers.batch_norm(out,
                                      act=act,
				                      momentum=bn_momentum,
                                      param_attr=ParamAttr(name=bn_name + "_scale"),
                                      bias_attr=ParamAttr(name=bn_name + "_offset"),
                                      moving_mean_name=bn_name + '_mean',
                                      moving_variance_name=bn_name + '_var')

    return out


def MLP(features, out_channels_list, bn=True, bn_momentum=0.95, act='relu', name=None):
    out = features
    for i, out_channels in enumerate(out_channels_list):
        out = conv_bn(out, out_channels, bn=bn, act=act, bn_momentum=bn_momentum, name=name + "_{}".format(i))
    return out


def PointnetSAModuleVotes(xyz, features = None, inds = None, mlp=[], npoint = None,
                          radius = None, nsample = None, bn = True, use_xyz = True,
                          pooling = 'max', sigma = None,  # for RBF pooling
                          normalize_xyz = False,  # noramlize local XYZ with radius
                          sample_uniformly = False, ret_unique_cnt = False):
    """
    PointNet MSG(Multi-Scale Group) Set Abstraction Module.
    Call with radiuss, nsamples, mlps as single element list for
    SSG(Single-Scale Group).

    Args:
        xyz (Variable): xyz coordiantes features with shape [B, N, 3]
        radius (float32): radius of ball
        nsample (int32): maximum number of gather features
        mlps ([int32]): out_channels_list
        feature (Variable): features with shape [B, C, N]
        bn (bool): whether perform batch norm after conv2d
    bn_momentum (float): momentum of batch norm
        use_xyz (bool): whether use xyz coordiantes features

    Returns:
        new_xyz (Variable): centriods features with shape [B, npoint, 3]
        out (Variable): features with shape [B, npoint, \sum_i{mlps[i][-1]}]
    """
    # assert len(radiuss) == len(nsamples) == len(mlps), \
    #         "radiuss, nsamples, mlps length should be same"
    if sigma is None:
        sigma = radius/2

    if npoint is not None:
        grouper = query_and_group(xyz, features, radius, nsample, use_xyz=use_xyz, ret_grouped_xyz=True,
                                  normalize_xyz=normalize_xyz, sample_uniformly=sample_uniformly,
                                  ret_unique_cnt=ret_unique_cnt)
    else:
        grouper = group_all(xyz, features, use_xyz=use_xyz, ret_grouped_xyz=True)

    # mlp_spec = mlp
    # if use_xyz and len(mlp_spec) > 0:
        # mlp_spec[0] += 3
    # mlp_module = MLP(mlp_spec, bn=bn)

    xyz_flipped = fluid.layers.transpose(xyz, perm=[0, 2, 1])
    if inds is None:
        inds = farthest_point_sampling(xyz, npoint)
    else:
        assert(inds.shape[1] == npoint)

    new_xyz = fluid.layers.transpose(gather_point(xyz_flipped, inds), perm=[0, 2, 1]) if npoint is not None else None

    if not ret_unique_cnt:
        grouped_features, grouped_xyz = grouper(xyz, new_xyz, features) # (B, C, npoint, nsample)
    else:
        # (B, C, npoint, nsample),(B, 3, npoint, nsample),(B,npoint)
        grouped_features, grouped_xyz, unique_cnt = grouper(xyz, new_xyz, features)



    # farthest_idx = farthest_point_sampling(xyz, npoint)
    # farthest_idx.stop_gradient = True
    # new_xyz = gather_point(xyz, farthest_idx) if npoint is not None else None
    #
    # outs = []
    # for i, (radius, nsample, mlp) in enumerate(zip(radiuss, nsamples, mlps)):
    #     out = query_and_group(xyz, new_xyz, radius, nsample, feature, use_xyz) if npoint is not None else group_all(xyz, feature, use_xyz)
    #     out = MLP(out, mlp, bn=bn, bn_momentum=bn_momentum, name=name + '_mlp{}'.format(i))
    #     out = fluid.layers.pool2d(out, pool_size=[1, out.shape[3]], pool_type='max')
    #     out = fluid.layers.squeeze(out, axes=[-1])
    #     outs.append(out)
    # out = fluid.layers.concat(outs, axis=1)

    # return (new_xyz, out)
    pass


class PointnetFPModule(object):
    def __init__(self):
        pass

if __name__=='__main__':
    # test pointnet2 functions

    pass