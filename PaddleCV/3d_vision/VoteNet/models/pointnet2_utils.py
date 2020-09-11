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
Contains PointNet++ utility functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import paddle.fluid.layers as layers

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))

from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant, Normal
from ext_op import *

__all__ = ["PointnetSAModuleVotes", "PointnetFPModule"]

def PointnetSAModuleVotes(xyz,
                          features=None,
                          inds=None,
                          mlps=None,
                          npoint=None,
                          radius=None,
                          nsample=None,
                          bn=True,
                          use_xyz=True,
                          pooling='max',
                          sigma=None,  # for RBF pooling
                          normalize_xyz=False,  # normalize local XYZ with radius
                          sample_uniformly=False,
                          ret_unique_cnt=False,
                          bn_momentum=0.9,
                          name=None,
                          end_points=None):
    """
    PointNet MSG(Multi-Scale Group) Set Abstraction Module.
    Call with radiuss, nsamples, mlps as single element list for
    SSG(Single-Scale Group).

    Args:
        xyz (Variable): xyz coordiantes features with shape [B, N, 3]
        feature (Variable): features with shape [B, C, N]
        radius (float32): radius of ball
        nsample (int32): maximum number of gather features
        mlp ([int32]): out_channels_list
        bn (bool): whether perform batch norm after conv2d
        use_xyz (bool): whether use xyz coordiantes features
	    bn_momentum (float): momentum of batch norm
	    pooling (string): pooling strategy
	    sigma (float): sigma for RBF pooling
	    normalize_xyz (bool): normalized local XYZ with radius
	    sample_uniformly (bool): whether sample uniformly
        ret_unique_cnt (bool):

    Returns:
        new_xyz (Variable): centriods features with shape [B, npoint, 3]
        new_features (Variable): features with shape [B, npoint, \sum_i{mlps[i][-1]}]
        inds (Variable):
        unique_cnt(optional Variable):
    """
    if sigma is None:
        sigma = radius/2

    # sample and grouping
    if npoint is not None:
        grouper = QueryAndGroup(radius=radius,
                                nsample=nsample,
                                use_xyz=use_xyz,
                                ret_grouped_xyz=True,
                                normalize_xyz=normalize_xyz,
                                sample_uniformly=sample_uniformly,
                                ret_unique_cnt=ret_unique_cnt,
                                end_points=end_points)
    else:
        grouper = GroupAll(use_xyz=use_xyz, ret_grouped_xyz=True)

    mlp_spec = mlps

    if inds is None:
        inds = farthest_point_sampling(xyz, npoint)  # [B, M], (M=nsample)
    else:
        assert(inds.shape[1] == npoint)

    new_xyz = gather_point(xyz, inds) if npoint is not None else None  # [B, M, 3], (M=nsample)

    # logger.info('PointnetSAModuleVotes: new_xyz.shape = {}'.format(new_xyz.shape))

    if not ret_unique_cnt:
        # (B, C, npoint, nsample),(B, 3, npoint, nsample, 3)
        grouped_features, grouped_xyz = grouper.build(xyz, new_xyz, features)
    else:
        # (B, C, npoint, nsample, C),(B, 3, npoint, nsample, 3),(B,npoint)
        grouped_features, grouped_xyz, unique_cnt = grouper.build(xyz, new_xyz, features)

    new_features = MLP(grouped_features, out_channels_list=mlp_spec, bn=bn, bn_momentum=bn_momentum, name=name+'.mlp_module')

    # Pooling
    if pooling == 'max':  # NCHW format
        new_features = layers.pool2d(new_features, pool_size=[1, new_features.shape[3]], pool_type='max')
    elif pooling == 'avg':
        new_features = layers.pool2d(new_features, pool_size=[1, new_features.shape[3]], pool_type='avg')
    elif pooling == 'rbf':
        # Use radial basis function kernel for weighted sum of features (normalized by nsample and sigma)
        # Ref: https://en.wikipedia.org/wiki/Radial_basis_function_kernel
        g_xyz_pow = layers.pow(grouped_xyz, factor=2)
        g_xyz_pow_sum = layers.reduce_sum(g_xyz_pow, dim=1, keep_dim=False)
        rbf = layers.exp(-1 * g_xyz_pow_sum / (sigma**2) / 2) # shape=(B, npoint, nsample)

        # new_features: shape = (B, mlp[-1], npoint, 1)
        new_features = layers.reduce_sum(new_features * layers.unsqueeze(rbf, axes=1), dim=-1, keep_dim=True) / float(nsample)

    new_features = layers.squeeze(new_features, axes=[-1])  # shape=[B, mlp_spec[-1], npoint]

    if not ret_unique_cnt:
        # new_xyz:shape=[B, nsample, 3], new_features:shape=[B, npoint, mlps[-1]], inds:shape=[B, nsample]
        return new_xyz, new_features, inds
    else:
        return new_xyz, new_features, inds, unique_cnt

def PointnetFPModule(unknown,
                     known,
                     unknown_feats,
                     known_feats,
                     mlps=None,
                     bn=True,
                     bn_momentum=0.9,
                     name=None,
                     batch_size=1,
                     end_points=None):
    """
    :param unknown: (B, n, 3) tensor of the xyz positions of the unknown features
    :param known: (B, m, 3) tensor of the xyz positions of the known features
    :param unknown_feats: (B, C1, n) tensor of the features to be propigated to
    :param known_feats: (B, C2, m) tensor of features to be propigated
    :param mlps: Pointnet module parameters
    :param bn: Use batchnorm
    :param bn_m: batchnorm momentum figure
    :return new_features: (B, n, mlps[-1]) tensor of the features of the unknown features
    """

    # logger.info('FPModule_{} unknown.shape = {}'.format(name, unknown.shape))
    # logger.info('FPModule_{} known.shape = {}'.format(name, known.shape))
    # logger.info('FPModule_{} unknown_feats.shape = {}'.format(name, unknown_feats.shape))
    # logger.info('FPModule_{} known_feats.shape = {}'.format(name, known_feats.shape))

    if known is not None:
        dist, idx = three_nn(unknown, known)  # dist:shape=[B, N, 3], idx:shape=[B, N, 3]
        dist.stop_gradient = True
        idx.stop_gradient = True
        dist_recip = 1.0 / (dist + 1e-8)
        norm = layers.reduce_sum(dist_recip, dim=2, keep_dim=True)  # norm:shape=[B, N, 1]
        weight = dist_recip / norm  # weight:shape=[B, N, 3]
        weight.stop_gradient = True

        # known_feats.shape = [B, M, C], weight.shape = [B, N, 3], idx.shape = [B, N, 3]
        known_feats_t = layers.transpose(known_feats, perm=[0, 2, 1])
        interpolated_feats = three_interp(input=known_feats_t, weight=weight, idx=idx)  # out_shape=[B, N, C]
    else:
        known_feats_t = layers.transpose(known_feats, perm=[0, 2, 1])
        interpolated_feats = layers.expand(known_feats,
                                           expand_times=[batch_size, unknown.shape[1], known_feats_t.shape[2]]) # [B, N, C]

    if unknown_feats is not None:
        unknown_feats_t = layers.transpose(unknown_feats, perm=[0, 2, 1])
        new_features = layers.concat([interpolated_feats, unknown_feats_t], axis=-1)  # shape=(B, N, C2 + C1)
    else:
        new_features = interpolated_feats

    # Convert feature tensor from 'NWC' to 'NCW'
    new_features = layers.transpose(new_features, perm=[0, 2, 1])  # shape=(B, C2 + C1, N)
    new_features = layers.unsqueeze(new_features, axes=-1)  # shape=(B, C2+C1, N, 1)
    # for i, num_out_channel in enumerate(mlps):
    new_features = MLP(new_features, out_channels_list=mlps, bn=bn, bn_momentum=bn_momentum, name=name+'.mlp')
    new_features = layers.squeeze(new_features, axes=[-1])  # shape=(B, C2+C1, N)

    # Convert feature tensor from 'NCW' to 'NWC'
    # new_features = layers.transpose(new_features, perm=[0, 2, 1])  # shape=[B, N, C]
    return new_features

class GroupAll(object):
    def __init__(self, use_xyz=True, ret_grouped_xyz=False):
        self.use_xyz = use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz

    def build(self, xyz, new_xyz, features=None):
        r"""
        Parameters
        ---------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            Ignored
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 1, N, C + 3) tensor
        """
        grouped_xyz = layers.unsqueeze(xyz, axes=1)
        if features is not None:
            features = layers.transpose(features, perm=[0, 2, 1])
            grouped_features = layers.unsqueeze(features, axes=1)
            if self.use_xyz:
                new_features = layers.concat(input=[grouped_xyz, grouped_features], axis=-1)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        if self.ret_grouped_xyz:
            return new_features, grouped_xyz
        else:
            return new_features

class QueryAndGroup(object):
    def __init__(self,
                 radius,
                 nsample,
                 use_xyz=True,
                 ret_grouped_xyz=False,
                 normalize_xyz=False,
                 sample_uniformly=False,
                 ret_unique_cnt=False,
                 end_points=None):
        """
        :param radius (float32): radius of ball
        :param nsample (int32): maximum number of gather features
        :param use_xyz (bool): whether use xyz coordiantes features
        :param ret_grouped_xyz (bool): whether return grouped_xyz variable
        :param normalize_xyz (bool): whether normalize the grouped_xyz var by dividing radius
        :param sample_uniformly (bool): whether use uniform sampling instead of using query_ball result
        :param ret_unique_cnt (bool): whether return the unique cnt
        """
        self.radius = radius
        self.nsample = nsample
        self.use_xyz = use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt
        if self.ret_unique_cnt:
            assert(self.sample_uniformly)
        self.end_points = end_points

    def build(self, xyz, new_xyz, features=None):
        """
            Perform query_ball and group_points operation

            Args:
                xyz (Variable): xyz coordiantes features with shape [B, N, 3]
                new_xyz (Variable): centriods features with shape [B, npoint, 3]
                features (Variable): features with shape [B, C, N]
            Returns:
                out (Variable): features with shape [B, npoint, nsample, C + 3]
            """

        # idx:shape=[B, npoint, nsample]
        idx = query_ball(input=xyz, new_points=new_xyz, radius=self.radius, n_sample=self.nsample)
        idx.stop_gradient = True

        # Added sample uniform option
        if self.sample_uniformly:
            unique_cnt = layers.zeros(shape=(idx.shape[0], idx.shape[1]), dtype='float32') # shape=[B, npoint]
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    _, unique_ind = layers.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = layers.randint(0, num_unique, shape=(self.nsample - num_unique,), dtype='int64')
                    all_ind = layers.concat([unique_ind, unique_ind[sample_ind]])
                    idx[i_batch, i_region, :] = all_ind

        xyz_t = layers.transpose(xyz, perm=[0, 2, 1])

        grouped_xyz = group_points(xyz_t, idx)  # output_shape=[B, 3, npoint, nsample]

        new_xyz_ex = layers.expand(layers.unsqueeze(layers.transpose(new_xyz, perm=[0, 2, 1]), -1), [1, 1, 1, grouped_xyz.shape[3]])

        grouped_xyz -= new_xyz_ex  # shape=[B, 3, npoint, nsample]

        if self.normalize_xyz:
            grouped_xyz /= self.radius

        if features is not None:
            grouped_features = group_points(features, idx)

            # new_features: shape = [B, 3 + C, npoint, nsample]
            new_features = layers.concat([grouped_xyz, grouped_features], axis=1) if self.use_xyz else grouped_features
        else:
            new_features = grouped_xyz

        ret = [new_features]
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            ret.append(unique_cnt)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)

def conv_bn(input, out_channels, bn=True, bn_momentum=0.9, act='relu', name=None):
    def _get_kaiming_init():
        fan_in = input.shape[1]
        std = (1.0 / fan_in / 3.0) ** 0.5
        return Normal(0., std, 0.)

    # param_attr = ParamAttr(name='{}_conv_weight'.format(name),
    #                        initializer=Constant(1.0))
    # bias_attr = ParamAttr(name='{}_conv_bias'.format(name),
    #                       initializer=Constant(0.0)) if not bn else None
    param_attr = ParamAttr(name=name+'.conv.weight',
                           initializer=_get_kaiming_init())
    bias_attr = ParamAttr(name=name+'.conv.bias') if not bn else False

    out = layers.conv2d(input,
                        num_filters=out_channels,
                        filter_size=1,
                        stride=1,
                        padding=0,
                        dilation=1,
                        param_attr=param_attr,
                        bias_attr=bias_attr,
                        act=act if not bn else None)
    if bn:
        bn_name = name + ".bn"
        out = layers.batch_norm(out,
                                act=act,
                                momentum=bn_momentum,
                                param_attr=ParamAttr(name=bn_name + ".bn.weight"),
                                bias_attr=ParamAttr(name=bn_name + ".bn.bias"),
                                moving_mean_name=bn_name + '.bn.mean',
                                moving_variance_name=bn_name + '.bn.var')

    return out

def MLP(features, out_channels_list, bn=True, bn_momentum=0.9, act='relu', name=None):
    out = features
    for i, out_channels in enumerate(out_channels_list):
        out = conv_bn(out, out_channels, bn=bn, act=act, bn_momentum=bn_momentum, name=name + ".layer{}".format(i))
    return out
