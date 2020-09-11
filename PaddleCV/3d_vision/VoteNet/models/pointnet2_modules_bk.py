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

import os
import numpy as np
import logging
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant, Normal
# from .ext_op import *
from .ext_op import *
from .loss_helper import *
# from utils import conv1d

__all__ = ["PointnetSAModuleVotes", "PointnetFPModule", "VoteNet", "VotingModule", "ProposalModule"]

logger = logging.getLogger(__name__)

MEAN_SIZE_ARR_PATH = os.path.join('dataset/scannet/meta_data/scannet_means.npz')
MEAN_SIZE_ARR = np.load(MEAN_SIZE_ARR_PATH)['arr_0']

MAX_NUM_OBJ = 64

# Checked!
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

        new_xyz_ex = fluid.layers.expand(layers.unsqueeze(layers.transpose(new_xyz, perm=[0, 2, 1]), -1), [1, 1, 1, grouped_xyz.shape[3]])

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

# Checked!
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

# Checked!
def conv1d(input,
           num_filters,
           filter_size,
           bn=True,
           bn_momentum=0.9,
           stride=1,
           padding=0,
           dilation=1,
           groups=None,
           use_cudnn=True,
           act='relu',
           conv_name=None,
           bn_name=None,
           data_format="NCHW"):
    """

    :param input: The input is 3-D Tensor with shape [N, C, W], the data type
            of input is float16 or float32 or float64.
    :param num_filters: The number of filter. It is as same as the output
            image channel.
    :param filter_size:
    :param stride:
    :param padding:
    :param dilation:
    :param groups:
    :param param_attr:
    :param bias_attr:
    :param use_cudnn:
    :param act:
    :param name:
    :param data_format:
    :return:
    """

    def _get_kaiming_init():
        fan_in = input.shape[1]
        std = (1.0 / fan_in / 3.0) ** 0.5
        return Normal(0., std, 0.)

    # Convert the input tensor from dim 3 to dim 4
    input_4dim = layers.unsqueeze(input, axes=-1)

    # Expand kernel dim from 1 to 2 by appending 1 to the end of the kernel list.
    if isinstance(filter_size, list):
        assert(len(filter_size) == 1)
        filter_size = filter_size.append(1)
    elif isinstance(filter_size, tuple):
        assert(len(filter_size) == 1)
        filter_size = list(filter_size)
        filter_size = filter_size.append(1)
    else:
        assert(isinstance(filter_size, int))
        filter_size = [filter_size, 1]

    param_attr = ParamAttr(name='{}.weight'.format(conv_name),
                           initializer=_get_kaiming_init())
    bias_attr = ParamAttr(name='{}.bias'.format(conv_name),
                          initializer=Constant(0.0))

    out_4dim = layers.conv2d(input=input_4dim,
                             num_filters=num_filters,
                             filter_size=filter_size,
                             stride=stride,
                             padding=padding,
                             dilation=dilation,
                             groups=groups,
                             param_attr=param_attr,
                             bias_attr=bias_attr,
                             use_cudnn=use_cudnn,
                             data_format=data_format)
    out_3dim = layers.squeeze(out_4dim, [-1])
    if bn:
        out_3dim = layers.batch_norm(out_3dim,
                                     act=act,
                                     momentum=bn_momentum,
                                     param_attr=ParamAttr(name=bn_name + '.weight'),
                                     bias_attr=ParamAttr(name=bn_name + '.bias'),
                                     moving_mean_name=bn_name + '.mean',
                                     moving_variance_name=bn_name + '.var')

    # Convert the output tensor from dim 4 back to dim 3
    return out_3dim

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

# Checked!
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

# Checked!
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

# Checked!
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

# Checked!
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

        return vote_xyz, vote_features

# Checked!
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

# Checked!
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

class testVoteNet(object):
    def __init__(self, num_class, num_points, num_heading_bin, num_size_cluster,
                 input_feature_dim=0, num_proposal=128, vote_factor=1, sampling='vote_fps', batch_size=1):
        self.num_class = num_class
        self.num_points = num_points
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.batch_size = batch_size

        # init Pointnet2Backbone
        self.backbone_net = Pointnet2Backbone(input_feature_dim=input_feature_dim)

        # init VotingModule
        self.vgen = VotingModule(self.vote_factor, 256, batch_size=batch_size, name='vgen')

        # init proposal module
        self.pnet = ProposalModule(num_class, num_heading_bin, num_size_cluster, num_proposal, sampling, name='pnet')

    def build(self, xyz, mean_size_arr=None):
        end_points = {}

        # Backbone point feature learning
        end_points = self.backbone_net.build(xyz, end_points=end_points)

        # =========== Hough voting ==========
        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features
        end_points['seed_inds'] = end_points['fp2_inds']

        # Convert features tensor from 'NWC' to 'NCW' format
        features = layers.transpose(features, perm=[0, 2, 1])  # NCW
        vote_xyz, vote_features = self.vgen.build(xyz, features, end_points)  # vote_features.shape = [B, out_dim, num_vote]
        features_norm = layers.sqrt(layers.reduce_sum(layers.pow(vote_features, factor=2.0), dim=1, keep_dim=False))
        vote_features = vote_features / layers.unsqueeze(features_norm, 1)  # features_norm.shape = [B, 1, num_vote]

        # Convert features tensor from 'NCW' to 'NWC'
        # vote_features = layers.transpose(vote_features, perm=[0, 2, 1])

        print('vote_features.shape = ', vote_features.shape)
        # end_points['vote_features'] = features_norm

        end_points['vote_xyz'] = vote_xyz
        end_points['vote_features'] = vote_features # [B, C, K] NCW format

        # Vote aggregation and detection
        end_points = self.pnet.build(vote_xyz, vote_features, end_points, batch_size=self.batch_size, mean_size_arr=mean_size_arr)

        return end_points

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

    def build_input(self, mode='train'):
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

        if mode == 'train':
            feed_list = [self.xyz, self.feature, self.center_label, self.heading_class_label, self.heading_residual_label,
                       self.size_class_label, self.size_residual_label, self.sem_cls_label, self.box_label_mask, self.vote_label,
                       self.vote_label_mask]

            self.loader = fluid.io.DataLoader.from_generator(
                feed_list=feed_list,
                capacity=64,
                use_double_buffer=True,
                iterable=False)
            self.feed_vars = feed_list
        elif mode == 'infer':
            feed_list = [self.xyz, self.feature]
        else:
            logger.error('Input mode can only either be train or infer.')
            exit(-1)

        return feed_list

    def build(self, mode='train'):
        end_points = {}

        # Backbone point feature learning
        end_points = self.backbone_net.build(self.xyz, self.feature, end_points)

        # =========== Hough voting ==========
        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features
        end_points['seed_inds'] = end_points['fp2_inds']

        vote_xyz, vote_features = self.vgen.build(xyz, features) # vote_features.shape = [B, out_dim, num_vote]
        features_norm = layers.sqrt(layers.reduce_sum(layers.pow(vote_features, factor=2.0), dim=1, keep_dim=False))
        vote_features = vote_features / layers.unsqueeze(features_norm, 1) # features_norm.shape = [B, 1, num_vote]

        end_points['vote_xyz'] = vote_xyz
        end_points['vote_features'] = vote_features

        # Vote aggregation and detection
        end_points = self.pnet.build(vote_xyz, vote_features, end_points, batch_size=self.batch_size)#, self.mean_size_arr)

        if mode == 'infer':
            return end_points

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

def prepare_input_data(num_points, num_class, input_feature_dim):

    xyz = fluid.data(name='xyz',
                     shape=[None, num_points, 3],
                     dtype='float32',
                     lod_level=0)
    feature = fluid.data(name='feature',
                         shape=[None, num_points, input_feature_dim],
                         dtype='float32',
                         lod_level=0)
    label = fluid.data(name='label',
                       shape=[None, num_points, 1],
                       dtype='int64',
                       lod_level=0)
    mean_size_arr = fluid.data(name='mean_size_arr',
                               shape=[num_class, 3],
                               dtype='float32',
                               lod_level=0)
    return xyz, feature, label, mean_size_arr

def test_backbone():
    # ================ Define param ================
    num_points = 4096
    num_class = 2

    # case #1 feature_dim = 0
    in_feature_dim = 0
    # ================ Define model ================
    xyz, feature, label = prepare_input_data(num_points, input_feature_dim=in_feature_dim)
    backbone = Pointnet2Backbone(input_feature_dim=in_feature_dim)

    seed_xyz, seed_features, seed_inds = backbone.build(xyz, feature)

    # ================ Define program ==============
    gpu = fluid.CUDAPlace(0)
    exe = fluid.Executor(gpu)
    exe.run(fluid.default_startup_program())

    # ================ Random gen input ============
    xyz_val = np.random.rand(1, num_points, 3).astype(np.float32)
    # feature_val =
    label_val = np.random.randint(0, num_class-1, size=(1, num_points, 1), dtype=np.int64)

    # ================ Run =========================
    outs = exe.run(
        feed={'xyz': xyz_val, 'label': label_val},
        fetch_list=[seed_xyz, seed_features, seed_inds]
    )

    print(outs.shape)


# if __name__ == '__main__':
#     sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
#     from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, DC
#     from loss_helper import get_loss
#
#     # Define model
#     model = VoteNet(10, 12, 10, np.random.random((10, 3))).cuda()
#
#     try:
#         # Define dataset
#         TRAIN_DATASET = SunrgbdDetectionVotesDataset('train', num_points=20000, use_v1=True)
#
#         # Model forward pass
#         sample = TRAIN_DATASET[5]
#         inputs = {'point_clouds': torch.from_numpy(sample['point_clouds']).unsqueeze(0).cuda()}
#     except:
#         print('Dataset has not been prepared. Use a random sample.')
#         inputs = {'point_clouds': torch.rand((20000, 3)).unsqueeze(0).cuda()}
#
#     end_points = model(inputs)
#     for key in end_points:
#         print(key, end_points[key])
#
#     try:
#         # Compute loss
#         for key in sample:
#             end_points[key] = torch.from_numpy(sample[key]).unsqueeze(0).cuda()
#         loss, end_points = get_loss(end_points, DC)
#         print('loss', loss)
#         end_points['point_clouds'] = inputs['point_clouds']
#         end_points['pred_mask'] = np.ones((1, 128))
#         dump_results(end_points, 'tmp', DC)
#     except:
#         print('Dataset has not been prepared. Skip loss and dump.')

def test_VoteNet():
    num_class = 10
    num_points = 20000
    num_heading_bin = 12
    num_size_cluster = 10
    input_feature_dim = 0
    batch_size = 1
    # mean_size_arr = layers.data(name='mean_size_arr', shape=[num_class, 3], dtype='float32') # np.random.random((num_class, 3))
    # inputs = layers.data(name='xyz')
    xyz = fluid.data(name='xyz', shape=[batch_size, num_points, 3], dtype='float32', lod_level=0)

    mean_size_arr_val = np.array([[0.02584677, 0.27209708, 0.91912236],
                                   [0.52118567, 0.09442533, 0.60527583],
                                   [0.18992851, 0.08463525, 0.74016808],
                                   [0.84663676, 0.71893957, 0.11271357],
                                   [0.82065344, 0.65177494, 0.11037908],
                                   [0.22033138, 0.64618324, 0.55625422],
                                   [0.66764257, 0.19630913, 0.23078685],
                                   [0.27257402, 0.0218716 , 0.2267122 ],
                                   [0.9639698 , 0.81415039, 0.87099402],
                                   [0.01661695, 0.71431973, 0.08257291]], dtype=np.float32)

    xyz_val = np.load('test_pcd.npz.npy').astype(np.float32)

    net = testVoteNet(num_class=num_class,
                      num_points=num_points,
                      num_heading_bin=num_heading_bin,
                      num_size_cluster=num_size_cluster,
                      input_feature_dim=input_feature_dim,
                      batch_size=batch_size)

    end_points = net.build(xyz, mean_size_arr=mean_size_arr_val)

    gpu = fluid.CUDAPlace(0)
    exe = fluid.Executor(gpu)
    exe.run(fluid.default_startup_program())

    # f_list = []
    #
    # for key in end_points:
    #     f_list.append(key)

    out = exe.run(
        feed={'xyz': xyz_val},
        fetch_list=[end_points['objectness_scores'],
                    end_points['center']]
    )

    center_tensor = out[1]

    print(center_tensor)

    # print(out)



def test_load_mean_size_arr():
    path = '../dataset/scannet/meta_data/scannet_means.npz'

    mean_size_arr = np.load(path)['arr_0']

    print('mean_size_arr.shape: {}'.format(mean_size_arr.shape))
    print('mean_size_arr[0]: {}'.format(mean_size_arr[0]))

if __name__ == '__main__':
    # test_backbone()
    # test_load_mean_size_arr()
    test_VoteNet()