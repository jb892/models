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
from utils import conv1d

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


def PointnetSAModuleVotes(xyz, features = None, inds = None, mlps=None, npoint = None,
                          radius = None, nsample = None, bn = True, use_xyz = True,
                          pooling = 'max', sigma = None,  # for RBF pooling
                          normalize_xyz = False,  # noramlize local XYZ with radius
                          sample_uniformly = False, ret_unique_cnt = False, bn_m=0.95, name=None):
    """
    PointNet MSG(Multi-Scale Group) Set Abstraction Module.
    Call with radiuss, nsamples, mlps as single element list for
    SSG(Single-Scale Group).

    Args:
        xyz (Variable): xyz coordiantes features with shape [B, N, 3]
        radius (float32): radius of ball
        nsample (int32): maximum number of gather features
        mlp ([int32]): out_channels_list
        feature (Variable): features with shape [B, C, N]
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
        grouper = query_and_group(xyz, features, radius, nsample, use_xyz=use_xyz, ret_grouped_xyz=True,
                                  normalize_xyz=normalize_xyz, sample_uniformly=sample_uniformly,
                                  ret_unique_cnt=ret_unique_cnt)
    else:
        grouper = group_all(xyz, features, use_xyz=use_xyz, ret_grouped_xyz=True)

    mlp_spec = mlps
    if use_xyz and len(mlp_spec) > 0:
        mlp_spec[0] += 3

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

    # MLP
    for i, num_out_channel in enumerate(mlps):
        grouped_features = MLP(grouped_features, out_channels_list=num_out_channel, bn=bn, bn_momentum=bn_m, name=name+'_mlp{}'.format(i))
    new_features = grouped_features # (B, mlp[-1], npoint, nsample)

    # Pooling
    if pooling == 'max':
        new_features_shape = fluid.layers.shape(new_features)
        new_features = fluid.layers.pool2d(new_features, pool_size=[1, new_features_shape[3]], pool_type='max')
    elif pooling == 'avg':
        new_features_shape = fluid.layers.shape(new_features)
        new_features = fluid.layers.pool2d(new_features, pool_size=[1, new_features_shape[3]], pool_type='avg')
    elif pooling == 'rbf':
        # Use radial basis function kernel for weighted sum of features (normalized by nsample and sigma)
        # Ref: https://en.wikipedia.org/wiki/Radial_basis_function_kernel
        g_xyz_pow = fluid.layers.pow(grouped_xyz, factor=2)
        g_xyz_pow_sum = fluid.layers.reduce_sum(g_xyz_pow, dim=1, keep_dim=False)

        # (B, npoint, nsample)
        rbf = fluid.layers.exp(-1 * g_xyz_pow_sum / (sigma**2) / 2)

        # (B, mlp[-1], npoint, 1)
        new_features = fluid.layers.reduce_sum(new_features * fluid.layers.unsqueeze(rbf, [1]), dim=-1, keep_dim=True) / float(nsample)

    new_features = fluid.layers.unsqueeze(new_features, [1])

    if not ret_unique_cnt:
        return new_xyz, new_features, inds
    else:
        return new_xyz, new_features, inds, unique_cnt


def PointnetFPModule(unknown, known, unknown_feats, known_feats, mlps=None, bn=True, bn_m=0.95, name=None):
    """
    :param unknown: (B, n, 3) tensor of the xyz positions of the unknown features
    :param known: (B, m, 3) tensor of the xyz positions of the known features
    :param unknown_feats: (B, C1, n) tensor of the features to be propigated to
    :param known_feats: (B, C2, m) tensor of features to be propigated
    :param mlps: Pointnet module parameters
    :param bn: Use batchnorm
    :return new_features: (B, mlps[-1], n) tensor of the features of the unknown features
    """
    if known is not None:
        dist, idx = three_nn(unknown, known)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = fluid.layers.reduce_sum(dist_recip, dim=2, keep_dim=True)
        weight = dist_recip / norm

        interpolated_feats = three_interp(input=known_feats, weight=weight, idx=idx)
    else:
        interpolated_feats = fluid.layers.expand(*known_feats.shape()[0:2], unknown.shape()[1])

    if unknown_feats is not None:
        new_features = fluid.layers.concat([interpolated_feats, unknown_feats], axis=1)
    else:
        new_features = interpolated_feats

    new_features = fluid.layers.unsqueeze(new_features, axes=-1)
    for i, num_out_channel in enumerate(mlps):
        new_features = MLP(new_features, out_channels_list=num_out_channel, bn=bn, bn_momentum=bn_m,
                           name=name+'_mlp{}'.format(i))

    return fluid.layers.squeeze(new_features, axes=-1)

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

class VotingModule(object):
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
        self.out_dim = self.in_dim # due to residual feature, in_dim has to be == out_dim

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
                     act=None) # (batch_size, (3+out_dim)*vote_factor, num_seed)

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

def decode_scores(net, end_points, num_class, num_heading_bin, num_size_cluster, mean_size_arr):
    net_transposed = net.transpose(2, 1)  # (batch_size, 1024, ..)
    batch_size = net_transposed.shape()[0]
    num_proposal = net_transposed.shape()[1]

    objectness_scores = net_transposed[:, :, 0:2]
    end_points['objectness_scores'] = objectness_scores

    base_xyz = end_points['aggregated_vote_xyz']  # (batch_size, num_proposal, 3)
    center = base_xyz + net_transposed[:, :, 2:5]  # (batch_size, num_proposal, 3)
    end_points['center'] = center

    heading_scores = net_transposed[:, :, 5:5 + num_heading_bin]
    heading_residuals_normalized = net_transposed[:, :, 5 + num_heading_bin:5 + num_heading_bin * 2]
    end_points['heading_scores'] = heading_scores  # Bxnum_proposalxnum_heading_bin
    end_points[
        'heading_residuals_normalized'] = heading_residuals_normalized  # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
    end_points['heading_residuals'] = heading_residuals_normalized * (
                np.pi / num_heading_bin)  # Bxnum_proposalxnum_heading_bin

    size_scores = net_transposed[:, :, 5 + num_heading_bin * 2:5 + num_heading_bin * 2 + num_size_cluster]
    size_residuals_normalized = net_transposed[:, :,
                                5 + num_heading_bin * 2 + num_size_cluster:5 + num_heading_bin * 2 + num_size_cluster * 4].view(
        [batch_size, num_proposal, num_size_cluster, 3])  # Bxnum_proposalxnum_size_clusterx3
    end_points['size_scores'] = size_scores
    end_points['size_residuals_normalized'] = size_residuals_normalized
    end_points['size_residuals'] = size_residuals_normalized * fluid.create_lod_tensor(
        mean_size_arr.astype(np.float32), place=fluid.CUDAPlace).unsqueeze(0).unsqueeze(0)

    sem_cls_scores = net_transposed[:, :, 5 + num_heading_bin * 2 + num_size_cluster * 4:]  # Bxnum_proposalx10
    end_points['sem_cls_scores'] = sem_cls_scores
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
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
                 input_feature_dim=0, num_proposal=128, vote_factor=1, sampling='vote_fps'):
        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling

        self.backbone_net = Pointnet2Backbone(input_feature_dim=input_feature_dim)

        self.vgen = VoteNet(self.vote_factor, 256)

        self.pnet = ProposalModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling)

    def build_input(self):
        self.xyz = fluid.data(name='xyz',
                              shape=[None, self.num_points, 3],
                              dtype='float32',
                              lod_level=0)
        self.feature = fluid.data(name='feature',
                                  shape=[None, self.num_points, self.input_feature_dim],
                                  dtype='float32',
                                  lod_level=0)
        self.label = fluid.data(name='label',
                                shape=[None, self.num_points, 1],
                                dtype='int64',
                                lod_level=0)
        self.loader = fluid.io.DataLoader.from_generator(
            feed_list=[self.xyz, self.feature, self.label],
            capacity=64,
            use_double_buffer=True,
            iterable=False)
        self.feed_vars = [self.xyz, self.feature, self.label]


    def build(self, input_xyz, input_feature=None):
        end_points = {}

        # Backbone point feature learning
        seed_xyz, seed_features, seed_inds = self.backbone_net(input_xyz, input_feature)

        end_points['seed_xyz'] = seed_xyz
        end_points['seed_features'] = seed_features
        end_points['seed_inds'] = seed_inds

        # Hough voting
        vote_xyz, vote_features = self.vgen(seed_xyz, seed_features)

        features_norm = fluid.layers.reduce_sum(fluid.layers.abs(vote_features), dim=1, keep_dim=True)
        vote_features = vote_features / features_norm

        end_points['vote_xyz'] = vote_xyz
        end_points['vote_features'] = vote_features

        # Vote aggregation and detection
        output = self.pnet(vote_xyz, vote_features)

        return output
