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
Contains functions of the loss computation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid
import numpy as np

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3 # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8] # put larger weights on positive objectness


def weighted_softmax_cross_entropy_loss(inputs, targets, weight=None, size_average=None, ignore_idx=-100, reduce=None,
                                        reduction=None):
    # inputs: [N, C, d1, d2, ...]
    # target: [N, ] a list of class index
    # weight: [N, ]
    assert(inputs.shape[0] == len(weight) == len(targets))

    num_of_class = inputs.shape[1]

    # Apply log_softmax
    input_log_sm = fluid.layers.log_softmax(input=inputs, axis=1)

    # Convert target index list to one-hot encode
    targets_onehot_mat = fluid.layers.one_hot(targets, depth=num_of_class)

    input_log_sm_abs = fluid.layers.abs(input_log_sm)

    loss_mat = input_log_sm_abs * targets_onehot_mat

    weight_mat = weight * targets_onehot_mat

    loss_mat = weight_mat * loss_mat

    loss = fluid.layers.reduce_sum(loss_mat, dim=1)

    return loss

def gather_dim1(val_tensor, inds_tensor):
    """
    Apply gather operation to dim=1
    :param val_tensor: input tensor
    :param inds_tensor: idx list

    :return:
    """
    assert(len(val_tensor)==len(inds_tensor)) # input tensor and inds tensor must have the same length

    tmp = []
    for idx, val in enumerate(val_tensor):
        tmp.append(fluid.layers.gather(input=val, index=inds_tensor[idx]))
    out_tensor = (fluid.layers.concat(input=tmp))

    return out_tensor

def huber_loss(error, delta=1.0):
    """
    Args:
        error: Torch tensor (d1,d2,...,dk)
    Returns:
        loss: Torch tensor (d1,d2,...,dk)

    x = error = pred - gt or dist(pred,gt)
    0.5 * |x|^2                 if |x|<=d
    0.5 * d^2 + d * (|x|-d)     if |x|>d
    Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py
    """
    abs_error = fluid.layers.abs(error)
    # quadratic = torch.min(abs_error, torch.FloatTensor([delta]))
    quadratic = fluid.layers.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    loss = 0.5 * quadratic ** 2 + delta * linear
    return loss

def nn_distance(pc1, pc2, l1smooth=False, delta=1.0, l1=False):
    """
    Input:
        pc1: (B,N,C) torch tensor
        pc2: (B,M,C) torch tensor
        l1smooth: bool, whether to use l1smooth loss
        delta: scalar, the delta used in l1smooth loss
    Output:
        dist1: (B,N) torch float32 tensor
        idx1: (B,N) torch int64 tensor
        dist2: (B,M) torch float32 tensor
        idx2: (B,M) torch int64 tensor
    """
    N = pc1.shape[1]
    M = pc2.shape[1]
    pc1_expand_tile = fluid.layers.expand(fluid.layers.unsqueeze(pc1, axes=2), expand_times=[1, 1, M, 1])
    pc2_expand_tile = fluid.layers.expand(fluid.layers.unsqueeze(pc2, axes=1), expand_times=[1, N, 1, 1])
    pc_diff = pc1_expand_tile - pc2_expand_tile

    if l1smooth:
        pc_dist = fluid.layers.reduce_sum(huber_loss(pc_diff, delta), dim=-1)  # (B,N,M)
    elif l1:
        pc_dist = fluid.layers.reduce_sum(fluid.layers.abs(pc_diff), dim=-1)  # (B,N,M)
    else:
        pc_dist = fluid.layers.reduce_sum(pc_diff ** 2, dim=-1)  # (B,N,M)
    dist1, idx1 = fluid.layers.reduce_sum(pc_dist, dim=2)  # (B,N)
    dist2, idx2 = fluid.layers.reduce_min(pc_dist, dim=1)  # (B,M)
    return dist1, idx1, dist2, idx2

# def demo_nn_distance():
#     np.random.seed(0)
#     pc1arr = np.random.random((1, 5, 3))
#     pc2arr = np.random.random((1, 6, 3))
#     pc1 = fluid.create_lod_tensor(pc1arr.astype(np.float32))
#     pc2 = fluid.create_lod_tensor(pc2arr.astype(np.float32))
#     dist1, idx1, dist2, idx2 = nn_distance(pc1, pc2)
#     print(dist1)
#     print(idx1)
#     dist = np.zeros((5, 6))
#     for i in range(5):
#         for j in range(6):
#             dist[i, j] = np.sum((pc1arr[0, i, :] - pc2arr[0, j, :]) ** 2)
#     print(dist)
#     print('-' * 30)
#     print('L1smooth dists:')
#     dist1, idx1, dist2, idx2 = nn_distance(pc1, pc2, True)
#     print(dist1)
#     print(idx1)
#     dist = np.zeros((5, 6))
#     for i in range(5):
#         for j in range(6):
#             error = np.abs(pc1arr[0, i, :] - pc2arr[0, j, :])
#             quad = np.minimum(error, 1.0)
#             linear = error - quad
#             loss = 0.5 * quad ** 2 + 1.0 * linear
#             dist[i, j] = np.sum(loss)
#     print(dist)

def cfloat(input_tensor):
    """
    Convert tensor type to float32
    """
    return fluid.layers.cast(input_tensor, dtype='float32')

def clong(input_tensor):
    """
    Convert tensor type to long (int64)
    """
    return fluid.layers.cast(input_tensor, dtype='int64')

def compute_vote_loss(end_points):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        end_points: dict (read-only)

    Returns:
        vote_loss: scalar Tensor

    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.

        Each seed point may vote for multiple translations v1,v2,v3
        A seed point may also be in the boxes of multiple objects:
        o1,o2,o3 with corresponding GT votes c1,c2,c3

        Then the loss for this seed point is:
            min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
    """

    # Load ground truth votes and assign them to seed points
    batch_size = end_points['seed_xyz'].shape[0]
    num_seed = end_points['seed_xyz'].shape[1]  # B,num_seed,3
    vote_xyz = end_points['vote_xyz']  # B,num_seed*vote_factor,3
    seed_inds = fluid.layers.cast(end_points['seed_inds'], dtype='int64')  # B,num_seed in [0,num_points-1]

    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    seed_gt_votes_mask = gather_dim1(end_points['vote_label_mask'], seed_inds)

    seed_inds_expand = fluid.layers.expand(fluid.layers.reshape(seed_inds, shape=[batch_size, num_seed, 1]), expand_times=[1, 1, 3 * GT_VOTE_FACTOR])

    seed_gt_votes = gather_dim1(end_points['vote_label'], seed_inds_expand)
    seed_gt_votes += fluid.layers.expand(end_points['seed_xyz'], expand_times=[1, 1, 3])

    # Compute the min of min of distance
    vote_xyz_reshape = fluid.layers.reshape(vote_xyz, shape=[batch_size * num_seed, -1,
                                            3])  # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape = fluid.layers.reshape(seed_gt_votes, shape=[batch_size * num_seed, GT_VOTE_FACTOR,
                                               3])  # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = fluid.layers.reduce_min(dist2, dim=1)  # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist = fluid.layers.reshape(votes_dist, shape=[batch_size, num_seed])
    vote_loss = fluid.layers.reduce_sum(votes_dist * cfloat(seed_gt_votes_mask)) / (fluid.layers.reduce_sum(cfloat(seed_gt_votes_mask)) + 1e-6)
    return vote_loss


def compute_objectness_loss(end_points):
    """ Compute objectness loss for the proposals.

    Args:
        end_points: dict (read-only)

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        object_assignment: (batch_size, num_seed) Tensor with long int
            within [0,num_gt_object-1]
    """
    # Associate proposal and GT objects by point-to-point distances
    aggregated_vote_xyz = end_points['aggregated_vote_xyz']
    gt_center = end_points['center_label'][:, :, 0:3]
    B = gt_center.shape[0]
    K = aggregated_vote_xyz.shape[1]
    K2 = gt_center.shape[1]
    dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center)  # dist1: BxK, dist2: BxK2

    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
    euclidean_dist1 = fluid.layers.sqrt(dist1 + 1e-6)
    objectness_label = fluid.layers.zeros(shape=(B, K), dtype='int64', force_cpu=False)
    objectness_mask = fluid.layers.zeros(shape=(B, K), force_cpu=False)
    objectness_label[euclidean_dist1 < NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1 < NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1 > FAR_THRESHOLD] = 1

    # Compute objectness loss
    # TODO: CrossEntropyLoss
    objectness_scores = end_points['objectness_scores']

    weight_tensor = fluid.Tensor()
    weight_tensor.set(OBJECTNESS_CLS_WEIGHTS, fluid.CUDAPlace(0))

    criterion = fluid.layers.softmax_with_cross_entropy(weight_tensor, reduction='none')
    objectness_scores_T = fluid.layers.transpose(objectness_scores, perm=[0, 2, 1])

    objectness_loss = criterion(objectness_scores_T, objectness_label)
    objectness_loss = fluid.layers.reduce_sum(objectness_loss * objectness_mask) / (fluid.layers.reduce_sum(objectness_mask) + 1e-6)

    # Set assignment
    object_assignment = ind1  # (B,K) with values in 0,1,...,K2-1

    return objectness_loss, objectness_label, objectness_mask, object_assignment


def compute_box_and_sem_cls_loss(end_points, config):
    """ Compute 3D bounding box and semantic classification loss.

    Args:
        end_points: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_cls_loss
        size_reg_loss
        sem_cls_loss
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    object_assignment = end_points['object_assignment']
    batch_size = object_assignment.shape[0]

    # Compute center loss
    pred_center = end_points['center']
    gt_center = end_points['center_label'][:, :, 0:3]
    dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center)  # dist1: BxK, dist2: BxK2
    box_label_mask = end_points['box_label_mask']
    objectness_label = cfloat(end_points['objectness_label'])
    centroid_reg_loss1 = \
        fluid.layers.reduce_sum(dist1 * objectness_label) / (fluid.layers.reduce_sum(objectness_label) + 1e-6)
    centroid_reg_loss2 = \
        fluid.layers.reduce_sum(dist2 * box_label_mask) / (fluid.layers.reduce_sum(box_label_mask) + 1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2

    # Compute heading loss
    heading_class_label = gather_dim1(end_points['heading_class_label'], object_assignment)  # select (B,K) from (B,K2)


    # TODO: CrossEntropyLoss
    criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
    heading_class_loss = criterion_heading_class(end_points['heading_scores'].transpose(2, 1),
                                                 heading_class_label)  # (B,K)


    heading_class_loss = fluid.layers.reduce_sum(heading_class_loss * objectness_label) / (fluid.layers.reduce_sum(objectness_label) + 1e-6)

    heading_residual_label = gather_dim1(end_points['heading_residual_label'], object_assignment)  # select (B,K) from (B,K2)

    heading_residual_normalized_label = heading_residual_label / (np.pi / num_heading_bin)


    # TODO: convert to one-hot encoding
    # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
    heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1], num_heading_bin).zero_()
    heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1),
                                   1)  # src==1 so it's *one-hot* (B,K,num_heading_bin)

    heading_residual_normalized_loss = huber_loss(
        fluid.layers.reduce_sum(end_points['heading_residuals_normalized'] * heading_label_one_hot,
                  -1) - heading_residual_normalized_label, delta=1.0)  # (B,K)

    heading_residual_normalized_loss = fluid.layers.reduce_sum(heading_residual_normalized_loss * objectness_label) / (
                fluid.layers.reduce_sum(objectness_label) + 1e-6)

    # Compute size loss
    size_class_label = gather_dim1(end_points['size_class_label'], object_assignment)  # select (B,K) from (B,K2)


    # TODO: CrossEntropyLoss
    criterion_size_class = nn.CrossEntropyLoss(reduction='none')
    size_scores_T = fluid.layers.transpose(end_points['size_scores'], perm=[0, 2, 1])
    size_class_loss = criterion_size_class(size_scores_T, size_class_label)  # (B,K)



    size_class_loss = fluid.layers.reduce_sum(size_class_loss * objectness_label) / (fluid.layers.reduce_sum(objectness_label) + 1e-6)

    object_assignment_unsqueeze_reshape = fluid.layers.expand(fluid.layers.unsqueeze(object_assignment, axes=-1),
                                                              expand_times=[1, 1, 3])
    size_residual_label = gather_dim1(end_points['size_residual_label'],
                                      object_assignment_unsqueeze_reshape) # select (B,K,3) from (B,K2,3)


    # TODO: convert to one-hot encoding
    size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
    size_label_one_hot.scatter_(2, fluid.layers.unsqueeze(size_class_label, axes=-1), 1)  # src==1 so it's *one-hot* (B,K,num_size_cluster)

    size_label_one_hot_tiled = fluid.layers.expand(fluid.layers.unsqueeze(size_label_one_hot, axes=-1),
                                                   expand_times=[1, 1, 1, 3]) # (B,K,num_size_cluster,3)

    predicted_size_residual_normalized = fluid.layers.reduce_sum(end_points['size_residuals_normalized'] * size_label_one_hot_tiled,
                                                   2)  # (B,K,3)

    mean_size_arr_expanded = fluid.create_lod_tensor(fluid.layers.cast(mean_size_arr, dtype='float32'), place=fluid.CUDAPlace(0))
    mean_size_arr_expanded = fluid.layers.unsqueeze(fluid.layers.unsqueeze(mean_size_arr_expanded, axes=0), axes=0) # (1,1,num_size_cluster,3)

    mean_size_label = fluid.layers.reduce_sum(size_label_one_hot_tiled * mean_size_arr_expanded, dim=2)  # (B,K,3)
    size_residual_label_normalized = size_residual_label / mean_size_label  # (B,K,3)
    size_residual_normalized_loss = fluid.layers.reduce_mean(
        huber_loss(predicted_size_residual_normalized - size_residual_label_normalized, delta=1.0),
        dim=-1)  # (B,K,3) -> (B,K)
    size_residual_normalized_loss = fluid.layers.reduce_sum(size_residual_normalized_loss * objectness_label) / (
                fluid.layers.reduce_sum(objectness_label) + 1e-6)

    # 3.4 Semantic cls loss
    sem_cls_label = gather_dim1(end_points['sem_cls_label'], object_assignment)  # select (B,K) from (B,K2)

    # TODO: CrossEntropyLoss
    criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
    sem_cls_loss = criterion_sem_cls(end_points['sem_cls_scores'].transpose(2, 1), sem_cls_label)  # (B,K)

    sem_cls_loss = fluid.layers.reduce_sum(sem_cls_loss * objectness_label) / (fluid.layers.reduce_sum(objectness_label) + 1e-6)

    return center_loss, heading_class_loss, heading_residual_normalized_loss, size_class_loss, size_residual_normalized_loss, sem_cls_loss


def get_loss(end_points, config):
    """ Loss functions

    Args:
        end_points: dict
            {
                seed_xyz, seed_inds, vote_xyz,
                center,
                heading_scores, heading_residuals_normalized,
                size_scores, size_residuals_normalized,
                sem_cls_scores, #seed_logits,#
                center_label,
                heading_class_label, heading_residual_label,
                size_class_label, size_residual_label,
                sem_cls_label,
                box_label_mask,
                vote_label, vote_label_mask
            }
        config: dataset config instance
    Returns:
        loss: pytorch scalar tensor
        end_points: dict
    """

    # Vote loss
    vote_loss = compute_vote_loss(end_points)
    end_points['vote_loss'] = vote_loss

    # Obj loss
    objectness_loss, objectness_label, objectness_mask, object_assignment = \
        compute_objectness_loss(end_points)
    end_points['objectness_loss'] = objectness_loss
    end_points['objectness_label'] = objectness_label
    end_points['objectness_mask'] = objectness_mask
    end_points['object_assignment'] = object_assignment
    total_num_proposal = objectness_label.shape[0] * objectness_label.shape[1]
    end_points['pos_ratio'] = \
        fluid.layers.reduce_sum(cfloat(objectness_label)) / float(total_num_proposal)
    end_points['neg_ratio'] = \
        fluid.layers.reduce_sum(cfloat(objectness_mask)) / float(total_num_proposal) - end_points['pos_ratio']

    # Box loss and sem cls loss
    center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss, sem_cls_loss = \
        compute_box_and_sem_cls_loss(end_points, config)
    end_points['center_loss'] = center_loss
    end_points['heading_cls_loss'] = heading_cls_loss
    end_points['heading_reg_loss'] = heading_reg_loss
    end_points['size_cls_loss'] = size_cls_loss
    end_points['size_reg_loss'] = size_reg_loss
    end_points['sem_cls_loss'] = sem_cls_loss
    box_loss = center_loss + 0.1 * heading_cls_loss + heading_reg_loss + 0.1 * size_cls_loss + size_reg_loss
    end_points['box_loss'] = box_loss

    # Final loss function
    loss = vote_loss + 0.5 * objectness_loss + box_loss + 0.1 * sem_cls_loss
    loss *= 10
    end_points['loss'] = loss

    # --------------------------------------------
    # Some other statistics
    obj_pred_val = fluid.layers.argmax(end_points['objectness_scores'], 2)  # B,K
    obj_acc = cfloat(fluid.layers.reduce_sum((obj_pred_val == clong(objectness_label))) * objectness_mask) / (
                fluid.layers.reduce_sum(objectness_mask) + 1e-6)
    end_points['obj_acc'] = obj_acc

    return loss, end_points


if __name__ == '__main__':
    # demo_nn_distance()

    pass