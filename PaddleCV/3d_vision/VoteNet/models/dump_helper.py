# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from .pc_util import softmax, gather_numpy, write_ply, write_oriented_bbox, write_ply_color

DUMP_CONF_THRESH = 0.5  # Dump boxes with obj prob larger than that.

def dump_results(end_points, dump_dir, config, inference_switch=False):
    ''' Dump results.

    Args:
        end_points: dict
            {..., pred_mask}
            pred_mask is a binary mask array of size (batch_size, num_proposal) computed by running NMS and empty box removal
    Returns:
        None
    '''
    if not os.path.exists(dump_dir):
        os.system('mkdir %s'%(dump_dir))

    # INPUT
    point_clouds = end_points['point_clouds']
    batch_size = point_clouds.shape[0]

    # NETWORK OUTPUTS
    seed_xyz = np.array(end_points['seed_xyz'], dtype=np.float32)  # (B,num_seed,3)
    if 'vote_xyz' in end_points:
        vote_xyz = np.array(end_points['vote_xyz'], dtype=np.float32)  # (B,num_seed,3)
        aggregated_vote_xyz = np.array(end_points['aggregated_vote_xyz'], dtype=np.float32)
    objectness_scores = np.array(end_points['objectness_scores'], dtype=np.float32)  # (B,K,2)
    pred_center = np.array(end_points['center'], dtype=np.float32)  # (B,K,3)
    heading_scores = np.array(end_points['heading_scores'], dtype=np.float32)
    pred_heading_class = np.argmax(heading_scores, axis=-1)  # B,num_proposal
    pred_heading_residual = gather_numpy(end_points['heading_residuals'], 2, np.expand_dims(pred_heading_class, -1)) # B,num_proposal,1
    pred_heading_class = pred_heading_class  # B,num_proposal
    pred_heading_residual = np.squeeze(pred_heading_residual, 2) # B,num_proposal
    pred_size_class = np.argmax(np.array(end_points['size_scores'], dtype=np.float32), -1) # B,num_proposal
    pred_size_class_expand = np.tile(np.expand_dims(np.expand_dims(pred_size_class, -1), -1), (1, 1, 1, 3))
    pred_size_residual = gather_numpy(np.array(end_points['size_residuals'], dtype=np.float32),
                                      2, pred_size_class_expand) # B,num_proposal,1,3
    pred_size_residual = np.squeeze(pred_size_residual, 2) # B,num_proposal,3

    # OTHERS
    pred_mask = end_points['pred_mask'] # B,num_proposal
    idx_beg = 0

    for i in range(batch_size):
        pc = point_clouds[i, :, :]
        objectness_prob = softmax(objectness_scores[i, :, :])[:, 1] # (K,)

        # Dump various point clouds
        write_ply(pc, os.path.join(dump_dir, '%06d_pc.ply'%(idx_beg+i)))
        write_ply(seed_xyz[i,:,:], os.path.join(dump_dir, '%06d_seed_pc.ply'%(idx_beg+i)))
        if 'vote_xyz' in end_points:
            write_ply(end_points['vote_xyz'][i,:,:], os.path.join(dump_dir, '%06d_vgen_pc.ply'%(idx_beg+i)))
            write_ply(aggregated_vote_xyz[i,:,:], os.path.join(dump_dir, '%06d_aggregated_vote_pc.ply'%(idx_beg+i)))
            write_ply(aggregated_vote_xyz[i,:,:], os.path.join(dump_dir, '%06d_aggregated_vote_pc.ply'%(idx_beg+i)))
        write_ply(pred_center[i,:,0:3], os.path.join(dump_dir, '%06d_proposal_pc.ply'%(idx_beg+i)))
        if np.sum(objectness_prob>DUMP_CONF_THRESH)>0:
            write_ply(pred_center[i,objectness_prob>DUMP_CONF_THRESH,0:3], os.path.join(dump_dir, '%06d_confident_proposal_pc.ply'%(idx_beg+i)))

        # OPTIONALL, also dump prediction and gt details
        if 'batch_pred_map_cls' in end_points:
            # fout = open(os.path.join(dump_dir, '%06d_pred_map_cls.txt' % (i)), 'w')
            # for t in end_points['batch_pred_map_cls'][i]:
            #     fout.write(str(t[0]) + ' ')
            #     fout.write(",".join([str(x) for x in list(t[1].flatten())]))
            #     fout.write(' ' + str(t[2]))
            #     fout.write('\n')
            # fout.close()
            cur_sem_cls_labels = end_points['batch_pred_map_cls'][i]

        # Dump predicted bounding boxes
        if np.sum(objectness_prob>DUMP_CONF_THRESH)>0:
            num_proposal = pred_center.shape[1]
            obbs = []
            for j in range(num_proposal):
                obb = config.param2obb(pred_center[i,j,0:3], pred_heading_class[i,j], pred_heading_residual[i,j],
                                pred_size_class[i,j], pred_size_residual[i,j])
                obbs.append(obb)
            if len(obbs)>0:
                obbs = np.vstack(tuple(obbs)) # (num_proposal, 7)
                write_oriented_bbox(obbs[objectness_prob>DUMP_CONF_THRESH,:], os.path.join(dump_dir, '%06d_pred_confident_bbox.ply'%(idx_beg+i)))
                write_oriented_bbox(obbs[np.logical_and(objectness_prob>DUMP_CONF_THRESH, pred_mask[i,:]==1),:], os.path.join(dump_dir, '%06d_pred_confident_nms_bbox.ply'%(idx_beg+i)))
                write_oriented_bbox(obbs[pred_mask[i,:]==1,:], os.path.join(dump_dir, '%06d_pred_nms_bbox.ply'%(idx_beg+i)))
                write_oriented_bbox(obbs, os.path.join(dump_dir, '%06d_pred_bbox.ply'%(idx_beg+i)))

    # Return if it is at inference time. No dumping of groundtruths
    if inference_switch:
        return

    # LABELS
    gt_center = end_points['center_label'] # (B,MAX_NUM_OBJ,3)
    gt_mask = end_points['box_label_mask'] # B,K2
    gt_heading_class = end_points['heading_class_label'] # B,K2
    gt_heading_residual = end_points['heading_residual_label'] # B,K2
    gt_size_class = end_points['size_class_label'] # B,K2
    gt_size_residual = end_points['size_residual_label'] # B,K2,3
    objectness_label = end_points['objectness_label'] # (B,K,)
    objectness_mask = end_points['objectness_mask'] # (B,K,)

    for i in range(batch_size):
        if np.sum(objectness_label[i,:])>0:
            write_ply(pred_center[i,objectness_label[i,:]>0,0:3], os.path.join(dump_dir, '%06d_gt_positive_proposal_pc.ply'%(idx_beg+i)))
        if np.sum(objectness_mask[i,:])>0:
            write_ply(pred_center[i,objectness_mask[i,:]>0,0:3], os.path.join(dump_dir, '%06d_gt_mask_proposal_pc.ply'%(idx_beg+i)))
        write_ply(gt_center[i,:,0:3], os.path.join(dump_dir, '%06d_gt_centroid_pc.ply'%(idx_beg+i)))
        write_ply_color(pred_center[i,:,0:3], objectness_label[i,:], os.path.join(dump_dir, '%06d_proposal_pc_objectness_label.obj'%(idx_beg+i)))

        # Dump GT bounding boxes
        obbs = []
        for j in range(gt_center.shape[1]):
            if gt_mask[i,j] == 0: continue
            obb = config.param2obb(gt_center[i,j,0:3], gt_heading_class[i,j], gt_heading_residual[i,j],
                            gt_size_class[i,j], gt_size_residual[i,j])
            obbs.append(obb)
        if len(obbs)>0:
            obbs = np.vstack(tuple(obbs)) # (num_gt_objects, 7)
            write_oriented_bbox(obbs, os.path.join(dump_dir, '%06d_gt_bbox.ply'%(idx_beg+i)))

    # OPTIONALL, also dump prediction and gt details
    if 'batch_pred_map_cls' in end_points:
        for ii in range(batch_size):
            fout = open(os.path.join(dump_dir, '%06d_pred_map_cls.txt'%(ii)), 'w')
            for t in end_points['batch_pred_map_cls'][ii]:
                fout.write(str(t[0])+' ')
                fout.write(",".join([str(x) for x in list(t[1].flatten())]))
                fout.write(' '+str(t[2]))
                fout.write('\n')
            fout.close()
    if 'batch_gt_map_cls' in end_points:
        for ii in range(batch_size):
            fout = open(os.path.join(dump_dir, '%06d_gt_map_cls.txt'%(ii)), 'w')
            for t in end_points['batch_gt_map_cls'][ii]:
                fout.write(str(t[0])+' ')
                fout.write(",".join([str(x) for x in list(t[1].flatten())]))
                fout.write('\n')
            fout.close()
