#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import time
import shutil
import argparse
import ast
import logging
import numpy as np

import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.framework as framework
import paddle.fluid.layers.learning_rate_scheduler as lr_scheduler

from models import *
from data.scannet_reader import ScannetDetectionReader
from utils import *
from models.ap_helper import APCalculator, parse_groundtruths, parse_predictions

logging.root.handlers = []
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("VoteNet 3D Object Detection train script")
    parser.add_argument(
        '--model',
        type=str,
        default='MSG',
        help='SSG or MSG model to train, default MSG')
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='training batch size, default 8')
    parser.add_argument(
        '--num_points',
        type=int,
        default=20000,
        help='number of points in a sample, default: 20000')
    parser.add_argument(
        '--num_target',
        type=int,
        default=256,
        help='Proposal number [default: 256]')
    parser.add_argument(
        '--vote_factor',
        type=int,
        default=1,
        help='Vote factor [default: 1]')
    parser.add_argument(
        '--cluster_sampling',
        type=str,
        default='vote_fps',
        help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
    parser.add_argument(
        '--ap_iou_thresh',
        type=float,
        default=0.25,
        help='AP IoU threshold [default: 0.25]')
    parser.add_argument(
        '--num_classes',
        type=int,
        default=18,
        help='number of classes in dataset, default: 18')
    parser.add_argument(
        '--num_heading_bin',
        type=int,
        default=1,
        help='number of heading bin, default: 1')
    parser.add_argument(
        '--num_size_cluster',
        type=int,
        default=18,
        help='number of size of cluster, default: 18')
    # parser.add_argument(
    #     '--lr',
    #     type=float,
    #     default=0.001,
    #     help='initial learning rate, default 0.001')
    # parser.add_argument(
    #     '--lr_decay',
    #     type=float,
    #     default=0.5,
    #     help='learning rate decay gamma, default 0.5')
    parser.add_argument(
        '--lr_decay_steps',
        type=str,
        default='80,120,160',
        help='When to decay the learning rate (in epochs) [default: 80,120,160]')
    parser.add_argument(
        '--lr_decay_rates',
        type=str,
        default='0.001,0.0001,0.00001,0.000001',
        help='Decay rates for lr decay [default: 0.001,0.0001,0.00001,0.000001]')
    parser.add_argument(
        '--use_height',
        type=bool,
        default=True,
        help='Use height signal in input.')
    parser.add_argument(
        '--use_color',
        type=bool,
        default=False,
        help='Use RGB color in input.')
    parser.add_argument(
        '--bn_momentum',
        type=float,
        default=0.5,
        help='initial batch norm momentum, default 0.99')
    parser.add_argument(
        '--decay_steps',
        type=int,
        default=2000,
        help='learning rate and batch norm momentum decay steps, default 2k')
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.,
        help='L2 regularization weight decay coeff, default 0.')
    parser.add_argument(
        '--dump_results',
        action='store_true',
        help='Dump results.')
    parser.add_argument(
        '--dump_dir',
        type=str,
        default=None,
        help='Dump dir to save sample outputs [default: None]')
    parser.add_argument(
        '--epoch',
        type=int,
        default=181,
        help='epoch number. default 181.')
    parser.add_argument(
        '--data_dir',
        type=str,
        # default='dataset/Indoor3DSemSeg/indoor3d_sem_seg_hdf5_data',
        default='data',
        help='dataset directory')
    parser.add_argument(
        '--mean_size_arr_path',
        type=str,
        default='../dataset/scannet/meta_data/scannet_means.npz',
        help='ScanNet mean npy file path')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='checkpoints_seg',
        help='directory name to save train snapshoot')
    parser.add_argument(
        '--resume',
        type=str,
        default=None,  #'checkpoints_seg/30/votenet_det',
        help='path to resume training based on previous checkpoints. '
             'None for not resuming any checkpoints.')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=10,
        help='mini-batch interval for logging (in step).')
    parser.add_argument(
        '--enable_ce',
        action='store_true',
        help='The flag indicating whether to run the task '
             'for continuous evaluation.')
    parser.add_argument(
        '--eval_interval',
        type=int,
        default=10,
        help='mini-batch interval for validation (in epoch).')
    args = parser.parse_args()
    return args

def train():
    args = parse_args()
    print_arguments(args)
    # check whether the installed paddle is compiled with GPU
    check_gpu(args.use_gpu)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    # Used for AP calculation
    DATASET_CONFIG = ScannetDatasetConfig()

    CONFIG_DICT = {
        'remove_empty_box': False,
        'use_3d_nms': True,
        'nms_iou': 0.25,
        'use_old_type_nms': False,
        'cls_nms': True,
        'per_class_proposal': True,
        'conf_thresh': 0.05,
        'batch_size': args.batch_size,
        'dataset_config': DATASET_CONFIG
    }

    # Prepare get_decay_momentum function
    def get_decay_momentum(momentum_init, decay_steps, decay_rate):
        global_step = lr_scheduler._decay_step_counter()
        momentum = fluid.layers.create_global_var(
            shape=[1],
            value=float(momentum_init),
            dtype='float32',
            # set persistable for save checkpoints and resume
            persistable=True,
            name="momentum")
        div_res = global_step / decay_steps
        decayed_momentum = 1.0 - momentum_init * (decay_rate ** div_res)
        fluid.layers.assign(decayed_momentum, momentum)

        return momentum

    # build model
    if args.enable_ce:
        SEED = 102
        fluid.default_main_program().random_seed = SEED
        framework.default_startup_program().random_seed = SEED

    # Build train model
    startup = fluid.Program()
    train_prog = fluid.Program()

    NUM_INPUT_FEATURE_CHANNEL = int(args.use_color) * 3 + int(args.use_height) * 1

    with fluid.program_guard(train_prog, startup):
        with fluid.unique_name.guard():
            train_model = VoteNet(
                num_class=args.num_classes,
                num_points=args.num_points,
                num_heading_bin=args.num_heading_bin,
                num_size_cluster=args.num_size_cluster,
                num_proposal=args.num_target,
                input_feature_dim=NUM_INPUT_FEATURE_CHANNEL,
                vote_factor=args.vote_factor,
                sampling=args.cluster_sampling,
                batch_size=args.batch_size,
                bn_momentum=get_decay_momentum(0.5, 20 * 150, 0.5) # 20 epoch, each epoch has 150 steps
            )

            train_model.build_input()
            train_model.build()
            train_feeds = train_model.get_feeds()
            train_outputs = train_model.get_outputs('train')
            train_loader = train_model.get_loader()
            train_loss = train_outputs['loss']

            LR_DECAY_STEPS = [int(x)*150 for x in args.lr_decay_steps.split(',')]
            LR_VALUES = [float(x) for x in args.lr_decay_rates.split(',')]

            lr = layers.piecewise_decay(LR_DECAY_STEPS, LR_VALUES)

            params = []
            for var in train_prog.list_vars():
                if fluid.io.is_parameter(var):
                    params.append(var.name)
            optimizer = fluid.optimizer.Adam(
                learning_rate=lr,
                regularization=fluid.regularizer.L2Decay(args.weight_decay)
            )
            optimizer.minimize(train_loss, parameter_list=params)
    train_keys, train_values = parse_outputs(train_outputs)

    # Build test model
    test_prog = fluid.Program()
    with fluid.program_guard(test_prog, startup):
        with fluid.unique_name.guard():
            test_model = VoteNet(
                num_class=args.num_classes,
                num_points=args.num_points,
                num_heading_bin=args.num_heading_bin,
                num_size_cluster=args.num_size_cluster,
                num_proposal=args.num_target,
                input_feature_dim=NUM_INPUT_FEATURE_CHANNEL,
                vote_factor=args.vote_factor,
                sampling=args.cluster_sampling,
                batch_size=args.batch_size
            )

            test_model.build_input()
            test_model.build()
            test_feeds = test_model.get_feeds()
            test_outputs = test_model.get_outputs('test')
            test_loader = test_model.get_loader()
    test_prog = test_prog.clone(True)
    test_keys, test_values = parse_outputs(test_outputs)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup)

    start_epoch = 0
    if args.resume:
        if not os.path.isdir(args.resume):
            assert os.path.exists("{}.pdparams".format(args.resume)), \
                "Given resume weight {}.pdparams not exist.".format(args.resume)
            assert os.path.exists("{}.pdopt".format(args.resume)), \
                "Given resume optimizer state {}.pdopt not exist.".format(args.resume)
        fluid.load(train_prog, args.resume, exe)
        start_epoch = int(args.resume.split('/')[1]) + 1

    build_strategy = fluid.BuildStrategy()
    build_strategy.memory_optimize = False
    build_strategy.enable_inplace = False
    build_strategy.fuse_all_optimizer_ops = False
    train_compile_prog = fluid.compiler.CompiledProgram(
        train_prog).with_data_parallel(loss_name=train_loss.name,
                                       build_strategy=build_strategy)
    test_compile_prog = fluid.compiler.CompiledProgram(test_prog)

    def save_model(exe, prog, path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        logger.info("Save model to {}".format(path))
        fluid.save(prog, path)

    scannet_reader_tr = ScannetDetectionReader(num_points=args.num_points,
                                               use_color=args.use_color,
                                               use_height=args.use_height,
                                               augment=True,
                                               mode='train')
    scannet_reader_te = ScannetDetectionReader(num_points=args.num_points,
                                               use_color=args.use_color,
                                               use_height=args.use_height,
                                               augment=False,
                                               mode='val')
    train_reader = scannet_reader_tr.get_reader(args.batch_size)
    test_reader = scannet_reader_te.get_reader(args.batch_size)
    train_loader.set_sample_list_generator(train_reader, place)
    test_loader.set_sample_list_generator(test_reader, place)

    train_stat = Stat()
    test_stat = Stat()

    # ce_time = 0
    # ce_loss = []

    for epoch_id in range(start_epoch, args.epoch):
        try:
            train_loader.start()
            train_iter = 0
            train_periods = []

            while True:
                cur_time = time.time()
                train_outs = exe.run(train_compile_prog, fetch_list=train_values + [lr.name])
                period = time.time() - cur_time
                train_periods.append(period)
                train_stat.update(train_keys, train_outs[:-1])
                if train_iter % args.log_interval == 0:
                    log_str = ""
                    for name, values in zip(train_keys + ['learning_rate'], train_outs):
                        log_str += "{}: {:.5f},\n".format(name, np.mean(values))
                        # if name == 'loss':
                        #     ce_loss.append(np.mean(values))
                    logger.info(
                        "[TRAIN] Epoch {}, batch {}: {}time: {:.2f}".format(epoch_id, train_iter, log_str, period))
                train_iter += 1

        except fluid.core.EOFException:
            logger.info("[TRAIN] Epoch {} finished, {}average time: {:.2f}".format(epoch_id, train_stat.get_mean_log(),
                                                                                   np.mean(train_periods[1:])))
            ce_time = np.mean(train_periods[1:])
            save_model(exe, train_prog, os.path.join(args.save_dir, str(epoch_id), "votenet_det"))

            # evaluation
            if not args.enable_ce and epoch_id % args.eval_interval == 0:
                ap_calculator = APCalculator(ap_iou_thresh=args.ap_iou_thresh, class2type_map=DATASET_CONFIG.class2type)
                try:
                    test_loader.start()
                    test_iter = 0
                    test_periods = []
                    while True:
                        cur_time = time.time()
                        test_outs = exe.run(test_compile_prog, fetch_list=test_values)
                        period = time.time() - cur_time
                        test_periods.append(period)
                        test_stat.update(test_keys[:13], test_outs[:13])

                        # Calculate AP figures:
                        batch_pred_map_cls = parse_predictions(dict(zip(test_keys[13:21], test_outs[13:21])), CONFIG_DICT)
                        batch_gt_map_cls = parse_groundtruths(dict(zip(test_keys[21:], test_outs[21:])), CONFIG_DICT)

                        ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

                        if test_iter % args.log_interval == 0:
                            log_str = ""
                            for name, value in zip(test_keys[:13], test_outs[:13]):
                                log_str += "{}: {:.4f}, \n".format(name, np.mean(value))
                            logger.info("[TEST] Epoch {}, batch {}: {}time: {:.2f}".format(epoch_id, test_iter, log_str,
                                                                                           period))
                        test_iter += 1
                except fluid.core.EOFException:
                    ap_output = ap_calculator.compute_metrics()
                    log_str = ""
                    for key in ap_output:
                        log_str += '{}: {:.4f}, \n'.format(key, ap_output[key])
                    logger.info(
                        "[TEST] Epoch {} finished, {}, {} average time: {:.2f}".format(epoch_id, test_stat.get_mean_log(),
                                                                                       log_str, np.mean(test_periods[1:])))
                finally:
                    test_loader.reset()
                    test_stat.reset()
                    test_periods = []

        finally:
            train_loader.reset()
            train_stat.reset()
            train_periods = []

    # # only for ce
    # if args.enable_ce:
    #     card_num = get_cards()
    #     _loss = 0
    #     _time = 0
    #     try:
    #         _time = ce_time
    #         _loss = np.mean(ce_loss[1:])
    #     except:
    #         print("ce info error")
    #     print("kpis\ttrain_seg_%s_duration_card%s\t%s" % (args.model, card_num, _time))
    #     print("kpis\ttrain_seg_%s_loss_card%s\t%f" % (args.model, card_num, _loss))

def test_train():
    # Params list
    USE_HEIGHT = True
    AUGMENT = False
    MAX_NUM_OBJ = 64

    # 1. Prepare the same input data
    scene_name = 'scene0000_00'
    data_folder_path = 'dataset/scannet/scannet_train_detection_data/'
    scene_path = os.path.join(data_folder_path, scene_name)

    mesh_vertices = np.load(scene_path + '_vert.npy')
    instance_labels = np.load(scene_path + '_ins_label.npy')
    semantic_labels = np.load(scene_path + '_sem_label.npy')
    instance_bboxes = np.load(scene_path + '_bbox.npy')

    point_cloud = mesh_vertices[:, :3]
    features = np.empty()

    if USE_HEIGHT:
        floor_height = np.percentile(point_cloud[:, 2], 0.99)  # Find the floor height along the z-axis/gravity direction
        height = np.expand_dims(point_cloud[:, 2] - floor_height)
        features = height

    # ------------------------------- LABELS ------------------------------
    target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
    target_bboxes_mask = np.zeros((MAX_NUM_OBJ))
    angle_classes = np.zeros((MAX_NUM_OBJ,))
    angle_residuals = np.zeros((MAX_NUM_OBJ,))
    size_classes = np.zeros((MAX_NUM_OBJ,))
    size_residuals = np.zeros((MAX_NUM_OBJ, 3))

    point_cloud, choices = random_sampling(point_cloud, num_points, return_choices=True)
    instance_labels = instance_labels[choices]
    semantic_labels = semantic_labels[choices]

    features = features[choices]

    target_bboxes_mask[0:instance_bboxes.shape[0]] = 1
    target_bboxes[0:instance_bboxes.shape[0], :] = instance_bboxes[:, 0:6]

    # ------------------------------- DATA AUGMENTATION ------------------------------
    if AUGMENT:
        if np.random.random() > 0.5:
            # Flipping along the YZ plane
            point_cloud[:, 0] = -1 * point_cloud[:, 0]
            target_bboxes[:, 0] = -1 * target_bboxes[:, 0]

        if np.random.random() > 0.5:
            # Flipping along the XZ plane
            point_cloud[:, 1] = -1 * point_cloud[:, 1]
            target_bboxes[:, 1] = -1 * target_bboxes[:, 1]

            # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
        rot_mat = rotz(rot_angle)
        point_cloud = np.dot(point_cloud, np.transpose(rot_mat))
        target_bboxes = rotate_aligned_boxes(target_bboxes, rot_mat)

        # compute votes *AFTER* augmentation
        # generate votes
        # Note: since there's no map between bbox instance labels and
        # pc instance_labels (it had been filtered
        # in the data preparation step) we'll compute the instance bbox
        # from the points sharing the same instance label.
        point_votes = np.zeros([num_points, 3])
        point_votes_mask = np.zeros(num_points)
        for i_instance in np.unique(instance_labels):
            # find all points belong to that instance
            ind = np.where(instance_labels == i_instance)[0]
            # find the semantic label
            if semantic_labels[ind[0]] in nyu40ids:
                x = point_cloud[ind, :]
                center = 0.5 * (x.min(0) + x.max(0))
                point_votes[ind, :] = center - x
                point_votes_mask[ind] = 1.0
        point_votes = np.tile(point_votes, (1, 3))  # make 3 votes identical

        class_ind = [np.where(nyu40ids == x)[0][0] for x in instance_bboxes[:, -1]]
        # NOTE: set size class as semantic class. Consider use size2class.
        size_classes[0:instance_bboxes.shape[0]] = class_ind
        size_residuals[0:instance_bboxes.shape[0], :] = \
            target_bboxes[0:instance_bboxes.shape[0], 3:6] - mean_size_arr[class_ind, :]

        # ret_dict = {}
        point_cloud = point_cloud.astype(np.float32)
        center_label = target_bboxes.astype(np.float32)[:, 0:3]
        heading_class_label = angle_classes.astype(np.int64)
        heading_residual_label = angle_residuals.astype(np.float32)
        size_class_label = size_classes.astype(np.int64)
        size_residual_label = size_residuals.astype(np.float32)
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_bboxes_semcls[0:instance_bboxes.shape[0]] = \
            [nyu40id2class[x] for x in instance_bboxes[:, -1][0:instance_bboxes.shape[0]]]
        sem_cls_label = target_bboxes_semcls.astype(np.int64)
        box_label_mask = target_bboxes_mask.astype(np.float32)
        vote_label = point_votes.astype(np.float32)
        vote_label_mask = point_votes_mask.astype(np.int64)


    # 2. Build the model

    # 3. Compare the loss result

    # 4. Spot what fuck is wrong

    pass

def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def get_cards():
    num = 0
    cards = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cards != '':
        num = len(cards.split(","))
    return num

if __name__ == '__main__':
    train()