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
Contains paddle utility functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import six
import logging
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.param_attr import ParamAttr
import numpy as np

__all__ = ["conv1d"]

logger = logging.getLogger(__name__)

def conv1d(input,
           num_filters,
           filter_size,
           bn=True,
           bn_momentum=0.99,
           stride=1,
           padding=0,
           dilation=1,
           groups=None,
           # param_attr=None,
           # bias_attr=None,
           use_cudnn=True,
           act='relu',
           name=None,
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

    param_attr = ParamAttr(name='{}_conv1d_weight'.format(name), )
    bias_attr = ParamAttr(name='{}_conv1d_bias'.format(name)) \
        if not bn else False

    out_4dim = layers.conv2d(input=input_4dim,
                               num_filters=num_filters,
                               filter_size=filter_size,
                               stride=1,
                               padding=0,
                               dilation=1,
                               groups=groups,
                               param_attr=param_attr,
                               bias_attr=bias_attr,
                               use_cudnn=use_cudnn,
                               name=name,
                               data_format=data_format)
                               # act=act if not bn else None)
    if bn:
        bn_name = name + '_bn'
        out_4dim = layers.batch_norm(out_4dim,
                                       act=act,
                                       momentum=bn_momentum,
                                       param_attr=ParamAttr(name=bn_name + '_scale'),
                                       bias_attr=ParamAttr(name=bn_name + '_offset'),
                                       moving_mean_name=bn_name + '_mean',
                                       moving_variance_name=bn_name + '_var')
    if act == "relu":
        out_4dim = layers.relu(out_4dim)

    # Convert the output tensor from dim 4 back to dim 3
    out_3dim = layers.squeeze(out_4dim, -1)
    return out_3dim


def test():
    # Test goes here
    data = fluid.data(name='data', shape=[8, 128], dtype='float32')
    label = fluid.data(name='label', shape=[8, 1], dtype='int64')

    new_data = layers.reshape(x=data, shape=[2, 4, 128])
    sh = new_data.shape[:2]
    nn_data = new_data[:, :, :3]
    int_32_label = layers.cast(label, dtype='int32')


    import numpy as np
    val = np.random.rand(2, 3)
    # t = fluid.Tensor()
    # t.set(val, fluid.CUDAPlace(0))
    t = fluid.data(name='t', shape=[2, 3], )

    t = layers.transpose(t, perm=[0, 2, 1])

    fc = layers.fc(input=data, size=100)
    out = layers.softmax_with_cross_entropy(logits=fc, label=label)

    print(out.shape)

def test_paddle_ops():
    a = fluid.data(name="a", shape=[None, 1], dtype='int64')
    b = fluid.data(name="b", shape=[None, 1], dtype='int64')

    result = {}

    result['add'] = layers.elementwise_add(a, b)
    result['sub'] = layers.elementwise_sub(a, b)

    cpu = fluid.CPUPlace()
    exe = fluid.Executor(cpu)
    exe.run(fluid.default_startup_program())

    import numpy
    data_1 = int(input("Please enter an integer: a="))
    data_2 = int(input("Please enter an integer: b="))
    x = numpy.array([[data_1]])
    y = numpy.array([[data_2]])

    outs = exe.run(
        feed={'a': x, 'b': y},
        fetch_list=[result['add'], result['sub']]
    )

    print("%d+%d=%d" % (data_1, data_2, outs[0][0]))
    print("%d-%d=%d" % (data_1, data_2, outs[1][0]))

def test_paddle_mat_op():
    a = fluid.data(name="a", shape=[None, 4], dtype='float32')
    b = fluid.data(name="b", shape=[None, 4], dtype='float32')

    result = {}
    result['add'] = a+b
    result['sub'] = a-b

    cpu = fluid.CPUPlace()
    exe = fluid.Executor(cpu)
    exe.run(fluid.default_startup_program())

    x = np.random.rand(2, 4).astype(np.float32)
    y = np.random.rand(2, 4).astype(np.float32)

    outs = exe.run(
        feed={'a': x, 'b': y},
        fetch_list=[result['add'], result['sub']]
    )

def test_paddle_tensor():
    gpu = fluid.CUDAPlace(0)

    # OBJECTNESS_CLS_WEIGHTS = np.ndarray(shape=(2,), buffer=np.array([0.2, 0.8]), dtype=float)
    a = fluid.data(name='a', shape=(3, ), dtype='float32')
    b = fluid.data(name='b', shape=(3, ), dtype='float32')

    # sum_t = a[0] * 0.2 # weight_tensor

    # last_c0 = a[:, 0]

    conca = layers.concat([a, b], axis=-1)
    conca = layers.reshape(conca, shape=[2, -1])
    trans = layers.transpose(conca, perm=[1, 0])

    print(len(a.shape))

    exe = fluid.Executor(gpu)
    exe.run(fluid.default_startup_program())

    a_np = np.random.rand(3, ).astype(np.float32)
    b_np = np.random.rand(3, ).astype(np.float32)
    # print(a_np[0])

    out = exe.run(
        feed={'a': a_np, 'b': b_np},
        fetch_list=[a, b, conca, trans]
    )

    print(out)

def test_gather_nd_op():
    x = fluid.layers.data(name='x', shape=[4, 3, 3], dtype='float32')
    index = fluid.layers.data(name='index', shape=[4, 2], dtype='int32')
    output = fluid.layers.gather_nd(x, index)

    use_gpu = True
    if use_gpu:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()

    exe = fluid.executor.Executor(place=place)
    exe.run(fluid.default_startup_program())
    x = np.random.uniform(size=(4, 3, 3)).astype('float32')
    print("X=", x)
    # d = np.array([[0, 0], [1, 1], [2, 2], [3, 1]]).astype('int')
    d = np.random.randint(3, size=[4, 2]).astype(np.int64)
    print('d=', d)

    [y_] = exe.run(fluid.default_main_program(), feed={'x': x, 'index': d}, fetch_list=[output])
    print('M', y_.shape)
    print('Y', y_)

def test_gather_op():
    BATCH_SIZE = 2

    gpu = fluid.CUDAPlace(0)
    x = fluid.data(name='x', shape=(None, 2, 3), dtype='float32')
    idx = fluid.data(name='idx', shape=(None, 10, 3), dtype='int64')

    x_val = np.random.rand(BATCH_SIZE, 2, 3).astype(np.float32)
    idx_val = np.random.randint(2, size=[BATCH_SIZE, 10, 3]).astype(np.int64)

    print(x_val)
    print(idx_val)

    x = layers.transpose(x, perm=[1, 0, 2])
    idx = layers.transpose(idx, perm=[1, 0, 2])

    # tmp = []
    # for i in range(BATCH_SIZE):
    #     tmp.append(layers.gather(x[i], idx[i]))
    # result = layers.concat(tmp, axis=0)

    result = layers.gather(x, idx)

    exe = fluid.Executor(gpu)
    exe.run(fluid.default_startup_program())

    out = exe.run(
        feed={'x': x_val, 'idx': idx_val},
        fetch_list=[result]
    )

    print(out)

def test_item_assignment():
    # Graph Organizing
    x = fluid.layers.data(name='x', shape=[2], dtype='float64')
    y = fluid.layers.data(name='y', shape=[2], dtype='float64')
    result = fluid.layers.less_than(x=x, y=y)
    # The comment lists another available method.
    # result = fluid.layers.fill_constant(shape=[2], dtype='float64', value=0)
    # fluid.layers.less_than(x=x, y=y, cond=result)

    # Create an executor using CPU as example
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())

    # Execute
    x_i = np.array([[1, 2], [3, 4]]).astype(np.float64)
    y_i = np.array([[2, 2], [1, 3]]).astype(np.float64)
    result_value, = exe.run(fluid.default_main_program(), feed={'x': x_i, 'y': y_i}, fetch_list=[result])
    print(result_value)  # [[True, False], [False, False]]

def test_mat():
    x = fluid.layers.data(name='x', shape=[2, 2], dtype='float64')
    # y = fluid.layers.data(name='y', shape=[2], dtype='float64')
    # result = fluid.layers.less_than(x=x, y=y)

    result = x[x > 0.3]

    # Create an executor using CPU as example
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())

    # Execute
    x_i = np.random.rand(2, 2).astype(np.float64)

    print(x_i)

    # y_i = np.array([[2, 2], [1, 3]]).astype(np.float64)
    result_value, = exe.run(fluid.default_main_program(), feed={'x': x_i}, fetch_list=[result])
    print(result_value)  # [[True, False], [False, False]]

def weighted_softmax_cross_entropy_loss(inputs, targets, weight=None, size_average=None, ignore_idx=-100, reduce=None,
                                        reduction=None):
    # inputs: [N, d1, Class]
    # target: [N, d1] a list of class index
    # weight: [Class]
    logger.info('inputs.shape = {}'.format(inputs.shape))
    logger.info('targets.shape = {}'.format(targets.shape))
    # logger.info('weight.shape = {}'.format(weight.shape))

    assert(inputs.shape[0] == targets.shape[0])

    # lets print the inputs.shape and targets.shape here
    # logger.info('Weighted_softmax_cross_entropy_loss: inputs.shape={}, targets.shape={}'.format(inputs.shape, targets.shape))

    num_of_class = inputs.shape[-1]

    # Apply log_softmax
    input_log_sm = -layers.log_softmax(input=inputs, axis=-1)

    # Convert target index list to one-hot encode
    targets_onehot_mat = fluid.one_hot(targets, depth=num_of_class)

    loss_mat = targets_onehot_mat * input_log_sm

    assert len(weight) == targets_onehot_mat.shape[-1], 'Error: length of weight array is not equal mat channel size.'

    loss_mat_t = layers.transpose(loss_mat, perm=[2, 1, 0])
    loss_mat_t_0 = layers.unsqueeze(layers.transpose(weight[0] * loss_mat_t[0], perm=[1, 0]), -1)
    loss_mat_t_1 = layers.unsqueeze(layers.transpose(weight[1] * loss_mat_t[1], perm=[1, 0]), -1)
    loss_mat = layers.concat([loss_mat_t_0, loss_mat_t_1], axis=-1)

    loss = layers.reduce_sum(loss_mat, dim=-1)
    return loss

def test_cross_entropy_loss():
    BATCH_SIZE = 2
    X_SHAPE = (BATCH_SIZE, 5, 2)
    Y_SHAPE = (BATCH_SIZE, 5)
    WEIGHTS = [0.2, 0.8]

    x = fluid.data(name='x', shape=X_SHAPE, dtype='float32')  # pred
    y = fluid.data(name='y', shape=Y_SHAPE, dtype='int64')  # label

    # x_val = np.random.rand(BATCH_SIZE, 5, 2).astype(np.float32)
    # y_val = np.random.randint(2, size=[BATCH_SIZE, 5]).astype(np.int64)

    x_val = np.array([[[0.1835, 0.2570],
                     [0.3595, 0.8258],
                     [0.8185, 0.8752],
                     [0.8778, 0.5042],
                     [0.5456, 0.9268]],
                    [[0.0696, 0.3445],
                     [0.4098, 0.1119],
                     [0.1921, 0.6646],
                     [0.1789, 0.8803],
                     [0.9501, 0.3315]]]).astype(np.float32)
    y_val = np.array([[0, 1, 0, 0, 0],
                      [1, 1, 1, 0, 1]]).astype(np.int64)

    answer = np.array([[0.1461, 0.3896, 0.1444, 0.1047, 0.1804],
                       [0.4521, 0.6825, 0.3876, 0.2208, 0.8396]]).astype(np.float32)
    print(x_val)
    print(y_val)
    print(answer)

    result = weighted_softmax_cross_entropy_loss(x, y, WEIGHTS)

    gpu = fluid.CUDAPlace(0)
    exe = fluid.Executor(gpu)
    exe.run(fluid.default_startup_program())

    out = exe.run(
        feed={'x': x_val, 'y': y_val},
        fetch_list=[result]
    )

    print(out)

def test_greaterop():
    BATCH_SIZE = 2
    X_SHAPE = (BATCH_SIZE, 3)
    Y_SHAPE = (BATCH_SIZE, 3)
    alpha = 0.3

    x = layers.zeros(shape=X_SHAPE, dtype='float32') #fluid.data(name='x', shape=X_SHAPE, dtype='float32')  # pred
    y = fluid.data(name='y', shape=Y_SHAPE, dtype='float32')  # label

    # x_val = np.random.rand(BATCH_SIZE, 5, 2).astype(np.float32)
    # y_val = np.random.randint(2, size=[BATCH_SIZE, 5]).astype(np.int64)

    y_val = np.array([[0.1313, 0.1197, 0.2880],
                    [0.1332, 0.6165, 0.7328]]).astype(np.float32)

    mask = layers.cast(y > alpha, 'int32')


    # print(x_val)
    print(y_val)
    # print(answer)

    # result = weighted_softmax_cross_entropy_loss(x, y, WEIGHTS)

    gpu = fluid.CUDAPlace(0)
    exe = fluid.Executor(gpu)
    exe.run(fluid.default_startup_program())

    out = exe.run(
        feed={'y': y_val},
        fetch_list=[mask]
    )

    print(out)



if __name__=='__main__':
    # test_paddle_ops()
    # test_paddle_mat_op()
    # test_paddle_tensor()
    # test_gather_op()
    # test_item_assignment()
    # test_mat()
    # test_cross_entropy_loss()
    test_greaterop()