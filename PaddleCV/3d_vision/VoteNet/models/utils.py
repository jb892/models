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

def test_gather_op4():
    gpu = fluid.CUDAPlace(0)
    channel = 3

    x = fluid.data(name='x', shape=(3, 3), dtype='float32')
    idx = fluid.data(name='idx', shape=[1, 3], dtype='int64')
    # label = fluid.data(name='label', shape=[3], dtype="int64")

    # label = layers.unsqueeze(label, 0)
    # label = layers.unsqueeze(label, 0)
    # label = layers.expand(label, expand_times=[3, 1])

    x = layers.transpose(x, perm=[1, 0])
    result = layers.gather(x, idx)

    # idx_ = layers.unsqueeze(idx, 0)
    # idx_ = layers.expand(idx_, expand_times=[3, 1])

    # mask = layers.equal(x=label, y=idx_)

    # result = layers.masked_select(x, mask)
    # result = x[mask]

    # bool_tensor = layers.zeros(shape=[3, 3], dtype='float32')
    # bool_tensor[idx] = 1.0

    x_val = np.array([[0.9427, 0.0364, 0.2587],
                     [0.4433, 0.3639, 0.4383],
                     [0.5494, 0.4386, 0.2218]]).astype(np.float32)

    idx_val = np.array([[0, 0, 2]]).astype(np.int64)

    # label_val = np.array(range(channel)).astype(np.int64)

    # idx_val = np.expend_dims(idx_val, axis=0)

    print(x_val)
    print(idx_val)

    exe = fluid.Executor(gpu)
    exe.run(fluid.default_startup_program())

    out = exe.run(
        feed={'x': x_val, 'idx': idx_val},
        fetch_list=[result]
    )

    print(out)

def test_gather_op_own_forloop():
    BATCH_SIZE = 2

    gpu = fluid.CUDAPlace(0)

    x = fluid.data(name='x', shape=(None, 3, 3), dtype='float32')
    idx = fluid.data(name='idx', shape=(None, 5, 3), dtype='int64')

    x_val = np.random.rand(BATCH_SIZE, 3, 3).astype(np.float32)
    idx_val = np.random.randint(3, size=[BATCH_SIZE, 5, 3]).astype(np.int64)

    print(x_val)
    print(idx_val)

    # out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
    for i in range(BATCH_SIZE):
        for j in range(idx.shape[1]):
            for k in range(idx.shape[2]):
                pass



def test_gather_op_own():
    BATCH_SIZE = 2

    gpu = fluid.CUDAPlace(0)

    x = fluid.data(name='x', shape=(None, 3, 3), dtype='float32')
    idx = fluid.data(name='idx', shape=(None, 5, 3), dtype='int64')

    x_val = np.random.rand(BATCH_SIZE, 3, 3).astype(np.float32)
    idx_val = np.random.randint(3, size=[BATCH_SIZE, 5, 3]).astype(np.int64)

    print(x_val)
    print(idx_val)

    tmp = []
    # for i in range(BATCH_SIZE):
    #     x_T = layers.transpose(x[i], perm=[1, 0])
    #     print('x_T.shape = {}'.format(x_T.shape))
    #     print('idx[i].shape = {}'.format(idx[i].shape))
    #     idx_ = layers.reshape(idx[i], shape=[idx[i].shape[0] * idx[i].shape[1]])
    #     tmp.append(layers.gather(x_T, idx_))
    for i in range(BATCH_SIZE):
        tmp_2 = []
        print('idx[i].shape[0] = {}'.format(idx[i].shape[0]))
        print('idx[i].shape = {}'.format(idx[i].shape))
        print('x[i].shape = {}'.format(x[i].shape))
        tmp.append(layers.gather(x[i], idx[i][:3, :]))

        # for j in range(idx[i].shape[0]):
        #     r = layers.gather(x[i], idx[i][j])
        #     tmp_2.append(r)
        # tmp_2_concat = layers.concat(tmp_2)
        # print('tmp_2_concat.shape = {}'.format(tmp_2_concat.shape))
        # tmp.append(tmp_2_concat)
    result = layers.concat(tmp, axis=0)

    # result = layers.concat(tmp, axis=0)
    # result = layers.reshape(result, shape=[BATCH_SIZE, idx.shape[1], idx.shape[2]])

    exe = fluid.Executor(gpu)
    exe.run(fluid.default_startup_program())

    out = exe.run(
        feed={'x': x_val, 'idx': idx_val},
        fetch_list=[result]
    )

    print(out)

def test_gather_op3():
    BATCH_SIZE = 2

    gpu = fluid.CUDAPlace(0)
    x = fluid.data(name='x', shape=(None, 3, 3), dtype='float32')
    idx = fluid.data(name='idx', shape=(None, 3, 3), dtype='int64')

    x_val = np.random.rand(BATCH_SIZE, 3, 3).astype(np.float32)
    idx_val = np.random.randint(3, size=[BATCH_SIZE, 3, 3]).astype(np.int64)

    print(x_val)
    print(idx_val)

    idx_oneslike = layers.cast(layers.ones_like(idx), 'float32')
    print('idx shape= {}'.format(idx_oneslike.shape))

    x = layers.concat([x, idx_oneslike], axis=0)

    print('shape = {}'.format(x.shape))

    tmp = []
    for i in range(BATCH_SIZE):
        # idx_oneslike = layers.cast(layers.ones_like(idx[i]), 'float32')
        # tmp.append(layers.gather(layers.concat([x[i], idx_oneslike], axis=0), idx[i]))
        tmp.append(layers.gather(layers.unsqueeze(layers.transpose(x[i], perm=[1, 0]), -1), idx[i][0]))
    result = layers.concat(tmp)
    # result = layers.reshape(result, shape=[BATCH_SIZE, idx.shape[1]])

    # result = layers.gather(x, idx)

    exe = fluid.Executor(gpu)
    exe.run(fluid.default_startup_program())

    out = exe.run(
        feed={'x': x_val, 'idx': idx_val},
        fetch_list=[result]
    )

    print(out)

def test_gather_op2():
    BATCH_SIZE = 2

    gpu = fluid.CUDAPlace(0)
    x = fluid.data(name='x', shape=(None, 2), dtype='float32')
    idx = fluid.data(name='idx', shape=(None, 5), dtype='int64')

    x_val = np.random.rand(BATCH_SIZE, 2).astype(np.float32)
    idx_val = np.random.randint(2, size=[BATCH_SIZE, 5]).astype(np.int64)

    print(x_val)
    print(idx_val)

    tmp = []
    for i in range(BATCH_SIZE):
        tmp.append(layers.gather(x[i], idx[i]))
    result = layers.concat(tmp)
    result = layers.reshape(result, shape=[BATCH_SIZE, idx.shape[1]])

    # result = layers.gather(x, idx)

    exe = fluid.Executor(gpu)
    exe.run(fluid.default_startup_program())

    out = exe.run(
        feed={'x': x_val, 'idx': idx_val},
        fetch_list=[result]
    )

    print(out)

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

if __name__=='__main__':
    # test_paddle_ops()
    # test_paddle_mat_op()
    # test_paddle_tensor()
    # test_gather_op()
    test_gather_op2()
    # test_gather_op3()
    # test_gather_op4()
    # test_gather_nd_op()
    # test_gather_op_own()
    # test_gather_op_own_forloop()