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

    pass

if __name__=='__main__':
    # TODO: test pointnet2 functions
    # test_paddle_ops()
    test_paddle_mat_op()
