#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import unittest
import numpy as np
import paddle.fluid as fluid
import pointnet_lib

# def gather_dim_np(points, index):
#     result = []
#     for i in range(len(index)):
#         a = points[i][index[i]]
#         result.append(a.tolist())
#     return result

class TestGatherDimOp(unittest.TestCase):
    def test_check_output(self):
        x_shape = (2, 3, 3)
        x_type = 'float32'
        idx_shape = (2, 5, 3)
        idx_type = 'int32'

        x = fluid.data(
            name='x', shape=x_shape, dtype=x_type)
        idx = fluid.data(
            name='idx', shape=idx_shape, dtype=idx_type)
        y = pointnet_lib.gather_dim(x, idx)

        # x_np = np.random.uniform(-10, 10, x_shape).astype(x_type)
        # idx_np = np.random.randint(0, x_shape[1], idx_shape).astype(idx_type)
        
        x_np = np.array([[[0.9427, 0.0364, 0.2587],
                         [0.4433, 0.3639, 0.4383],
                         [0.5494, 0.4386, 0.2218]],
                        [[0.1443, 0.9749, 0.3620],
                         [0.6472, 0.0879, 0.7137],
                         [0.2322, 0.3581, 0.8765]]]).astype(np.float32)

        idx_np = np.array( [[[0, 0, 2],
                             [2, 0, 2],
                             [1, 1, 2],
                             [1, 0, 0],
                             [2, 0, 2]],
                            [[2, 2, 0],
                             [1, 1, 1],
                             [2, 2, 2],
                             [1, 0, 1],
                             [1, 0, 1]]]).astype(np.int32)
        
        out_np = np.array( [[[0.9427, 0.0364, 0.2218],
                             [0.5494, 0.0364, 0.2218],
                             [0.4433, 0.3639, 0.2218],
                             [0.4433, 0.0364, 0.2587],
                             [0.5494, 0.0364, 0.2218]],
                            [[0.2322, 0.3581, 0.3620],
                             [0.6472, 0.0879, 0.7137],
                             [0.2322, 0.3581, 0.8765],
                             [0.6472, 0.9749, 0.7137],
                             [0.6472, 0.9749, 0.7137]]])

        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        outs = exe.run(feed={'x': x_np, 'idx': idx_np}, fetch_list=[y])
        
        print(outs[0])

        self.assertTrue(np.allclose(outs[0], out_np))


if __name__ == "__main__":
    unittest.main()
