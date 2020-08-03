/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class GatherDimOp : public framework::OperatorWithKernel {
public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) shoud not be null");
    auto x_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE(x_dims.size() == 3 && x_dims[2] == 3,
                   "Input(X) of GatherDimOp should be 3-D Tensor, the last "
                   "dimension must be 3");
    auto index_dims = ctx->GetInputDim("Index");
    PADDLE_ENFORCE(index_dims.size() == 3 && index_dims[2] == 3,
                   "Index of GatherDimOp should be 3-D Tensor, the last dim must be 3");
    PADDLE_ENFORCE(index_dims[0] == x_dims[0], 
    		    "Input(X) and Index should have the same batch_size.");
    ctx->SetOutputDim("Output", {x_dims[0], index_dims[1], 3});
  }

protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type = ctx.Input<Tensor>("X")->type();
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

class GatherDimOpMaker : public framework::OpProtoAndCheckerMaker {
public:
  void Make() override {
    AddInput("X",
             "Input points with shape (batch, n, 3), n is input "
             "points's num");
    AddInput("Index",
             "input index with shape (batch, m, 3), m is output idx's num");
    AddOutput("Output", "output points with shape(batch, m, 3)");
    AddComment(
        R"Doc(
        Gather Dim Operator.
        Out is obtained by gathering entries of X indexed by Index and 
        concatenate them together.

        Example:
        X = [[[0.9427, 0.0364, 0.2587],
		 [0.4433, 0.3639, 0.4383],
		 [0.5494, 0.4386, 0.2218]],
		[[0.1443, 0.9749, 0.3620],
		 [0.6472, 0.0879, 0.7137],
		 [0.2322, 0.3581, 0.8765]]]
		 
        Index = [[[0, 0, 2],
		 [2, 0, 2],
		 [1, 1, 2],
		 [1, 0, 0],
		 [2, 0, 2]],
		[[2, 2, 0],
		 [1, 1, 1],
		 [2, 2, 2],
		 [1, 0, 1],
		 [1, 0, 1]]]

        Then:
        Out = [[[0.9427, 0.0364, 0.2218],
		 [0.5494, 0.0364, 0.2218],
		 [0.4433, 0.3639, 0.2218],
		 [0.4433, 0.0364, 0.2587],
		 [0.5494, 0.0364, 0.2218]],
		[[0.2322, 0.3581, 0.3620],
		 [0.6472, 0.0879, 0.7137],
		 [0.2322, 0.3581, 0.8765],
		 [0.6472, 0.9749, 0.7137],
		 [0.6472, 0.9749, 0.7137]]])Doc");
  }
};

class GatherDimOpGrad : public framework::OperatorWithKernel {
public:
  using framework::OperatorWithKernel::OperatorWithKernel;

protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Index"), "Input(Index) should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Output")),
                   "Input(Output@GRAD) should not be null");
    auto dim_x = ctx->GetInputDim("X");
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), dim_x);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        ctx.Input<Tensor>(framework::GradVarName("Output"))->type(),
        ctx.GetPlace());
  }
};

template <typename T>
class GatherDimGradDescMaker : public framework::SingleGradOpMaker<T> {
public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("gather_dim_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Index", this->Input("Index"));
    op->SetInput(framework::GradVarName("Output"), this->OutputGrad("Output"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(gather_dim,
                  ops::GatherDimOp,
                  ops::GatherDimOpMaker,
                  ops::GatherDimGradDescMaker<paddle::framework::OpDesc>,
                  ops::GatherDimGradDescMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(gather_dim_grad, ops::GatherDimOpGrad);
