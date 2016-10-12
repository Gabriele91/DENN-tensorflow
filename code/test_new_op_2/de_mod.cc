#include "tensorflow/core/framework/op.h"

REGISTER_OP("NewGen")
    .Input("to_zero: double")
    .Output("zeroed: double");

#include "tensorflow/core/framework/op_kernel.h"

#include <iostream>
#include <typeinfo>

using namespace tensorflow;

class NewGenOp : public OpKernel {
 public:
  explicit NewGenOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    std::cout << typeid(context->input(0)).name() << '\n';
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat_inner_dims<double>();
    std::cout << typeid(input).name() << '\n';

    
    const int size = input_tensor.shape().dims();  // to check dimensions
    const int NP = input_tensor.shape().dim_size(0);
    const int D = input_tensor.shape().dim_size(1);
    std::cout << NP << " " << D << '\n';

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->flat_inner_dims<double>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output(i) = 0;
    }

    // Preserve the first input value if possible.
    if (N > 0) output(0, 2) = input(0, 1);
  }
};

REGISTER_KERNEL_BUILDER(Name("NewGen").Device(DEVICE_CPU), NewGenOp);