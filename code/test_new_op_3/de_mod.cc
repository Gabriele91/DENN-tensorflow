#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include <iostream>
#include <stdlib.h> /* srand, rand */

#include <typeinfo>

using namespace tensorflow;

REGISTER_OP("TrialVectors")
    .Input("population: double")
    .Input("weight: double")
    .Input("crossover: double")
    .Output("new_generation: double");

class TrialVectorsOp : public OpKernel
{
public:
  explicit TrialVectorsOp(OpKernelConstruction *context) : OpKernel(context) {}

  void Compute(OpKernelContext *context) override
  {
    //
    //std::cout << typeid(context->input(0)).name() << '\n';
    //

    // Grab the input tensors
    const Tensor &population_tensor = context->input(0);
    const Tensor &W_tensor = context->input(1);
    const Tensor &CR_tensor = context->input(2);

    auto population = population_tensor.flat_inner_dims<double>();
    const auto W = W_tensor.flat<double>();
    const double CR = CR_tensor.flat<double>()(0);
    // std::cout << W(0) << " " << CR << '\n';

    // const int size = population_tensor.shape().dims(); // to check dimensions
    const int NP = population_tensor.shape().dim_size(0);
    const int D = population_tensor.shape().dim_size(1);
    // std::cout << NP << " " << D << '\n';

    // Output tensor
    // Create an output tensor
    Tensor *new_generation_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, population_tensor.shape(),
                                                     &new_generation_tensor));

    auto new_generation = new_generation_tensor->flat_inner_dims<double>();

    // Generate new population
    int i_a = 0,
        i_b = 0,
        i_c = 0;

    for (int index = 0; index < NP; ++index)
    {
      // std::cout << index << " " << '\n';
      threeRandIndicesDiffFrom(NP, index, i_a, i_b, i_c);
      int random_index = irand(D);
      // std::cout << index << " " << i_a << " " << i_b << " " << i_c << " " << '\n';
      for (int elm = 0; elm < D; ++elm)
      {
        if (random() < CR ||  random_index == elm)
        {
          new_generation(index, elm) = (population(i_a, elm) - population(i_b, elm)) * W(elm) + population(i_c, elm);
        }
        else
        {
          new_generation(index, elm) = population(index, elm);
        }
      }
    }

    // Set all but the first element of the output tensor to 0.
    // const int N = population.size();
    // for (int i = 1; i < N; i++)
    // {
    //   new_generation(i) = 0;
    // }

    // Preserve the first input value if possible.
    // if (N > 0)
    //   new_generation(0, 2) = population(0, 1);
  }

private:
  //random integer in [0,size)
  inline int irand(int size)
  {
    return rand() % size;
  }
  //first, second, third are integers in [0,size) different among them and with respect to diffFrom
  void threeRandIndicesDiffFrom(int size, int diffFrom, int &first, int &second, int &third)
  {
    //3 calls to the rng
    first = (diffFrom + 1 + irand(size - 1)) % size; //first in [0,size[ excluded diffFrom
    int min, med, max;
    if (first < diffFrom)
    {
      min = first;
      max = diffFrom;
    }
    else
    {
      min = diffFrom;
      max = first;
    }
    second = (min + 1 + irand(size - 2)) % size;
    if (second >= max || second < min)
      second = (second + 1) % size;
    if (second < min)
    {
      med = min;
      min = second;
    }
    else if (second > max)
    {
      med = max;
      max = second;
    }
    else
      med = second;
    third = (min + 1 + irand(size - 3)) % size;
    if (third < min || third >= max - 1)
      third = (third + 2) % size;
    else if (third >= med)
      third++; //no modulo since I'm sure to not overflow size
  }
};

REGISTER_KERNEL_BUILDER(Name("TrialVectors").Device(DEVICE_CPU), TrialVectorsOp);

REGISTER_OP("AssignIndividual")
    .Input("population: double")
    .Input("individual: double")
    .Input("position: int32");

class AssignIndividualOp : public OpKernel
{
public:
  explicit AssignIndividualOp(OpKernelConstruction *context) : OpKernel(context) {}

  void Compute(OpKernelContext *context) override
  {
    //
    //std::cout << typeid(context->input(0)).name() << '\n';
    //

    // Grab the input tensors
    const Tensor &population_tensor_const = context->input(0);
    Tensor& population_tensor = const_cast<Tensor&>(population_tensor_const);
    const Tensor &individual_tensor = context->input(1);
    const Tensor &position_tensor = context->input(2);

    auto population = population_tensor.flat_inner_dims<double>();
    const auto individual = individual_tensor.flat<double>();
    const double position = position_tensor.flat<int32>()(0);
    const int D = population_tensor.shape().dim_size(1);

    for (int elm = 0; elm < D; ++elm)
    {
      population(position, elm) = individual(elm);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("AssignIndividual").Device(DEVICE_CPU), AssignIndividualOp);