#include <config.h>
#include <dennop.h>

using namespace tensorflow;

REGISTER_OP("DENN")
.Attr("T: {float, double}")
.Attr("space: int")
.Attr("graph: string")
.Attr("JDE: float = 0.1")
.Attr("smoothing: list(shape)")
.Attr("smoothing_n_pass: int = 0")
.Attr("inheritance: float = 1.0")
.Attr("f_min: float = -1.0")
.Attr("f_max: float = +1.0")
.Attr("f_inputs: list(string)")
.Attr("f_input_labels: string = 'y'")
.Attr("f_input_features: string = 'x'")
.Attr("f_name_execute_net: string = 'cross_entropy:0'")
.Attr("DE: {"
      "'rand/1/bin', "
      "'rand/1/exp', "
      "'rand/2/bin', "
      "'rand/2/exp', "
      "'best/1/bin', "
      "'best/1/exp', "
      "'best/2/bin', "
      "'best/2/exp', "
      "'current-to-best/1/bin', "
      "'current-to-best/1/exp'  "
      "} = 'rand/1/bin'")
//input 
.Input("info: int32") //[ NUM_GEN, CALC_FIRST_EVAL ]
.Input("batch_labels: T")
.Input("batch_data: T")
//population
.Input("f: T")
.Input("cr: T")
.Input("population_first_eval: T")
.Input("populations_list: space * T")
//output
.Output("final_f: T")
.Output("final_cr: T")
.Output("final_populations: space * T")
.Output("final_eval: T");


REGISTER_KERNEL_BUILDER(Name("DENN").Device(DEVICE_CPU).TypeConstraint<float>("T"), DENNOp<float>);
REGISTER_KERNEL_BUILDER(Name("DENN").Device(DEVICE_CPU).TypeConstraint<double>("T"), DENNOp<double>);

