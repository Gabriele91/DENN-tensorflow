#include <config.h>
#include <dennop.h>

using namespace tensorflow;

REGISTER_OP("DENN")
.Attr("space: int")
.Attr("graph: string")
.Attr("CR: float = 0.5")
.Attr("f_min: float = -1.0")
.Attr("f_max: float = +1.0")
.Attr("f_inputs: list(string)")
.Attr("f_input_labels: string = 'y'")
.Attr("f_input_features: string = 'x'")
.Attr("f_name_train: string = 'cross_entropy:0'")
.Attr("DE: {"
      "'rand/1/bin', "
      "'rand/1/exp', "
      "'rand/2/bin', "
      "'rand/2/exp',  "
      "'best/1/bin', "
      "'best/1/exp', "
      "'best/2/bin', "
      "'best/2/exp'  "
      "} = 'rand/1/bin'")
.Input("info: int32") //[ NUM_GEN, CALC_FIRST_EVAL ]
.Input("bach_labels: double")
.Input("bach_data: double")
.Input("population_first_eval: double")
.Input("w_list: space * double")
.Input("populations_list: space * double")
.Output("final_populations: space * double")
.Output("final_eval: double");


REGISTER_KERNEL_BUILDER(Name("DENN").Device(DEVICE_CPU), DENNOp<double>);

