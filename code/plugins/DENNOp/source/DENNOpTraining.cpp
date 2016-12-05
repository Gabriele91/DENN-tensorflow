#include <dennop_training.h>

using namespace tensorflow;

REGISTER_OP("DENN")
.Attr("space: int")
.Attr("graph: string")
.Attr("CR: float = 0.5")
.Attr("fmin: float = -1.0")
.Attr("fmax: float = +1.0")
.Attr("names: list(string)")
.Attr("DE: {'rand/1/bin', 'rand/1/exp', 'rand/2/bin', 'rand/2/exp'} = 'rand/1/bin'")
.Input("info: int32") //[ NUM_GEN, CALC_FIRST_EVAL ]
.Input("population_first_eval: double")
.Input("w_list: space * double")
.Input("populations_list: space * double")
.Output("final_populations: space * double")
.Output("final_eval: double");


REGISTER_KERNEL_BUILDER(Name("DENN").Device(DEVICE_CPU), DENNOpTraining);
