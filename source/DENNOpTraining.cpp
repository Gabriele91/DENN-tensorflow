#include <dennop_training.h>

using namespace tensorflow;

REGISTER_OP("DennTraining")
.Attr("T: {float, double}")
.Attr("space: int")
.Attr("graph: string")
//dataset
.Attr("dataset: string")
//de
.Attr("JDE: float = 0.1")
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
//smoothing
.Attr("smoothing: list(shape)")
.Attr("smoothing_n_pass: int = 0")
//reset
.Attr("reset_type: {'none','execute'} = 'none'")
.Attr("reset_counter: int  = 0")
.Attr("reset_f: string")
.Attr("reset_cr: string")
.Attr("reset_rand_pop: list(string)")
//insert best 
.Attr("reinsert_best: bool = false")
//clamp
.Attr("f_min: float = -1.0")
.Attr("f_max: float = +1.0")
//NN
.Attr("f_input_labels: string = 'y'")
.Attr("f_input_features: string = 'x'")
.Attr("f_inputs: list(string)")
.Attr("f_name_execute_net: string = 'cross_entropy:0'")
.Attr("f_name_validation: string = 'accuracy:0'")
.Attr("f_name_test: string = 'accuracy:0'")
// Input 
.Input("info: int32") //[ NUM_GEN, NUM_GEN_STEP, CALC_FIRST_EVAL ]
.Input("f: T")
.Input("cr: T")
.Input("population_first_eval: T")
.Input("populations_list: space * T")
// Output 
.Output("final_eval_of_best: T")
.Output("final_eval_of_best_of_best: T")

.Output("final_best_f: T")
.Output("final_best_cr: T")
.Output("final_best: space * T")

.Output("final_f: T")
.Output("final_cr: T")
.Output("final_populations: space * T")
;

REGISTER_KERNEL_BUILDER(Name("DennTraining").Device(DEVICE_CPU).TypeConstraint<float>("T"), DENNOpTraining<float>);
REGISTER_KERNEL_BUILDER(Name("DennTraining").Device(DEVICE_CPU).TypeConstraint<double>("T"), DENNOpTraining<double>);
