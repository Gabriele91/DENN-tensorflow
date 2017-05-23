#include <dennop_ada_boost_training.h>

using namespace tensorflow;

REGISTER_OP("DennAdaTraining")
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
//EXECUTE COMPARATION Y and Y_
.Attr("f_input_correct_predition: string = 'x'")
.Attr("f_correct_predition: string = 'ada_label_diff:0'")
//EXECUTE CROSS ENTROPY
.Attr("f_input_cross_entropy_c: string = 'cross_entropy_x:0'")
.Attr("f_input_cross_entropy_y: string = 'cross_entropy_y:0'")
.Attr("f_cross_entropy: string = 'cross_entropy:0'")
//Validation and Test fn
.Attr("f_name_validation: string = 'accuracy:0'")
.Attr("f_name_test: string = 'accuracy:0'")
//ADA 
.Attr("ada_boost_alpha: float = 0.5")
.Attr("ada_boost_c: float = 1.0")
.Attr("ada_reset_c_on_change_bacth: bool = true")
//inputs 
.Input("info: int32") //[ NUM_GEN, NUM_GEN_STEP ]
.Input("f: T")
.Input("cr: T")
.Input("populations_list: space * T")
//output
.Output("final_eval_of_best: T")
.Output("final_eval_of_best_of_best: T")

.Output("final_best_f: T")
.Output("final_best_cr: T")
.Output("final_best: space * T")

.Output("final_f: T")
.Output("final_cr: T")
.Output("final_populations: space * T")
;

REGISTER_KERNEL_BUILDER(Name("DennAdaTraining").Device(DEVICE_CPU).TypeConstraint<float>("T"), DENNOpAdaBoostTraining<float>);
REGISTER_KERNEL_BUILDER(Name("DennAdaTraining").Device(DEVICE_CPU).TypeConstraint<double>("T"), DENNOpAdaBoostTraining<double>);
