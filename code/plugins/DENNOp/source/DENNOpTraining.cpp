#include <dennop_training.h>

using namespace tensorflow;

REGISTER_OP("DENN")
.Attr("space: int")
.Attr("graph: string")
.Attr("dataset: string")
.Attr("CR: float = 0.5")
.Attr("f_min: float = -1.0")
.Attr("f_max: float = +1.0")
.Attr("f_input_labels: string = 'y'")
.Attr("f_input_features: string = 'x'")
.Attr("f_inputs: list(string)")
.Attr("f_name_train: string = 'evaluate_train'")
.Attr("f_name_validation: string = 'evaluate_validation'")
.Attr("f_name_test: string = 'evaluate_test'")
.Attr("DE: {'rand/1/bin', 'rand/1/exp', 'rand/2/bin', 'rand/2/exp'} = 'rand/1/bin'")
.Input("info: int32") //[ NUM_GEN, NUM_GEN_STEP, CALC_FIRST_EVAL ]
.Input("population_first_eval: double")
.Input("w_list: space * double")
.Input("populations_list: space * double")
.Output("final_population: double");


REGISTER_KERNEL_BUILDER(Name("DENN").Device(DEVICE_CPU), DENNOpTraining);

