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
.Attr("f_name_train: string = 'cross_entropy:0'")
.Attr("f_name_validation: string = 'accuracy:0'")
.Attr("f_name_test: string = 'accuracy:0'")
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
.Input("info: int32") //[ NUM_GEN, NUM_GEN_STEP, CALC_FIRST_EVAL ]
.Input("population_first_eval: double")
.Input("w_list: space * double")
.Input("populations_list: space * double")
.Output("best_population: space * double")
;


REGISTER_KERNEL_BUILDER(Name("DENN").Device(DEVICE_CPU), DENNOpTraining<double>);

