#include <config.h>
#include <dennop_ada_boost.h>

using namespace tensorflow;

REGISTER_OP("DennAda")
.Attr("T: {float, double}")
.Attr("space: int")
.Attr("graph: string")
.Attr("JDE: float = 0.1")
.Attr("smoothing: list(shape)")
.Attr("smoothing_n_pass: int = 0")
.Attr("ada_boost_alpha: float = 0.5")
.Attr("f_min: float = -1.0")
.Attr("f_max: float = +1.0")
//EXECUTE NN
.Attr("f_inputs: list(string)")
.Attr("f_input_labels: string = 'y'")
.Attr("f_input_features: string = 'x'")
.Attr("f_name_execute_net: string = 'cross_entropy:0'")
//EXECUTE COMPARATION Y and Y_
.Attr("f_input_correct_predition: string = 'x'")
.Attr("f_correct_predition: string = 'ada_label_diff:0'")
//EXECUTE CROSS ENTROPY
.Attr("f_input_cross_entropy_c: string = 'cross_entropy_x:0'")
.Attr("f_input_cross_entropy_y: string = 'cross_entropy_y:0'")
.Attr("f_cross_entropy: string = 'cross_entropy:0'")
//
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

//Input
.Input("info: int32") //[ NUM_GEN ]
//batch
.Input("batch_labels: T")
.Input("batch_data: T")
//population
.Input("f: T")
.Input("cr: T")
.Input("populations_list: space * T")
//ada
.Input("c: T")

//Output
//population
.Output("final_f:T")
.Output("final_cr:T")
.Output("final_populations: space * T")
//ada
.Output("final_c: T");

REGISTER_KERNEL_BUILDER(Name("DennAda").Device(DEVICE_CPU).TypeConstraint<float>("T"), DENNOpAdaBoost<float>);
REGISTER_KERNEL_BUILDER(Name("DennAda").Device(DEVICE_CPU).TypeConstraint<double>("T"), DENNOpAdaBoost<double>);

