
#include "config.h"
#include <vector>
#include <string>

namespace tensorflow
{
    //types
    using NameList       =  std::vector< std::string >;
    using TensorList     =  std::vector< tensorflow::Tensor >;
    using TensorListList =  std::vector< TensorList >;
    using TensorInput    =  std::pair< std::string, tensorflow::Tensor >;
    using TensorInputs   =  std::vector< TensorInput >;
}