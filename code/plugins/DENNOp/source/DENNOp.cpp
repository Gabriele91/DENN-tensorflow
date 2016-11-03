#include <string>
#include <iostream>
#include <stdlib.h> /* srand, rand */
#include <memory>
#include <cmath>
#include <algorithm>
#include <typeinfo>

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/env.h"


#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("DENN")
.Attr("space: int")
.Attr("graph: string")
.Attr("CR: float")
.Attr("fmin: float")
.Attr("fmax: float")
.Input("num_gen: int32")
.Input("w_list: space * double")
.Input("populations_list: space * double")
.Output("final_populations: space * double")
.Output("final_eval: double");

/***
*
* TESTING: == / c::concatDim0 / splitDim0::splitDim0
*
* if(!(concatDim0(splitDim0(population)) == population))
* {
*   context->CtxFailure
*   (
*      {tensorflow::error::Code::ABORTED,"concatDim0 != splitDim0"}
*   );
* }
*
***/

bool operator ==(const tensorflow::Tensor& left,const tensorflow::Tensor& right)
{
    //type
    if(left.dtype() != right.dtype()) return false;
    //shape size
    if(left.shape().dims()  != right.shape().dims()) return false;
    //shape dims size
    for(size_t i=0;i!=left.shape().dims();++i)
    {
        if(left.shape().dim_size(i)!=right.shape().dim_size(i)) return false;
    }
    //get memory (N.B. string fail)
    StringPiece left_data  = left.tensor_data();
    StringPiece right_data = right.tensor_data();
    //compare
    return std::memcmp(left_data.data(), right_data.data(),  left_data.size()) == 0;
}


class DENNOp : public OpKernel
{
    using NameList      =  std::vector< std::string >;
    using TensorList    =  std::vector< tensorflow::Tensor >;
    using TensorInput   =  std::pair< std::string, tensorflow::Tensor >;
    using TensorInputs  =  std::vector< TensorInput >;
    
    
    //session
    std::unique_ptr< Session >            m_session;
    //clamp
    double                                m_f_min{ -1.0 };
    double                                m_f_max{  1.0 };
    //update factors
    double                                m_CR   {  0.5 };
    // population variables
    int                                   m_space_size{ 1 };
    
public:
    
    explicit DENNOp(OpKernelConstruction *context) : OpKernel(context)
    {
        // space size
        OP_REQUIRES_OK(context, context->GetAttr("space", &m_space_size));
        //get graph path
        std::string graph_proto_string;
        // Get the index of the value to preserve
        OP_REQUIRES_OK(context, context->GetAttr("graph", &graph_proto_string));
        // float params temp
        float
        f_CR,
        f_f_min,
        f_f_max;
        // get CR
        OP_REQUIRES_OK(context, context->GetAttr("CR", &f_CR));
        // get f min
        OP_REQUIRES_OK(context, context->GetAttr("fmin", &f_f_min));
        // get f max
        OP_REQUIRES_OK(context, context->GetAttr("fmax", &f_f_max));
        // params float to double
        m_CR      = f_CR;
        m_f_min  = f_f_min;
        m_f_max  = f_f_max;
        //options
        SessionOptions options;
        //session
        m_session = std::unique_ptr< Session >(tensorflow::NewSession(options));
        //read operation
        GraphDef graph_def;
        /**
         * REF
         * https://github.com/tensorflow/tensorflow/blob/c8a45a8e236776bed1d14fd71f3b6755bd63cc58/tensorflow/core/platform/env.cc#L316
         */
        
        ::tensorflow::protobuf::TextFormat::ParseFromString(graph_proto_string, &graph_def);
        
        m_session->Create(graph_def);
    }
    
    void Compute(OpKernelContext *context) override
    {
        // get input
        const Tensor& t_metainfo_i = context->input(0);
        //info
        const int num_gen = t_metainfo_i.flat<int>()(0);
        // start input
        const size_t start_input = 1;
        // W
        std::vector < Tensor >  W_list;
        // pupulation inputs
        for(int i=0; i != m_space_size; ++i)
        {
            W_list.push_back(context->input(start_input+i));
        }
        // pupulations
        std::vector < std::vector <Tensor> >  current_populations_list;
        // pupulation inputs
        for(int i=0; i != m_space_size; ++i)
        {
            const Tensor& population = context->input(start_input+i+m_space_size);
            current_populations_list.push_back(splitDim0(population));
        }
        //new pupulation
        std::vector < std::vector <Tensor> >  new_populations_list(m_space_size);
        // pupulation inputs
        for(int i=0; i != m_space_size; ++i)
        {
            new_populations_list[i].resize(current_populations_list[i].size());
        }
        //Size of population
        const size_t NP = current_populations_list[0].size();
        //Alloc first eval
        std::vector < double > current_eval_result(NP);
        //First eval
        for(int index = 0; index!=NP ;++index)
        {
            current_eval_result[index] = execute_evaluate(index,current_populations_list);
        }
        //loop
        for(int i=0;i!=num_gen;++i)
        {
            trialVectorsOp(context,W_list,current_populations_list,new_populations_list);
            //change pop
            for(int index = 0; index!=NP ;++index)
            {
                //Evaluation
                double new_eval = execute_evaluate(index,current_populations_list);
                //Choice
                if(new_eval < current_eval_result[index])
                {
                    for(int p=0; p!=current_populations_list.size(); ++p)
                    {
                        current_populations_list[p][index] = new_populations_list[p][index];
                    }
                    current_eval_result[index] = new_eval;
                }
            }
        }
        // Output populations
        for(int i=0; i != m_space_size; ++i)
        {
            Tensor* new_generation_tensor = nullptr;
            Tensor current_pop = concatDim0(current_populations_list[i]);
            OP_REQUIRES_OK(context, context->allocate_output(i, current_pop.shape(), &new_generation_tensor));
            (*new_generation_tensor) = current_pop;
        }
        // Output the last eval
        {
            //ptr
            Tensor* out_eval = nullptr;
            TensorShape out_shape({(int)current_eval_result.size()});
            //alloc
            OP_REQUIRES_OK(context, context->allocate_output(m_space_size,out_shape, &out_eval));
            //ref to data
            StringPiece to_data = (*out_eval).tensor_data();
            //copy
            std::memcpy(const_cast<char*>(to_data.data()), current_eval_result.data(),  current_eval_result.size()*sizeof(double));
        }
    }
    
protected:
    
    
    static std::vector<Tensor> splitDim0(const Tensor& tensor)
    {
        //output
        std::vector<Tensor> result;
        //num of dimations
        const int NUM_OF_D = tensor.shape().dims();
        //ref from data
        StringPiece from_data = tensor.tensor_data();
        //test
        if(NUM_OF_D)
        {
            //shape
            TensorShape shape;
            //type of shape
            if(NUM_OF_D>1)
                for(int i=1;i!=NUM_OF_D;++i) { shape.AddDim(tensor.dim_size(i)); }
            else
                shape.AddDim(1);
            //start offset
            int64 offset = 0;
            //copy
            for(int i=0;i!=tensor.dim_size(0); ++i)
            {
                //alloc
                result.emplace_back(tensor.dtype(), shape);
                Tensor* split = &result[result.size() - 1];
                //ref to data
                StringPiece to_data = split->tensor_data();
                //copy
                std::memcpy(const_cast<char*>(to_data.data()), from_data.data() + offset,  to_data.size());
                //offset
                offset += to_data.size();
            }
        }
        //return
        return result;
    }
    
    static Tensor concatDim0(const std::vector< Tensor >& list_tensor)
    {
        //base tensor
        const Tensor& tensor0 = list_tensor[0];
        const TensorShape& tensor0_shape = tensor0.shape();
        //new shape
        TensorShape output_shape; output_shape.AddDim((int)list_tensor.size());
        //add base dims
        for(int i=0;i!=tensor0_shape.dims();++i)
        { output_shape.AddDim(tensor0_shape.dim_size(i)); }
        //Alloc output shape
        Tensor out_tensor(tensor0.dtype(),output_shape);
        //start offset
        int64 offset = 0;
        //ref to data
        StringPiece to_data = out_tensor.tensor_data();
        //copy
        for(size_t i=0;i!=list_tensor.size();++i)
        {
            //ref to data
            StringPiece from_data = list_tensor[i].tensor_data();
            //copy
            std::memcpy(const_cast<char*>(to_data.data()) + offset, from_data.data(),  from_data.size());
            //offset
            offset += from_data.size();
        }
        
        return out_tensor;
    }
    
    //call
    double execute_evaluate(const int NP_i,
                            const std::vector < std::vector<Tensor> >& populations_list)
    {
        
        TensorList f_on_values;
        //create input
        TensorInputs input;
        //append
        for(size_t p=0; p!=populations_list.size(); ++p)
        {
            input.push_back({
                std::string("target_")+std::to_string((long long)p),
                populations_list[p][NP_i]
            });
        }
        //execute
        auto
        status= m_session->Run(
                               //input
                               input,
                               //function
                               NameList{ "evaluate:0" } ,
                               //one
                               NameList{ },
                               //output
                               &f_on_values
                               );
        
        //output error
        if(!status.ok()) std::cout << status.ToString() << std::endl;
        //results
        return f_on_values[0].flat<double>()(0);
    }
    //choiceOp
    bool choiceOp(const int NP_i,
                  const std::vector < std::vector<Tensor> >& cur_populations_list,
                  const std::vector < std::vector<Tensor> >& new_populations_list)
    {
        //results
        double reduce_pop = execute_evaluate(NP_i,cur_populations_list);
        double reduce_ind = execute_evaluate(NP_i,new_populations_list);
        //alloc output
        //copy
        if(reduce_pop < reduce_ind) return false;
        else                        return true;
    }
            
    double f_clamp(const double t) const
    {
        return std::min(std::max(t, m_f_min), m_f_max);
    }
    
    //trialVectorsOp
    void
    trialVectorsOp(OpKernelContext *context,
                   const std::vector < Tensor >&              W_list,
                   const std::vector < std::vector<Tensor> >& cur_populations_list,
                         std::vector < std::vector<Tensor> >& new_populations_list)
    {
        //for all populations
        for(size_t p=0; p!=cur_populations_list.size(); ++p)
        {
            //get values
            const auto W = W_list[p].flat<double>();
            // size of a population
            const int NP = cur_populations_list[p].size();
            // Generate new population
            int
            i_a = 0,
            i_b = 0,
            i_c = 0;
            //for all
            for (int index = 0; index < NP; ++index)
            {
                //ref to population
                const std::vector< Tensor >& population = cur_populations_list[p];
                //Num of dimations
                const int NUM_OF_D = population[index].shape().dims();
                //Compute flat dimension
                int D = NUM_OF_D ? 1 : 0;
                //compute D
                for(int i=0; i < NUM_OF_D; ++i)
                {
                    D *= population[index].shape().dim_size(i);
                }
                // alloc
                new_populations_list[p][index] = Tensor(DataType::DT_DOUBLE, population[index].shape());
                //ref new gen
                auto new_generation = new_populations_list[p][index].flat_inner_dims<double>();
                //do rand indices
                threeRandIndicesDiffFrom(NP, index, i_a, i_b, i_c);
                //random index
                int random_index = irand(D);
                //compute
                for (int elm = 0; elm < D; ++elm)
                {
                    if (random() < m_CR || random_index == elm)
                    {
                        const double a = population[i_a].flat<double>()(elm);
                        const double b = population[i_b].flat<double>()(elm);
                        const double c = population[i_c].flat<double>()(elm);
                        new_generation(elm) = f_clamp( (a-b) * W(elm) + c );
                    }
                    else
                    {
                        new_generation(elm) = population[index].flat_inner_dims<double>()(elm);
                    }
                }
            }
        }
    }
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

REGISTER_KERNEL_BUILDER(Name("DENN").Device(DEVICE_CPU), DENNOp);

