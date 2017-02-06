#pragma once 
#define DENN_USE_SOCKET_DEBUG 
#include "config.h"
#include "dennop.h"
#include "dataset_loader.h"


namespace tensorflow
{
    template< class value_t = double > 
    class DENNOpAdaBoost : public DENNOp< value_t >
    {

        using DENNOp_t = DENNOp< value_t >;
        //types
        using NameList      =  typename DENNOp_t::NameList;
        using TensorList    =  typename DENNOp_t::TensorList;
        using TensorInput   =  typename DENNOp_t::TensorInput;
        using TensorInputs  =  typename DENNOp_t::TensorInputs;


    public:

        //init DENN from param
        explicit DENNOpAdaBoost(OpKernelConstruction *context) : DENNOp< value_t >(context)
        {
            //get alpha in float
            float alpha = 0.5f;
            OP_REQUIRES_OK(context, context->GetAttr("ada_boost_alpha", &alpha));
            //cast alpha to value_t alpha 
            m_alpha = value_t(alpha);
            // Get name of function of test execution
            OP_REQUIRES_OK(context, context->GetAttr("f_input_correct_predition", &m_name_input_correct_predition));
            OP_REQUIRES_OK(context, context->GetAttr("f_correct_predition", &m_name_correct_predition));
            // Get name of execute function
            OP_REQUIRES_OK(context, context->GetAttr("f_cross_entropy", &m_name_cross_entropy));
            OP_REQUIRES_OK(context, context->GetAttr("f_input_cross_entropy", &m_name_input_cross_entropy));
        }


        //star execution from python
        virtual void Compute(OpKernelContext *context) override
        {        
            // start input
            const size_t start_input = 4;
            // Take C [ start input + W + population ]
            m_C = context->input(start_input + this->m_space_size * 2);
            // Alloc C Counter
            m_C_counter = Tensor(data_type<int>(), m_C.shape());
            //compute DENN
            DENNOp_t::Compute(context);
            //return C
            Tensor* new_generation_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(this->m_space_size+1, m_C.shape(), &new_generation_tensor));
            (*new_generation_tensor) = m_C;
        }

        /**
        * Start differential evolution
        * @param context
        * @param num_gen, number of generation
        * @param W_list, DE weights
        * @param new_populations_list, (input) cache memory of the last population generated 
        * @param current_populations_list, (input/output) population
        * @param current_eval_result, (input/output) evaluation of population
        */
        virtual void RunDe
        (
            OpKernelContext *context,
            const int num_gen,
            const std::vector < Tensor >& W_list,
            std::vector < std::vector <Tensor> >& new_populations_list,
            std::vector < std::vector <Tensor> >& current_populations_list,
            Tensor& current_eval_result
        )
        {
            //Get np 
            const int NP = current_populations_list[0].size();
            //Pointer to memory
            auto ref_current_eval_result = current_eval_result.flat<value_t>();
            //loop
            for(int i=0;i!=num_gen;++i)
            {
                //Create new population
                switch(this->m_pert_vector)
                {
                    case DENNOp_t::PV_RANDOM: this->RandTrialVectorsOp(context, NP,W_list,current_populations_list,new_populations_list); break;
                    case DENNOp_t::PV_BEST: this->BestTrialVectorsOp(context, NP,W_list,current_eval_result,current_populations_list,new_populations_list); break;
                    default: return /* FAILED */;
                }
                //Change old population (if required)
                for(int index = 0; index!=NP ;++index)
                {
                    //Evaluation
                    value_t new_eval = ExecuteEvaluateTrainAda(context, index, new_populations_list);
                    //Choice
                    if(new_eval < ref_current_eval_result(index))
                    {
                        for(int p=0; p!=current_populations_list.size(); ++p)
                        {
                            current_populations_list[p][index] = new_populations_list[p][index];
                        }
                        ref_current_eval_result(index) = new_eval;
                    }
                }   
                #if 0
                SOCKET_DEBUG(
                             m_debug.write(i);
                )
                #endif
                //////////////////////////////////////////////////////////////
                // TO DO: Update C and to 0 C_counter
                // C(gen+1) = (1-alpha) * C(gen) + alpha * (Ni / Np)
                //////////////////////////////////////////////////////////////
                //get values
                auto raw_C         = m_C.flat<float>();
                auto raw_C_counter = m_C_counter.flat<float>();
                //
                for(size_t i = 0; i!=m_C.shape().dim_size(0); ++i)
                {
                    raw_C(i) = (value_t(1.0)-m_alpha) * raw_C(i) + m_alpha * (value_t(raw_C_counter(i)) / NP);
                }
            }
        }
 
    
        //execute evaluate train function (tensorflow function)   
        virtual value_t ExecuteEvaluateTrainAda
        (
            OpKernelContext* context,
            const int NP_i,
            const std::vector < std::vector<Tensor> >& populations_list
        ) 
        {
            NameList function
            {
                  this->m_name_execute_net
                , this->m_name_correct_predition
                , this->m_name_cross_entropy    
            };
            return ExecuteEvaluateAdaBoost(context, NP_i, populations_list, function);
        }

        //execute evaluate function (tensorflow function)
        virtual value_t ExecuteEvaluateAdaBoost
        (
            OpKernelContext* context,
            const int NP_i,
            const std::vector < std::vector<Tensor> >& populations_list,
            const NameList& functions_list
        )
        {
            //Output
            TensorList 
              output_y
            , output_correct_values
            , cross_value;
            //Set input
            if NOT(this->SetCacheInputs(populations_list, NP_i))
            {
                context->CtxFailure({tensorflow::error::Code::ABORTED,"Run evaluate: error to set inputs"});
            }

            //execute network
            {
                auto
                status= this->m_session->Run(//input
                                        this->m_inputs_tensor_cache,
                                        //function
                                        NameList{
                                            functions_list[0] //m_name_execute_net
                                        },
                                        //one
                                        NameList{ },
                                        //output
                                        &output_y
                                    );
                //output error
                if NOT(status.ok())
                {
                    context->CtxFailure({tensorflow::error::Code::ABORTED,"Run execute network: "+status.ToString()});
                }
            }

            //execute m_name_correct_predition
            {
                auto
                status= this->m_session->Run(//input
                                        TensorInputs
                                        {
                                            { this->m_input_labels, output_y[0] },                    // y
                                            { this->m_input_labels, this->GetLabelsInCacheInputs() }  // y_
                                        },
                                        //function
                                        NameList{
                                            functions_list[1] //m_name_correct_predition
                                        },
                                        //one
                                        NameList{ },
                                        //output
                                        &output_correct_values
                                    );
                //output error
                if NOT(status.ok())
                {
                    context->CtxFailure({tensorflow::error::Code::ABORTED,"Run correct predition: "+status.ToString()});
                }
            }
            
            //compute N_i of C(gen+1) = (1-alpha) * C(gen) + alpha * (Ni / Np)
            {
                //
                const Tensor& y_correct = output_correct_values[0];
                auto raw_y_correct = y_correct.flat<bool>();
                auto raw_C_counter = m_C_counter.flat<int>();
                //
                for(size_t i = 0; i != y_correct.shape().dim_size(0); ++i)
                {
                    if NOT(raw_y_correct(i)) ++raw_C_counter(i);
                }
            }

            //Compute Y*C
            {
                //get Y
                auto& curr_y = output_y[0];
                auto raw_y  = curr_y.template flat<value_t>();
                auto raw_c  = m_C.template flat<value_t>();
                auto raw_y_ = this->GetLabelsInCacheInputs().template flat<value_t>();
                //for all values
                for(size_t i = 0; i != curr_y.shape().dim_size(0); ++i)
                {
                    raw_y(i) *= raw_c(i);
                }
            }

            //compure cross (Y*C)
            {   
                auto
                status= this->m_session->Run( //input
                                        TensorInputs
                                        {
                                            { this->m_name_input_cross_entropy ,  output_y[0]  }
                                        },
                                        //function
                                        NameList{
                                            functions_list[2] //m_name_cross_entropy
                                        },
                                        //one
                                        NameList{ },
                                        //output
                                        &cross_value
                                    );

                //output error
                if NOT(status.ok())
                {
                    context->CtxFailure({tensorflow::error::Code::ABORTED,"Run cross eval: "+status.ToString()});
                }
            }
         
            //results
            return cross_value.size() ? cross_value[0].template flat<value_t>()(0) : 0.0;
        }

    protected:

        //names of functions
        std::string m_name_input_correct_predition; //name of : Y
        std::string m_name_correct_predition;       //F(Y,Y_) -> C_ : where Y_ is labels and C_ is a vector of booleans

        std::string m_name_input_cross_entropy;    //name of Y*C
        std::string m_name_cross_entropy;          //name of : F(Y*C) -> cross(Y) 
        //ada boost factor 
        value_t m_alpha;
        Tensor  m_C;
        Tensor  m_C_counter;

    };
}
