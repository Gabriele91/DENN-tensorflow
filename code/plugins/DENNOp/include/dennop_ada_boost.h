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
            // Get name of function of test execution
            OP_REQUIRES_OK(context, context->GetAttr("f_input_correct_predition", &m_name_input_correct_predition));
            OP_REQUIRES_OK(context, context->GetAttr("f_correct_predition", &m_name_correct_predition));
            // Get name of execute function
            OP_REQUIRES_OK(context, context->GetAttr("f_cross_entropy", &m_name_cross_entropy));
            OP_REQUIRES_OK(context, context->GetAttr("f_input_cross_entropy", &m_name_input_cross_entropy));
        }
 
        //execute evaluate train function (tensorflow function)   
        virtual value_t ExecuteEvaluateTrain
        (
            OpKernelContext* context,
            const int NP_i,
            const std::vector < std::vector<Tensor> >& populations_list
        ) const
        {
            NameList function
            {
                  m_name_execute_net
                , m_name_correct_predition
                , m_name_cross_entropy    
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
        ) const
        {
            //Output
            TensorList 
              output_y, 
            , output_correct_values
            , cross_value;
            //Set input
            if NOT(SetCacheInputs(populations_list, NP_i))
            {
                context->CtxFailure({tensorflow::error::Code::ABORTED,"Run evaluate: error to set inputs"});
            }

            //execute network
            {
                auto
                status= m_session->Run(//input
                                        m_inputs_tensor_cache,
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
                status= m_session->Run(//input
                                        TensorInput
                                        {
                                            { m_input_labels, output_y },                 // y
                                            { m_input_labels, GetLabelsInCacheInputs() }  // y_
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
            
            //compute N_i of C(gen+1) = C(gen) + delta * (Ni / Np) 
            {
                //
                auto& y_correct     = output_correct_values[0];
                auto& raw_y_correct = y_correct.flat<bool>();
                auto& raw_n         = m_C_Counter.flat<int>();
                //
                for(size_t i = 0; i != y_correct.shape().dim_size(i); ++i)
                {
                    if NOT(raw_y_correct[i]) ++raw_n[i];
                }
            }

            //Compute Y*C
            {
                //get Y
                auto& y     = output_y[0];
                auto& raw_y = y.flat<value_t>();
                auto& raw_c = m_C.flat<value_t>();
                auto& raw_y_= GetLabelsInCacheInputs().flat<value_t>();
                //for all values
                for(size_t i = 0; i != y.shape().dim_size(i); ++i) raw_y(i) *= raw_c(i);
            }

            //compure cross (Y*C)
            {   
                auto
                status= m_session->Run( //input
                                        TensorInput
                                        {
                                            { m_name_input_cross_entropy ,  y  }
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
            return cross_value.size() ? cross_value[0].flat<value_t>()(0) : 0.0;
        }

    protected:

        //names of functions
        std::string m_name_input_correct_predition; //name of : Y
        std::string m_name_correct_predition;       //F(Y,Y_) -> C_ : where Y_ is labels and C_ is a vector of booleans

        std::string m_name_input_cross_entropy     //name of Y*C
        std::string m_name_cross_entropy;          //name of : F(Y*C) -> cross(Y) 
        //ada boost factor 
        Tensor m_C;
        Tensor m_C_Counter;

    }