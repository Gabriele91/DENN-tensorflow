#pragma once 

#define DENN_USE_SOCKET_DEBUG 
#define DENN_USE_TRAINING

#include "config.h"
#include "dennop.h"
#include "dataset_loader.h"

namespace tensorflow
{
    class DENNOpTraining : public DENNOp
    {
    public:

        //init DENN from param
        explicit DENNOpTraining(OpKernelConstruction *context) : DENNOp(context)
        {
            // Get dataset path
            OP_REQUIRES_OK(context, context->GetAttr("f_input_labels", &m_input_labels));
            // Get dataset path
            OP_REQUIRES_OK(context, context->GetAttr("f_input_features", &m_input_features));
            // Get validation function
            OP_REQUIRES_OK(context, context->GetAttr("f_name_valida", &m_name_valida));
            // Get dataset path
            OP_REQUIRES_OK(context, context->GetAttr("dataset", &m_dataset_path));
            // Try to openfile
            if(!m_dataset.open(m_dataset_path))
            {
                context->CtxFailure({tensorflow::error::Code::ABORTED,"Attribute error: can't open dataset' "});
            }
        }

        //star execution from python
        virtual void Compute(OpKernelContext *context) override
        {
            // get input
            const Tensor& t_metainfo_i = context->input(0);
            //info 1: (NUM GEN)
            const int tot_gen = t_metainfo_i.flat<int>()(0);
            //info 2: (STEP GEN)
            const int sup_gen = t_metainfo_i.flat<int>()(1);
            //info 3; (COMPUTE FIRST VALUTATION OF POPULATION)
            const int calc_first_eval = t_metainfo_i.flat<int>()(2);
            //get population first eval
            const Tensor& population_first_eval = context->input(1);
            // start input
            const size_t start_input = 3;
            //super gen
            const int n_sub_gen = tot_gen / sup_gen;
            ////////////////////////////////////////////////////////////////////////////
            // W
            std::vector < Tensor >  W_list;
            // pupulation inputs
            for(int i=0; i != m_space_size; ++i)
            {
                W_list.push_back(context->input(start_input+i));
            }
            
            ////////////////////////////////////////////////////////////////////////////
            // populations
            std::vector < std::vector <Tensor> >  current_populations_list;
            // populations inputs
            for(int i=0; i != m_space_size; ++i)
            {
                const Tensor& population = context->input(start_input+i+m_space_size);
                current_populations_list.push_back(splitDim0(population));
            }
            //Test sizeof populations
            if(!TestPopulationSize(context,current_populations_list)) return;
            
            ////////////////////////////////////////////////////////////////////////////
            //Temp of new gen of populations
            std::vector < std::vector <Tensor> > new_populations_list;
            //Alloc temp vector of populations
            GenCachePopulation(current_populations_list,new_populations_list);
            
            ////////////////////////////////////////////////////////////////////////////
            //Tensor first evaluation of all ppopulations
            Tensor current_eval_result;
            // init evaluation
            DoFirstEvaluationIfRequired
            (
               context
             , calc_first_eval
             , current_populations_list
             , population_first_eval 
             , current_eval_result
            );
            
            ////////////////////////////////////////////////////////////////////////////
            // START STREAM
            m_dataset.start_read_bach();
            
            ////////////////////////////////////////////////////////////////////////////
            // Execute DE
            for(size_t i_sub_gen = 0; i_sub_gen != n_sub_gen; ++i_sub_gen)
            {
                //load bach
                if( !m_dataset.read_bach(m_bach) )
                {
                    context->CtxFailure({
                        tensorflow::error::Code::ABORTED,
                        "Error stream dataset: can't read ["+std::to_string(m_dataset.get_last_bach_info().m_bach_id)+"] bach' "
                    });
                }
                //execute
                RunDe
                (
                 // Input
                   context
                 , sub_gen
                 , W_list
                 // Cache
                 , new_populations_list
                 // In/Out
                 , current_populations_list
                 , current_eval_result
                );
                
                SOCKET_DEBUG(
                    m_debug.write("Stage["+std::string(i_sub_gen)+"] complete");
                )
            }
            ////////////////////////////////////////////////////////////////////////////
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
                //alloc
                OP_REQUIRES_OK(context, context->allocate_output(m_space_size,current_eval_result.shape(), &out_eval));
                //copy
                (*out_eval) = current_eval_result;
            }

        }
    
        //execute evaluate function (tensorflow function)
        virtual double ExecuteEvaluate
        (
            OpKernelContext* context,
            const int NP_i,
            const std::vector < std::vector<Tensor> >& populations_list
        ) const override
        {
            
            TensorList f_on_values;
            //create input
            TensorInputs input;
            //inputs bach
            input.push_bach({
                m_input_labels,
                m_bach.m_labels
            });
            input.push_bach({
                m_input_features,
                m_bach.m_features
            });
            //append
            for(size_t p=0; p!=populations_list.size(); ++p)
            {
                input.push_back({
                    m_inputs[p],
                    populations_list[p][NP_i]
                });
            }
            //execute
            auto
            status= m_session->Run(//input
                                   input,
                                   //function
                                   NameList{ m_name_train+":0" } ,
                                   //one
                                   NameList{ },
                                   //output
                                   &f_on_values
                                   );
            
            //output error
            if(!status.ok())
            {
                context->CtxFailure({tensorflow::error::Code::ABORTED,"Run evaluate: "+status.ToString()});
            }
            //results
            return f_on_values[0].flat<double>()(0);
        }
        
    protected:

        //dataset
        DataSetLoader< io_wrapper::zlib_file<> > m_dataset;
        //Bach
        DataSetRaw m_bach;
        //dataset path
        std::string m_dataset_path;
        //validation function
        std::string m_name_valida;
        //bach inputs
        std::string m_input_labels;
        std::string m_input_features;

    };
};
