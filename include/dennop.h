
#pragma once
#include "config.h"
#include "de_info.h"
#include "denn_util.h"
#include "tensorflow_alias.h"
#include "population_generator.h"
#include <string>
#include <iostream>
#include <memory>
#include <cmath>
#include <algorithm>
#include <typeinfo>

namespace tensorflow
{

template< class value_t = double > 
class DENNOp : public OpKernel 
{
public:

    //num of threads (OpenMP)
    const int N_THREADS = 4;
    
    //init DENN from param
    explicit DENNOp(OpKernelConstruction *context) : OpKernel(context)
    {
        // Space size
        OP_REQUIRES_OK(context, context->GetAttr("space", &m_space_size));
        // Get names of eval inputs
        OP_REQUIRES_OK(context, context->GetAttr("f_inputs", &m_inputs_names));
        // Get dataset path
        OP_REQUIRES_OK(context, context->GetAttr("f_input_labels", &m_input_labels));
        // Get dataset path
        OP_REQUIRES_OK(context, context->GetAttr("f_input_features", &m_input_features));
        // Get name of eval function
        OP_REQUIRES_OK(context, context->GetAttr("f_name_execute_net", &m_name_execute_net));
        // Test size == sizeof(names)
        if( m_space_size != int(m_inputs_names.size()) )
        {
            context->CtxFailure({tensorflow::error::Code::ABORTED,"Attribute error: sizeof(inputs names) != sizeof(populations) "});
        }
        
        // Get factors
        ParserAttr(context, m_de_factors);
        
        // Get DE type
        ParserAttr(context, m_de_info);

        // Create session from graph 
        ParserAttr(context, m_session);

    }
    
    //star execution from python
    virtual void Compute(OpKernelContext *context) override
    {
        ////////////////////////////////////////////////////////////////////////////
        // get input info
        const Tensor& t_metainfo_i = context->input(0);
        //info 1: (NUM GEN)
        const int num_gen = t_metainfo_i.flat<int>()(0);
        //info 2; (COMPUTE FIRST VALUTATION OF POPULATION)
        const int calc_first_eval = t_metainfo_i.flat<int>()(1);
        ////////////////////////////////////////////////////////////////////////////
        // get input batch labels
        const Tensor& t_batch_labels = context->input(1);
        // get input batch data
        const Tensor& t_batch_features = context->input(2);
        ////////////////////////////////////////////////////////////////////////////
        // F and CR
        TensorList  current_population_F_CR;
        // get F
        current_population_F_CR.emplace_back( context->input(3) );
        // get CR
        current_population_F_CR.emplace_back( context->input(4) );
        //debug 
        MSG_DEBUG( "|F,CR|: " << current_population_F_CR.size() )
        MSG_DEBUG( "|F|: " << current_population_F_CR[0].shape().dim_size(0) )
        MSG_DEBUG( "|CR|: " << current_population_F_CR[1].shape().dim_size(0) )
        ////////////////////////////////////////////////////////////////////////////
        //get population first eval
        const Tensor& population_first_eval = context->input(5);
        // start input
        const size_t start_input = 6;
        ////////////////////////////////////////////////////////////////////////////
        // populations
        TensorListList  current_population_list;
        // populations inputs
        for(int i=0; i != m_space_size; ++i)
        {
            const Tensor& population = context->input(start_input+i);
            current_population_list.push_back(splitDim0(population));
        }
        //Test sizeof populations
        if NOT(TestPopulationSize(context,current_population_list)) return;

        ////////////////////////////////////////////////////////////////////////////
        //Temp of new gen of populations
        TensorList     new_population_F_CR;
        TensorListList new_population_list;
        //Alloc temp vector of populations
        AllocNewPopulation(current_population_F_CR,
                           current_population_list, 
                           new_population_F_CR,
                           new_population_list);
        ////////////////////////////////////////////////////////////////////////////
        //Alloc input 
        AllocCacheInputs(current_population_list);
        //Copy batch in input
        SetDatasetInCacheInputs(t_batch_labels,t_batch_features);

        ////////////////////////////////////////////////////////////////////////////
        //Tensor first evaluation of all ppopulations
        Tensor current_eval_result;
        // init evaluation
        DoFirstEvaluationIfRequired
        (
              context
            , calc_first_eval
            , current_population_list
            , population_first_eval 
            , current_eval_result
        );

        ////////////////////////////////////////////////////////////////////////////
        // Execute DE
        RunDe
        (
            // Input
              context
            , num_gen
            // Cache
            , new_population_F_CR
            , new_population_list
            // In/Out
            , current_eval_result
            , current_population_F_CR
            , current_population_list        
        );

        ////////////////////////////////////////////////////////////////////////////
        // Output counter
        int output_id=0;
        // Output F CR
        for(int i=0; i != int(current_population_F_CR.size()); ++i)
        {
            Tensor* output_tensor = nullptr;
            Tensor& current_f_or_cr = current_population_F_CR[i];
            OP_REQUIRES_OK(context, context->allocate_output(output_id++, current_f_or_cr.shape(), &output_tensor));
            (*output_tensor) = current_f_or_cr;
        }
        // Output populations
        for(int i=0; i != m_space_size; ++i)
        {
            Tensor* output_tensor = nullptr;
            Tensor current_pop = concatDim0(current_population_list[i]);
            OP_REQUIRES_OK(context, context->allocate_output(output_id++, current_pop.shape(), &output_tensor));
            (*output_tensor) = current_pop;
        }
        // Output the last eval
        {
            //ptr
            Tensor* out_eval = nullptr;
            //alloc
            OP_REQUIRES_OK(context, context->allocate_output(output_id++,current_eval_result.shape(), &out_eval));
            //copy
            (*out_eval) = current_eval_result;
        }
    }    

    /**
     * Start differential evolution
     * @param context
     * @param num_gen, number of generation
     * @param new_population_list, (input) cache memory of the last population generated 
     * @param current_population_list, (input/output) population
     * @param current_eval_result, (input/output) evaluation of population
     */
    virtual bool RunDe
    (
        OpKernelContext *context,
        const int num_gen,
        // Cache
        TensorList&       new_population_F_CR,
        TensorListList&   new_population_list,
        // In/Out
        Tensor&           current_eval_result,
        TensorList&       current_population_F_CR,
        TensorListList&   current_population_list
    )
    {
        //Get np 
        const int NP = current_population_list[0].size();
        //Pointer to memory
        auto ref_current_eval_result = current_eval_result.flat<value_t>();
        //loop
        for(int i=0;i!=num_gen;++i)
        {
            //Create new population
            PopulationGenerator< value_t >
            (
                context, 
                //params
                this->m_de_info,
                this->m_de_factors,
                //gen info
                i,
                num_gen,
                // population in
                NP,
                current_eval_result,
                current_population_F_CR,
                current_population_list,
                // population out
                new_population_F_CR,
                new_population_list
            );
            //Change old population (if required)
            for(int index = 0; index!=NP ;++index)
            {
                //Evaluation
                value_t new_eval = ExecuteEvaluateTrain(context, index, new_population_list);
                //Choice
                if(new_eval < ref_current_eval_result(index))
                {
                    //save F and CR 
                    for(int i = 0; i != int(current_population_F_CR.size()); ++i)
                    {
                        //get refs
                        auto ref_f_cr = current_population_F_CR[i].flat<value_t>();
                        auto new_f_cr = new_population_F_CR[i].flat<value_t>();
                        //current F and CR <- new F and CR
                        ref_f_cr(index) = new_f_cr(index);
                    }
                    //x <- y
                    for(size_t p=0; p!=current_population_list.size(); ++p)
                    {
                        current_population_list[p][index] = new_population_list[p][index];
                    }
                    //save eval
                    ref_current_eval_result(index) = new_eval;
                }
            }   
            #if 1
            SOCKET_DEBUG(
                //m_debug.write(i);
                m_debug.write(1);
                //process message
                while(this->m_debug.get_n_recv_mgs())
                {
                    MSG_DEBUG("+++ Read message")
                    /*
                    MSG_INT,
                    MSG_FLOAT,
                    MSG_DOUBLE,
                    MSG_STRING,
                    MSG_CLOSE_CONNECTION
                    */
                    //get message 
                    auto raw_msg = this->m_debug.pop_recv_msg();
                    //parser message
                    debug::socket_messages_server::message_decoder msg( raw_msg );
                    //execute task by type
                    switch(msg.get_type())
                    {
                        //exit case
                        case debug::socket_messages_server::MSG_CLOSE_CONNECTION:
                            //exit
                            return false;
                        break;
                        //not used cases
                        default:
                        break;
                    }
                }
            )
            #endif
        }
        //continue
        return true;
    }
 
    /**
     * Do evaluation if required 
     * @param force_to_eval, if true, eval all populations in anyway
     * @param current_population_list, (input) populations
     * @param population_first_eval, (input) last evaluation of population
     * @param current_eval_result, (output) new evaluation of population
     */
    virtual void DoFirstEvaluationIfRequired
    (
         OpKernelContext *context,
         const bool force_to_eval,
         const TensorListList& current_population_list,
         const Tensor& population_first_eval,
         Tensor& current_eval_result
    )
    {
        //Get np 
        const int NP = current_population_list[0].size();
        //Population already evaluated?
        if(  !(force_to_eval)
           && population_first_eval.shape().dims() == 1
           && population_first_eval.shape().dim_size(0) == NP)
        {
            //copy eval
            current_eval_result = population_first_eval;
        }
        //else execute evaluation
        else
        {
            //Alloc
            current_eval_result = Tensor(data_type<value_t>(), TensorShape({(int)NP}));
            //First eval
            for(int index = 0; index!=NP ;++index)
            {
                current_eval_result.flat<value_t>()(index) = ExecuteEvaluateTrain(context, index, current_population_list);
            }
        }

    }


    //execute evaluate train function (tensorflow function)   
    virtual value_t ExecuteEvaluateTrain
    (
        OpKernelContext* context,
        const int NP_i,
        const TensorListList& populations_list
    ) const
    {
        NameList function{
             m_name_execute_net //+":0" 
        };
        return ExecuteEvaluate(context, NP_i, populations_list, function);
    }

    //execute evaluate function (tensorflow function)
    virtual value_t ExecuteEvaluate
    (
        OpKernelContext* context,
        const int NP_i,
        const TensorListList& populations_list,
        const NameList& functions_list
    ) const
    {
        //Output
        TensorList f_on_values;
        //Set input
        if NOT(SetCacheInputs(populations_list, NP_i))
        {
            context->CtxFailure({tensorflow::error::Code::ABORTED,"Run evaluate: error to set inputs"});
            ASSERT_DEBUG( 0 )
            return 0.0;
        }
        //execute
        auto
        status= m_session->Run(//input
                               m_inputs_tensor_cache,
                               //function
                               functions_list,
                               //one
                               NameList{ },
                               //output
                               &f_on_values
                               );
        
        //output error
        if NOT(status.ok())
        {
            context->CtxFailure({tensorflow::error::Code::ABORTED,"Run evaluate: "+status.ToString()});
            ASSERT_DEBUG( 0 )
            return 0.0;
        }
        //results
        return f_on_values.size() ? f_on_values[0].flat<value_t>()(0) : 0.0;
    }


protected:
       
    /**
    * Alloc m_inputs_tensor_cache
    * @param populations_list, (input) populations
    */
    virtual bool AllocCacheInputs(const TensorListList& populations_list) const 
    {
        //resize
        m_inputs_tensor_cache.resize(populations_list.size()+2);
        //add all names
        for(size_t p=0; p!=populations_list.size(); ++p)
        {
            m_inputs_tensor_cache[p].first = m_inputs_names[p];
        }
        m_inputs_tensor_cache[m_inputs_tensor_cache.size()-2].first = m_input_features;
        m_inputs_tensor_cache[m_inputs_tensor_cache.size()-1].first = m_input_labels;
        return true;
    }

    /**
     * Set tensors of pupulation in m_inputs_tensor_cache
     * @param populations_list, (input) populations
     * @param NP_i, (input) population index
     */
    virtual bool SetCacheInputs
    (
        const TensorListList& populations_list,
        const int NP_i
    ) const
    {
        //test size
        if(m_inputs_tensor_cache.size() != (populations_list.size()+2)) return false;
        //add all Tensor
        for(size_t p=0; p!=populations_list.size(); ++p)
        {
            m_inputs_tensor_cache[p].second = populations_list[p][NP_i];
        }
        return true;
    }
    
    /**
    * Set dataset in m_inputs_tensor_cache
    * @param labels, tensor of labels
    * @param features, tensor of features
    */
    virtual bool SetDatasetInCacheInputs
    (
        const Tensor& labels, 
        const Tensor& features
    ) const
    {
        //test size
        if(m_inputs_tensor_cache.size() < 2) return false;
        //add dataset in input
        m_inputs_tensor_cache[m_inputs_tensor_cache.size()-2].second  = features;
        m_inputs_tensor_cache[m_inputs_tensor_cache.size()-1].second  = labels;
        //ok
        return true;
    }
    
    //Return X saved in cache inputs
    const Tensor& GetFeaturesInCacheInputs() const 
    {
        return m_inputs_tensor_cache[m_inputs_tensor_cache.size()-2].second;
    }
    
    //Return Y_ saved in cache inputs
    const Tensor& GetLabelsInCacheInputs() const 
    {
        return m_inputs_tensor_cache[m_inputs_tensor_cache.size()-1].second;
    }

protected:

    //session
    std::unique_ptr< Session > m_session;
    //de info
    DeInfo                m_de_info;
    DeFactors< value_t >  m_de_factors;
    // population variables
    int                        m_space_size{ 1 };
    //input evaluate
    std::string                m_name_execute_net;
    NameList                   m_inputs_names;
    mutable TensorInputs       m_inputs_tensor_cache;
    //batch inputs
    std::string m_input_labels;
    std::string m_input_features;
    //debug
    SOCKET_DEBUG(
       debug::socket_messages_server m_debug;
    )
            
};

}
