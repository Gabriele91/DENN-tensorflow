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
    class DENNOpAdaBoost : public OpKernel 
    {

    public:

        //init DENN from param
        explicit DENNOpAdaBoost(OpKernelConstruction *context) : OpKernel(context)
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
            OP_REQUIRES_OK(context, context->GetAttr("f_input_cross_entropy_c", &m_name_input_cross_entropy_c));
            OP_REQUIRES_OK(context, context->GetAttr("f_input_cross_entropy_y", &m_name_input_cross_entropy_y));
        }


        //star execution from python
        virtual void Compute(OpKernelContext *context) override
        {        
            ////////////////////////////////////////////////////////////////////////////
            // get input info
            const Tensor& t_metainfo_i = context->input(0);
            //info 1: (NUM GEN)
            const int num_gen = t_metainfo_i.flat<int>()(0);
            ////////////////////////////////////////////////////////////////////////////
            // get input batch labels
            const Tensor& t_batch_labels = context->input(1);
            // get input batch data
            const Tensor& t_batch_features = context->input(2);
            ////////////////////////////////////////////////////////////////////////////
            // F and CR
            TensorList current_population_F_CR;
            // get F
            current_population_F_CR.emplace_back( context->input(3) );
            // get CR
            current_population_F_CR.emplace_back( context->input(4) );
            ////////////////////////////////////////////////////////////////////////////
            // start input population [INFO + BATCH ROWS + BATCH LABEL + F + CR]
            const size_t start_input_population = 5;
            // populations
            TensorListList  current_population_list;        
            // populations inputs
            for(int i=0; i != m_space_size; ++i)
            {
                const Tensor& population = context->input(start_input_population+i);
                current_population_list.push_back(splitDim0(population));
            }
            //Test sizeof populations
            if NOT(TestPopulationSize(context,current_population_list)) return;
            ////////////////////////////////////////////////////////////////////////////
            // Take C at [ INFO + BATCH ROWs + BATCH LABEL + F + CR + population ]
            Tensor C  = context->input(start_input_population + m_space_size);
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
            //Get np 
            const int NP = current_population_list[0].size();
            //Tensor first evaluation of all populations
            Tensor current_eval_result(data_type<value_t>(),TensorShape({int64(NP)}));
            //fill all to 0
            fill<value_t>(current_eval_result,0);
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
                , C
            );
            ////////////////////////////////////////////////////////////////////////////
            // Output counter
            int output_id=0;
            // Output F CR
            for(int i=0; i != int(current_population_F_CR.size()); ++i)
            {
                Tensor* new_generation_tensor = nullptr;
                Tensor& current_f_or_cr = current_population_F_CR[i];
                OP_REQUIRES_OK(context, context->allocate_output(output_id++, current_f_or_cr.shape(), &new_generation_tensor));
                (*new_generation_tensor) = current_f_or_cr;
            }
            // Output populations
            for(int i=0; i != m_space_size; ++i)
            {
                Tensor* new_generation_tensor = nullptr;
                Tensor current_pop = concatDim0(current_population_list[i]);
                OP_REQUIRES_OK(context, context->allocate_output(output_id++, current_pop.shape(), &new_generation_tensor));
                (*new_generation_tensor) = current_pop;
            }
            //return C
            {
                Tensor* new_generation_tensor = nullptr;
                OP_REQUIRES_OK(context, context->allocate_output(output_id++,  C.shape(), &new_generation_tensor));
                (*new_generation_tensor) = C;
            }
        }

        /**
        * Start differential evolution
        * @param context
        * @param num_gen, number of generation
        * @param new_population_F_CR, (input) cache memory of F and CR of last population generated
        * @param new_population_list, (input) cache memory of last population generated 
        * @param current_eval_result, (input/output) evaluation of population
        * @param current_population_F_CR, (input/output) F and CR of population
        * @param current_population_list, (input/output) population
        */
        virtual bool RunDe
        (
            OpKernelContext*  context,
            const int         num_gen,
            // Cache
            TensorList&       new_population_F_CR,
            TensorListList&   new_population_list,
            // In/Out
            Tensor&           current_eval_result,
            TensorList&       current_population_F_CR,
            TensorListList&   current_population_list,
            // ADA
            Tensor&           C
        )
        {
            //Get np 
            const int NP = current_population_list[0].size();
            // temp info about pop eval
            TensorList pop_EC(NP), pop_Y(NP);
            // execute networks of population 
            if NOT( ComputePopYAndEC(context, current_population_list, pop_Y, pop_EC))
            {
                //exit, wrong 
                return false;
            }
            //Pointer to memory of evals
            auto ref_current_eval_result = current_eval_result.flat<value_t>();
            //evaluete all population 
            for(int NP_i = 0; NP_i != NP; ++NP_i)
            {
                ref_current_eval_result(NP_i) = ExecuteEvaluateAdaBoost(  context, pop_Y[NP_i], C );
            }
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
                    //get y and ec
                    Tensor y;
                    Tensor ec;
                    if NOT(ComputeYAndEC(context, index, new_population_list, y, ec))
                    {
                        //exit, wrong 
                        return false;
                    }
                    //execute cross
                    value_t new_eval = ExecuteEvaluateAdaBoost(context, y, C);
                    //Choice
                    if(new_eval < ref_current_eval_result(index))
                    {
                        //save a individual (composiction of W, B list)
                        for(size_t p=0; p!=current_population_list.size(); ++p)
                        {
                            current_population_list[p][index] = new_population_list[p][index];
                        }
                        //save F and CR 
                        for(int i = 0; i != int(current_population_F_CR.size()); ++i)
                        {
                            //get refs
                            auto ref_f_cr = current_population_F_CR[i].flat<value_t>();
                            auto new_f_cr = new_population_F_CR[i].flat<value_t>();
                            //current F and CR <- new F and CR
                            ref_f_cr(index) = new_f_cr(index);
                        }
                        //save Y
                        pop_Y[index] = y;
                        //save EC 
                        pop_EC[index] = ec; 
                        //save Eval
                        ref_current_eval_result(index) = new_eval;
                    }
                }   
                //////////////////////////////////////////////////////////////
                // C(gen+1) = (1-alpha) * C(gen) + alpha * (Ni / Np)
                //////////////////////////////////////////////////////////////
                //alloc counter of all ec
                Tensor ec_counter(data_type<int>(), C.shape());
                //all to 0
                fill<int>(ec_counter,0);
                //get pointer
                auto raw_EC_counter = ec_counter.template flat<int>();
                //for all errors
                for(size_t i = 0; i!=pop_EC.size(); ++i)
                {
                    //get ec
                    auto raw_ec_i = pop_EC[i].template flat<bool>();
                    //for all ec
                    for(int j=0; j!=ec_counter.dim_size(0); ++j)
                    {
                        raw_EC_counter(j) += int(!raw_ec_i(j));
                    }
                }
                //get values
                auto raw_C = C.flat<value_t>();
                //new c
                for(int i = 0; i!= C.shape().dim_size(0); ++i)
                {
                    value_t  op0 = (value_t(1.0)-m_alpha) * raw_C(i);
                    value_t  op1 = m_alpha * (value_t(raw_EC_counter(i)) / NP);
                    raw_C(i) = op0 + op1;
                }

                #if 1
                SOCKET_DEBUG(
                    //m_debug.write(i);
                    this->m_debug.write(1);
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
        * Compute NN if req
        * @param populations_list, (input) populations
        */
        virtual bool ComputePopYAndEC
        (
            OpKernelContext *context,
            const TensorListList& populations_list,
            TensorList& pop_Y,
            TensorList& EC
        )
        {
            //Get np 
            const int NP = populations_list[0].size();
            //output 
            bool output = true;
            //execute
            for(int NP_i=0; NP_i!=NP; ++NP_i)
                output &= ComputeYAndEC(context, NP_i, populations_list, pop_Y[NP_i], EC[NP_i]);
            //return 
            return output;
        }

        /**
        * Compute NN if req
        * @param NP_i, (input) populations
        * @param populations_list, (input) populations
        * @param population y vectors, (output)
        * @param population ec vectors, (output)
        */
        virtual bool ComputeYAndEC
        (
            OpKernelContext *context,
            //input
            const size_t NP_i,
            const TensorListList& populations_list,
            //output 
            TensorList& out_list_y,
            TensorList& out_list_ec            
        )
        {
            return ComputeYAndEC(context, NP_i, populations_list, out_list_y[NP_i], out_list_ec[NP_i]);
        }

        /**
        * Compute Y and EC of a individual
        * @param NP_i, (input) id individual
        * @param populations_list, (input) population
        * @param y vector, (output)
        * @param ec vector, (output)
        */
        virtual bool ComputeYAndEC
        (
            OpKernelContext *context,
            //input
            const size_t NP_i,
            const TensorListList& populations_list,
            //output 
            Tensor& out_y,
            Tensor& out_ec            
        )
        {
            TensorList 
              output_y
            , output_correct_values;
            //Set input
            if NOT(this->SetCacheInputs(populations_list, NP_i))
            {
                context->CtxFailure({tensorflow::error::Code::ABORTED,"Run evaluate: error to set inputs"});
                ASSERT_DEBUG_MSG(0, "Run evaluate: error to set inputs");
                return false;
            }
            //execute network
            {
                auto
                status= m_session->Run
                (   //input
                    m_inputs_tensor_cache,
                    //function
                    NameList
                    {
                        this->m_name_execute_net
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
                    ASSERT_DEBUG_MSG(0, "Run execute network: " << status.ToString());
                    return false;
                }
                //return Y
                out_y = output_y[0];
            }
            //Run Diff
            {
                auto
                status= this->m_session->Run(
                                        //input
                                        TensorInputs
                                        {
                                            { this->m_name_input_correct_predition, output_y[0] }, // y
                                            { this->m_input_labels,                 m_labels    }  // y_
                                        },
                                        //function
                                        NameList{
                                            this->m_name_correct_predition
                                        },
                                        //one
                                        NameList{ },
                                        //output
                                        &output_correct_values
                                    );
                //output error
                if NOT(status.ok())
                {
                    context->CtxFailure({tensorflow::error::Code::ABORTED,"Run correct predition diff: "+status.ToString()});
                    ASSERT_DEBUG_MSG(0, "Run correct predition diff: " << status.ToString());
                    return false;
                }
                //return Y
                out_ec = output_correct_values[0];
            }
            //ok 
            return true;
        }

        //execute evaluate function (tensorflow function->cross entropy)
        virtual value_t ExecuteEvaluateAdaBoost
        (
            OpKernelContext* context,
            //input
            const Tensor& nn_Y,
            const Tensor& C
        )
        {
            #if 0
            MSG_DEBUG("pop_Y_i dim0: "  << nn_Y.shape().dim_size(0))
            MSG_DEBUG("C dim0: "  << C.shape().dim_size(0))
            MSG_DEBUG("m_labels dim0: "  << m_labels.shape().dim_size(0))
            #endif
            //Output
            TensorList  cross_value;
            //compure cross (Y*C)
            {   
                auto
                status= this->m_session->Run( //input
                                        TensorInputs
                                        {
                                            { this->m_name_input_cross_entropy_c ,  C        },  // c
                                            { this->m_name_input_cross_entropy_y ,  nn_Y     },  // y
                                            { this->m_input_labels,                 m_labels }   // y_
                                        },
                                        //function
                                        NameList
                                        {
                                            this->m_name_cross_entropy    
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
                    ASSERT_DEBUG_MSG( 0, "Run cross eval: " << status.ToString());
                    return value_t(-1);
                }
            }
            //get cross validation
            value_t cross_res = cross_value.size() ? cross_value[0].template flat<value_t>()(0) : -1.0;
            //results
            return cross_res;
        }

        /**
        * Alloc m_inputs_tensor_cache
        * @param populations_list, (input) populations
        */
        virtual bool AllocCacheInputs(const TensorListList& populations_list) const 
        {
            //resize
            m_inputs_tensor_cache.resize(populations_list.size()+1);
            //add all names
            for(size_t p=0; p!=populations_list.size(); ++p)
            {
                m_inputs_tensor_cache[p].first = this->m_inputs_names[p];
            }
            m_inputs_tensor_cache[m_inputs_tensor_cache.size()-1].first = this->m_input_features;
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
            if(m_inputs_tensor_cache.size() != (populations_list.size()+1)) return false;
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
            m_inputs_tensor_cache[m_inputs_tensor_cache.size()-1].second  = features;
            //add ref to tensor
            m_labels = labels;
            //ok
            return true;
        }

    protected:

        //session
        std::unique_ptr< Session > m_session;

        //de info
        DeInfo                m_de_info;
        DeFactors< value_t >  m_de_factors;

        // population variables
        int                        m_space_size{ 1 };

        //batch inputs
        std::string m_input_labels;
        std::string m_input_features;

        //execute network names and inputs
        std::string                m_name_execute_net;
        NameList                   m_inputs_names;
        mutable TensorInputs       m_inputs_tensor_cache;

        //names of functions
        std::string m_name_input_correct_predition; //name of : Y
        std::string m_name_correct_predition;       //F(Y,Y_) -> C_ : where Y_ is labels and C_ is a vector of booleans

        std::string m_name_input_cross_entropy_c;  //name of Y
        std::string m_name_input_cross_entropy_y;  //name of C
        std::string m_name_cross_entropy;          //name of : F(Y*C) -> cross(Y) 

        //ada boost factor 
        value_t      m_alpha;

        //:D
        mutable Tensor m_labels;

        //debug
        SOCKET_DEBUG(
            debug::socket_messages_server m_debug;
        )

    };
}
