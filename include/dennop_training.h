#pragma once 
#define DENN_USE_SOCKET_DEBUG 
#include "config.h"
#include "dennop.h"
#include "dataset_loader.h"
#include "training_util.h"

namespace tensorflow
{
    template< class value_t = double > 
    class DENNOpTraining : public DENNOp< value_t >
    {

        using DENNOp_t = DENNOp< value_t >;

    public:

        //init DENN from param
        explicit DENNOpTraining(OpKernelConstruction *context) : DENNOp< value_t >(context)
        {
            // Get validation function
            OP_REQUIRES_OK(context, context->GetAttr("f_name_validation", &m_name_validation));
            // Get test function
            OP_REQUIRES_OK(context, context->GetAttr("f_name_test", &m_name_test));
            // Get dataset path
            OP_REQUIRES_OK(context, context->GetAttr("dataset", &m_dataset_path));
            // Get reset
            std::string reset_type{"none"};
            float       reset_factor{100.0};
            int         reset_counter{0};
            NameList    reset_rand_pop;
            OP_REQUIRES_OK(context, context->GetAttr("reset_type", &reset_type));
            OP_REQUIRES_OK(context, context->GetAttr("reset_fector", &reset_factor));
            OP_REQUIRES_OK(context, context->GetAttr("reset_counter", &reset_counter));
            OP_REQUIRES_OK(context, context->GetAttr("reset_rand_pop", &reset_rand_pop));
            //init 
            new (&m_reset) DEReset<value_t>
            (
                  reset_type != "none"
                , (value_t)reset_factor
                , reset_counter
                , reset_rand_pop
            );
            // Try to openfile
            if NOT(m_dataset.open(m_dataset_path))
            {
                context->CtxFailure({tensorflow::error::Code::ABORTED,"Attribute error: can't open dataset' "});
            }
            //Load dataset
            m_dataset.read_test(m_test);
            m_dataset.read_validation(m_validation);
        }

        //star execution from python
        virtual void Compute(OpKernelContext *context) override
        {
            ////////////////////////////////////////////////////////////////////////////
            // get input
            const Tensor& t_metainfo_i = context->input(0);
            //info 1: (NUM GEN)
            const int tot_gen = t_metainfo_i.flat<int>()(0);
            //info 2: (STEP GEN)
            const int sub_gen = t_metainfo_i.flat<int>()(1);
            //info 3; (COMPUTE FIRST VALUTATION OF POPULATION)
            const int calc_first_eval = t_metainfo_i.flat<int>()(2);
            ////////////////////////////////////////////////////////////////////////////
            //get population first eval
            const Tensor& population_first_eval = context->input(1);
            ////////////////////////////////////////////////////////////////////////////
            // start input
            const size_t start_input = 2;
            //super gen
            const int n_sub_gen = tot_gen / sub_gen;
            ////////////////////////////////////////////////////////////////////////////
            // populations
            TensorListList  current_populations_list;
            // populations inputs
            for(int i=0; i != this->m_space_size; ++i)
            {
                const Tensor& population = context->input(start_input+i);
                current_populations_list.push_back(splitDim0(population));
            }
            //Test sizeof populations
            if NOT(DENNOp_t::TestPopulationSize(context,current_populations_list)) return;
            
            ////////////////////////////////////////////////////////////////////////////
            //Temp of new gen of populations
            TensorListList new_populations_list;
            //Alloc temp vector of populations
            DENNOp_t::GenCachePopulation(current_populations_list,new_populations_list);            
            
            ////////////////////////////////////////////////////////////////////////////
            //Alloc input 
            DENNOp_t::AllocCacheInputs(current_populations_list);

            ////////////////////////////////////////////////////////////////////////////
            // START STREAM
            m_dataset.start_read_bach();
            // Load first bach
            if NOT(LoadNextBach(context)) return ;//false;

            ////////////////////////////////////////////////////////////////////////////
            // Tensor of first evaluation of all populations
            Tensor current_eval_result;
            // init evaluation
            DENNOp_t::DoFirstEvaluationIfRequired
            (
               context
             , calc_first_eval
             , current_populations_list
             , population_first_eval 
             , current_eval_result
            );
            ////////////////////////////////////////////////////////////////////////////
            CacheBest< value_t > best;
            std::vector< value_t > list_eval_of_best;
            std::vector< value_t > list_eval_of_best_of_best;
            //Get np 
            const int NP = current_populations_list[0].size();
            ////////////////////////////////////////////////////////////////////////////
            // Execute DE
            for
            (
                //gen counter
                size_t i_sub_gen = 0;          
                //exit case
                i_sub_gen != n_sub_gen;    
                //next    
                ++i_sub_gen, 
                LoadNextBach(context) 
            )
            {
                //execute
                DENNOp_t::RunDe
                (
                 // Input
                   context
                 , sub_gen
                 // Cache
                 , new_populations_list
                 // In/Out
                 , current_populations_list
                 , current_eval_result
                );

                //Execute validation test 
                for(int individual_id = 0; individual_id != NP; ++individual_id)
                {         
                    //Change input 
                    SetValidationDataInCacheInputs();
                    //execute evaluation
                    value_t eval = ExecuteEvaluateValidation(context, individual_id, current_populations_list);
                    //add 
                    best.test_best(eval, individual_id, current_populations_list);
                }
                
                //find best
                int cur_best_id;
                value_t cur_best_eval;
                FindBest(context, current_populations_list,cur_best_id,cur_best_eval);
                
                //test 
                value_t cur_test_eval = ExecuteEvaluateTest(context, cur_best_id, current_populations_list);

                //update best 
                best.test_best(cur_best_eval,cur_best_id,current_populations_list);

                //add into vector
                list_eval_of_best.push_back(cur_test_eval);
                list_eval_of_best_of_best.push_back(best.m_eval);
                
                //Execute reset 
                CheckReset(context,best, current_populations_list);


                SOCKET_DEBUG(
                    //process message
                    while(this->m_debug.get_n_recv_mgs())
                    {
                        MSG_DEBUG("+++ Read message")
                        /*
                        msg types
                        {
                            MSG_INT,
                            MSG_FLOAT,
                            MSG_DOUBLE,
                            MSG_STRING,
                            MSG_CLOSE_CONNECTION
                        };
                        */
                        //get message
                        debug::socket_messages_server::message_decoder msg( this->m_debug.pop_recv_msg() );
                        //execute task by type
                        switch(msg.get_type())
                        {
                            //exit case
                            case debug::socket_messages_server::MSG_CLOSE_CONNECTION:
                                i_sub_gen = n_sub_gen-1;
                            break;
                            //not used cases
                            default:
                            break;
                        }
                    }
                )
            }
            ////////////////////////////////////////////////////////////////////////////
            int output_id=0;
            ////////////////////////////////////////////////////////////////////////////
            // Output list_eval_of_best and list_eval_of_best_of_best
            OutputVector(context,output_id++,list_eval_of_best);
            OutputVector(context,output_id++,list_eval_of_best_of_best);
            ////////////////////////////////////////////////////////////////////////////
            // Output best pop
            for(int i=0; i != this->m_space_size; ++i)
            {
                //Output ptr
                Tensor* new_generation_tensor = nullptr;
                //Get output tensor
                const Tensor& best_population = best.m_individual[i];
                //Alloc
                OP_REQUIRES_OK(context, context->allocate_output(output_id++, best_population.shape(), &new_generation_tensor));
                //copy tensor
                (*new_generation_tensor) = best_population;
            }
            ////////////////////////////////////////////////////////////////////////////
            // Output populations
            for(int i=0; i != this->m_space_size; ++i)
            {
                Tensor* new_generation_tensor = nullptr;
                Tensor current_pop = concatDim0(current_populations_list[i]);
                OP_REQUIRES_OK(context, context->allocate_output(output_id++, current_pop.shape(), &new_generation_tensor));
                (*new_generation_tensor) = current_pop;
            }

        }

        /**
        * Load next bach
        */
        bool LoadNextBach(OpKernelContext *context)
        {
            //Load bach
            if( !m_dataset.read_bach(m_bach) )
            {
                context->CtxFailure({
                    tensorflow::error::Code::ABORTED,
                    "Error stream dataset: can't read ["+std::to_string(m_dataset.get_last_bach_info().m_bach_id)+"] bach' "
                });
                return false;
            }

            //Set bach in input
            if( !SetBachInCacheInputs() )
            {
                context->CtxFailure({
                    tensorflow::error::Code::ABORTED,
                    "Error add bach data in inputs"
                });
                return false;
            }
            return true;
        }

        
        /**
        * Execute a reset if necessary
        * @param Context
        * @param Cache Best
        * @param populations, list of populations
        */
        void CheckReset(OpKernelContext *context,
                        const CacheBest<value_t>& best,
                        TensorListList& populations)
        {
            //reset
            if(m_reset.IsEnable() && m_reset.CanExecute(best.m_eval))
            {
                //todo
                populations.clear();
                //compute random 
                {
                    //output
                    TensorList output_pop;
                    //execute
                    auto
                    status= this->m_session->Run
                    (   //input
                        TensorInputs{},
                        //function
                        m_reset.GetRandFunctions(),
                        //one
                        NameList{ },
                        //output
                        &output_pop
                    );
                    //output error
                    if NOT(status.ok())
                    {
                        context->CtxFailure({tensorflow::error::Code::ABORTED,"Run execute random: "+status.ToString()});
                        return /* fail */;
                    }
                    //return population
                    for(int i=0; i!=this->m_space_size ;++i)
                    {
                        populations.push_back(splitDim0(output_pop[i]));
                    }
                }
                //push best
                for(int layer_id=0; layer_id!=populations.size(); ++layer_id)
                {
                    populations[layer_id][best.m_id] = best.m_individual[layer_id];
                }
            }
        }

        /**
        * copy vector to tensor as output
        */
        void OutputVector(OpKernelContext *context, int output, std::vector < value_t >& list_values)
        {
            //Output ptr
            Tensor* new_generation_tensor = nullptr;
            //alloc
            OP_REQUIRES_OK(context, context->allocate_output(output, TensorShape({int64(list_values.size())}), &new_generation_tensor));
            //copy
            auto output_ptr = new_generation_tensor->flat<value_t>();
            //copy all
            for(int i = 0; i!= (int)list_values.size(); ++i)
            {
                output_ptr(i) = list_values[i];
            }
        }


        /**
         * Find best individual in populations
         * @param Context
         * @param populations, list of populations
         * @param output id 
         * @param output eval
         * @return population index
         */
        void FindBest(OpKernelContext *context,
                      const TensorListList& populations,
                      int&    best_id,
                      value_t& best_eval
                      )
        {          
            //Get np 
            const int NP = populations[0].size();
            //Change input 
            SetValidationDataInCacheInputs();
            //set id best
            best_id = 0;  
            //execute evaluation
            best_eval = ExecuteEvaluateValidation(context, best_id, populations);
            //Execute validation test to all pop
            for(int individual_id = 1; individual_id < NP; ++individual_id)
            {         
                //Change input 
                SetValidationDataInCacheInputs();
                //execute evaluation
                value_t eval = ExecuteEvaluateValidation(context, individual_id, populations);
                //is the best?
                if(best_eval < eval)
                {
                    best_id   = individual_id;
                    best_eval = eval;
                }
            }
        }

        /**
         * Test best population
         * @param Context
         * @param current_populations_list, list of populations
         * @param population index
         */
        value_t TestBest
        (
            OpKernelContext *context,
            const TensorListList& current_populations_list,
            int best_index
        ) const
        {
            return ExecuteEvaluateTest(context, best_index, current_populations_list);
        }


        //execute evaluate validation function (tensorflow function)   
        virtual value_t ExecuteEvaluateValidation
        (
            OpKernelContext* context,
            const int NP_i,
            const TensorListList& populations_list
        ) const
        {
            NameList function{ 
                m_name_validation //+":0"
            };
            return DENNOp_t::ExecuteEvaluate(context, NP_i, populations_list, function);
        }

        //execute evaluate test function (tensorflow function)   
        virtual value_t ExecuteEvaluateTest
        (
            OpKernelContext* context,
            const int NP_i,
            const TensorListList& populations_list
        ) const
        {
            NameList function{
                 m_name_test //+":0" 
            };
            return DENNOp_t::ExecuteEvaluate(context, NP_i, populations_list, function);
        }

    protected:
        /**
        * Set dataset in m_inputs_tensor_cache
        */
        virtual bool SetBachInCacheInputs() const
        {
            return DENNOp_t::SetDatasetInCacheInputs( m_bach.m_labels, m_bach.m_features);
        }

        /**
        * Set validation data in m_inputs_tensor_cache
        */
        virtual bool SetValidationDataInCacheInputs() const
        {
            return DENNOp_t::SetDatasetInCacheInputs( m_validation.m_labels, m_validation.m_features);
        }

        /**
        * Set test data in m_inputs_tensor_cache
        */
        virtual bool SetTestDataInCacheInputs() const
        { 
            return DENNOp_t::SetDatasetInCacheInputs(m_test.m_labels, m_test.m_features);
        }


    protected:

        //dataset
        DataSetLoader< io_wrapper::zlib_file<> > m_dataset;
        //Bach
        DataSetRaw m_bach;
        DataSetRaw m_validation;
        DataSetRaw m_test;
        //dataset path
        std::string m_dataset_path;
        //validation function
        std::string m_name_validation;
        //test function
        std::string m_name_test;
        //reset 
        DEReset<value_t> m_reset;

    };
};
