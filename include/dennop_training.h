#pragma once 
#define DENN_USE_SOCKET_DEBUG 
#include "config.h"
#include "dennop.h"
#include "dataset_loader.h"

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
            struct CacheBest
            {
                //attrs
                bool m_init{ false };
                value_t m_eval{ 0 };
                TensorList m_individual;
                //add 
                void test_best(value_t eval,int id, const TensorListList& pop)
                {
                    //case to copy the individual
                    if(!m_init || eval > m_eval)
                    {
                        //pop all
                        m_individual.clear();
                        //copy 
                        for(const TensorList& layer : pop)
                        {
                            m_individual.push_back(layer[id]);
                        }
                        //set init to true 
                        m_init = true;
                        m_eval = eval;
                    }
                }
            }
            best;
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
            // Output best pop
            for(int i=0; i != this->m_space_size; ++i)
            {
                //Output ptr
                Tensor* new_generation_tensor = nullptr;
                //Get output tensor
                const Tensor& best_population = best.m_individual[i];
                //Alloc
                OP_REQUIRES_OK(context, context->allocate_output(i, best_population.shape(), &new_generation_tensor));
                //copy tensor
                (*new_generation_tensor) = best_population;
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
         * Find best population in populations
         * @param Context
         * @param current_populations_list, list of populations
         * @return population index
         */
        int FindBest
        (
            OpKernelContext *context,
            const TensorListList& current_populations_list
        ) const
        {
            //Get np 
            const int NP = current_populations_list[0].size();
            //Change input 
            SetValidationDataInCacheInputs();
            //First eval
            value_t val_best= ExecuteEvaluateValidation(context, 0, current_populations_list);
            int     id_best = 0;
            //search best
            for(int index = 1; index < NP ;++index)
            {
                //eval
                value_t eval_cur = ExecuteEvaluateValidation(context, index, current_populations_list);
                //best? (the accuracy is increased)
                if(val_best < eval_cur)
                {
                    val_best= eval_cur;
                    id_best = index;
                }
            }
            return id_best;
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

    };
};
