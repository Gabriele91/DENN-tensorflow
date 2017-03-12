#pragma once 
#define DENN_USE_SOCKET_DEBUG 
#include "config.h"
#include "dennop_ada_boost.h"
#include "dataset_loader.h"
#include "training_util.h"
#include <assert.h>

namespace tensorflow
{
    template< class value_t = double > 
    class DENNOpAdaBoostTraining : public DENNOpAdaBoost< value_t >
    {

        using DENNOpAdaBoost_t = DENNOpAdaBoost< value_t >;

        //Struct of additional data of a batch
        struct BatchValuesAda
        {
            bool       m_init = false;
            Tensor     m_C;
            TensorList m_EC;
            TensorList m_pop_Y;
        };

    public:

        //init DENN from param
        explicit DENNOpAdaBoostTraining(OpKernelConstruction *context) : DENNOpAdaBoost< value_t >(context)
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
            std::string reset_f;
            std::string reset_cr;
            NameList    reset_rand_pop;
            OP_REQUIRES_OK(context, context->GetAttr("reset_type", &reset_type));
            OP_REQUIRES_OK(context, context->GetAttr("reset_fector", &reset_factor));
            OP_REQUIRES_OK(context, context->GetAttr("reset_counter", &reset_counter));
            OP_REQUIRES_OK(context, context->GetAttr("reset_rand_pop", &reset_rand_pop));
            OP_REQUIRES_OK(context, context->GetAttr("reset_f", &reset_f));
            OP_REQUIRES_OK(context, context->GetAttr("reset_cr", &reset_cr));
            //init 
            new (&m_reset) DEReset<value_t>
            (
                  reset_type != "none"
                , (value_t)reset_factor
                , reset_counter
                , reset_f
                , reset_cr
                , reset_rand_pop
            );
            //reinsert best 
            OP_REQUIRES_OK(context, context->GetAttr("reinsert_best", &m_reinsert_best));
            
            // Try to openfile
            if NOT(m_dataset.open(m_dataset_path))
            {
                context->CtxFailure({tensorflow::error::Code::ABORTED,"Attribute error: can't open dataset' "});
            }

            //Load dataset
            m_dataset.read_test(m_test);
            m_dataset.read_validation(m_validation);

            /* get ADA C INIT value */
            float ada_boost_c = 0.0;
            OP_REQUIRES_OK(context, context->GetAttr("ada_boost_c",&ada_boost_c));
            //cast C value
            m_ada_C_init_value = value_t(ada_boost_c);
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
            //super gen
            const int n_sub_gen = tot_gen / sub_gen;
            ////////////////////////////////////////////////////////////////////////////
            // F and CR
            TensorList  current_population_F_CR;
            // get F
            current_population_F_CR.emplace_back( context->input(1) );
            // get CR
            current_population_F_CR.emplace_back( context->input(2) );
            //debug 
            MSG_DEBUG( "|F,CR|: " << current_population_F_CR.size() )
            MSG_DEBUG( "|F|: " << current_population_F_CR[0].shape().dim_size(0) )
            MSG_DEBUG( "|CR|: " << current_population_F_CR[1].shape().dim_size(0) )
            ////////////////////////////////////////////////////////////////////////////
            // start input
            const size_t start_input = 3;
            // populations
            TensorListList  current_population_list;
            // populations inputs
            for(int i=0; i != this->m_space_size; ++i)
            {
                const Tensor& population = context->input(start_input+i);
                current_population_list.emplace_back(splitDim0(population));
            }
            //Test sizeof populations
            if NOT(TestPopulationSize(context,current_population_list)) return;
            //Get np 
            const int NP = current_population_list[0].size(); 
            //debug 
            MSG_DEBUG( "NP: " << NP )
            ////////////////////////////////////////////////////////////////////////////
            //Temp of new gen of populations
            TensorList     new_population_F_CR;
            TensorListList new_population_list;
            //Alloc temp vector of populations
            AllocNewPopulation(current_population_F_CR,
                               current_population_list, 
                               new_population_F_CR,
                               new_population_list);
            MSG_DEBUG( "new|F,CR|: " << new_population_F_CR.size() )
            MSG_DEBUG( "new|F|: " << new_population_F_CR[0].shape().dim_size(0) )
            MSG_DEBUG( "new|CR|: " << new_population_F_CR[1].shape().dim_size(0) )
            MSG_DEBUG( "|new_population_list|: " << new_population_list.size() )
            MSG_DEBUG( "|new_population_list[0]|: " << new_population_list[0].size() )
            MSG_DEBUG( "|new_population_list[0][0]|: " << new_population_list[0][0].dim_size(0) )
            ////////////////////////////////////////////////////////////////////////////
            //Alloc input 
            this->AllocCacheInputs(current_population_list);

            ////////////////////////////////////////////////////////////////////////////
            // START STREAM
            m_dataset.start_read_batch();
            // Load first batch
            if NOT(LoadNextBatch(context, current_population_list)) return ;//false;
            //Set batch in input
            if( !SetBatchInCacheInputs() )
            {
                context->CtxFailure({
                    tensorflow::error::Code::ABORTED,
                    "Error add batch data in inputs"
                });
                return;
            }
            ////////////////////////////////////////////////////////////////////////////
            CacheBest< value_t > best;
            std::vector< value_t > list_eval_of_best;
            std::vector< value_t > list_eval_of_best_of_best;
            value_t cur_test_eval = 0.0;
            value_t best_test_eval = 0.0;

            ////////////////////////////////////////////////////////////////////////////
            // The start state
            {
                //find best
                int     cur_best_id;
                value_t cur_best_eval;
                int     cur_worst_id;
                value_t cur_worst_eval;
                FindBestAndWorst
                (
                    context, 
                    current_population_list,
                    cur_best_id,
                    cur_best_eval,
                    cur_worst_id,
                    cur_worst_eval,
                    GetLastAdaBatchValues()
                );
                //test best 
                best.TestBest(cur_best_eval, cur_best_id, current_population_F_CR, current_population_list);
                //Test 
                SetTestDataInCacheInputs();
                cur_test_eval  = 
                best_test_eval = ExecuteEvaluateTest(context, cur_best_id, current_population_list);
                //add results into vector
                list_eval_of_best.push_back(cur_test_eval);
                list_eval_of_best_of_best.push_back(best_test_eval);
            }
            ////////////////////////////////////////////////////////////////////////////
            // Tensor first evaluation of all populations
            Tensor current_eval_result(data_type<value_t>(),TensorShape({int64(NP)}));
            //fill all to 0
            fill<value_t>(current_eval_result,0);
            //loop    
            bool de_loop = true;
            ////////////////////////////////////////////////////////////////////////////
            // Execute DE
            for
            (
                //gen counter
                size_t i_sub_gen = 0;          
                //exit case
                i_sub_gen != n_sub_gen && de_loop;    
                //next    
                ++i_sub_gen, 
                LoadNextBatch(context, current_population_list) 
            )
            {
                //Set batch in input
                if( !SetBatchInCacheInputs() )
                {
                    context->CtxFailure({
                        tensorflow::error::Code::ABORTED,
                        "Error add batch data in inputs"
                    });
                    return;
                }
                //Get current ada values 
                BatchValuesAda& ada_batch_values = GetLastAdaBatchValues();
                //execute
                de_loop = this->RunDe
                (
                 // Input
                   context
                 , sub_gen
                 // Cache
                 , new_population_F_CR
                 , new_population_list
                 // In/Out
                 , current_eval_result
                 , current_population_F_CR
                 , current_population_list
                 //ADA
                 , ada_batch_values.m_C 
                 , ada_batch_values.m_EC
                 , ada_batch_values.m_pop_Y
                );

                //find best
                int     cur_best_id;
                value_t cur_best_eval;
                int     cur_worst_id;
                value_t cur_worst_eval;
                FindBestAndWorst
                (
                    context, 
                    current_population_list,
                    cur_best_id,
                    cur_best_eval,
                    cur_worst_id,
                    cur_worst_eval,
                    ada_batch_values
                );

                //test 
                SetTestDataInCacheInputs();
                cur_test_eval = ExecuteEvaluateTest(context, cur_best_id, current_population_list);

                //update best 
                if( best.TestBest(cur_best_eval, cur_best_id, current_population_F_CR, current_population_list) )
                {
                   best_test_eval = cur_test_eval;
                }
                else if(m_reinsert_best)
                {
                    //replace wrost
                    for(int layer_id=0; layer_id!=current_population_list.size(); ++layer_id)
                    {
                        current_population_list[layer_id][cur_worst_id] = best.m_individual[layer_id];
                    }
                    //replace F and CR 
                    current_population_F_CR[0].flat<value_t>()(cur_worst_id) = best.m_F;
                    current_population_F_CR[1].flat<value_t>()(cur_worst_id) = best.m_CR;
                    //recompute wrost Y & EC 
                    this->ComputeYAndEC(
                          context 
                        , cur_worst_id
                        , current_population_list 
                        , ada_batch_values.m_EC
                        , ada_batch_values.m_pop_Y
                    );
                }

                //add into vector
                list_eval_of_best.push_back(cur_test_eval);
                list_eval_of_best_of_best.push_back(best_test_eval);
                //Execute reset 
                if(CheckReset(context, best, current_population_F_CR, current_population_list))
                {
                    //recompute all 
                    RecomputePopYandEC(context, current_population_list);
                }
            }
            ////////////////////////////////////////////////////////////////////////////
            int output_id=0;
            ////////////////////////////////////////////////////////////////////////////
            // Output list_eval_of_best and list_eval_of_best_of_best
            OutputVector<value_t>(context,output_id++,list_eval_of_best);
            OutputVector<value_t>(context,output_id++,list_eval_of_best_of_best);
            ////////////////////////////////////////////////////////////////////////////
            // Output best F value 
            OutputValue<value_t>(context,output_id++, best.m_F);
            // Output best CR value 
            OutputValue<value_t>(context,output_id++, best.m_CR);
            // Output best pop
            for(int i=0; i != this->m_space_size; ++i)
            {
                //Output ptr
                Tensor* output_tensor = nullptr;
                //Get output tensor
                const Tensor& best_layer_list = best.m_individual[i];
                //Alloc
                OP_REQUIRES_OK(context, context->allocate_output(output_id++, best_layer_list.shape(), &output_tensor));
                //copy tensor
                (*output_tensor) = best_layer_list;
            }
            ////////////////////////////////////////////////////////////////////////////          
            // Output F CR
            for(int i=0; i != current_population_F_CR.size(); ++i)
            {
                Tensor* output_tensor = nullptr;
                Tensor& current_f_or_cr = current_population_F_CR[i];
                OP_REQUIRES_OK(context, context->allocate_output(output_id++, current_f_or_cr.shape(), &output_tensor));
                (*output_tensor) = current_f_or_cr;
            }
            // Output populations
            for(int i=0; i != this->m_space_size; ++i)
            {
                Tensor* output_tensor = nullptr;
                Tensor current_pop = concatDim0(current_population_list[i]);
                OP_REQUIRES_OK(context, context->allocate_output(output_id++, current_pop.shape(), &output_tensor));
                (*output_tensor) = current_pop;
            }
        }

        /**
        * Load next batch
        */
        bool LoadNextBatch(OpKernelContext *context,
                           const TensorListList& populations)
        {
            //Load batch
            if( !m_dataset.read_batch(m_batch) )
            {
                context->CtxFailure({
                    tensorflow::error::Code::ABORTED,
                    "Error stream dataset: can't read ["+std::to_string(m_dataset.get_last_batch_info().m_batch_id)+"] batch' "
                });
                return false;
            }

            #if 0
            MSG_DEBUG("batch id: "  << m_dataset.get_last_batch_info().m_batch_id )
            MSG_DEBUG("batch rows dim0: "  << m_dataset.get_last_batch_info().m_n_row )
            MSG_DEBUG("nclass of batch class "  << m_dataset.get_main_header_info().m_n_classes )
            #endif
            //Init ada values
            InitAdaBatchValuesIfRequired
            (
                  context
                , populations
                , m_dataset.get_last_batch_info().m_batch_id
                , m_dataset.get_last_batch_info().m_n_row
                , m_dataset.get_main_header_info().m_n_classes
            );

            return true;
        }

        
        /**
        * Execute a reset if necessary
        * @param Context
        * @param Cache Best
        * @param populations, list of populations
        */
        bool CheckReset(OpKernelContext *context,
                        const CacheBest<value_t>& best,
                        TensorList&     pop_F_CR,
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
                        context->CtxFailure({tensorflow::error::Code::ABORTED,"Run execute random population: "+status.ToString()});
                        MSG_DEBUG("Run execute random population, fail")
                        ASSERT_DEBUG( 0 )
                        return false;
                    }
                    //return population
                    for(int i=0; i!=this->m_space_size ;++i)
                    {
                        populations.push_back(splitDim0(output_pop[i]));
                    }
                }
                //reset F 
                {
                    //output
                    TensorList f_out;
                    //execute
                    auto
                    status= this->m_session->Run
                    (   //input
                        TensorInputs{},
                        //function
                        m_reset.GetResetF(),
                        //one
                        NameList{ },
                        //output
                        &f_out
                    );
                    //output error
                    if NOT(status.ok())
                    {
                        context->CtxFailure({tensorflow::error::Code::ABORTED,"Run execute reset F: "+status.ToString()});
                        MSG_DEBUG("Run execute reset F, fail")
                        ASSERT_DEBUG( 0 )
                        return false;
                    }
                    //return population
                    pop_F_CR[0] = f_out[0];
                }
                //reset CR 
                {
                    //output
                    TensorList cr_out;
                    //execute
                    auto
                    status= this->m_session->Run
                    (   //input
                        TensorInputs{},
                        //function
                        m_reset.GetResetCR(),
                        //one
                        NameList{ },
                        //output
                        &cr_out
                    );
                    //output error
                    if NOT(status.ok())
                    {
                        context->CtxFailure({tensorflow::error::Code::ABORTED,"Run execute reset CR: "+status.ToString()});
                        MSG_DEBUG("Run execute reset CR, fail")
                        ASSERT_DEBUG( 0 )
                        return false;
                    }
                    //return population
                    pop_F_CR[1] = cr_out[0];
                }
                //push best
                for(int layer_id=0; layer_id!=populations.size(); ++layer_id)
                {
                    populations[layer_id][best.m_id] = best.m_individual[layer_id];
                }
                #if 0
                MSG_DEBUG("F len " << pop_F_CR[0].shape().dims())
                MSG_DEBUG("F size " << pop_F_CR[0].shape().dim_size(0))
                MSG_DEBUG("CR len " << pop_F_CR[1].shape().dims())
                MSG_DEBUG("CR size " << pop_F_CR[1].shape().dim_size(0))
                #endif
                //replace F and CR on best id
                auto ref_F = pop_F_CR[0].flat<value_t>();
                auto ref_CR = pop_F_CR[1].flat<value_t>();
                ref_F(best.m_id) = best.m_F;
                ref_CR(best.m_id) = best.m_CR;
                //population is reset 
                return true;
            }
            //population isn't reset 
            return false;
        }


        /**
         * Find best individual in populations
         * @param Context
         * @param populations, list of populations
         * @param output id 
         * @param output eval
         * @return population index
         */
        void FindBestAndWorst
        (
            OpKernelContext *context,
            const TensorListList& populations,
            int&     best_id,
            value_t& best_eval,
            int&     worst_id,
            value_t& worst_eval,
            BatchValuesAda& ada_values
        )
        {          
            //Get np 
            const int NP = populations[0].size();
            //Change input 
            SetValidationDataInCacheInputs();
            //set id best
            worst_id = best_id = 0;  
            //execute evaluation
            worst_eval = best_eval = ExecuteEvaluateValidation(context, best_id, populations);
            //Execute validation test to all pop
            for(int individual_id = 1; individual_id < NP; ++individual_id)
            {         
                //execute evaluation
                value_t eval = ExecuteEvaluateValidation(context, individual_id, populations);
                //is the best?
                if(best_eval < eval)
                {
                    best_id   = individual_id;
                    best_eval = eval;
                }
                //is the worst?
                if(eval < worst_eval)
                {
                    worst_id   = individual_id;
                    worst_eval = eval;
                }
            }
        }

        /**
         * Test best population
         * @param Context
         * @param current_population_list, list of populations
         * @param population index
         */
        value_t TestBest
        (
            OpKernelContext *context,
            const TensorListList& current_population_list,
            int best_index
        ) const
        {
            return ExecuteEvaluateTest(context, best_index, current_population_list);
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
            return ExecuteEvaluate(context, NP_i, populations_list, function);
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
            if NOT(this->SetCacheInputs(populations_list, NP_i))
            {
                context->CtxFailure({tensorflow::error::Code::ABORTED,"Run evaluate: error to set inputs"});
                MSG_DEBUG("Run evaluate: error to set inputs, fail")
                ASSERT_DEBUG(0)
                return -1.0;
            }
            //add labels 
            this->m_inputs_tensor_cache.push_back({ this->m_input_labels, this->m_labels });
            //execute
            auto
            status= this->m_session->Run
            (
                //input
                this->m_inputs_tensor_cache,
                //function
                functions_list,
                //one
                NameList{ },
                //output
                &f_on_values
            );
            //remove m_input_labels
            this->m_inputs_tensor_cache.pop_back();
            //output error
            if NOT(status.ok())
            {
                context->CtxFailure({tensorflow::error::Code::ABORTED,"Run evaluate: "+status.ToString()});
                MSG_DEBUG("Run evaluate, fail: "<<status.ToString())
                ASSERT_DEBUG(0)
                return -1.0;
            }
            //results
            return f_on_values.size() ? f_on_values[0].flat<value_t>()(0) : 0.0;
        }

    protected:
        /**
        * Set dataset in m_inputs_tensor_cache
        */
        virtual bool SetBatchInCacheInputs() const
        {
            return DENNOpAdaBoost_t::SetDatasetInCacheInputs( m_batch.m_labels, m_batch.m_features);
        }


        /**
        * Set validation data in m_inputs_tensor_cache
        */
        virtual bool SetValidationDataInCacheInputs() const
        {
            return DENNOpAdaBoost_t::SetDatasetInCacheInputs( m_validation.m_labels, m_validation.m_features);
        }

        /**
        * Set test data in m_inputs_tensor_cache
        */
        virtual bool SetTestDataInCacheInputs() const
        { 
            return DENNOpAdaBoost_t::SetDatasetInCacheInputs(m_test.m_labels, m_test.m_features);
        }

        /**
        * Recompute Y and EC 
        * OpKernelContext *context,
        * const TensorListList& populations
        */
        void RecomputePopYandEC
        (
            OpKernelContext *context,
            const TensorListList& populations
        )
        {
            //Set batch in input
            SetBatchInCacheInputs();
            //id batch 
            int id = m_dataset.get_last_batch_info().m_batch_id;
            //compute Pop Y & EC
            this->ComputePopYAndEC
            (
                  context
                , populations
                , m_cache_ada[id].m_pop_Y
                , m_cache_ada[id].m_EC
            );
        }

        /**
        * Alloc no C, EC, Y is required
        * @param populations
        * @param batch id 
        * @param batch size
        * @param number of classes 
        */ 
        BatchValuesAda& InitAdaBatchValuesIfRequired
        (
            OpKernelContext *context,
            const TensorListList& populations,
            int id_batch, 
            int len_batch,
            int len_class
        )
        {
            //alloc vector
            if(m_cache_ada.size() <= id_batch) m_cache_ada.resize(id_batch+1);
            //alloc variables
            if NOT(m_cache_ada[id_batch].m_init)
            {
                //NP 
                const int NP = populations[0].size();
                //init
                m_cache_ada[id_batch].m_init = true;
                //alloc C 
                m_cache_ada[id_batch].m_C = Tensor(data_type<value_t>(), TensorShape({int64(len_batch)}));
                fill<value_t>(m_cache_ada[id_batch].m_C, m_ada_C_init_value);
                //alloc EC 
                for(int i=0; i!=NP; ++i)
                {
                    m_cache_ada[id_batch].m_EC.push_back(Tensor(data_type<bool>(), TensorShape({int64(len_batch)})));
                    fill<bool>(m_cache_ada[id_batch].m_EC[i], false);
                }
                //alloc pop Y 
                for(int i=0; i!=NP; ++i)
                {
                    value_t const_y_init = 0;
                    m_cache_ada[id_batch].m_pop_Y.push_back(Tensor(data_type<value_t>(), TensorShape({int64(len_batch), int64(len_class)})));
                    fill<value_t>(m_cache_ada[id_batch].m_pop_Y[i], const_y_init);
                }
                //Set batch in input
                SetBatchInCacheInputs();
                //compute Pop Y & EC
                this->ComputePopYAndEC
                (
                      context
                    , populations
                    , m_cache_ada[id_batch].m_pop_Y
                    , m_cache_ada[id_batch].m_EC
                );
            }
            return m_cache_ada[id_batch];
        }

        BatchValuesAda& GetAdaBatchValues(int id_batch)
        {
            assert(id_batch < m_cache_ada.size());
            return m_cache_ada[id_batch];
        }


        BatchValuesAda& GetLastAdaBatchValues()
        {
            return  GetAdaBatchValues(m_dataset.get_last_batch_info().m_batch_id);
        }


    protected:

        //ada boost data on batchs
        std::vector < BatchValuesAda > m_cache_ada;
        //ada init C
        value_t m_ada_C_init_value{ 1.0 };
        //dataset
        DataSetLoader< io_wrapper::zlib_file<> > m_dataset;
        //Batch
        DataSetRaw m_batch;
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
        //reinsert best 
        bool m_reinsert_best{ false };

    };
};