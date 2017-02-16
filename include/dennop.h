
#pragma once
#include "config.h"
#include <string>
#include <iostream>
#include <memory>
#include <cmath>
#include <algorithm>
#include <typeinfo>
#define NOT(x) (!(x))

namespace tensorflow
{

template< class value_t = double > 
class DENNOp : public OpKernel 
{
public:

    //types
    using NameList      =  std::vector< std::string >;
    using TensorList    =  std::vector< tensorflow::Tensor >;
    using TensorInput   =  std::pair< std::string, tensorflow::Tensor >;
    using TensorInputs  =  std::vector< TensorInput >;

    //num of threads (OpenMP)
    const int N_THREADS = 4;
    
    //type of CR
    enum CRType
    {
        CR_BIN,
        CR_EXP
    };
    
    //type of DE
    enum DifferenceVector
    {
        DIFF_ONE,
        DIFF_TWO
    };
    
    //type of generator a new population
    enum PerturbedVector
    {
        PV_RANDOM,
        PV_BEST,
        PV_RAND_TO_BEST
    };

    //init DENN from param
    explicit DENNOp(OpKernelConstruction *context) : OpKernel(context)
    {
        // Space size
        OP_REQUIRES_OK(context, context->GetAttr("space", &m_space_size));
        // Get graph path
        std::string graph_proto_string;
        // Get the index of the value to preserve
        OP_REQUIRES_OK(context, context->GetAttr("graph", &graph_proto_string));
        // Get names of eval inputs
        OP_REQUIRES_OK(context, context->GetAttr("f_inputs", &m_inputs_names));
        // Get dataset path
        OP_REQUIRES_OK(context, context->GetAttr("f_input_labels", &m_input_labels));
        // Get dataset path
        OP_REQUIRES_OK(context, context->GetAttr("f_input_features", &m_input_features));
        // Get name of eval function
        OP_REQUIRES_OK(context, context->GetAttr("f_name_execute_net", &m_name_execute_net));
        // Test size == sizeof(names)
        if( m_space_size != m_inputs_names.size() )
        {
            context->CtxFailure({tensorflow::error::Code::ABORTED,"Attribute error: sizeof(names) != sizeof(populations) "});
        }
        // float params temp
        float
        f_CR,
        f_f_min,
        f_f_max;
        // get CR
        context->GetAttr("CR", &f_CR);
        // get f min
        context->GetAttr("f_min", &f_f_min);
        // get f max
        context->GetAttr("f_max", &f_f_max);
        // get DE
        std::string de_type;
        context->GetAttr("DE", &de_type);
        //parsing
        std::istringstream stream_de_type{de_type};
        string type_elm;
        //get values
        while (getline(stream_de_type, type_elm, '/'))
        {
                 if (type_elm == "rand")           m_pert_vector = PV_RANDOM;
            else if (type_elm == "best")           m_pert_vector = PV_BEST;
            else if (type_elm == "rand-to-best")   m_pert_vector = PV_RAND_TO_BEST;
            else if (type_elm == "1")              m_diff_vector = DIFF_ONE;
            else if (type_elm == "2")              m_diff_vector = DIFF_TWO;
            else if (type_elm == "bin")            m_cr_type = CR_BIN;
            else if (type_elm == "exp")            m_cr_type = CR_EXP;
        }
        // params float to value_t
        m_CR     = value_t(f_CR);
        m_f_min  = value_t(f_f_min);
        m_f_max  = value_t(f_f_max);
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
        //create graph
        m_session->Create(graph_def);
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
        // get input bach labels
        const Tensor& t_bach_labels = context->input(1);
        // get input bach data
        const Tensor& t_bach_features = context->input(2);
        ////////////////////////////////////////////////////////////////////////////
        //get population first eval
        const Tensor& population_first_eval = context->input(3);
        ////////////////////////////////////////////////////////////////////////////
        // start input
        const size_t start_input = 4;
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
        if NOT(TestPopulationSize(context,current_populations_list)) return;

        ////////////////////////////////////////////////////////////////////////////
        //Temp of new gen of populations
        std::vector < std::vector <Tensor> > new_populations_list;
        //Alloc temp vector of populations
        GenCachePopulation(current_populations_list,new_populations_list);

        ////////////////////////////////////////////////////////////////////////////
        //Alloc input 
        AllocCacheInputs(current_populations_list);
        //Copy bach in input
        SetDatasetInCacheInputs(t_bach_labels,t_bach_features);

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
        // Execute DE
        RunDe
        (
            // Input
              context
            , num_gen
            , W_list
            // Cache
            , new_populations_list
            // In/Out
            , current_populations_list
            , current_eval_result
        );

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
            switch(m_pert_vector)
            {
                case PV_RANDOM: RandTrialVectorsOp(context, NP,W_list,current_populations_list,new_populations_list); break;
                case PV_BEST: BestTrialVectorsOp(context, NP,W_list,current_eval_result,current_populations_list,new_populations_list); break;
                default: return /* FAILED */;
            }
            //Change old population (if required)
            for(int index = 0; index!=NP ;++index)
            {
                //Evaluation
                value_t new_eval = ExecuteEvaluateTrain(context, index, new_populations_list);
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
            #if 1
            SOCKET_DEBUG(
                //m_debug.write(i);
                m_debug.write(1);
            )
            #endif
        }
    }
 
    /**
     * Alloc cache populations
     * @param context
     * @param current_populations_list, (input) population
     * @param new_populations_list, (output) cache of new population
     */
    void GenCachePopulation
    (
        const std::vector < std::vector <Tensor> >& current_populations_list,
        std::vector < std::vector <Tensor> >& new_populations_list
    ) const
    {
        //new pupulations
        new_populations_list.resize(m_space_size);
        // pupulation inputs
        for(int i=0; i != m_space_size; ++i)
        {
            new_populations_list[i].resize(current_populations_list[i].size());
        }
    }

    /**
     * Do evaluation if required 
     * @param force_to_eval, if true, eval all populations in anyway
     * @param current_populations_list, (input) populations
     * @param population_first_eval, (input) last evaluation of population
     * @param current_eval_result, (output) new evaluation of population
     */
    virtual void DoFirstEvaluationIfRequired
    (
         OpKernelContext *context,
         const bool force_to_eval,
         const std::vector < std::vector <Tensor> >& current_populations_list,
         const Tensor& population_first_eval,
         Tensor& current_eval_result
    )
    {
        //Get np 
        const int NP = current_populations_list[0].size();
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
                current_eval_result.flat<value_t>()(index) = ExecuteEvaluateTrain(context, index, current_populations_list);
            }
        }

    }

    /**
     * Test if size of all populations is correct
     * @param current_populations_list, (input) population 
     * @return true if is correct
     */
    bool TestPopulationSize
    (
         OpKernelContext *context,
         const std::vector < std::vector <Tensor> >& current_populations_list 
    ) const
    {
        //Size of population
        const size_t NP = current_populations_list[0].size();
        //Test NP
        for(size_t i = 1; i < current_populations_list.size(); ++i)
        {
            if( current_populations_list[i].size() != NP )
            {
                context->CtxFailure({tensorflow::error::Code::ABORTED,"Input error: sizeof(populations["+std::to_string(i)+"]) != NP "});
                return false;
            }
        }
        return true;
    }


    //execute evaluate train function (tensorflow function)   
    virtual value_t ExecuteEvaluateTrain
    (
        OpKernelContext* context,
        const int NP_i,
        const std::vector < std::vector<Tensor> >& populations_list
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
        const std::vector < std::vector<Tensor> >& populations_list,
        const NameList& functions_list
    ) const
    {
        //Output
        TensorList f_on_values;
        //Set input
        if NOT(SetCacheInputs(populations_list, NP_i))
        {
            context->CtxFailure({tensorflow::error::Code::ABORTED,"Run evaluate: error to set inputs"});
        }
        #if 0
        //values
        for (auto t : m_inputs_tensor_cache)
        {
            std::cout 
            << "Input[" 
            << t.first 
            << "] = NDIM "
            << t.second.shape().dims()
            << std::endl;
        }
        #endif
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
        }
        //results
        return f_on_values.size() ? f_on_values[0].flat<value_t>()(0) : 0.0;
    }


protected:
       
    /**
     * Clamp DE final value
     * @param t value,
     * @return t between f_min and f_max
     */
    value_t f_clamp(const value_t t) const
    {
        return std::min(std::max(t, m_f_min), m_f_max);
    }  
       
    /**
     * RandTrialVectorsOp, create new generation (RAND DE METHOD)
     * @param context,
     * @param NP, size of population
     * @param List, Factor to DE
     * @param populations_list, input population list
     * @param new_populations_list (output), return of function
     */
    void RandTrialVectorsOp
    (
        OpKernelContext*                           context,
        const int                                  NP,
        const std::vector < Tensor >&              W_list,
        const std::vector < std::vector<Tensor> >& cur_populations_list,
        std::vector < std::vector<Tensor> >&       new_populations_list
    ) const
    {
        //name space random indices
        using namespace random_indices;
        // Generate vector of random indexes
        std::vector < int > randoms_i;
        //Select type of method
        switch (m_diff_vector)
        {
            default:
            case DIFF_ONE: randoms_i.resize(3);  break;
            case DIFF_TWO: randoms_i.resize(5);  break;
        }
        
        //for all populations
        for(size_t p=0; p!=cur_populations_list.size(); ++p)
        {
            //get values
            const auto W = W_list[p].flat<value_t>();
            //for all
        #ifdef ENABLE_PARALLEL_NEW_GEN
            #pragma omp parallel for num_threads(N_THREADS)
            for (int index = 0; index < NP; ++index)
        #else 
            for (int index = 0; index < NP; ++index)
        #endif
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
                new_populations_list[p][index] = Tensor(data_type<value_t>(), population[index].shape());
                //ref new gen
                auto new_generation = new_populations_list[p][index].flat_inner_dims<value_t>();
                //do rand indices
                threeRandIndicesDiffFrom(NP, index, randoms_i);
                //do random
                bool do_random = true;
                //random index
                int random_index = irand(D);
                //compute
                for (int i = 0, elm = 0; i < D; ++i)
                {
                    //start element 
                    if(m_cr_type != CR_EXP) 
                        elm = i;
                    else 
                        elm = (elm + random_index) % D;
                    //cross event
                    bool cross_event = random() < m_CR ;
                    //cases
                    if (do_random && (cross_event || random_index == elm))
                    {
                        switch (m_diff_vector)
                        {
                            default:
                            case DIFF_ONE:
                            {
                                const value_t a = population[randoms_i[0]].flat<value_t>()(elm);
                                const value_t b = population[randoms_i[1]].flat<value_t>()(elm);
                                const value_t c = population[randoms_i[2]].flat<value_t>()(elm);
                                new_generation(elm) = f_clamp( (a-b) * W(elm) + c );
                            }
                            break;
                            case DIFF_TWO:
                            {
                                const value_t first_diff  = population[randoms_i[0]].flat<value_t>()(elm) - population[randoms_i[1]].flat<value_t>()(elm);
                                const value_t second_diff = population[randoms_i[2]].flat<value_t>()(elm) - population[randoms_i[3]].flat<value_t>()(elm);
                                new_generation(elm) = f_clamp( (first_diff + second_diff) * W(elm) + population[randoms_i[4]].flat<value_t>()(elm) );
                            }
                            break;
                        }
                    }
                    else
                    {
                        new_generation(elm) = population[index].flat_inner_dims<value_t>()(elm);
                    }
                    //in exp case stop to rand values
                    if(do_random  &&  !cross_event && m_cr_type == CR_EXP)  do_random = false;
                }
            }
        }
    }

    /**
     * BestTrialVectorsOp, create new generation (BEST DE METHOD)
     * @param context,
     * @param NP, size of population
     * @param List, Factor of DE
     * @param populations_valutation, list(f_train(popularion) of all populations)
     * @param populations_list, input population list
     * @param new_populations_list (output), return of function
     */
    void BestTrialVectorsOp
    (
        OpKernelContext*                           context,
        const int                                  NP,
        const std::vector < Tensor >&              W_list,
              Tensor&                              cur_populations_eval,
        const std::vector < std::vector<Tensor> >& cur_populations_list,
        std::vector < std::vector<Tensor> >&       new_populations_list
    ) const
    {
        //name space random indices
        using namespace random_indices;
        // Generate vector of random indexes
        std::vector < int > randoms_i;
        //First best
        auto   pop_eval_ref = cur_populations_eval.flat<value_t>();
        value_t val_best     = pop_eval_ref(0);
        int    id_best      = 0;
        //search best
        for(int index = 1; index < NP ;++index)
        {
            //best?
            if(val_best < pop_eval_ref(index))
            {
                val_best= pop_eval_ref(index);
                id_best = index;
            }
        }
        //Select type of method
        switch (m_diff_vector)
        {
            default:
            case DIFF_ONE: randoms_i.resize(2);  break;
            case DIFF_TWO: randoms_i.resize(4);  break;
        }        
        //for all populations
        for(size_t p=0; p!=cur_populations_list.size(); ++p)
        {
            //get values
            const auto W = W_list[p].flat<value_t>();
            //for all
        #ifdef ENABLE_PARALLEL_NEW_GEN
            #pragma omp parallel for num_threads(N_THREADS)
            for (int index = 0; index < NP; ++index)
        #else 
            for (int index = 0; index < NP; ++index)
        #endif
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
                new_populations_list[p][index] = Tensor(data_type<value_t>(), population[index].shape());
                //ref new gen
                auto new_generation = new_populations_list[p][index].flat_inner_dims<value_t>();
                //do rand indices
                threeRandIndicesDiffFrom(NP, index, randoms_i);
                //do random
                bool do_random = true;
                //random index
                int random_index = irand(D);
                //compute
                for (int i = 0, elm = 0; i < D; ++i)
                {
                    //start element 
                    if(m_cr_type != CR_EXP) 
                        elm = i;
                    else 
                        elm = (elm + random_index) % D;
                    //cross event
                    bool cross_event = random() < m_CR ;
                    //cases
                    if (do_random && (cross_event || random_index == elm))
                    {
                        switch (m_diff_vector)
                        {
                            default:
                            case DIFF_ONE:
                            {
                                const value_t a = population[randoms_i[0]].flat<value_t>()(elm);
                                const value_t b = population[randoms_i[1]].flat<value_t>()(elm);
                                const value_t c = population[id_best].flat<value_t>()(elm);
                                new_generation(elm) = f_clamp( (a-b) * W(elm) + c );
                            }
                            break;
                            case DIFF_TWO:
                            {
                                const value_t first_diff  = population[randoms_i[0]].flat<value_t>()(elm) - population[randoms_i[1]].flat<value_t>()(elm);
                                const value_t second_diff = population[randoms_i[2]].flat<value_t>()(elm) - population[randoms_i[3]].flat<value_t>()(elm);
                                new_generation(elm) = f_clamp( (first_diff + second_diff) * W(elm) + population[id_best].flat<value_t>()(elm) );
                            }
                            break;
                        }
                    }
                    else
                    {
                        new_generation(elm) = population[index].flat_inner_dims<value_t>()(elm);
                    }
                    //in exp case stop to rand values
                    if(do_random  &&  !cross_event && m_cr_type == CR_EXP )  do_random = false;
                }
            }
        }
    }
    
    /**
    * Alloc m_inputs_tensor_cache
    * @param populations_list, (input) populations
    */
    virtual bool AllocCacheInputs(const std::vector < std::vector<Tensor> >& populations_list) const 
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
        const std::vector < std::vector<Tensor> >& populations_list,
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
    //clamp
    value_t                     m_f_min{ -1.0 };
    value_t                     m_f_max{  1.0 };
    //update factors
    value_t                     m_CR   {  0.5 };
    // population variables
    int                        m_space_size{ 1 };
    //input evaluate
    std::string                m_name_execute_net;
    NameList                   m_inputs_names;
    mutable TensorInputs       m_inputs_tensor_cache;
    //bach inputs
    std::string m_input_labels;
    std::string m_input_features;
    //DE types
    CRType           m_cr_type    { CR_BIN    };
    DifferenceVector m_diff_vector{ DIFF_ONE  };
    PerturbedVector  m_pert_vector{ PV_RANDOM };
    //debug
    SOCKET_DEBUG(
       debug::socket_messages_server m_debug;
    )
            
};

}
