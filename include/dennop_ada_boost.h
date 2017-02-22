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
            // start input w
            const size_t start_input_weigth = 3;
            // W
            TensorList  W_list;
            // pupulation inputs
            for(int i=0; i != this->m_space_size; ++i)
            {
                W_list.push_back(context->input(start_input_weigth+i));
            }
            ////////////////////////////////////////////////////////////////////////////
            // start input population
            const size_t start_input_population = start_input_weigth + this->m_space_size;
            // populations
            TensorListList  current_populations_list;        
            // populations inputs
            for(int i=0; i != this->m_space_size; ++i)
            {
                const Tensor& population = context->input(start_input_population+i);
                current_populations_list.push_back(splitDim0(population));
            }
            //Test sizeof populations
            if NOT(DENNOp_t::TestPopulationSize(context,current_populations_list)) return;
            ////////////////////////////////////////////////////////////////////////////
            // start input of C / EC / Y
            const size_t start_input_C_EC_Y = start_input_population + this->m_space_size;
            // Take C [ start input + W + population ]
            m_C      = context->input(start_input_C_EC_Y);
            // C errors list
            m_EC     = splitDim0(context->input(start_input_C_EC_Y+1));
            // C errors list
             m_pop_Y = splitDim0(context->input(start_input_C_EC_Y+2));
            ////////////////////////////////////////////////////////////////////////////
            //Temp of new gen of populations
            TensorListList new_populations_list;
            
            ////////////////////////////////////////////////////////////////////////////
            //Alloc temp vector of populations
            this->GenCachePopulation(current_populations_list,new_populations_list); 
            ////////////////////////////////////////////////////////////////////////////
            //Alloc input 
            this->AllocCacheInputs(current_populations_list);
            //Copy bach in input
            this->SetDatasetInCacheInputs(t_bach_labels,t_bach_features);

            ////////////////////////////////////////////////////////////////////////////
            // init evaluation
            if(calc_first_eval)
            {
                ComputePopY(context, current_populations_list);
            }
            //Get np 
            const int NP = current_populations_list[0].size();
            //Tensor first evaluation of all ppopulations
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
                , W_list
                // Cache
                , new_populations_list
                // In/Out
                , current_populations_list
                , current_eval_result
            );
            ////////////////////////////////////////////////////////////////////////////
            // Output populations
            for(int i=0; i != this->m_space_size; ++i)
            {
                Tensor* new_generation_tensor = nullptr;
                Tensor current_pop = concatDim0(current_populations_list[i]);
                OP_REQUIRES_OK(context, context->allocate_output(i, current_pop.shape(), &new_generation_tensor));
                (*new_generation_tensor) = current_pop;
            }
            //return C
            {
                Tensor* new_generation_tensor = nullptr;
                OP_REQUIRES_OK(context, context->allocate_output(this->m_space_size, m_C.shape(), &new_generation_tensor));
                (*new_generation_tensor) = m_C;
            }
            //return EC
            {
                const Tensor& all_EC = concatDim0(m_EC);
                Tensor* new_generation_tensor = nullptr;
                OP_REQUIRES_OK(context, context->allocate_output(this->m_space_size+1, all_EC.shape(), &new_generation_tensor));
                (*new_generation_tensor) = all_EC;
            }
            //return pop Y
            {
                const Tensor& all_pop_Y = concatDim0(m_pop_Y);
                Tensor* new_generation_tensor = nullptr;
                OP_REQUIRES_OK(context, context->allocate_output(this->m_space_size+2, all_pop_Y.shape(), &new_generation_tensor));
                (*new_generation_tensor) = all_pop_Y;
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
            OpKernelContext*  context,
            const int         num_gen,
            const TensorList& W_list,
            TensorListList&   new_populations_list,
            TensorListList&   current_populations_list,
            Tensor&           current_eval_result
        )
        {
            //Get np 
            const int NP = current_populations_list[0].size();
            //Pointer to memory
            auto ref_current_eval_result = current_eval_result.flat<value_t>();
            //evaluete all population 
            for(size_t NP_i = 0; NP_i != NP; ++NP_i)
            {
                ref_current_eval_result(NP_i) = ExecuteEvaluateAdaBoost(  context, m_pop_Y[NP_i] );
            }
            //loop
            for(int i=0;i!=num_gen;++i)
            {
                //Create new population
                PopulationGenerator< value_t >
                (
                    context, 
                    this->m_de_info,
                    this->m_de_factors,
                    NP,
                    W_list,
                    current_eval_result,
                    current_populations_list,
                    new_populations_list
                );
                //Change old population (if required)
                for(int index = 0; index!=NP ;++index)
                {
                    //get y and ec
                    Tensor y;
                    Tensor ec;
                    ComputeYAndEC(context, index, new_populations_list, y, ec);
                    //execute cross
                    value_t new_eval = ExecuteEvaluateAdaBoost(context, y);
                    //Choice
                    if(new_eval < ref_current_eval_result(index))
                    {
                        //save all populations (W, B)
                        for(int p=0; p!=current_populations_list.size(); ++p)
                            current_populations_list[p][index] = new_populations_list[p][index];
                        //save EC 
                        m_EC[index] = ec; 
                        //save Y
                        m_pop_Y[index] = y;
                        //save Eval
                        ref_current_eval_result(index) = new_eval;
                    }
                }   
                //////////////////////////////////////////////////////////////
                // TO DO: Update C and to 0 C_counter
                // C(gen+1) = (1-alpha) * C(gen) + alpha * (Ni / Np)
                //////////////////////////////////////////////////////////////
                //alloc counter of all ec
                Tensor ec_counter(data_type<int>(),m_C.shape());
                //all to 0
                fill<int>(ec_counter,0);
                //get pointer
                auto raw_EC_counter = ec_counter.template flat<int>();
                //for all errors
                for(size_t i = 0; i!=m_EC.size(); ++i)
                {
                    //get ec
                    auto raw_ec_i = m_EC[i].template flat<bool>();
                    //for all ec
                    for(size_t j=0; j!=ec_counter.dim_size(0); ++j)
                    {
                        raw_EC_counter(j) += int(!raw_ec_i(j));
                    }
                }
                //get values
                auto raw_C = m_C.flat<value_t>();
                //new c
                for(size_t i = 0; i!=m_C.shape().dim_size(0); ++i)
                {
                    value_t  op0 = (value_t(1.0)-m_alpha) * raw_C(i);
                    value_t  op1 = m_alpha * (value_t(raw_EC_counter(i)) / NP);
                    raw_C(i) = op0 + op1;
                }
            }
        }
 
        /**
        * Compute NN if req
        * @param populations_list, (input) populations
        */
        virtual void ComputePopY
        (
            OpKernelContext *context,
            const TensorListList& populations_list
        )
        {
            //Get np 
            const int NP = populations_list[0].size();
            //execute
            for(size_t NP_i=0; NP_i!=NP; ++NP_i)
                ComputeYAndEC(context, NP_i, populations_list, m_pop_Y[NP_i], m_EC[NP_i]);
        }

        /**
        * Compute NN if req
        * @param NP_i, (input) populations
        * @param populations_list, (input) populations
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
                return false;
            }
            //execute network
            {
                auto
                status= this->m_session->Run
                                    (   //input
                                        this->m_inputs_tensor_cache,
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
                    context->CtxFailure({tensorflow::error::Code::ABORTED,"Run correct predition: "+status.ToString()});
                    return true;
                }
                //return Y
                out_ec = output_correct_values[0];
            }
        }

        //execute evaluate function (tensorflow function->cross entropy)
        virtual value_t ExecuteEvaluateAdaBoost
        (
            OpKernelContext* context,
            //input
            const Tensor& nn_Y
        )
        {
            //Output
            TensorList  cross_value;
            //Temp Y*C 
            Tensor tmp_Y_C(nn_Y.dtype(),nn_Y.shape());
            //Compute Y*C            
            auto  raw_y_c  = tmp_Y_C.template flat_inner_dims<value_t>();
            auto  raw_y    = nn_Y.template flat_inner_dims<value_t>();
            auto  raw_c    = m_C.template flat<value_t>();
            //for all values
            for(size_t i = 0; i != tmp_Y_C.shape().dim_size(0); ++i)
            for(size_t j = 0; j != tmp_Y_C.shape().dim_size(1); ++j)
            {
                raw_y_c(i,j) = raw_y(i,j) * raw_c(i);
            }
            //compure cross (Y*C)
            {   
                auto
                status= this->m_session->Run( //input
                                        TensorInputs
                                        {
                                            { this->m_name_input_cross_entropy ,  tmp_Y_C  },  // y * c
                                            { this->m_input_labels,               m_labels }   // y_
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
            this->m_inputs_tensor_cache.resize(populations_list.size()+1);
            //add all names
            for(size_t p=0; p!=populations_list.size(); ++p)
            {
                this->m_inputs_tensor_cache[p].first = this->m_inputs_names[p];
            }
            this->m_inputs_tensor_cache[this->m_inputs_tensor_cache.size()-1].first = this->m_input_features;
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
            if(this->m_inputs_tensor_cache.size() != (populations_list.size()+1)) return false;
            //add all Tensor
            for(size_t p=0; p!=populations_list.size(); ++p)
            {
                this->m_inputs_tensor_cache[p].second = populations_list[p][NP_i];
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
            if(this->m_inputs_tensor_cache.size() < 2) return false;
            //add dataset in input
            this->m_inputs_tensor_cache[this->m_inputs_tensor_cache.size()-1].second  = features;
            //add ref to tensor
            m_labels = labels;
            //ok
            return true;
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
        TensorList  m_EC;
        TensorList  m_pop_Y;
        //:D
        mutable Tensor m_labels;

    };
}
