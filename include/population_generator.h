#include "config.h"
#include "image_filters.h"
#include "tensorflow_alias.h"
#include <string>
#include <iostream>
#include <memory>
#include <cmath>
#include <algorithm>
#include <typeinfo>

namespace tensorflow
{

    class GeneratorRandomVector
    {
    public:

        GeneratorRandomVector
        (
            const DeInfo& info,
            int NP
        )
        {
            //init np
            m_NP = NP;
            //Select type of method
            switch (info.m_diff_vector)
            {
                default:
                case DIFF_ONE: m_randoms_i.resize(3);  break;
                case DIFF_TWO: m_randoms_i.resize(5);  break;
            }
        }

        GeneratorRandomVector
        (
            const DeInfo& info,
            int NP,
            int best
        )
        : GeneratorRandomVector(info, NP)
        {
            m_Best = best;
        }

        void SetBest(int best)
        {
            m_Best = best;
        }

        void DoRand(int individual)
        {               
            //do rand indices
            if(m_Best > -1)
            { 
                //get last
                int last = (int)m_randoms_i.size() - 1;
                //rand [0,last)
                random_indices::threeRandIndicesDiffFrom(m_NP, individual, m_randoms_i, last);
                //last is the best
                m_randoms_i[last] = m_Best;
            }
            else 
            {
                random_indices::threeRandIndicesDiffFrom(m_NP, individual, m_randoms_i, m_randoms_i.size());
            }
            //index shuffle
            std::random_shuffle(m_randoms_i.begin(),m_randoms_i.end());
        }
    
        operator const std::vector< int >& () const
        {
            return m_randoms_i;
        }

        int operator [](size_t i) const
        {
            return m_randoms_i[i];
        }

        int GetBest() const
        {
            return m_Best;
        }

    protected:

        int m_Best { -1 };
        int m_NP   {  1 };
        std::vector< int > m_randoms_i;
    };

    
    template< class value_t = double >
    void DeMethod
    (
        const DeInfo&                              info,
        const DeFactors<value_t>&                  factors,
        const value_t                              F,
        const int                                  cur_x,
        const TensorList&                          cur_population,
        typename TTypes<value_t>::Matrix&          new_generation, 
        int                                        elm,
        const std::vector < int >&                 randoms_i
    )
    {
        switch (info.m_diff_vector)
        {
            case DIFF_ONE:
            {
                //donor vectors
                const value_t a = cur_population[randoms_i[0]].flat<value_t>()(elm);
                const value_t b = cur_population[randoms_i[1]].flat<value_t>()(elm);
                //genome
                const value_t c = cur_population[randoms_i[2]].flat<value_t>()(elm);
                //function
                if(info.m_pert_vector==PV_CURR_TO_BEST) 
                {
                    //target vector
                    const value_t x = cur_population[cur_x].flat<value_t>()(elm);
                    //CURR_TO_BEST function
                    new_generation(elm) = factors.f_clamp( x+F*(a-b)+F*(c-x));
                }
                else
                {
                    new_generation(elm) = factors.f_clamp( (a-b) * F + c );
                }
            }
            break;
            case DIFF_TWO:
            {
                //donor vectors
                const value_t first_diff  = cur_population[randoms_i[0]].flat<value_t>()(elm) - cur_population[randoms_i[1]].flat<value_t>()(elm);
                const value_t second_diff = cur_population[randoms_i[2]].flat<value_t>()(elm) - cur_population[randoms_i[3]].flat<value_t>()(elm);
                //genome
                const value_t c = cur_population[randoms_i[4]].flat<value_t>()(elm);
                //end
                new_generation(elm) = factors.f_clamp( (first_diff + second_diff) * F + c );
            }
            break;
        }
    }

    template< class value_t = double >
    void DeCrossOver    
    (
        const DeInfo&                              info,
        const DeFactors<value_t>&                  factors,
        const value_t                              F,
        const value_t                              CR,
        const int                                  cur_x,
        const TensorList&                          cur_population,
        typename TTypes<value_t>::Matrix&          new_generation, 
        const int                                  D,
        const std::vector < int >&                 randoms_i
    )
    {                
        //do random
        bool do_random = true;
        //random index
        int random_index = random_indices::irand(D);
        //compute
        for (int i = 0, elm = 0; i < D; ++i)
        {
            //cross event
            bool cross_event = random_indices::random0to1() < CR ;
            //decide by data_type
            switch(info.m_cr_type)
            {
                case CR_EXP:
                    //start element 
                    elm = (i + random_index) % D;
                    //cases
                    if (do_random && (cross_event || random_index == elm))
                    {
                        DeMethod< value_t >(info, factors, F, cur_x, cur_population, new_generation, elm, randoms_i);
                    }
                    else
                    {
                        new_generation(elm) = cur_population[cur_x].flat_inner_dims<value_t>()(elm);
                    }
                    //in exp case stop to rand values
                    if(do_random  &&  !cross_event)  do_random = false;
                break;
                default:
                    //start element 
                    elm = i;
                    //cases
                    if (cross_event || random_index == elm)
                    {
                        DeMethod< value_t >(info, factors, F, cur_x, cur_population, new_generation, elm, randoms_i);
                    }
                    else
                    {
                        new_generation(elm) = cur_population[cur_x].flat_inner_dims<value_t>()(elm);
                    }
                break;
            }
        }
    }


    template< class value_t = double >
    void LayerGenerator
    (
        const DeInfo&                              info,
        const DeFactors<value_t>&                  factors,
        //old
        const int                                  NP,
        const TensorList&                          cur_population,
        //new
        const TensorList&                          new_F_CR,
              TensorList&                          new_population,
        GeneratorRandomVector&                     v_random
    )
    {
    #ifdef ENABLE_PARALLEL_NEW_GEN
        #pragma omp parallel for
        for (int index = 0; index < NP; ++index)
    #else 
        for (int index = 0; index < NP; ++index)
    #endif
        {
            //Num of dimations
            const int NUM_OF_D = cur_population[index].shape().dims();
            //Compute flat dimension
            int D = NUM_OF_D ? 1 : 0;
            //compute D
            for(int i=0; i < NUM_OF_D; ++i)
            {
                D *= cur_population[index].shape().dim_size(i);
            }
            //get ref 
            typename TTypes<value_t>::Matrix ref_new_population = new_population[index].flat_inner_dims<value_t>();
            //get F 
            value_t F  = new_F_CR[0].flat<value_t>()(index);
            value_t CR = new_F_CR[1].flat<value_t>()(index);
            //do rand indices
            v_random.DoRand(index);
            //crossover
            DeCrossOver< value_t >
            (
                info, 
                factors, 
                F,
                CR,
                index,
                cur_population, 
                ref_new_population, 
                D, 
                v_random
            );
        }
    }

    /**
     * ALLOC
     */
    void AllocNewPopulation
    (
        const TensorList&         cur_population_F_CR,
        const TensorListList&     cur_population_list,
              TensorList&         new_population_F_CR,
              TensorListList&     new_population_list
    )
    {
        //alloc
        new_population_list.resize(cur_population_list.size());
        //alloc new pop 
        for (size_t l_type=0; l_type!=cur_population_list.size(); ++l_type)
        {
            //alloc
            new_population_list[l_type].resize(cur_population_list[l_type].size());
            //ref to pupulation of layer
            const TensorList& cur_population = cur_population_list[l_type];
            TensorList& new_population = new_population_list[l_type];
            //alloc
            for (size_t index = 0; index < cur_population_list[l_type].size(); ++index)
            {
                #if 0
                new (&new_population[index]) Tensor(cur_population[index].dtype(), cur_population[index].shape());
                #else 
                new_population[index] = Tensor(cur_population[index].dtype(), cur_population[index].shape());
                #endif 
            }
        }
        //resize 
        new_population_F_CR.resize(cur_population_F_CR.size());
        //new F <- cur F
        //new CR <- cur CR
        for(size_t i = 0; i != cur_population_F_CR.size(); ++i)
        {
            #if 0
            new (&new_population_F_CR[i]) Tensor(cur_population_F_CR[i].dtype(), cur_population_F_CR[i].shape());
            #else 
            new_population_F_CR[i] = Tensor(cur_population_F_CR[i].dtype(), cur_population_F_CR[i].shape());
            #endif
        }
    }

    /**
    * Reset population
    */
    void ResetPopulation
    (
        TensorList&         new_population_F_CR,
        TensorListList&     new_population_list
    )
    {
        //reset pop 
        for (size_t l_type=0; l_type!=new_population_list.size(); ++l_type)
        {            
            //population 
            TensorList& new_population = new_population_list[l_type];
            //alloc
            for (size_t index = 0; index < new_population.size(); ++index)
            {
                new_population[index] = Tensor(new_population[index].dtype(), new_population[index].shape());
            }
        }
        //reset F, CR
        for(size_t i = 0; i != new_population_F_CR.size(); ++i)
        {
            new_population_F_CR[i] = Tensor(new_population_F_CR[i].dtype(), new_population_F_CR[i].shape());
        }
    } 

    /**
     * PopulationGenerator, create new generation
     * @param context,
     * @param de info,
     * @param de factor,
     * @param NP, size of population
     * @param F, Factor
     * @param cur_population_eval, input population evaluation list
     * @param cur_population_F_CR, input population F and CR values
     * @param cur_population_list, input population list
     * @param new_population_F_CR (output), return new F and CR
     * @param new_population_list (output), return new population
     */
    template < class value_t = double > 
    void PopulationGenerator
    (
        OpKernelContext*          context,
        const DeInfo&             info,
        const DeFactors<value_t>& factors,
        const int                 generation_i,
        const int                 generations,
        const int                 NP,
              Tensor&             cur_population_eval,
        const TensorList&         cur_population_F_CR,
        const TensorListList&     cur_population_list,
              TensorList&         new_population_F_CR,
              TensorListList&     new_population_list
    )
    {
        //Select type of method
        GeneratorRandomVector v_random
        (
            info, NP
        );
        //best?
        if(info.m_pert_vector == PV_BEST || info.m_pert_vector == PV_CURR_TO_BEST)
        {        
            //First best
            auto    pop_eval_ref = cur_population_eval.flat<value_t>();
            value_t val_best     = pop_eval_ref(0);
            int     id_best      = 0;
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
            //set best
            v_random.SetBest(id_best);
        }
        //realloc 
        #if 1
        ResetPopulation(new_population_F_CR,new_population_list);
        #endif
        //JDE
        {
            auto cur_f_list_ref  = cur_population_F_CR[0].flat<value_t>();
            auto cur_cr_list_ref = cur_population_F_CR[1].flat<value_t>();
            auto new_f_list_ref  = new_population_F_CR[0].flat<value_t>();
            auto new_cr_list_ref = new_population_F_CR[1].flat<value_t>();
            for(int pop_i = 0; pop_i !=NP; ++pop_i)
            {
                value_t rand_value = random_indices::random0to1();
                //MSG_DEBUG("rand_value:" << rand_value << " < factors.m_JDE:" << factors.m_JDE)
                if(rand_value <= factors.m_JDE)
                {
                    //new F
                    new_f_list_ref( pop_i ) = random_indices::random0to1() * 2.0;
                    //new CR
                    new_cr_list_ref( pop_i ) = random_indices::random0to1() * 1.0;
                }
                else //copy
                {
                    //new F
                    new_f_list_ref( pop_i )  = cur_f_list_ref( pop_i );
                    //new CR
                    new_cr_list_ref( pop_i ) = cur_cr_list_ref( pop_i );
                }
                //MSG_DEBUG("F[i]:" << new_f_list_ref( pop_i ) << ", CR[i]:" << new_cr_list_ref( pop_i )<< " i:" << pop_i);
            }
        }
        //for all layers types
        for(size_t l_type=0; l_type!=cur_population_list.size(); ++l_type)
        {
            //ref to population
            const TensorList& cur_population = cur_population_list[l_type];
            //ref to pupulation of layer
            TensorList& new_population = new_population_list[l_type];
            //gen a new layer by method selected 
            LayerGenerator< value_t >
            (
                info,
                factors,
                //current
                NP,
                cur_population,
                //new
                new_population_F_CR,
                new_population,
                //random values generator
                v_random
            );
        }
        //enable smooting?
        if NOT( factors.m_smoothing_n_pass ) return;
        //compute when use smoothing
        int basket_size   = generations / factors.m_smoothing_n_pass;
        int mid_basket    = std::ceil(float(basket_size-1) / 2.0f);
        bool do_smoothing = !basket_size || (generation_i % basket_size) == mid_basket;
        //smoothing
        if(cur_population_list.size() && do_smoothing)
        {
            //types
            for(size_t l_type=0; l_type!=cur_population_list.size(); ++l_type)
            {
                //ref to new pupulation
                TensorList& new_population = new_population_list[l_type];
                //smoothing if enable 
                if(factors.CanSmoothing(l_type))
                {
                #ifdef ENABLE_PARALLEL_NEW_GEN
                    #pragma omp parallel for
                    for (int index = 0; index < NP; ++index)
                #else 
                    for (int index = 0; index < NP; ++index)
                #endif
                    {
                        ApplayFilterAVG<value_t>(new_population[index],factors.GetShape(l_type));
                    }
                }
            }
        }
    }
}