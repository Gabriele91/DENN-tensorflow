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

        void set_best(int best)
        {
            m_Best = best;
        }

        void do_rand(int individual)
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
        }
    
        operator const std::vector< int >& () const
        {
            return m_randoms_i;
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
                const value_t a = cur_population[randoms_i[0]].flat<value_t>()(elm);
                const value_t b = cur_population[randoms_i[1]].flat<value_t>()(elm);
                const value_t c = cur_population[randoms_i[2]].flat<value_t>()(elm);
                new_generation(elm) = factors.f_clamp( (a-b) * F + c );
            }
            break;
            case DIFF_TWO:
            {
                const value_t first_diff  = cur_population[randoms_i[0]].flat<value_t>()(elm) - cur_population[randoms_i[1]].flat<value_t>()(elm);
                const value_t second_diff = cur_population[randoms_i[2]].flat<value_t>()(elm) - cur_population[randoms_i[3]].flat<value_t>()(elm);
                const value_t c = cur_population[randoms_i[4]].flat<value_t>()(elm);
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
            bool cross_event = random() < CR ;
            //decide by data_type
            switch(info.m_cr_type)
            {
                case CR_EXP:
                    //start element 
                    elm = (i + random_index) % D;
                    //cases
                    if (do_random && (cross_event || random_index == elm))
                    {
                        DeMethod< value_t >(info, factors, F, cur_population, new_generation, elm, randoms_i);
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
                        DeMethod< value_t >(info, factors, F, cur_population, new_generation, elm, randoms_i);
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
            v_random.do_rand(index);
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
        //alloc new pop 
        for (size_t l_type=0; l_type!=cur_population_list.size(); ++l_type)
        {
            //alloc
            new_population_list[l_type].resize(cur_population_list[l_type].size());
            //ref to pupulation of layer
            const TensorList& cur_population = cur_population_list[l_type];
            TensorList& new_population = new_population_list[l_type];
            //alloc
            for (int index = 0; index < cur_population_list[l_type].size(); ++index)
            {
                new (&new_population[index]) Tensor(cur_population[index].dtype(), cur_population[index].shape());
            }
        }
        //resize 
        new_population_F_CR.resize(cur_population_F_CR.size());
        //new F <- cur F
        //new CR <- cur CR
        for(size_t i = 0; i != cur_population_F_CR.size(); ++i)
        {
            new (&new_population_F_CR[i]) Tensor(cur_population_F_CR[i].dtype(), cur_population_F_CR[i].shape());
        }
    }

    /**
     * PopulationGenerator, create new generation
     * @param context,
     * @param de info,
     * @param de factor,
     * @param NP, size of population
     * @param F, Factor
     * @param cur_populations_eval, input population evaluation list
     * @param cur_populations_F_CR, input population F and CR values
     * @param cur_populations_list, input population list
     * @param new_populations_F_CR (output), return new F and CR
     * @param new_populations_list (output), return new population
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
              Tensor&             cur_populations_eval,
        const TensorList&         cur_populations_F_CR,
        const TensorListList&     cur_populations_list,
              TensorList&         new_populations_F_CR,
              TensorListList&     new_populations_list
    )
    {
        //Select type of method
        GeneratorRandomVector v_random
        (
            info, NP
        );
        //best?
        if(info.m_pert_vector == PV_BEST)
        {        
            //First best
            auto    pop_eval_ref = cur_populations_eval.flat<value_t>();
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
            v_random.set_best(id_best);
        }
        //JDE
        auto cur_f_list_ref  = cur_populations_F_CR[0].flat<value_t>();
        auto cur_cr_list_ref = cur_populations_F_CR[1].flat<value_t>();
        auto new_f_list_ref  = new_populations_F_CR[0].flat<value_t>();
        auto new_cr_list_ref = new_populations_F_CR[1].flat<value_t>();
        for(int pop_i = 0; pop_i !=NP; ++pop_i)
        {
            if(random() <= factors.m_JDE)
            {
                //new F
                new_f_list_ref( pop_i ) = random() * 2.0;
                //new CR
                new_cr_list_ref( pop_i ) = random() * 1.0;
            }
            else //copy
            {
                //new F
                new_f_list_ref( pop_i )  = cur_f_list_ref( pop_i );
                //new CR
                new_cr_list_ref( pop_i ) = cur_cr_list_ref( pop_i );
            }
        }
        //for all layers types
        for(size_t l_type=0; l_type!=cur_populations_list.size(); ++l_type)
        {
            //ref to population
            const TensorList& cur_population = cur_populations_list[l_type];
            //ref to pupulation of layer
            TensorList& new_population = new_populations_list[l_type];
            //gen a new layer by method selected 
            LayerGenerator< value_t >
            (
                info,
                factors,
                //current
                NP,
                cur_population,
                //new
                new_populations_F_CR,
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
        if(cur_populations_list.size() && do_smoothing)
        {
            //types
            for(size_t l_type=0; l_type!=cur_populations_list.size(); ++l_type)
            {
                //ref to new pupulation
                TensorList& new_population = new_populations_list[l_type];
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
                        ApplayFilterAVG(new_population[index],factors.GetShape(l_type));
                    }
                }
            }
        }
    }
}