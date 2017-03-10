#include <cmath>

namespace tensorflow
{
    template< class value_t = double > 
    class DEReset 
    {
    public:

        DEReset()
        : m_enable(false)
        , m_factor(100.0)
        , m_counter(0)
        , m_current_counter(0)
        , m_value((value_t)0)
        {
            
        }

        DEReset
        (
            bool    enable,
            value_t factor,
            int     counter,
            const std::string& reset_f,
            const std::string& reset_cr,
            const NameList& rand_functions
        )
        : m_enable(enable)
        , m_factor(factor)
        , m_counter(counter)
        , m_reset_F({reset_f})
        , m_reset_CR({reset_cr})
        , m_rand_functions(rand_functions)
        , m_current_counter(0)
        , m_value((value_t)0)
        {
        }

        bool IsEnable() const
        { 
            return m_enable;
        }

        bool CanExecute(value_t value)
        {
            if(std::abs(m_value - value) < m_factor)
            {
                //dec counter
                ++m_current_counter;
                //test 
                if(m_counter <= m_current_counter)
                {
                    //reset 
                    m_current_counter = 0;
                    m_value = value;
                    //return true
                    return true;
                }
            }
            else 
            {
                //reset counter
                m_current_counter = 0;
            }
            //update value 
            m_value = value;
            //not reset
            return false;
        }

        const NameList& GetRandFunctions() const
        {
            return m_rand_functions;
        }

        const NameList& GetResetF() const
        {
            return m_reset_F;
        }

        const NameList& GetResetCR() const
        {
            return m_reset_CR;
        }

    protected:
        //const values
        bool     m_enable;
        value_t  m_factor;
        int      m_counter;   
        NameList m_reset_F;
        NameList m_reset_CR;
        NameList m_rand_functions;         
        //runtime
        int     m_current_counter;
        value_t m_value;
    };

    template< class value_t = double > 
    class CacheBest
    {
    public:


        /*
        * Copy a individual if pass the test 
        * @param accuracy
        * @param index of individual in population
        * @param CRs and Fs of population 
        * @param population 
        * @return true if test is passed  
        **/
        bool TestBest(value_t eval, int id, const TensorList& pop_F_CR, const TensorListList& pop)
        {
            return TestBest
            (
                  eval
                , pop_F_CR[0].flat<value_t>()(id)
                , pop_F_CR[1].flat<value_t>()(id)
                , id
                , pop
            );
        }
        /*
        * Copy a individual if pass the test 
        * @param accuracy
        * @param F
        * @param CR
        * @param index of individual in population
        * @param population 
        * @return true if test is passed  
        **/
        bool TestBest(value_t eval, value_t F, value_t CR, int id, const TensorListList& pop)
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
                m_F    = F; 
                m_CR   = CR;
                m_id   = id;
                //is changed
                return true;
            }
            return false;
        }
        
        //attrs
        bool       m_init{ false };
        int        m_id  { -1 };
        value_t    m_eval{  0 };
        value_t    m_F   {  0 };
        value_t    m_CR  {  0 };
        TensorList m_individual;
    };

    /**
    * copy vector to output tensor
    */
    template < typename value_t = double >
    static void OutputVector(OpKernelContext *context, int output, std::vector < value_t >& list_values)
    {
        //Output ptr
        Tensor* output_tensor = nullptr;
        //alloc
        OP_REQUIRES_OK(context, context->allocate_output(output, TensorShape({int64(list_values.size())}), &output_tensor));
        //copy
        auto output_ptr = output_tensor->flat<value_t>();
        //copy all
        for(int i = 0; i!= (int)list_values.size(); ++i)
        {
            output_ptr(i) = std::move(list_values[i]);
        }
    }

    /**
    * copy value to output tensor
    */
    template < typename value_t = double >
    static void OutputValue(OpKernelContext *context, int output, value_t& value)
    {
        //Output ptr
        Tensor* output_tensor = nullptr;
        //alloc
        OP_REQUIRES_OK(context, context->allocate_output(output, TensorShape({int64(1)}), &output_tensor));
        //copy
        auto output_ptr = output_tensor->flat<value_t>();
        //copy
        output_ptr(0) = std::move(value);
    }
}