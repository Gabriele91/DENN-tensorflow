namespace
{
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

    struct DeInfo
    {
        //DE types
        CRType           m_cr_type    { CR_BIN    };
        DifferenceVector m_diff_vector{ DIFF_ONE  };
        PerturbedVector  m_pert_vector{ PV_RANDOM };
    };

    template < class value_t > 
    struct DeFactors
    {
        //clamp
        value_t                     m_f_min{ -1.0 };
        value_t                     m_f_max{  1.0 };
        //update factors
        value_t                     m_CR   {  0.5 };
        /**
        * Clamp DE final value
        * @param t value,
        * @return t between f_min and f_max
        */
        value_t f_clamp(const value_t t) const
        {
            return std::min(std::max(t, m_f_min), m_f_max);
        }  
    };
}