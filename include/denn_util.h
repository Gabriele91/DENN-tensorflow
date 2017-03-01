#import "config.h"
#import "tensorflow_alias.h"
namespace tensorflow
{
    //CREATE SESSION FROM GRAPH
    void ParserAttr(OpKernelConstruction *context, std::unique_ptr< Session >& session)
    {
        // Get graph path
        std::string graph_proto_string;
        // Get the index of the value to preserve
        OP_REQUIRES_OK(context, context->GetAttr("graph", &graph_proto_string));
        //session options
        SessionOptions options;
        //session
        session = std::unique_ptr< Session >(tensorflow::NewSession(options));
        //read proto graph of operations
        GraphDef graph_def;
        /**
         * REF
         * https://github.com/tensorflow/tensorflow/blob/c8a45a8e236776bed1d14fd71f3b6755bd63cc58/tensorflow/core/platform/env.cc#L316
         */
        ::tensorflow::protobuf::TextFormat::ParseFromString(graph_proto_string, &graph_def);
        //create graph from proto graph 
        session->Create(graph_def);
    }

    //GET DE FACTORS
    template < class value_t > 
    void ParserAttr(OpKernelConstruction *context, DeFactors<value_t>& de_factors)
    { 
        // float params temp
        float
        f_CR,
        f_F,
        f_f_min,
        f_f_max;
        // int params temp
        int smoothing_n_pass;
        //shape param temp
        std::vector<TensorShapeProto> shapes_smoothing;
        // get CR
        context->GetAttr("CR", &f_CR);
        // get F
        context->GetAttr("F", &f_F);
        // get f min
        context->GetAttr("f_min", &f_f_min);
        // get f max
        context->GetAttr("f_max", &f_f_max);
        // get smoothing pass
        context->GetAttr("smoothing_n_pass", &smoothing_n_pass);
        // get smoothing shapes
        context->GetAttr("smoothing", &shapes_smoothing);
        // params float to value_t
        de_factors.m_CR     = value_t(f_CR);
        de_factors.m_F      = value_t(f_F);
        de_factors.m_f_min  = value_t(f_f_min);
        de_factors.m_f_max  = value_t(f_f_max);
        //smoothing
        de_factors.m_smoothing_n_pass = smoothing_n_pass;
        de_factors.SetShapesSmoothing( shapes_smoothing );
    }

    //GET DE INFO
    void ParserAttr(OpKernelConstruction *context, DeInfo& de_info)
    {
        // get DE
        std::string de_type;
        context->GetAttr("DE", &de_type);
        //parsing
        de_info.SetTypeFromString(de_type);
    }

    /**
     * Test if size of all populations is correct
     * @param current_populations_list, (input) population 
     * @return true if is correct
     */
    static bool TestPopulationSize
    (
         OpKernelContext *context,
         const TensorListList& current_populations_list 
    )
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
}