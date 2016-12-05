#pragma once 

#define DENN_USE_SOCKET_DEBUG 
#define DENN_USE_TRAINING

#include "config.h"
#include "dennop.h"
#include "dataset_loader.h"

namespace tensorflow
{
    class DENNOpTraining : public DENNOp
    {
    public:

        //init DENN from param
        explicit DENNOpTraining(OpKernelConstruction *context) : DENNOp(context)
        {
            // Get dataset path
            OP_REQUIRES_OK(context, context->GetAttr("dataset", &m_dataset_path));
            // Try to openfile
            if(!m_dataset.open(m_dataset_path))
            {
                context->CtxFailure({tensorflow::error::Code::ABORTED,"Attribute error: can't open dataset' "});
            }
        }

        //star execution from python
        virtual void Compute(OpKernelContext *context) override
        {
            DENNOp::Compute(context);
        }
    
    protected:

        //dataset
        DataSetLoader< io_wrapper::zlib_file<> > m_dataset;
        //dataset path
        std::string m_dataset_path;

    };
};