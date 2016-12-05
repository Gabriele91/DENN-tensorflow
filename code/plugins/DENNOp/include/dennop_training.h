#pragma once 

#define DENN_USE_SOCKET_DEBUG 
#define DENN_USE_TRAINING

#include "config.h"
#include "dennop.h"

namespace tensorflow
{
    class DENNOpTraining : public DENNOp
    {
    public:


        //init DENN from param
        explicit DENNOpTraining(OpKernelConstruction *context) : DENNOp(context)
        {

        }

        //star execution from python
        virtual void Compute(OpKernelContext *context) override
        {
            DENNOp::Compute(context);
        }
    };
};