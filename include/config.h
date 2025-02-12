#pragma once

// SESSION TENSORFLOW
#include "tensorflow/core/public/session.h"

// PROTOBUF TENSORFLOW
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/env.h"

// CORE TENSORFLOW
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"

// FRAMEWORK TENSORFLOW
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/shape_inference.h"

// LOCAL UTILITIES TENSORFLOW
#include "tensorflow_util.h"
#include "random_indices_util.h"

// DEBUG UTILITIES
#ifdef DENN_USE_SOCKET_DEBUG
    #include "socket_messages_server.h"
    #define SOCKET_DEBUG(x) x
#else
    #define SOCKET_DEBUG(x)
#endif


#ifdef  _DEBUG
    #include <iostream>
    #include <assert.h>
    #define MSG_DEBUG( _msg_ ) std::cout<< _msg_ <<std::endl;
    #define ASSERT_DEBUG( _code_ ) assert(_code_);
    #define ASSERT_DEBUG_MSG( _code_, _msg_ )\
        { if (!(_code_)) { std::cout << _msg_ << std::endl; } assert(_code_); } 
#else
    #include <iostream>
    #include <assert.h>
    #define MSG_DEBUG( _msg_ ) 
    #define ASSERT_DEBUG( _code_ ) 
    #define ASSERT_DEBUG_MSG( _code_, _msg_ )\
        { if (!(_code_)) { std::cout << _msg_ << std::endl; } assert(_code_); } 
#endif

#ifndef NOT
    #define NOT(x) (!(x))
#endif 

//LOADER UTILS
#if defined( _MSC_VER )
    #define ASPACKED( __Declaration__ ) __pragma( pack(push, 1) ) __Declaration__ __pragma( pack(pop) )
#else 
    #define ASPACKED( __Declaration__ ) __Declaration__ __attribute__((__packed__))
#endif