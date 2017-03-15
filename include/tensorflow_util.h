#pragma once
/***
* 
* TESTING: == / c::concatDim0 / splitDim0::splitDim0
*
* if(!(concatDim0(splitDim0(population)) == population))
* {
*   context->CtxFailure
*   (
*      {tensorflow::error::Code::ABORTED,"concatDim0 != splitDim0"}
*   );
* }
*
***/
#include <vector>
#include <assert.h>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow
{
    /*
    DT_FLOAT 	tf.float32 	32 bits floating point.
    DT_DOUBLE 	tf.float64 	64 bits floating point.
    DT_INT8 	tf.int8 	8 bits signed integer.
    DT_INT16 	tf.int16 	16 bits signed integer.
    DT_INT32 	tf.int32 	32 bits signed integer.
    DT_INT64 	tf.int64 	64 bits signed integer.
    DT_UINT8 	tf.uint8 	8 bits unsigned integer.
    DT_UINT16 	tf.uint16 	16 bits unsigned integer.
    DT_STRING 	tf.string 	Variable length byte arrays. Each element of a Tensor is a byte array.
    DT_BOOL 	tf.bool 	Boolean.
    DT_COMPLEX64 	tf.complex64 	Complex number made of two 32 bits floating points: real and imaginary parts.
    DT_COMPLEX128 	tf.complex128 	Complex number made of two 64 bits floating points: real and imaginary parts.
    DT_QINT8 	tf.qint8 	8 bits signed integer used in quantized Ops.
    DT_QINT32 	tf.qint32 	32 bits signed integer used in quantized Ops.
    DT_QUINT8 	tf.quint8 	8 bits unsigned integer used in quantized Ops.
    */
    //data type 
    template < typename TYPE > DataType data_type() { assert(0); return DataType::DT_DOUBLE; /* fake return */ }
    template <> DataType data_type<float>() { return DataType::DT_FLOAT; }
    template <> DataType data_type<double>() { return DataType::DT_DOUBLE; }
    template <> DataType data_type<int8>() { return DataType::DT_INT8; }
    template <> DataType data_type<int16>() { return DataType::DT_INT16; }
    template <> DataType data_type<int32>() { return DataType::DT_INT32; }
    template <> DataType data_type<uint8>() { return DataType::DT_UINT8; }
    template <> DataType data_type<uint16>() { return DataType::DT_UINT16; }
    template <> DataType data_type<std::string>() { return DataType::DT_STRING; }
    template <> DataType data_type<bool>() { return DataType::DT_BOOL; }

    inline bool operator ==(const tensorflow::Tensor& left,const tensorflow::Tensor& right)
    {
        //type
        if(left.dtype() != right.dtype()) return false;
        //shape size
        if(left.shape().dims()  != right.shape().dims()) return false;
        //shape dims size
        for(size_t i=0;i!=left.shape().dims();++i)
        {
            if(left.shape().dim_size(i)!=right.shape().dim_size(i)) return false;
        }
        //get memory (N.B. string fail)
        StringPiece left_data  = left.tensor_data();
        StringPiece right_data = right.tensor_data();
        //compare
        return std::memcmp(left_data.data(), right_data.data(),  left_data.size()) == 0;
    }
    
    template < typename T = double > 
    inline void fill(Tensor& tensor, const T& v)
    {
        #if 0
        //get data
        StringPiece to_data = tensor.tensor_data();
        //get ptr 
        char* ptr_data = (char*)to_data.data();
        //for all values
        for(size_t i=0; i < to_data.size(); i+=sizeof(T))
        {
            std::memcpy(ptr_data+i,&v,sizeof(T));
        }
        #else 
        //raw ptr
        typename TTypes< T >::Flat tensor_data = tensor.flat< T >();
        //write
        for(int i=0; i!=tensor_data.size(); ++i)
        {
            tensor_data(i) = v;
        }
        //end
        #endif 
    }
 
    inline std::vector<Tensor> splitDim0(const Tensor& tensor)
    {
        //output
        std::vector<Tensor> result;
        //num of dimations
        const int NUM_OF_D = tensor.shape().dims();
        //ref from data
        StringPiece from_data = tensor.tensor_data();
        //test
        if(NUM_OF_D)
        {
            //shape
            TensorShape shape;
            //type of shape
            if(NUM_OF_D>1)
                for(int i=1;i!=NUM_OF_D;++i) { shape.AddDim(tensor.dim_size(i)); }
            else
                shape.AddDim(1);
            //start offset
            int64 offset = 0;
            //copy
            for(int i=0;i!=tensor.dim_size(0); ++i)
            {
                //alloc
                result.emplace_back(tensor.dtype(), shape);
                Tensor* split = &result[result.size() - 1];
                //ref to data
                StringPiece to_data = split->tensor_data();
                //copy
                std::memcpy(const_cast<char*>(to_data.data()), from_data.data() + offset,  to_data.size());
                //offset
                offset += to_data.size();
            }
        }
        //return
        return result;
    }
    
    inline Tensor concatDim0(const std::vector< Tensor >& list_tensor)
    {
        //base tensor
        const Tensor& tensor0 = list_tensor[0];
        const TensorShape& tensor0_shape = tensor0.shape();
        //new shape
        TensorShape output_shape; output_shape.AddDim((int)list_tensor.size());
        //add base dims
        for(int i=0;i!=tensor0_shape.dims();++i)
        { output_shape.AddDim(tensor0_shape.dim_size(i)); }
        //Alloc output shape
        Tensor out_tensor(tensor0.dtype(),output_shape);
        //start offset
        int64 offset = 0;
        //ref to data
        StringPiece to_data = out_tensor.tensor_data();
        //copy
        for(size_t i=0;i!=list_tensor.size();++i)
        {
            //ref to data
            StringPiece from_data = list_tensor[i].tensor_data();
            //copy
            std::memcpy(const_cast<char*>(to_data.data()) + offset, from_data.data(),  from_data.size());
            //offset
            offset += from_data.size();
        }
        
        return out_tensor;
    }
}
