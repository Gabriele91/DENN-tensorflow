#pragma once 
#include "config.h"
#include "io_wrapper.h"
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow
{
    ASPACKED(struct DataSetHeader
    {   
        unsigned short m_version;
        int m_n_batch;
        int m_n_features;
        int m_n_classes;
        int m_type;
        int m_seed;
        float m_train_percentage;
        unsigned int m_test_offset;
        unsigned int m_validation_offset;
        unsigned int m_train_offset;

        tensorflow::DataType get_dtype() const
        {
            return 
            m_type == 1 
            ? tensorflow::DataType::DT_FLOAT
            : tensorflow::DataType::DT_DOUBLE
            ;
        }

        tensorflow::TensorShape get_shape_features(size_t n_values) const
        {
            tensorflow::TensorShape shape;
            shape.AddDim(n_values);
            shape.AddDim(m_n_features);
            return shape;
        }

        tensorflow::TensorShape get_shape_labels(size_t n_values) const
        {
            tensorflow::TensorShape shape;
            shape.AddDim(n_values);
            shape.AddDim(m_n_classes);
            return shape;
        }

    });

    ASPACKED(struct DataSetTestHeader
    {
        unsigned int m_n_row;
    });

    ASPACKED(struct DataSetValidationHeader
    {
        unsigned int m_n_row;
    });

    ASPACKED(struct DataSetTrainHeader
    {
        unsigned int m_batch_id;
        unsigned int m_n_row;
    });

    struct DataSetRaw
    {
        Tensor m_features;
        Tensor m_labels;
    };

    template < class IO >
    class DataSetLoader 
    {
    
    public:

        DataSetLoader()
        {
        }

        DataSetLoader(const std::string& path_file)
        {
            open(path_file);
        }

        bool open(const std::string& path_file)
        {
            if(m_file.open(path_file,"rb"))
            {
                m_file.read(&m_header,sizeof(DataSetHeader),1);
                return true;
            }
            return false;
        }

        bool is_open() const 
        {
            return m_file.is_open();
        }

        bool read_test(DataSetRaw& t_out)
        {
            if(is_open())
            {
                //save file pos
                size_t cur_pos = m_file.tell();
                //set file to test offset 
                m_file.seek_set(m_header.m_test_offset);
                //read header
                m_file.read(&m_test_header,sizeof(DataSetTestHeader),1);
                //read data
                bool status = read_raw(t_out, m_test_header.m_n_row);
                //return back
                m_file.seek_set(cur_pos);
                //return
                return status;
            }
            return false;
        }

        bool read_validation(DataSetRaw& t_out)
        {
            if(is_open())
            {
                //save file pos
                size_t cur_pos = m_file.tell();
                //set file to validation offset 
                m_file.seek_set(m_header.m_validation_offset);
                //read header
                m_file.read(&m_val_header,sizeof(DataSetValidationHeader),1);
                //read data
                bool status = read_raw(t_out, m_val_header.m_n_row);
                //return back
                m_file.seek_set(cur_pos);
                //return
                return status;
            }
            return false;
        }
        
        const DataSetHeader& get_main_header_info() const
        {
            return m_header;
        }

        const DataSetTrainHeader& get_last_batch_info() const
        {
            return m_train_header;
        }

        bool start_read_batch()
        {
            if(is_open())
            {
                //set file to train offset 
                m_file.seek_set(m_header.m_train_offset);
                //ok
                return true;
            }
            return false;
        }

        bool read_batch(DataSetRaw& t_out,bool loop = true)
        {
            if(is_open())
            {
                //read header
                m_file.read(&m_train_header,sizeof(DataSetTrainHeader),1);
                //read data
                bool status = read_raw(t_out, m_train_header.m_n_row);
                //if loop enable and batch is the last
                if(loop  
                &&(m_train_header.m_batch_id+1) == m_header.m_n_batch)
                {
                    //restart
                    start_read_batch();
                }
                return status;
            }
            return false;
        }

    protected:

        bool read_raw(DataSetRaw& t_out,const unsigned int size)
        {
            //alloc output
            t_out.m_features = Tensor
            (
                m_header.get_dtype(),
                m_header.get_shape_features(size)
            );
            //read features
            m_file.read
            (   
                  (void*)(t_out.m_features.tensor_data().data())
                , t_out.m_features.tensor_data().size()
                , 1
            );
            //alloc output
            t_out.m_labels = Tensor
            (
                m_header.get_dtype(),
                m_header.get_shape_labels(size)
            );
            //read labels
            m_file.read
            (
                  (void*)(t_out.m_labels.tensor_data().data())
                , t_out.m_labels.tensor_data().size()
                , 1
            );
            //
            return true;
        }

        IO m_file;
        DataSetHeader           m_header;
        DataSetTestHeader       m_test_header;
        DataSetValidationHeader m_val_header;
        DataSetTrainHeader      m_train_header;
    };
}