#include "config.h"

namespace tensorflow
{
    template < class value_t = double >
    void ApplayFilterAVG(Tensor& inout_image,const TensorShape& shape)
    {
        //must to be 3D shape [height, width, k_size]
        assert(shape.dims() == 3);
        //size [IMAGE VECTOR, N_IMGS]
        assert(inout_image.shape().dims() == 2);
        //get size
        int height = shape.dim_size(0);
        int width  = shape.dim_size(1);
        int k_size = shape.dim_size(2);
        // MSG_DEBUG(height << " " << width << " " << k_size)    
        //new image
        Tensor out_image(data_type<value_t>(), inout_image.shape());
        //get ref
        #if 0
        //get data
        value_t* in_matrix = (value_t*)inout_image.tensor_data().data();
        value_t* out_matrix = (value_t*)out_image.tensor_data().data();
        #else 
        auto in_matrix  = inout_image.flat<value_t>();
        auto out_matrix = out_image.flat<value_t>();
        #endif
        //count classes
        auto class_size = inout_image.shape().dim_size(1);
        //size kernel
        int k_start = -(k_size-1) / 2;
        int k_end   =  (k_size-1) / 2;
        //counter
        value_t sum     = 0;
        value_t counter = 0;
        //applay
        for(int class_offset = 0; class_offset != class_size; ++class_offset)
        {
            //
            for(int img_y = 0; img_y != height; ++img_y)
            for(int img_x = 0; img_x != width;  ++img_x)
            {
                for(int k_y = k_start; k_y != k_end; ++k_y)
                {   
                    //compute y
                    int y = img_y + k_y;
                    //test
                    if( y > -1 && y < height )
                    {
                        for(int k_x = k_start; k_x != k_end; ++k_x)
                        {
                            //compute x
                            int x = img_x + k_x;
                            //test 
                            if( x > -1 && x < width )
                            {
                                //add
                                sum += in_matrix(x * class_size + class_offset + y * width);
                                //count
                                ++counter;
                            }
                        }
                    }
                }
                //save
                out_matrix(img_x * class_size + class_offset + img_y * width) = sum / counter;
                //reset
                sum     = 0;
                counter = 0;
            }
        }
        //copy new inmage in old image 
        inout_image = out_image;
    }

}