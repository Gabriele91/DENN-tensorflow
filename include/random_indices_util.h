#pragma once 
#include <vector>
#include <cstdlib>

namespace random_indices
{

    //random integer in [0,size)
    inline int irand(int size)
    {
        return std::rand() % size;
    }
    
    //random value in flooting point [0,1]
    inline double random0to1()
    {
        return ((double)std::rand()/(double)RAND_MAX);
    }

    //(start to start) in range of [0,size]
    inline int randRangeCircleDiffFrom(int size, int diff)
    {
        int output = (diff+1+irand(size)) % size;
        if(output == diff) return (output+1) % size;
        return output;
    }
    
    //n random indices (not equals)
    inline void threeRandIndicesDiffFrom(int size, int diff, std::vector< int >& indexs, int indexs_size)
    {
        //test
        assert(size >= indexs_size);
        //size of a batch
        int batch_size = size / indexs_size;
        int batch_current = 0;
        int batch_next    = 0;
        //compute rands
        for(int i=0; i!=indexs_size; ++i)
        {
            batch_current = batch_size*i;
            batch_next    = batch_size*(i+1);
            
            if(i==indexs_size-1)
            {
                int reminder = size % indexs_size;
                batch_next += reminder;
                batch_size += reminder;
            }
            
            if(batch_current <= diff && diff < batch_next)
            {
                indexs[i] = batch_current + randRangeCircleDiffFrom(batch_size, diff-batch_current);
            }
            else
            {
                indexs[i] = batch_current + irand(batch_next-batch_current);
            }
        }
    }

};