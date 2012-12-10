#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <algorithm>
//#include <sort>
#include "../treebuild/rtc.h"
#include "../treebuild/cudamem.h"

//Compile: nvcc -O3 -o sortTest  sortTest.cu -arch=sm_30 -Xptxas=-v 

#define IDSHIFT 24
#define VALMASK 0x0000000F

// -arch=sm_35 -Xptxas=-v -lcudadevrt -rdc=true -g  -DPLUMMER -DNPERLEAF=16 -maxrregcount=32

struct cmp_3bits{
  bool operator () (const uint &a, const uint &b){
    //Strip the ID
    int anew = a & VALMASK;
    int bnew = b & VALMASK;
    return anew < bnew;
  }
};

int hostSortTest(uint *in_data, uint *out_data, uint n)
{
  //Add the index to the data
  for(int i=0; i < n; i++)
  {
    out_data[i] = (i << IDSHIFT) | in_data[i];
  }

  //sort
  std::stable_sort(out_data, out_data+n, cmp_3bits());


  return 0;
}

void kernelSuccess(const char kernel[] = "kernel")
{
  const int ret = (cudaDeviceSynchronize() != cudaSuccess);
  if (ret)
  {
    fprintf(stderr, "%s launch failed: %s\n", kernel, cudaGetErrorString(cudaGetLastError()));
    assert(0);
  }
}

#define PRINT_STATS 

//Basic bitonic sort, taken from the Advanced Quicksort example in the SDK as reference implementation
static __global__ void testSortKernel_bitonicSDK(const int n, uint *input, uint *output)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  #ifdef PRINT_STATS
    int loopCount = 0;
    int swaps     = 0;
  #endif

  __shared__ int sortbuf[256];

  sortbuf[tid] = (tid << IDSHIFT) | input[tid];
  __syncthreads();

//   bitonicsort_kernel(input, output, 0, n); 
  //Call sort rountines
 // Now the sort loops                                                                                                                                     
  // Here, "k" is the sort level (remember bitonic does a multi-level butterfly style sort)                                                                 
  // and "j" is the partner element in the butterfly.                                                                                                       
  // Two threads each work on one butterfly, because the read/write needs to happen                                                                         
  // simultaneously                                                                                                                                         
  for(unsigned int k=2; k<=blockDim.x; k*=2)  // Butterfly stride increments in powers of 2                                                                 
  {                                                                                                                                                         
    for(unsigned int j=k>>1; j>0; j>>=1) // Strides also in powers of to, up to <k                                                                          
    {                                                                                                                                                       
      unsigned int swap_idx = threadIdx.x ^ j; // Index of element we're compare-and-swapping with                                                          
      unsigned my_elem      = sortbuf[threadIdx.x];                                                                                                              
      unsigned swap_elem    = sortbuf[swap_idx];                                                                                                               
                                                                                                                                                            
      __syncthreads();                                                                                                                                      

      #ifdef PRINT_STATS
        loopCount++;
      #endif
                                                                                                                               
      // The k'th bit of my threadid (and hence my sort item ID)                                                                                            
      // determines if we sort ascending or descending.                                                                                                     
      // However, since threads are reading from the top AND the bottom of                                                                                  
      // the butterfly, if my ID is > swap_idx, then ascending means mine<swap.                                                                             
      // Finally, if either my_elem or swap_elem is out of range, then it                                                                                   
      // ALWAYS acts like it's the largest number.                                                                                                          
      // Confusing? It saves us two writes though.                                                                                                          
      unsigned int ascend  = k * (swap_idx < threadIdx.x);                                                                                                   
      unsigned int descend = k * (swap_idx > threadIdx.x);                                                                                                  
      bool swap = false;           
      if((threadIdx.x & k) == ascend)                                                                                                                       
        {                                                                                                                                                     
          if((my_elem & VALMASK) > (swap_elem & VALMASK))                                                                                                                             
            swap = true;                                                                                                                                      
        }                                                                                                                                                     
        if((threadIdx.x & k) == descend)                                                                                                                      
        {                                                                                                                                                     
          if((my_elem & VALMASK) < (swap_elem & VALMASK))                                                                                                                                  
            swap = true;                                                                                                                                      
        }                                                                                                                                                     
                                                                                                                                                              
        // If we had to swap, then write my data to the other element's position.                                                                             
        // Don't forget to track out-of-range status too!                                                                                                     
        if(swap)                                                                                                                                              
        {                                                                                                                                                     
          sortbuf[swap_idx] = my_elem;    
          #ifdef PRINT_STATS
            swaps++;
          #endif                                                                                                                    
        }                                                                                                                                                     
                                                                                                                                                              
        __syncthreads();                                                                                                                                      
      }                                                                                                                                                       
    }           


    #ifdef PRINT_STATS
      printf("[%d, %d ]\t Loops: %d Swaps: %d \n", bid, tid, loopCount, swaps);
    #endif

  //Combine the value and the thread-id into the results

   output[tid] = sortbuf[tid];

}

int main(int argc, char * argv [])
{
  const int n = 256;
 
  host_mem<uint> h_input, h_output, h_check;
  h_input.alloc(n);
  h_output.alloc(n);
  h_check.alloc(n);
 
  cuda_mem<uint> d_input, d_output;
  d_input.alloc(n);
  d_output.alloc(n);

  //Allocate some data
  for(int i=0; i < n; i++)
  {
    h_input[i] = ((int)(1000*drand48())) % 8;
    //fprintf(stderr, "%d\t->\t%d\n", i, h_input[i]);
  }

  d_input.h2d(h_input);

  //Call the sort kernel
  const int NBLOCKS  = 1;
  const int NTHREADS = 256;
  testSortKernel_bitonicSDK<<<NBLOCKS,NTHREADS>>>(n, d_input, d_output);
  kernelSuccess("testSortKernel_bitonicSDK");


  d_output.d2h(h_output);



  //Compute result on the host
  hostSortTest(h_input, h_check, n);

  const int printStride = 1;
  for(int i=0; i < n; i+=printStride)
  {
    for(int j=0; j < printStride; j++)
    {
      if(i+j < 256)
      {
        //Extract id and value
        uint id  = h_output[i+j] >> IDSHIFT;
        int val  = h_output[i+j] & VALMASK;

        uint hid  = h_check[i] >> IDSHIFT;
        int hval  = h_check[i] & VALMASK;

        int match_id = 0;
        int match_val= 0;
        if(id == hid)         
          match_id = 1;
        if(val == hval)
          match_val = 1;

        //fprintf(stderr, "(%d, %d)\t\t", val, id);
        fprintf(stderr, "GPU: (%d, %d)\tCPU: (%d, %d)  Match-ID: %d  Match-val: %d",
                         val, id, hval, hid, match_id, match_val);

      }
    }
    fprintf(stderr, "\n");
  }


  //Compare the host and device results
  for(int i=0; i < n; i++)
  {
    //Extract id and value
    uint id  = h_check[i] >> IDSHIFT;
    int val  = h_check[i] & VALMASK;
   
 //   fprintf(stderr, "%d\t->\t%d  |  %d \n", i, val, id);
  }

}