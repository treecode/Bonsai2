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

#define WARP_SIZE2 5
#define WARP_SIZE 32

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

//Custom sort method
static __device__ __forceinline__ int lanemask_lt()
{
  int mask;
  asm("mov.u32 %0, %lanemask_lt;" : "=r" (mask));
  return mask;
}
static __device__ __forceinline__ int2 warpBinExclusiveScan(const bool p)
{
  const unsigned int b = __ballot(p);
  return make_int2(__popc(b & lanemask_lt()), __popc(b));
}

static __device__ __forceinline__ int warpBinReduce(const bool p)
{
  const unsigned int b = __ballot(p);
  return __popc(b);
}

static __device__ __forceinline__ uint shfl_scan_add_step(const uint partial, const uint up_offset)
{
  uint result;
  asm(
      "{.reg .u32 r0;"
      ".reg .pred p;"
      "shfl.up.b32 r0|p, %1, %2, 0;"
      "@p add.u32 r0, r0, %3;"
      "mov.u32 %0, r0;}"
      : "=r"(result) : "r"(partial), "r"(up_offset), "r"(partial));
  return result;
}


static __global__ void testSortKernel(const int n, uint *input, uint *output)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  const int laneIdx = threadIdx.x & (WARP_SIZE-1);
  const int warpIdx = threadIdx.x >> WARP_SIZE2;

  __shared__ int sortbuf[256];

  //Put the to be sorted values into shared memory
  uint value = (tid << IDSHIFT) | input[tid];
// sortbuf[tid]
  __syncthreads();

  //Histogram count, first each warp makes a local histogram
  bool use       = false;

  int2 histogram[8]; //x will contain the offset within the warp
                     //y will contain the total sum of the warp

  const int val = (value & 0x0000000F);
  //Count per radix the offset and number of values
  #pragma unroll
  for(int i=0; i < 8; i++)
  //for(int i=0; i < 2; i++)
  {
    use          = ((value & 0x0000000F) == i);
    histogram[i] = warpBinExclusiveScan(use);
  }

  //Now compute the prefix sums across our warps
  //note that we have 8 values by 8 histogram values
  //store this 8x8 into 64 lanes.
  //warp0_hist0, warp1_hist0, warp2_hist0, ...warp6_hist7, warp7_hist7
  //this should allows us to prefix sum using two warps
  if(laneIdx < 8)
  {
    //printf("Test: %d : %d : %d \n", threadIdx.x, histogram[laneIdx].y, laneIdx*8+warpIdx);
    sortbuf[laneIdx*8+warpIdx] = histogram[laneIdx].y;
  }
  __syncthreads();

  //Compute the exclusive prefix sum, using binary reduction
  //            0, 1, 2, 3, 4, 5, 6, 7
  //            0 + 1, 2+3, 4+5, 6+7
  //              A  + B  ,  C  + D
  //                 E    +    F
  int offset;
  if(warpIdx < 2)
  {
    offset =  sortbuf[laneIdx+WARP_SIZE*warpIdx];
    #pragma unroll
      //for(int i = 0; i < 3; i++) /* log2(8) steps */
      for(int i = 0; i < 5; i++) /* log2(32) steps */
          offset = shfl_scan_add_step(offset, 1 << i);


    //sortbuf[laneIdx+WARP_SIZE*warpIdx] = offset;
   //Now we have two warps with prefix sums, we need to add the final value
   //of first warp to the values of the second warp

   if(threadIdx.x==31) sortbuf[64] = offset;

   offset -= sortbuf[laneIdx+WARP_SIZE*warpIdx]; //Make exclusive

//     printf("[%d, %d , %d ]\t offset: %d %d \n", bid, tid, 
// 		    laneIdx+WARP_SIZE*warpIdx,  sortbuf[laneIdx+WARP_SIZE*warpIdx],offset);
  }
  __syncthreads();

  if(warpIdx == 1)
  {
    offset+=sortbuf[64];
//     printf("[%d, %d , %d ]\t offset2: %d %d \n", bid, tid, 
// 		    laneIdx+WARP_SIZE*warpIdx,  sortbuf[laneIdx+WARP_SIZE*warpIdx],offset);
  }
  
  if(warpIdx < 2)
  {
    sortbuf[laneIdx+WARP_SIZE*warpIdx] = offset;
//    printf("[%d, %d , %d ]\t offset2: %d %d \n", bid, tid, 
//                 laneIdx+WARP_SIZE*warpIdx,  sortbuf[laneIdx+WARP_SIZE*warpIdx],offset);
  }
  __syncthreads();

  //Now each thread reads their storage location in the following way:
  //Value to read is one of the eight bins and depends on the warp
  
   //val*8 + warpIdx
   int storeLocation = sortbuf[val*8 + warpIdx] + histogram[val].x;


   output[storeLocation] = value;

//    printf("[%d, %d ]\t input[tid]: %d histo: %d sortbuf: %d  storeLoc: %d \n",
//           bid, tid,  input[tid], histogram[val].x, sortbuf[val*8 + warpIdx], 
// 	  storeLocation);

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

  int histoCount[8] = {0};
  //Allocate some data
  for(int i=0; i < n; i++)
  {
    h_input[i] = ((int)(1000*drand48())) % 8;
    //fprintf(stderr, "%d\t->\t%d\n", i, h_input[i]);
    histoCount[h_input[i]]++;
  }

  for(int i=0; i < 8; i++)
	  fprintf(stderr,"Data-stats: %d\t%d\n",
			  i, histoCount[i]);

  d_input.h2d(h_input);

  //Call the sort kernel
  const int NBLOCKS  = 1;
  const int NTHREADS = 256;

//   testSortKernel_bitonicSDK<<<NBLOCKS,NTHREADS>>>(n, d_input, d_output);
//   kernelSuccess("testSortKernel_bitonicSDK");

  testSortKernel<<<NBLOCKS,NTHREADS>>>(n, d_input, d_output);
  kernelSuccess("testSortKernel");
  d_output.d2h(h_output);



//   exit(0);
  //Compute result on the host
  hostSortTest(h_input, h_check, n);

  const int printStride = 1;
  for(int i=0; i < n; i+=printStride)
  {
    for(int j=0; j < printStride; j++)
    {
      if(i+j < NTHREADS)
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
