#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <algorithm>
//#include <sort>
#include "../treebuild/rtc.h"
#include "../treebuild/cudamem.h"

//Compile: nvcc -O3 -o sortTest  sortTest.cu -arch=sm_30 -Xptxas=-v
//Compile: nvcc -O3 -o sortTest  sortTest.cu -arch=sm_35 -Xptxas=-v

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

//#define PRINT_STATS

//Basic bitonic sort, taken from the Advanced Quicksort example in the SDK as reference implementation
static __global__ void testSortKernel_bitonicSDK(const int n, uint *input, uint *output)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int idx = bid*blockDim.x + tid;

  #ifdef PRINT_STATS
    int loopCount = 0;
    int swaps     = 0;
  #endif

  __shared__ int sortbuf[256];

  sortbuf[tid] = (tid << IDSHIFT) | input[idx];
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

   output[idx] = sortbuf[tid];
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

// static __device__ __forceinline__ int warpBinReduce(const bool p)
// {
//   const unsigned int b = __ballot(p);
//   return __popc(b);
// }

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
  const int idx = bid*blockDim.x + tid;

  const int laneIdx = threadIdx.x & (WARP_SIZE-1);
  const int warpIdx = threadIdx.x >> WARP_SIZE2;

  __shared__ int sortbuf[65]; //2*32 + 1, +1 for value exchange

  //Put the to be sorted values into shared memory
  const int val = input[idx];
 
  int2 histogram;     //x will contain the offset within the warp
                      //y will contain the total sum of the warp

  //Count per radix the offset and number of values
  #pragma unroll
  for(int i=0; i < 8; i++)
  {
    int2 scanRes = warpBinExclusiveScan((val == i));

    if(laneIdx == i) //Lane 0 to 8 directly write the scan sum to shared-mem
    {
      sortbuf[laneIdx*8+warpIdx] = scanRes.y;
    }
    if(val == i) histogram = scanRes; 
  }
  __syncthreads(); //Let the writes to sortbuf be complete for all 8 warps

  //Now compute the prefix sums across our warps
  //note that we have 8 values by 8 histogram values
  //store this 8x8 into 64 lanes.
  //warp0_hist0, warp1_hist0, warp2_hist0, ...warp6_hist7, warp7_hist7
  //this allows us to compute the prefix sum using two warps


  //Compute the exclusive prefix sum, using binary reduction / shuffles
  int offset;
  if(warpIdx < 2)
  {
    offset =  sortbuf[laneIdx+WARP_SIZE*warpIdx];
    #pragma unroll
      for(int i = 0; i < 5; i++) /* log2(32) steps */
        offset = shfl_scan_add_step(offset, 1 << i);

    //Now we have two warps with prefix sums, we need to add the final value
    //of the first warp to the values of the second warp. Use the unused location
    if(threadIdx.x==31) sortbuf[64] = offset;

    offset -= sortbuf[laneIdx+WARP_SIZE*warpIdx]; //Make exclusive
  }
  __syncthreads(); //Wait on sortbuf[64] to be stored

  if(warpIdx == 1) offset+=sortbuf[64]; //Complete cross-warp prefix-sum
  
  //Prefix sum complete, write out the results
  if(warpIdx < 2) sortbuf[laneIdx+WARP_SIZE*warpIdx] = offset;

  __syncthreads();
  
  //Now each thread reads their storage location in the following way:
  //Value to read is one of the eight bins, namely the one associated to
  //the value and the is offset by the warp. This is then increased with 
  //the offset within the current warp as computed using shfl_scan_add_step 
  int storeLocation = sortbuf[val*8 + warpIdx] + histogram.x; 

  #if 1
    __shared__ int valbuf[256]; //2*32 + 1, +1 for value exchange
    //Scatter in shared-mem and then coalesced output to gmem
    valbuf[storeLocation] =  (tid << IDSHIFT) | val; //Use for CPU/GPU comparison
    //valbuf[storeLocation] =  val; //Use for production
    __syncthreads();
    output[bid*blockDim.x+tid] = valbuf[tid];
  #else
    //Scattered output to gmem
    output[bid*blockDim.x+storeLocation] = (tid << IDSHIFT) | val; //Use for CPU/GPU comparison
    //output[bid*blockDim.x+storeLocation] = val; //Production
  #endif
}

static __global__ void testSortKernelAtomic(const int n, uint *input, uint *output)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int idx = bid*blockDim.x + tid;

  const int laneIdx = threadIdx.x & (WARP_SIZE-1);
  const int warpIdx = threadIdx.x >> WARP_SIZE2;

  __shared__ int sortbuf[256]; //2*32 + 1, +1 for value exchange
  sortbuf[tid] = 0;
  __syncthreads();

  //Put the to be sorted values into shared memory
  const int val = input[idx];
 
  int2 histogram;     //x will contain the offset within the warp
                      //y will contain the total sum of the warp

  //Count per radix the offset and number of values
  #pragma unroll
  for(int i=0; i < 8; i++)
  {
    int2 scanRes = warpBinExclusiveScan((val == i));

    if(laneIdx == i) //Lane 0 to 8 directly write the scan sum to shared-mem
    {
      sortbuf[laneIdx*8+warpIdx] = scanRes.y;
      
      //Sum values of 0-4, 4*8 atomic ops...
      if(i < 4) atomicAdd(&sortbuf[65],scanRes.y);
      
    }
    if(val == i) histogram = scanRes; 
  }
  __syncthreads(); //Let the writes to sortbuf be complete for all 8 warps

  //Now compute the prefix sums across our warps
  //note that we have 8 values by 8 histogram values
  //store this 8x8 into 64 lanes.
  //warp0_hist0, warp1_hist0, warp2_hist0, ...warp6_hist7, warp7_hist7
  //this allows us to compute the prefix sum using two warps


  //Compute the exclusive prefix sum, using binary reduction / shuffles
  int offset;
  if(warpIdx < 2)
  {
    offset =  sortbuf[laneIdx+WARP_SIZE*warpIdx];    
    #pragma unroll
      for(int i = 0; i < 5; i++) /* log2(32) steps */
        offset = shfl_scan_add_step(offset, 1 << i);
   
    //Now we have two warps with prefix sums, we need to add the final value
    //of the first warp to the values of the second warp. Use the unused location

    offset -= sortbuf[laneIdx+WARP_SIZE*warpIdx]; //Make exclusive
    offset += sortbuf[64+warpIdx];                //Sum of other warp ( based on atomic)

    sortbuf[laneIdx+WARP_SIZE*warpIdx] = offset;
  }
  __syncthreads(); //Wait on sortbuf[64] to be stored
  
  //Now each thread reads their storage location in the following way:
  //Value to read is one of the eight bins, namely the one associated to
  //the value and the is offset by the warp. This is then increased with 
  //the offset within the current warp as computed using shfl_scan_add_step 
  int storeLocation = sortbuf[val*8 + warpIdx] + histogram.x; 

  #if 1
    __shared__ int valbuf[256]; //2*32 + 1, +1 for value exchange
    //Scatter in shared-mem and then coalesced output to gmem
    valbuf[storeLocation] =  (tid << IDSHIFT) | val; //Use for CPU/GPU comparison
    //valbuf[storeLocation] =  val; //Use for production
    __syncthreads();
    output[bid*blockDim.x+tid] = valbuf[tid];
  #else
    //Scattered output to gmem
    output[bid*blockDim.x+storeLocation] = (tid << IDSHIFT) | val; //Use for CPU/GPU comparison
    //output[bid*blockDim.x+storeLocation] = val; //Production
  #endif
}

static __global__ void testSortKernel1WarpPrefix(const int n, uint *input, uint *output)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int idx = bid*blockDim.x + tid;

  const int laneIdx = threadIdx.x & (WARP_SIZE-1);
  const int warpIdx = threadIdx.x >> WARP_SIZE2;

  __shared__ int sortbuf[64]; //2*32 + 1, +1 for value exchange

  //Put the to be sorted values into shared memory
  const int val = input[idx];
 
  int2 histogram;     //x will contain the offset within the warp
                      //y will contain the total sum of the warp

  //Count per radix the offset and number of values
  #pragma unroll
  for(int i=0; i < 8; i++)
  {
    int2 scanRes = warpBinExclusiveScan((val == i));

    if(laneIdx == i) //Lane 0 to 8 directly write the scan sum to shared-mem
    {
      sortbuf[laneIdx*8+warpIdx] = scanRes.y;
    }
    if(val == i) histogram = scanRes; 
  }
  __syncthreads(); //Let the writes to sortbuf be complete for all 8 warps

  //Now compute the prefix sums across our warps
  //note that we have 8 values by 8 histogram values
  //store this 8x8 into 64 lanes.
  //warp0_hist0, warp1_hist0, warp2_hist0, ...warp6_hist7, warp7_hist7
  //this allows us to compute the prefix sum using two warps


  //Compute the exclusive prefix sum, using binary reduction / shuffles
  if(warpIdx == 0) //Only warp 0 does this, this elimintas _syncthreads in between
  {
    int offset1 =  sortbuf[laneIdx+WARP_SIZE*0];
    int offset2 =  sortbuf[laneIdx+WARP_SIZE*1];
    #pragma unroll
      for(int i = 0; i < 5; i++) /* log2(32) steps */
        offset1 = shfl_scan_add_step(offset1, 1 << i);

    //Now we have prefix sums of first half, we need to add the final value
    //to the second half result. 
    int bcast = __shfl(offset1,31);

    offset1                     -= sortbuf[laneIdx+WARP_SIZE*0]; //Make exclusive
    sortbuf[laneIdx+WARP_SIZE*0] = offset1;

    //prefix sum on second half    
    #pragma unroll
      for(int i = 0; i < 5; i++) /* log2(32) steps */
        offset2 = shfl_scan_add_step(offset2, 1 << i);
    
    offset2 -= sortbuf[laneIdx+WARP_SIZE*1]; //Make exclusive
    offset2 += bcast;
        
    sortbuf[laneIdx+WARP_SIZE*1] = offset2;
    
  }
  __syncthreads(); //Wait on warp0 to be done


  //Now each thread reads their storage location in the following way:
  //Value to read is one of the eight bins, namely the one associated to
  //the value and the is offset by the warp. This is then increased with 
  //the offset within the current warp as computed using shfl_scan_add_step 
  int storeLocation = sortbuf[val*8 + warpIdx] + histogram.x; 

  #if 1
    __shared__ int valbuf[256]; //2*32 + 1, +1 for value exchange
    //Scatter in shared-mem and then coalesced output to gmem
    valbuf[storeLocation] =  (tid << IDSHIFT) | val; //Use for CPU/GPU comparison
    //valbuf[storeLocation] =  val; //Use for production
    __syncthreads();
    output[bid*blockDim.x+tid] = valbuf[tid];
  #else
    //Scattered output to gmem
    output[bid*blockDim.x+storeLocation] = (tid << IDSHIFT) | val; //Use for CPU/GPU comparison
    //output[bid*blockDim.x+storeLocation] = val; //Production
  #endif
}


static __global__ void testSortKernel1WarpPrefixIncludingSum(const int n, uint *input, uint *output)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int idx = bid*blockDim.x + tid;

  const int laneIdx = threadIdx.x & (WARP_SIZE-1);
  const int warpIdx = threadIdx.x >> WARP_SIZE2;

  __shared__ int sortbuf[64]; //2*32 + 1, +1 for value exchange
  __shared__ int numPerOctant[8];

  //Put the to be sorted values into shared memory
  const int val = input[idx];
 
  int2 histogram;     //x will contain the offset within the warp
                      //y will contain the total sum of the warp

  //Count per radix the offset and number of values
  #pragma unroll
  for(int i=0; i < 8; i++)
  {
    int2 scanRes = warpBinExclusiveScan((val == i));

    if(laneIdx == i) //Lane 0 to 8 directly write the scan sum to shared-mem
    {
      sortbuf[laneIdx*8+warpIdx] = scanRes.y;
    }
    if(val == i) histogram = scanRes; 
  }
  __syncthreads(); //Let the writes to sortbuf be complete for all 8 warps

  //Now compute the prefix sums across our warps
  //note that we have 8 values by 8 histogram values
  //store this 8x8 into 64 lanes.
  //warp0_hist0, warp1_hist0, warp2_hist0, ...warp6_hist7, warp7_hist7
  //this allows us to compute the prefix sum using two warps


  //Compute the exclusive prefix sum, using binary reduction / shuffles
  if(warpIdx == 0) //Only warp 0 does this, this elimintas _syncthreads in between
  {
    int offset1 =  sortbuf[laneIdx+WARP_SIZE*0];
    int offset2 =  sortbuf[laneIdx+WARP_SIZE*1];
    #pragma unroll
      for(int i = 0; i < 3; i++) /* log2(8) steps */
        offset1 = shfl_scan_add_step(offset1, 1 << i);

    //store the sum of the octants. Or call atomicAdd 
    if(laneIdx & 7) numPerOctant[laneIdx >> 3] = offset1;

    offset1 = shfl_scan_add_step(offset1, 1 << 3);
    offset1 = shfl_scan_add_step(offset1, 1 << 4);

    //Now we have prefix sums of first half, we need to add the final value
    //to the second half result. 
    int bcast = __shfl(offset1,31);

    offset1                     -= sortbuf[laneIdx+WARP_SIZE*0]; //Make exclusive
    sortbuf[laneIdx+WARP_SIZE*0] = offset1;

    //prefix sum on second half    
    #pragma unroll
      for(int i = 0; i < 3; i++) /* log2(8) steps */
        offset2 = shfl_scan_add_step(offset2, 1 << i);

    //store the sum of the octants. Or call atomicAdd 
    if(laneIdx & 7) numPerOctant[4+ (laneIdx >> 3)] = offset2;

    offset2 = shfl_scan_add_step(offset2, 1 << 3); 
    offset2 = shfl_scan_add_step(offset2, 1 << 4);

    offset2 -= sortbuf[laneIdx+WARP_SIZE*1]; //Make exclusive
    offset2 += bcast;
        
    sortbuf[laneIdx+WARP_SIZE*1] = offset2;
    
  }
  __syncthreads(); //Wait on warp0 to be done

  //Now each thread reads their storage location in the following way:
  //Value to read is one of the eight bins, namely the one associated to
  //the value and the is offset by the warp. This is then increased with 
  //the offset within the current warp as computed using shfl_scan_add_step 
  int storeLocation = sortbuf[val*8 + warpIdx] + histogram.x; 

  #if 1
    __shared__ int valbuf[256]; //2*32 + 1, +1 for value exchange
    //Scatter in shared-mem and then coalesced output to gmem
    valbuf[storeLocation] =  (tid << IDSHIFT) | val; //Use for CPU/GPU comparison
    //valbuf[storeLocation] =  val; //Use for production
    __syncthreads();
    if(threadIdx.x < 8) //TODO, this is just to prevent compiler optimization
      output[bid*blockDim.x+tid] = valbuf[tid]+numPerOctant[threadIdx.x];
    else
      output[bid*blockDim.x+tid] = valbuf[tid];
  #else
    //Scattered output to gmem
    output[bid*blockDim.x+storeLocation] = (tid << IDSHIFT) | val; //Use for CPU/GPU comparison
    //output[bid*blockDim.x+storeLocation] = val; //Production
  #endif
}

int main(int argc, char * argv [])
{
  const int nPerThread = 256;
  int nBlocks          = 1024;
  

  if(argc > 1)
    nBlocks = atoi(argv[1]);

  const int n = nPerThread*nBlocks;

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
	  fprintf(stdout,"Data-stats: %d\t%d\n",
			  i, histoCount[i]);

  d_input.h2d(h_input);

  //Call the sort kernel
  const int NBLOCKS  = nBlocks;
  const int NTHREADS = nPerThread; //Should be 256!!

  double t0 = rtc();
  testSortKernel_bitonicSDK<<<NBLOCKS,NTHREADS>>>(n, d_input, d_output);
  kernelSuccess("testSortKernel_bitonicSDK");
  double t1 = rtc();
  testSortKernel<<<NBLOCKS,NTHREADS>>>(n, d_input, d_output);
  kernelSuccess("testSortKernel");
  double t2 = rtc();
  testSortKernelAtomic<<<NBLOCKS,NTHREADS>>>(n, d_input, d_output);
  kernelSuccess("testSortKernelAtomic");
  double t2b = rtc();
  testSortKernel1WarpPrefix<<<NBLOCKS,NTHREADS>>>(n, d_input, d_output);
  kernelSuccess("testSortKernel1WarpPrefix");
  double t2c = rtc();
  testSortKernel1WarpPrefixIncludingSum<<<NBLOCKS,NTHREADS>>>(n, d_input, d_output);
  kernelSuccess("testSortKernel1WarpPrefixIncludingSum");
  double t2d = rtc();


  d_output.d2h(h_output);


  //Compute result on the host
  double t3 = rtc();
  for(int i=0; i < n; i+=256)
  {
    hostSortTest(h_input+i, h_check+i, 256);
  }
  double t4 = rtc();

  const int printStride = 1;
  int matchCount = 0;
  int matchValCount = 0;
  int matchIDCount  = 0;
  for(int i=0; i < n; i+=printStride)
  {
    if(i < n)
    {
      //Extract id and value
      uint id  = h_output[i] >> IDSHIFT;
      int val  = h_output[i] & VALMASK;

      uint hid  = h_check[i] >> IDSHIFT;
      int hval  = h_check[i] & VALMASK;

      int match_id = 0;
      int match_val= 0;
      if(id == hid)
        match_id = 1;
      if(val == hval)
        match_val = 1;
    
      matchValCount += match_val;
      matchIDCount  += match_id;
      if(match_id && match_val) matchCount++;

   
//       if(match_id == 0 || match_val == 0)
//         fprintf(stderr, "Index: %d Error GPU: (%d, %d)\tCPU: (%d, %d)  Match-ID: %d  Match-val: %d \n",
//                         i, val, id, hval, hid, match_id, match_val);
  
    }
  }

  if(matchValCount == n)
    fprintf(stdout, "SUCCESS \n");
  else
    fprintf(stdout, "FAILED  \n");

  fprintf(stdout,"Total items: %d  Match-full: %d Match-val: %d Match-id: %d \n", 
                  n, matchCount, matchValCount, matchIDCount);
  fprintf(stdout,"Time host: %lg  Time bitonic: %lg   Time-radix: %lg Time-radix atomic: %lg Time-radix 1warp: %lg\n", 
                  t4-t3, t1-t0, t2-t1, t2b-t2, t2c-t2b);
  fprintf(stdout,"Time-radix: %lg %f MPtcl/s\n", 
                  t2-t1,  ((1/(t2-t1))*n)/1000000);
  fprintf(stdout,"Time-radix-Atomic: %lg %f MPtcl/s\n", 
                  t2b-t2,  ((1/(t2b-t2))*n)/1000000);
  fprintf(stdout,"Time-radix-one warp: %lg %f MPtcl/s\n", 
                  t2c-t2b,  ((1/(t2c-t2b))*n)/1000000);
  fprintf(stdout,"Time-radix-one warp + sum: %lg %f MPtcl/s\n", 
                  t2d-t2c,  ((1/(t2d-t2c))*n)/1000000);
}
