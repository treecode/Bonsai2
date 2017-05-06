#pragma once
  


template<typename real_t>
static __device__ __forceinline__ real_t shfl_xor(const real_t x, const int lane, const int warpSize = WARP_SIZE);

  template<>
 __device__ __forceinline__ double shfl_xor<double>(const double x, const int lane, const int warpSize)
{
  return __hiloint2double(
      __shfl_xor(__double2hiint(x), lane, warpSize),
      __shfl_xor(__double2loint(x), lane, warpSize));
}
  template<>
 __device__ __forceinline__ float shfl_xor<float>(const float x, const int lane, const int warpSize)
{
  return __shfl_xor(x, lane, warpSize);
}

/*********************/

  template<typename Tex, typename T>
static void bindTexture(Tex &tex, const T *ptr, const int size)
{
  tex.addressMode[0] = cudaAddressModeWrap;
  tex.addressMode[1] = cudaAddressModeWrap;
  tex.filterMode     = cudaFilterModePoint;
  tex.normalized     = false;
  CUDA_SAFE_CALL(cudaBindTexture(0, tex, ptr, size*sizeof(T)));
}

  template<typename Tex>
static void unbindTexture(Tex &tex)
{
  CUDA_SAFE_CALL(cudaUnbindTexture(tex));
}

/*********************/

static __forceinline__ __device__ double atomicAdd_double(double *address, const double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do
  {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
        __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

/**************************/

template<typename real_t>
  static __device__ __forceinline__ 
void addBoxSize(typename vec<3,real_t>::type &_rmin, typename vec<3,real_t>::type &_rmax, const Position<real_t> pos)
{
  typename vec<3,real_t>::type rmin = {pos.x, pos.y, pos.z};
  typename vec<3,real_t>::type rmax = rmin;

#pragma unroll
  for (int i = WARP_SIZE2-1; i >= 0; i--)
  {
    rmin.x = min(rmin.x, shfl_xor(rmin.x, 1<<i, WARP_SIZE));
    rmax.x = max(rmax.x, shfl_xor(rmax.x, 1<<i, WARP_SIZE));

    rmin.y = min(rmin.y, shfl_xor(rmin.y, 1<<i, WARP_SIZE));
    rmax.y = max(rmax.y, shfl_xor(rmax.y, 1<<i, WARP_SIZE));

    rmin.z = min(rmin.z, shfl_xor(rmin.z, 1<<i, WARP_SIZE));
    rmax.z = max(rmax.z, shfl_xor(rmax.z, 1<<i, WARP_SIZE));
  }

  _rmin.x = min(_rmin.x, rmin.x);
  _rmin.y = min(_rmin.y, rmin.y);
  _rmin.z = min(_rmin.z, rmin.z);

  _rmax.x = max(_rmax.x, rmax.x);
  _rmax.y = max(_rmax.y, rmax.y);
  _rmax.z = max(_rmax.z, rmax.z);
}

/************ scan **********/

static __device__ __forceinline__ int lanemask_lt()
{
  int mask;
  asm("mov.u32 %0, %lanemask_lt;" : "=r" (mask));
  return mask;
}
static __device__ __forceinline__ uint shfl_scan_add_step(uint partial, uint up_offset)
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
  template <const int levels>
static __device__ __forceinline__ uint inclusive_scan_warp(const int sum)
{
  uint mysum = sum;
#pragma unroll
  for(int i = 0; i < levels; ++i)
    mysum = shfl_scan_add_step(mysum, 1 << i);
  return mysum;
}

static __device__ __forceinline__ int2 warpIntExclusiveScan(const int value)
{
  const int sum = inclusive_scan_warp<WARP_SIZE2>(value);
  return make_int2(sum-value, __shfl(sum, WARP_SIZE-1, WARP_SIZE));
}

/************** binary scan ***********/

static __device__ __forceinline__ int warpBinExclusiveScan1(const bool p)
{
  const unsigned int b = __ballot(p);
  return __popc(b & lanemask_lt());
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

/******************* segscan *******/

static __device__ __forceinline__ int lanemask_le()
{
  int mask;
  asm("mov.u32 %0, %lanemask_le;" : "=r" (mask));
  return mask;
}
static __device__ __forceinline__ int ShflSegScanStepB(
    int partial,
    uint distance,
    uint up_offset)
{
  asm(
      "{.reg .u32 r0;"
      ".reg .pred p;"
      "shfl.up.b32 r0, %1, %2, 0;"
      "setp.le.u32 p, %2, %3;"
      "@p add.u32 %1, r0, %1;"
      "mov.u32 %0, %1;}"
      : "=r"(partial) : "r"(partial), "r"(up_offset), "r"(distance));
  return partial;
}
  template<const int SIZE2>
static __device__ __forceinline__ int inclusive_segscan_warp_step(int value, const int distance)
{
  for (int i = 0; i < SIZE2; i++)
    value = ShflSegScanStepB(value, distance, 1<<i);
  return value;
}
static __device__ __forceinline__ int2 inclusive_segscan_warp(
    const int packed_value, const int carryValue)
{
  const int  flag = packed_value < 0;
  const int  mask = -flag;
  const int value = (~mask & packed_value) + (mask & (-1-packed_value));

  const int flags = __ballot(flag);

  const int dist_block = __clz(__brev(flags));

  const int laneIdx = threadIdx.x & (WARP_SIZE - 1);
  const int distance = __clz(flags & lanemask_le()) + laneIdx - 31;
  const int val = inclusive_segscan_warp_step<WARP_SIZE2>(value, min(distance, laneIdx)) +
    (carryValue & (-(laneIdx < dist_block)));
  return make_int2(val, __shfl(val, WARP_SIZE-1, WARP_SIZE));
}


