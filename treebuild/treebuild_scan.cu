#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>
#include "rtc.h"
#include "plummer.h"
#include "cudamem.h"

#define __out 

#ifndef NWARPS_OCTREE2
#define NWARPS_OCTREE2 3
#endif

#define NWARPS2 NWARPS_OCTREE2
#define NWARPS  (1<<NWARPS2)

#define WARP_SIZE2 5
#define WARP_SIZE  32

void kernelSuccess(const char kernel[] = "kernel")
{
  const int ret = (cudaDeviceSynchronize() != cudaSuccess);
  if (ret)
  {
    fprintf(stderr, "%s launch failed: %s\n", kernel, cudaGetErrorString(cudaGetLastError()));
    assert(0);
  }
}

template<int N, typename T> struct vec;
template<> struct vec<4,float>  { typedef float4  type; };
template<> struct vec<4,double> { typedef double4 type; };

template<typename T> struct int_type;
template<> struct int_type<float>  { typedef int       type; };
template<> struct int_type<double> { typedef long long type; };


struct CellData
{
  private:
    enum {NLEAF_SHIFT = 29};
    enum {NLEAF_MASK  = (0x1FU << NLEAF_SHIFT)};
    uint4 packed_data;
  public:
    __device__ CellData(
        const unsigned int parentCell,
        const unsigned int nBeg,
        const unsigned int nEnd,
        const unsigned int first = 0xFFFFFFFF,
        const unsigned int n = 0xFFFFFFFF)
    {
      int packed_firstleaf_n = 0xFFFFFFFF;
      if (n != 0xFFFFFFFF)
        packed_firstleaf_n = first | ((unsigned int)n << NLEAF_SHIFT);
      packed_data = make_uint4(parentCell, packed_firstleaf_n, nBeg, nEnd);
    }

    __device__ int n()      const {return packed_data.y >> NLEAF_SHIFT;}
    __device__ int first()  const {return packed_data.y  & NLEAF_MASK;}
    __device__ int parent() const {return packed_data.x;}
    __device__ int pbeg()   const {return packed_data.z;}
    __device__ int pend()   const {return packed_data.w;}

    __device__ bool isLeaf() const {return packed_data.y == 0xFFFFFFFF;}
    __device__ bool isNode() const {return !isLeaf();}
};

template<typename T> 
struct Particle4
{
  typedef typename int_type<T>::type intx;
  private:
#if 0  /* fails with CUDA 5.0 */
  union
  {
    typename vec<4,T>::type packed_data;
    struct {double _x,_y,_z; intx _id;};
  };
#else
  typename vec<4,T>::type packed_data;
#endif
  public:

  __host__ __device__ T x   ()  const { return packed_data.x;}
  __host__ __device__ T y   ()  const { return packed_data.y;}
  __host__ __device__ T z   ()  const { return packed_data.z;}
  __host__ __device__ T mass()  const { return packed_data.w;}
//  __host__ __device__ intx id() const { return _id; }
  __forceinline__ __device__ int get_idx() const;
  __forceinline__ __device__ int set_idx(const int i);
  __forceinline__ __device__ int get_oct() const;
  __forceinline__ __device__ int set_oct(const int oct);

  __host__ __device__ T& x    () { return packed_data.x;}
  __host__ __device__ T& y    () { return packed_data.y;}
  __host__ __device__ T& z    () { return packed_data.z;}
  __host__ __device__ T& mass () { return packed_data.w;}
//  __host__ __device__ intx& id() { return _id; }
};

__device__ unsigned int nnodes = 0;
__device__ unsigned int nleaves = 0;
__device__ unsigned int nlevels = 0;
__device__ unsigned int nbodies_leaf = 0;
__device__ unsigned int ncells = 0;


__device__   int *memPool;
__device__   CellData *cellDataList;
__device__   Particle4<float> *ptclVel;
__constant__ int d_node_max;
__constant__ int d_cell_max;
__device__ unsigned long long io_words;


template<> __device__ __forceinline__ int Particle4<float>::get_idx() const
{
  return (__float_as_int(packed_data.w) >> 4) & 0xF0000000;
}
template<> __device__ __forceinline__ int Particle4<float>::get_oct() const
{
  return __float_as_int(packed_data.w) & 0xF;
}
template<> __device__ __forceinline__ int Particle4<float>::set_idx(const int idx)
{
  const int oct = get_oct();
  packed_data.w = __int_as_float((idx << 4) | oct);
  return idx;
}
template<> __device__ __forceinline__ int Particle4<float>::set_oct(const int oct)
{
  const int idx = get_idx();
  packed_data.w = __int_as_float((idx << 4) | oct);
  return oct;
}

template<typename T>
struct Position
{
  T x, y, z;
  __host__ __device__ Position() {}
  __host__ __device__ Position(const T _x) : x(_x), y(_x), z(_x) {}

  __host__ __device__ Position(const T _x, const T _y, const T _z) : x(_x), y(_y), z(_z) {}
  static __host__ __device__ Position min(const Position &lhs, const Position &rhs) 
  {
    return Position( 
        fmin(lhs.x, rhs.x),
        fmin(lhs.y, rhs.y),
        fmin(lhs.z, rhs.z));
  }
  static __host__ __device__ Position max(const Position &lhs, const Position &rhs) 
  {
    return Position( 
        fmax(lhs.x, rhs.x),
        fmax(lhs.y, rhs.y),
        fmax(lhs.z, rhs.z));
  }
  __forceinline__ __device__ void shfl(const Position &p, const int i);
};

  template<typename T>
static __forceinline__ __device__ T myshfl(const T var0, T var, const int srcLane)
{
  var = __shfl(var, srcLane, WARP_SIZE);
  return srcLane < WARP_SIZE ? var : var0;
}

  template<>
__forceinline__ __device__ void Position<float>::shfl(const Position<float> &p, const int i)
{
  x = myshfl(x, p.x, i);
  y = myshfl(y, p.y, i);
  z = myshfl(z, p.z, i);
}


  template<typename T>
static __forceinline__ __device__ Position<T> get_volatile(const volatile Position<T>  &v)
{
  return Position<T>(v.x, v.y, v.z);
};

template<typename T>
struct BoundingBox
{
  Position<T> min, max;
  __device__ BoundingBox() {}
  __device__ BoundingBox(const Position<T> &_min, const Position<T> &_max) : min(_min), max(_max) {}
  __device__ Position<T> centre() const {return Position<T>(T(0.5)*(max.x + min.x), T(0.5)*(max.y + min.y), T(0.5)*(max.z + min.z)); }
  __device__ Position<T>  hsize() const {return Position<T>(T(0.5)*(max.x - min.x), T(0.5)*(max.y - min.y), T(0.5)*(max.z - min.z)); }
};
template<typename T>
struct Box
{
  Position<T> centre;
  T hsize;
  __device__ Box() {}
  __device__ Box(const Position<T> &c, T hs) : centre(c), hsize(hs) {}
};

  template<typename T>
static __device__ __forceinline__ int Octant(const Position<T> &lhs, const Position<T> &rhs)
{
  return 
    ((lhs.x <= rhs.x) << 0) +
    ((lhs.y <= rhs.y) << 1) +
    ((lhs.z <= rhs.z) << 2);
};

  template<typename T>
static __device__ __forceinline__ Box<T> ChildBox(const Box<T> &box, const int oct)
{
  const T s = T(0.5) * box.hsize;
  return Box<T>(Position<T>(
        box.centre.x + s * ((oct&1) ? T(1.0) : T(-1.0)),
        box.centre.y + s * ((oct&2) ? T(1.0) : T(-1.0)),
        box.centre.z + s * ((oct&4) ? T(1.0) : T(-1.0))
        ), 
      s);
}

static __forceinline__ __device__ void computeGridAndBlockSize(dim3 &grid, dim3 &block, const int np)
{
  const int NTHREADS = (1<<NWARPS_OCTREE2) * WARP_SIZE;
  block = dim3(NTHREADS);
  assert(np > 0);
  grid = dim3(min(max(np/(NTHREADS*4),1), 512));
}

static __device__ __forceinline__ int lanemask_lt()
{
  int mask;
  asm("mov.u32 %0, %lanemask_lt;" : "=r" (mask));
  return mask;
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

static __device__ __forceinline__ int2 warpBinExclusiveScan(const bool p)
{
  const unsigned int b = __ballot(p);
  return make_int2(__popc(b & lanemask_lt()), __popc(b));
}
static __device__ __forceinline__ int warpBinExclusiveScan1(const bool p)
{
  const unsigned int b = __ballot(p);
  return __popc(b & lanemask_lt());
}
static __device__ __forceinline__ int warpBinReduce(const bool p)
{
  const unsigned int b = __ballot(p);
  return __popc(b);
}


template<int NLEAF, typename T>
static __global__ void buildOctantSingle(
    Box<T> box,
    const int cellParentIndex,
    const int cellIndexBase,
    const int octantMask,
    __out int *octCounterBase,
    Particle4<T> *ptcl,
    Particle4<T> *buff,
    const int level)
{
  typedef typename vec<4,T>::type T4;
  const int laneIdx = threadIdx.x & (WARP_SIZE-1);
  const int warpIdx = threadIdx.x >> WARP_SIZE2;

  const int octant2process = (octantMask >> (3*blockIdx.y)) & 0x7;

  int *octCounter = octCounterBase + blockIdx.y*(8+8+8+64+8);

  const int data  = octCounter[laneIdx];
  const int nCell = __shfl(data, 8+warpIdx, WARP_SIZE);
  const int nBeg  = __shfl(data, 1, WARP_SIZE);
  const int nEnd  = __shfl(data, 2, WARP_SIZE);

  int cellCounter = __shfl(data, 8+8+warpIdx, WARP_SIZE);

  /* each of the 8 warps are responsible for each of the octant */
  box = ChildBox(box, octant2process);
  const Box<T> childBox = ChildBox(box, warpIdx);

  /* counter in shmem for each of the octant */
  int nChildren[8] = {0};

  __shared__ Particle4<T> dataX[8*WARP_SIZE];

  /* process particle array */
  Particle4<T> p4list = ptcl[min(nBeg + threadIdx.x, nEnd-1)];

  {
    dataX[threadIdx.x] = p4list;
    __syncthreads(); 

#pragma unroll
    for (int k = 0; k < 8; k++)  /* process particles in shared memory */
    {
      if (nBeg + (k<<WARP_SIZE2) >= nEnd) break;
      const int locid = (k<<WARP_SIZE2) + laneIdx;
      const int  addr = nBeg + locid;
      const bool mask = addr < nEnd;

      Particle4<T> p4 = dataX[locid];

      /* use prefix sums to compute offset to where scatter particles */
      const int     use = mask && (p4.get_oct() == warpIdx);
      const int2 offset = warpBinExclusiveScan(use);  /* x is write offset, y is element count */

      if (offset.y > 0)
      {
        const int addrB = cellCounter;
        cellCounter += offset.y;

        int subOctant = -1;
        const int  p4subOctant = Octant(childBox.centre, Position<T>(p4.x(),p4.y(),p4.z()));
        p4.set_oct(p4subOctant);
        if (use)
        {
          subOctant = p4subOctant;
          buff[addrB+offset.x] = p4;         /* float4 vector stores   */
        }

#pragma unroll
        for (int k = 0; k < 8; k++)
          nChildren[k] += warpBinReduce(subOctant == k);
      }
    }

    __syncthreads();
  }
  /* done processing particles, store counts atomically in gmem */

  int nSubCell;
#pragma unroll
  for (int k = 0; k < 8; k++)
    if (laneIdx == k)
      nSubCell = nChildren[k];

  /* last block finished, analysis of the data and schedule new kernel for children */

  __syncthreads();  /* must be present, otherwise race conditions occurs between parent & children */

  int *shmem = (int*)dataX; 
  if (warpIdx == 0)
    shmem[laneIdx] = 0;

  __syncthreads();

  if (threadIdx.x == 0)
    atomicCAS(&nlevels, level, level+1);

  const int nEnd1 = cellCounter;
  const int nBeg1 = nEnd1 - nCell;

  if (laneIdx == 0)
    shmem[warpIdx] = nCell;
  __syncthreads();

  const int npCell = laneIdx < 8 ? shmem[laneIdx] : 0;

  /* compute number of children that needs to be further split, and cmopute their offsets */
  const int2 nSubNodes = warpBinExclusiveScan(npCell > NLEAF);
  const int2 nLeaves   = warpBinExclusiveScan(npCell > 0 && npCell <= NLEAF);
  if (warpIdx == 0 && laneIdx < 8)
  {
    shmem[8 +laneIdx] = nSubNodes.x;
    shmem[16+laneIdx] = nLeaves.x;
  }

  int nCellmax = npCell;
#pragma unroll
  for (int i = 2; i >= 0; i--)
    nCellmax = max(nCellmax, __shfl_xor(nCellmax, 1<<i, WARP_SIZE));

  /* if there is at least one cell to split, increment nuumber of the nodes */
  if (threadIdx.x == 0 && nSubNodes.y > 0)
  {
    shmem[16+8] = atomicAdd(&nnodes,nSubNodes.y);
#if 0   /* temp solution, a better one is to use RingBuffer */
    assert(shmem[16+8] < d_node_max);
#endif
  }

  /* writing linking info, parent, child and particle's list */
  const int nChildrenCell = warpBinReduce(npCell > 0);
  if (threadIdx.x == 0 && nChildrenCell > 0)
  {
    const int cellFirstChildIndex = atomicAdd(&ncells, nChildrenCell);
    /*** keep in mind, the 0-level will be overwritten ***/
    const CellData cellData(cellParentIndex, nBeg, nEnd, cellFirstChildIndex, nChildrenCell);
    cellDataList[cellIndexBase + blockIdx.y] = cellData;
    shmem[16+9] = cellFirstChildIndex;
  }

  __syncthreads();
  const int cellFirstChildIndex = shmem[16+9];
  /* compute atomic data offset for cell that need to be split */
  const int next_node = shmem[16+8];
  int *octCounterNbase = &memPool[next_node*(8+8+8+64+8)];

  const int nodeOffset = shmem[8 +warpIdx];
  const int leafOffset = shmem[16+warpIdx];

  /* if cell needs to be split, populate it shared atomic data */
  if (nCell > NLEAF)
  {
    int *octCounterN = octCounterNbase + nodeOffset*(8+8+8+64+8);

    /* number of particles in each cell's subcells */
    //    const int nSubCell = laneIdx < 8 ? octCounter[8+16+warpIdx*8 + laneIdx] : 0;

    /* compute offsets */
    int cellOffset = nSubCell;
#pragma unroll
    for(int i = 0; i < 3; i++)  /* log2(8) steps */
      cellOffset = shfl_scan_add_step(cellOffset, 1 << i);
    cellOffset -= nSubCell;

    /* store offset in memory */

    cellOffset = __shfl_up(cellOffset, 8, WARP_SIZE);
    if (laneIdx < 8) cellOffset = nSubCell;
    else            cellOffset += nBeg1;
    cellOffset = __shfl_up(cellOffset, 8, WARP_SIZE);

    if (laneIdx <  8) cellOffset = 0;
    if (laneIdx == 1) cellOffset = nBeg1;
    if (laneIdx == 2) cellOffset = nEnd1;

    if (laneIdx < 24)
      octCounterN[laneIdx] = cellOffset;
  }

  /***************************/
  /*  launch  child  kernel  */
  /***************************/

  /* warps coorperate so that only 1 kernel needs to be launched by a thread block
   * with larger degree of paralellism */
  if (nSubNodes.y > 0 && warpIdx == 0)
  {
    /* build octant mask */
    int octant_mask = npCell > NLEAF ?  (laneIdx << (3*nSubNodes.x)) : 0;
#pragma unroll
    for (int i = 4; i >= 0; i--)
      octant_mask |= __shfl_xor(octant_mask, 1<<i, WARP_SIZE);

    if (threadIdx.x == 0)
    {
      dim3 grid, block;
      computeGridAndBlockSize(grid, block, nCellmax);
      cudaStream_t stream;
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

      grid.y = nSubNodes.y;  /* each y-coordinate of the grid will be busy for each parent cell */
      grid.x = 1;
      buildOctantSingle<NLEAF,T><<<grid,block,0,stream>>>
        (box, cellIndexBase+blockIdx.y, cellFirstChildIndex,
         octant_mask, octCounterNbase, buff, ptcl, level+1);
    }
  }

  /******************/
  /* process leaves */
  /******************/

  if (nCell <= NLEAF && nCell > 0)
  {
    if (laneIdx == 0)
    {
      assert(nEnd1 - nBeg1 <= NLEAF);
      atomicAdd(&nleaves,1);
      atomicAdd(&nbodies_leaf, nEnd1-nBeg1);
      const CellData leafData(cellIndexBase+blockIdx.y, nBeg, nEnd1);
      cellDataList[cellFirstChildIndex + nSubNodes.y + leafOffset] = leafData;
    }
    if ((level&1) == 0)
    {
      for (int i = nBeg1+laneIdx; i < nEnd1; i += WARP_SIZE)
        if (i < nEnd1)
        {
          Particle4<T> pos = buff[i];
          Particle4<T> vel = ptclVel[pos.get_idx()];
#ifdef PSHFL_SANITY_CHECK
          pos.mass() = T(pos.get_idx());
#else
          pos.mass() = vel.mass();
#endif
          ptcl[i] = pos;
          buff[i] = vel;
        }
    }
    else
    {
      for (int i = nBeg1+laneIdx; i < nEnd1; i += WARP_SIZE)
        if (i < nEnd1)
        {
          Particle4<T> pos = buff[i];
          Particle4<T> vel = ptclVel[pos.get_idx()];
#ifdef PSHFL_SANITY_CHECK
          pos.mass() = T(pos.get_idx());
#else
          pos.mass() = vel.mass();
#endif
          buff[i] = pos;
          ptcl[i] = vel;
        }
    }

  }

}

/****** this is the main functions that build the tree recursively *******/

template<int NLEAF, typename T, bool STOREIDX>
static __global__ void buildOctant(
    Box<T> box,
    const int cellParentIndex,
    const int cellIndexBase,
    const int octantMask,
    __out int *octCounterBase,
    Particle4<T> *ptcl,
    Particle4<T> *buff,
    const int level = 0)
{
  typedef typename vec<4,T>::type T4;
  /* compute laneIdx & warpIdx for each of the threads:
   *   the thread block contains only 8 warps
   *   a warp is responsible for a single octant of the cell 
   */   
  const int laneIdx = threadIdx.x & (WARP_SIZE-1);
  const int warpIdx = threadIdx.x >> WARP_SIZE2;

  /* We launch a 2D grid:
   *   the y-corrdinate carries info about which parent cell to process
   *   the x-coordinate is just a standard approach for CUDA parallelism 
   */
  const int octant2process = (octantMask >> (3*blockIdx.y)) & 0x7;

  /* get the pointer to atomic data that for a given octant */
  int *octCounter = octCounterBase + blockIdx.y*(8+8+8+64+8);

  /* read data about the current cell */
  const int data  = octCounter[laneIdx];
  const int nBeg  = __shfl(data, 1, WARP_SIZE);
  const int nEnd  = __shfl(data, 2, WARP_SIZE);
  /* if we are not at the root level, compute the geometric box
   * of the cell */
  if (!STOREIDX)
    box = ChildBox(box, octant2process);


  /* countes number of particles in each octant of a child octant */
  __shared__ int nShChildrenFine[NWARPS][9][8];
  __shared__ int nShChildren[8][8];

  Box<T> *shChildBox = (Box<T>*)&nShChildren[0][0];

  int *shdata = (int*)&nShChildrenFine[0][0][0];
#pragma unroll 
  for (int i = 0; i < 9*9*NWARPS; i += NWARPS*WARP_SIZE)
    if (i + threadIdx.x < 8*9*NWARPS)
      shdata[i + threadIdx.x] = 0;

  if (laneIdx == 0 && warpIdx < 8)
    shChildBox[warpIdx] = ChildBox(box, warpIdx);

  __syncthreads();

  /* process particle array */
  const int nBeg_block = nBeg + blockIdx.x * blockDim.x;
  for (int i = nBeg_block; i < nEnd; i += gridDim.x * blockDim.x)
  {
    Particle4<T> p4 = ptcl[min(i+threadIdx.x, nEnd-1)];

    int p4octant = p4.get_oct();
    if (STOREIDX)
    {
      p4.set_idx(i + threadIdx.x);
      p4octant = Octant(box.centre, Position<T>(p4.x(), p4.y(), p4.z()));
    }

    p4octant = i+threadIdx.x < nEnd ? p4octant : 0xF; 

    if (p4octant < 8)
    {
      const int p4subOctant = Octant(shChildBox[p4octant].centre, Position<T>(p4.x(), p4.y(), p4.z()));
      p4.set_oct(p4subOctant);
    }

    /* compute number of particles to write in each octant */
    int np = 0;
#pragma unroll
    for (int octant = 0; octant < 8; octant++)
    {
      const int sum = warpBinReduce(p4octant == octant);
      if (octant == laneIdx)
        np = sum;
    }

    /* accumulate atomic counters with number of particles to be written into an octant */
    int addrB0;
    if (np > 0)
      addrB0 = atomicAdd(&octCounter[8+8+laneIdx], np);

    /* compute offset for each particle */ 
    int cntr = 32; 
    int addrW = -1;
#pragma unroll
    for (int octant = 0; octant < 8; octant++)
    {
      const int sum = warpBinReduce(p4octant == octant);
      if (sum > 0)
      {
        const int offset = warpBinExclusiveScan1(p4octant == octant);
        const int addrB = __shfl(addrB0, octant, WARP_SIZE);
        if (p4octant == octant)
          addrW = addrB + offset;
        cntr -= sum;
        if (cntr == 0) break;
      }
    }

    /* writes sorted particels into coalesced manner */
    if (addrW >= 0)
      buff[addrW] = p4;

    /* now count number of particle in suboctants of each octant */

#if 0  

    if (p4octant < 8)
      atomicAdd(&nShChildrenFine[warpIdx][p4octant][p4.get_oct()],1);

#else  /* does the same thing but without atomics , however there is too much work.. */
    /* question, how to further optimize this functionality */

    cntr = 32;
#pragma unroll
    for (int octant = 0; octant < 8; octant++)
    {
      if (cntr == 0) break;

      const int sum = warpBinReduce(p4octant == octant);
      if (sum > 0)
      {
        const int subOctant = p4octant == octant ? p4.get_oct() : -1;
#pragma unroll
        for (int k = 0; k < 8; k += 4)
        {
          const int4 sum = make_int4(
              warpBinReduce(k   == subOctant),
              warpBinReduce(k+1 == subOctant),
              warpBinReduce(k+2 == subOctant),
              warpBinReduce(k+3 == subOctant));
          if (laneIdx == 0) 
          {
            int4 value = *(int4*)&nShChildrenFine[warpIdx][octant][k];
            value.x += sum.x;
            value.y += sum.y;
            value.z += sum.z;
            value.w += sum.w;
            *(int4*)&nShChildrenFine[warpIdx][octant][k] = value;
          }
        }
        cntr -= sum;
      }
    }
#endif
  }

  if (warpIdx >= 8) return;

  __syncthreads();

  if (laneIdx < NWARPS && warpIdx < 8)
#pragma unroll
    for (int k = 0; k < 8; k += 4)
    {
      int4 nSubOctant = *(int4*)&nShChildrenFine[laneIdx][warpIdx][k];
#pragma unroll
      for (int i = NWARPS2-1; i >= 0; i--)
      {
        nSubOctant.x += __shfl_xor(nSubOctant.x, 1<<i, NWARPS);
        nSubOctant.y += __shfl_xor(nSubOctant.y, 1<<i, NWARPS);
        nSubOctant.z += __shfl_xor(nSubOctant.z, 1<<i, NWARPS);
        nSubOctant.w += __shfl_xor(nSubOctant.w, 1<<i, NWARPS);
      }
      if (laneIdx == 0)
        *(int4*)&nShChildren[warpIdx][k] = nSubOctant;
    }

  __syncthreads();

  if (laneIdx < 8 && warpIdx < 8)
    if (nShChildren[warpIdx][laneIdx] > 0)
      atomicAdd(&octCounter[8+16+warpIdx*8 + laneIdx], nShChildren[warpIdx][laneIdx]);

  __syncthreads();  /* must be present, otherwise race conditions occurs between parent & children */


  /* detect last thread block for unique y-coordinate of the grid:
   * mind, this cannot be done on the host, because we don't detect last 
   * block on the grid, but instead the last x-block for each of the y-coordainte of the grid
   * this should increase the degree of parallelism
   */

  int *shmem = &nShChildren[0][0];
  if (warpIdx == 0)
    shmem[laneIdx] = 0;

  int &lastBlock = shmem[0];
  if (threadIdx.x == 0)
  {
    const int ticket = atomicAdd(octCounter, 1);
    lastBlock = (ticket == gridDim.x-1);
  }
  __syncthreads();

  if (!lastBlock) return;

  __syncthreads();

  /* okay, we are in the last thread block, do the analysis and decide what to do next */

  if (warpIdx == 0)
    shmem[laneIdx] = 0;

  if (threadIdx.x == 0)
    atomicCAS(&nlevels, level, level+1);

  __syncthreads();

  /* compute beginning and then end addresses of the sorted particles  in the child cell */

  const int nCell = __shfl(data, 8+warpIdx, WARP_SIZE);
  const int nEnd1 = octCounter[8+8+warpIdx];
  const int nBeg1 = nEnd1 - nCell;

  if (laneIdx == 0)
    shmem[warpIdx] = nCell;
  __syncthreads();

  const int npCell = laneIdx < 8 ? shmem[laneIdx] : 0;

  /* compute number of children that needs to be further split, and cmopute their offsets */
  const int2 nSubNodes = warpBinExclusiveScan(npCell > NLEAF);
  const int2 nLeaves   = warpBinExclusiveScan(npCell > 0 && npCell <= NLEAF);
  if (warpIdx == 0 && laneIdx < 8)
  {
    shmem[8 +laneIdx] = nSubNodes.x;
    shmem[16+laneIdx] = nLeaves.x;
  }

  int nCellmax = npCell;
#pragma unroll
  for (int i = 2; i >= 0; i--)
    nCellmax = max(nCellmax, __shfl_xor(nCellmax, 1<<i, WARP_SIZE));

  /* if there is at least one cell to split, increment nuumber of the nodes */
  if (threadIdx.x == 0 && nSubNodes.y > 0)
  {
    shmem[16+8] = atomicAdd(&nnodes,nSubNodes.y);
#if 0   /* temp solution, a better one is to use RingBuffer */
    assert(shmem[16+8] < d_node_max);
#endif
  }

  /* writing linking info, parent, child and particle's list */
  const int nChildrenCell = warpBinReduce(npCell > 0);
  if (threadIdx.x == 0 && nChildrenCell > 0)
  {
    const int cellFirstChildIndex = atomicAdd(&ncells, nChildrenCell);
    /*** keep in mind, the 0-level will be overwritten ***/
    const CellData cellData(cellParentIndex, nBeg, nEnd, cellFirstChildIndex, nChildrenCell);
    cellDataList[cellIndexBase + blockIdx.y] = cellData;
    shmem[16+9] = cellFirstChildIndex;
  }

  __syncthreads();
  const int cellFirstChildIndex = shmem[16+9];
  /* compute atomic data offset for cell that need to be split */
  const int next_node = shmem[16+8];
  int *octCounterNbase = &memPool[next_node*(8+8+8+64+8)];

  const int nodeOffset = shmem[8 +warpIdx];
  const int leafOffset = shmem[16+warpIdx];

  /* if cell needs to be split, populate it shared atomic data */
  if (nCell > NLEAF)
  {
    int *octCounterN = octCounterNbase + nodeOffset*(8+8+8+64+8);

    /* number of particles in each cell's subcells */
    const int nSubCell = laneIdx < 8 ? octCounter[8+16+warpIdx*8 + laneIdx] : 0;

    /* compute offsets */
    int cellOffset = nSubCell;
#pragma unroll
    for(int i = 0; i < 3; i++)  /* log2(8) steps */
      cellOffset = shfl_scan_add_step(cellOffset, 1 << i);
    cellOffset -= nSubCell;

    /* store offset in memory */

    cellOffset = __shfl_up(cellOffset, 8, WARP_SIZE);
    if (laneIdx < 8) cellOffset = nSubCell;
    else            cellOffset += nBeg1;
    cellOffset = __shfl_up(cellOffset, 8, WARP_SIZE);

    if (laneIdx <  8) cellOffset = 0;
    if (laneIdx == 1) cellOffset = nBeg1;
    if (laneIdx == 2) cellOffset = nEnd1;

    if (laneIdx < 24)
      octCounterN[laneIdx] = cellOffset;
  }

  /***************************/
  /*  launch  child  kernel  */
  /***************************/

  /* warps coorperate so that only 1 kernel needs to be launched by a thread block
   * with larger degree of paralellism */
  if (nSubNodes.y > 0 && warpIdx == 0)
  {
    /* build octant mask */
    int octant_mask = npCell > NLEAF ?  (laneIdx << (3*nSubNodes.x)) : 0;
#pragma unroll
    for (int i = 4; i >= 0; i--)
      octant_mask |= __shfl_xor(octant_mask, 1<<i, WARP_SIZE);

    if (threadIdx.x == 0)
    {
      dim3 grid, block;
      computeGridAndBlockSize(grid, block, nCellmax);
      cudaStream_t stream;
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

      grid.y = nSubNodes.y;  /* each y-coordinate of the grid will be busy for each parent cell */
#if NWARPS==8
      if (nCellmax <= block.x)
      {
        grid.x = 1;
        buildOctantSingle<NLEAF,T><<<grid,block,0,stream>>>
          (box, cellIndexBase+blockIdx.y, cellFirstChildIndex,
           octant_mask, octCounterNbase, buff, ptcl, level+1);
      }
      else
#endif
        buildOctant<NLEAF,T,false><<<grid,block,0,stream>>>
          (box, cellIndexBase+blockIdx.y, cellFirstChildIndex,
           octant_mask, octCounterNbase, buff, ptcl, level+1);
    }
  }

  /******************/
  /* process leaves */
  /******************/

  if (nCell <= NLEAF && nCell > 0)
  {
    if (laneIdx == 0)
    {
      assert(nEnd1 - nBeg1 <= NLEAF);
      atomicAdd(&nleaves,1);
      atomicAdd(&nbodies_leaf, nEnd1-nBeg1);
      const CellData leafData(cellIndexBase+blockIdx.y, nBeg1, nEnd1);
      cellDataList[cellFirstChildIndex + nSubNodes.y + leafOffset] = leafData;
    }
    if (!(level&1))
    {
      for (int i = nBeg1+laneIdx; i < nEnd1; i += WARP_SIZE)
        if (i < nEnd1)
        {
          Particle4<T> pos = buff[i];
          Particle4<T> vel = ptclVel[pos.get_idx()];
#ifdef PSHFL_SANITY_CHECK
          pos.mass() = T(pos.get_idx());
#else
          pos.mass() = vel.mass();
#endif
          ptcl[i] = pos;
          buff[i] = vel;
        }
    }
    else
    {
      for (int i = nBeg1+laneIdx; i < nEnd1; i += WARP_SIZE)
        if (i < nEnd1)
        {
          Particle4<T> pos = buff[i];
          Particle4<T> vel = ptclVel[pos.get_idx()];
#ifdef PSHFL_SANITY_CHECK
          pos.mass() = T(pos.get_idx());
#else
          pos.mass() = vel.mass();
#endif
          buff[i] = pos;
          ptcl[i] = vel;
        }
    }
  }
}

/******* compute multipole moments ****/

  template<int NTHREADS>
static __device__ double reduceBlock(double sum)
{
  extern volatile __shared__ double sh[];
  const int tid = threadIdx.x;

  sh[tid] = sum;
  __syncthreads();

  if (NTHREADS >= 512)
  {
    if (tid < 256) sum = sh[tid] = sum + sh[tid + 256];
    __syncthreads();
  }
  if (NTHREADS >= 256)
  {
    if (tid < 128) sum = sh[tid] = sum + sh[tid + 128];
    __syncthreads();
  }
  if (NTHREADS >= 128)
  {
    if (tid < 64) sum = sh[tid] = sum + sh[tid + 64];
    __syncthreads();
  }
  if (tid < 32)
  {
    if (NTHREADS >= 64)  sum = sh[tid] = sum + sh[tid + 32];
    if (NTHREADS >= 32)  sum = sh[tid] = sum + sh[tid + 16];
    if (NTHREADS >= 16)  sum = sh[tid] = sum + sh[tid +  8];
    if (NTHREADS >=  8)  sum = sh[tid] = sum + sh[tid +  4];
    if (NTHREADS >=  4)  sum = sh[tid] = sum + sh[tid +  2];
    if (NTHREADS >=  2)  sum = sh[tid] = sum + sh[tid +  1];
  }
  __syncthreads();
  return sh[0];
};

template<int NTHREADS, typename T>
static __global__ 
void computeNodeProperties(
    const int n,
    const CellData     *cellDataList,
    const Particle4<T> *ptclPosList,
    typename vec<4,T>::type *cellCOM,
    typename vec<4,T>::type *cellQMxx_yy_zz_m,
    typename vec<4,T>::type *cellQMxy_xz_yz)
{
  typedef typename vec<4,T>::type T4;
  const int cellIdx = blockIdx.x;
  const CellData cellData = cellDataList[cellIdx];

  double4 monopoleM = {0.0, 0.0, 0.0, 0.0};
  double3 Qxx_yy_zz = {0.0, 0.0, 0.0};
  double3 Qxy_xz_yz = {0.0, 0.0, 0.0};
  for (int i = cellData.pbeg(); i < cellData.pend(); i += blockDim.x)
  {
    const bool mask = (i + threadIdx.x) < cellData.pend();
    Particle4<T> ip;
    if (mask) ip = ptclPosList[i + threadIdx.x];

    double mass = mask ? ip.mass() : 0.0; 
    double3 pos = make_double3(ip.x(), ip.y(), ip.z());

    monopoleM.x += mass * pos.x;
    monopoleM.y += mass * pos.y;
    monopoleM.z += mass * pos.z;
    monopoleM.w += mass;

    Qxx_yy_zz.x += mass * pos.x*pos.x;
    Qxx_yy_zz.y += mass * pos.y*pos.y;
    Qxx_yy_zz.z += mass * pos.z*pos.z;

    Qxy_xz_yz.x += mass * pos.x*pos.y;
    Qxy_xz_yz.y += mass * pos.x*pos.z;
    Qxy_xz_yz.z += mass * pos.y*pos.z;
  }

#if 0
  monopoleM.x = reduceBlock<NTHREADS>(monopoleM.x); __syncthreads();
  monopoleM.y = reduceBlock<NTHREADS>(monopoleM.y); __syncthreads();
  monopoleM.z = reduceBlock<NTHREADS>(monopoleM.z); __syncthreads();
  monopoleM.w = reduceBlock<NTHREADS>(monopoleM.w); __syncthreads();

  Qxx_yy_zz.x = reduceBlock<NTHREADS>(Qxx_yy_zz.x); __syncthreads();
  Qxx_yy_zz.y = reduceBlock<NTHREADS>(Qxx_yy_zz.y); __syncthreads();
  Qxx_yy_zz.z = reduceBlock<NTHREADS>(Qxx_yy_zz.z); __syncthreads();

  Qxy_xz_yz.x = reduceBlock<NTHREADS>(Qxy_xz_yz.x); __syncthreads();
  Qxy_xz_yz.y = reduceBlock<NTHREADS>(Qxy_xz_yz.y); __syncthreads();
  Qxy_xz_yz.z = reduceBlock<NTHREADS>(Qxy_xz_yz.z); __syncthreads();
#endif

  //  assert(monopoleM.w > 0.0);
  const double invMass = monopoleM.w;

  T4 icellCOM;
  icellCOM.x = T(monopoleM.x * invMass);
  icellCOM.y = T(monopoleM.y * invMass);
  icellCOM.z = T(monopoleM.z * invMass);
  icellCOM.w = -1.0;
  if (threadIdx.x == 0)
    cellCOM[cellIdx] = icellCOM;

  T4 icellQxx_yy_zz_m;
  icellQxx_yy_zz_m.x = T(Qxx_yy_zz.x);
  icellQxx_yy_zz_m.y = T(Qxx_yy_zz.y);
  icellQxx_yy_zz_m.z = T(Qxx_yy_zz.z);
  icellQxx_yy_zz_m.w = T(monopoleM.w);
  if (threadIdx.x == 0)
    cellQMxx_yy_zz_m[cellIdx] = icellQxx_yy_zz_m;

  T4 icellQxy_xz_yz;
  icellQxy_xz_yz.x = T(Qxy_xz_yz.x);
  icellQxy_xz_yz.y = T(Qxy_xz_yz.y);
  icellQxy_xz_yz.z = T(Qxy_xz_yz.z);
  icellQxy_xz_yz.w = T(0.0);
  if (threadIdx.x == 0)
    cellQMxy_xz_yz[cellIdx] = icellQxy_xz_yz;
}


/****** not very tuned kernels to do preparatory stuff ********/


template<typename T, const int NTHREADS>
static __device__ void reduceBlock(
    volatile Position<T> *shmin,
    volatile Position<T> *shmax,
    Position<T> bmin,
    Position<T> bmax)
{
  const int tid = threadIdx.x;

#define STORE {\
  shmin[tid].x = bmin.x; shmin[tid].y = bmin.y; shmin[tid].z = bmin.z; \
  shmax[tid].x = bmax.x; shmax[tid].y = bmax.y; shmax[tid].z = bmax.z; }

  STORE;
  __syncthreads();

  // do reduction in shared mem
  if (NTHREADS >= 512)
  {
    if (tid < 256)
    {
      bmin = Position<T>::min(bmin, get_volatile<T>(shmin[tid+256]));
      bmax = Position<T>::max(bmax, get_volatile<T>(shmax[tid+256]));
      STORE;
    }
    __syncthreads();
  }

  if (NTHREADS >= 256)
  {
    if (tid < 128)
    {
      bmin = Position<T>::min(bmin, get_volatile<T>(shmin[tid+128]));
      bmax = Position<T>::max(bmax, get_volatile<T>(shmax[tid+128]));
      STORE;
    }
    __syncthreads();
  }

  if (NTHREADS >= 128)
  {
    if (tid <  64)
    {
      bmin = Position<T>::min(bmin, get_volatile<T>(shmin[tid+64]));
      bmax = Position<T>::max(bmax, get_volatile<T>(shmax[tid+64]));
      STORE;
    }
    __syncthreads();
  }


  if (tid < 32)
  {
    if (NTHREADS >=  64)
    {
      bmin = Position<T>::min(bmin, get_volatile<T>(shmin[tid+32]));
      bmax = Position<T>::max(bmax, get_volatile<T>(shmax[tid+32]));
      STORE;
    }
    if (NTHREADS >=  32)
    {
      bmin = Position<T>::min(bmin, get_volatile<T>(shmin[tid+16]));
      bmax = Position<T>::max(bmax, get_volatile<T>(shmax[tid+16]));
      STORE;
    }
    if (NTHREADS >=  16)
    {
      bmin = Position<T>::min(bmin, get_volatile<T>(shmin[tid+8]));
      bmax = Position<T>::max(bmax, get_volatile<T>(shmax[tid+8]));
      STORE;
    }
    if (NTHREADS >=   8)
    {
      bmin = Position<T>::min(bmin, get_volatile<T>(shmin[tid+4]));
      bmax = Position<T>::max(bmax, get_volatile<T>(shmax[tid+4]));
      STORE;
    }
    if (NTHREADS >=   4)
    {
      bmin = Position<T>::min(bmin, get_volatile<T>(shmin[tid+2]));
      bmax = Position<T>::max(bmax, get_volatile<T>(shmax[tid+2]));
      STORE;
    }
    if (NTHREADS >=   2)
    {
      bmin = Position<T>::min(bmin, get_volatile<T>(shmin[tid+1]));
      bmax = Position<T>::max(bmax, get_volatile<T>(shmax[tid+1]));
      STORE;
    }
  }
#undef STORE

  __syncthreads();
}

template<const int NTHREADS, const int NBLOCKS, typename T>
static __global__ void computeBoundingBox(
    const int n,
    __out Position<T> *minmax_ptr,
    __out Box<T>      *box_ptr,
    __out int *retirementCount,
    const Particle4<T> *ptclPos)
{
  __shared__ Position<T> shmin[NTHREADS], shmax[NTHREADS];

  const int gridSize = NTHREADS*NBLOCKS*2;
  int i = blockIdx.x*NTHREADS*2 + threadIdx.x;

  Position<T> bmin(T(+1e10)), bmax(T(-1e10));

  while (i < n)
  {
    const Particle4<T> p = ptclPos[i];
    const Position<T> pos(p.x(), p.y(), p.z());
    bmin = Position<T>::min(bmin, pos);
    bmax = Position<T>::max(bmax, pos);
    if (i + NTHREADS < n)
    {
      const Particle4<T> p = ptclPos[i + NTHREADS];
      const Position<T> pos(p.x(), p.y(), p.z());
      bmin = Position<T>::min(bmin, pos);
      bmax = Position<T>::max(bmax, pos);
    }
    i += gridSize;
  }

  reduceBlock<T, NTHREADS>(shmin, shmax, bmin, bmax);
  if (threadIdx.x == 0) 
  {
    bmin = shmin[0];
    bmax = shmax[0];
    minmax_ptr[blockIdx.x          ] = bmin;
    minmax_ptr[blockIdx.x + NBLOCKS] = bmax;
  }

  __shared__ bool lastBlock;
  __threadfence();

  if (threadIdx.x == 0)
  {
    const int ticket = atomicInc((unsigned int*)retirementCount, NBLOCKS);
    lastBlock = (ticket == NBLOCKS - 1);
  }

  __syncthreads();

  if (lastBlock)
  {
    Position<T> bmin(T(+1e10)), bmax(T(-1e10));
    int i = threadIdx.x;
    while (i < NBLOCKS)
      if (i < NBLOCKS)
      {
        bmin = Position<T>::min(bmin, minmax_ptr[i        ]);
        bmax = Position<T>::max(bmax, minmax_ptr[i+NBLOCKS]);
        i += NTHREADS;
      };

    reduceBlock<T, NTHREADS>(shmin, shmax, bmin, bmax);
    __syncthreads();

    if (threadIdx.x == 0)
    {
      bmin = shmin[0];
      bmax = shmax[0];
      const Position<T> cvec((bmax.x+bmin.x)*T(0.5), (bmax.y+bmin.y)*T(0.5), (bmax.z+bmin.z)*T(0.5));
      const Position<T> hvec((bmax.x-bmin.x)*T(0.5), (bmax.y-bmin.y)*T(0.5), (bmax.z-bmin.z)*T(0.5));
      const T h = fmax(hvec.z, fmax(hvec.y, hvec.x));
      T hsize = T(1.0);
      while (hsize > h) hsize *= T(0.5);
      while (hsize < h) hsize *= T(2.0);
#if 0
      hsize *= T(128.0);
#endif

      const int NMAXLEVEL = 20;

      const T hquant = hsize / T(1<<NMAXLEVEL);
      const long long nx = (long long)(cvec.x/hquant);
      const long long ny = (long long)(cvec.y/hquant);
      const long long nz = (long long)(cvec.z/hquant);

      const Position<T> centre(hquant * T(nx), hquant * T(ny), hquant * T(nz));

      *box_ptr = Box<T>(centre, hsize);
    }
  }
}

  template<int NTHREADS, int NBLOCKS, typename T>
static __global__ void computeDomainSize(const int n, const Particle4<T> *ptclPos, Box<T> *domain)
{
  Position<T> *minmax_ptr = new Position<T>[2*NBLOCKS];
  int *retirementCount = new int;
  *retirementCount = 0;
  computeBoundingBox<NTHREADS,NBLOCKS,T><<<NBLOCKS,NTHREADS>>>(
      n, minmax_ptr, domain, retirementCount, ptclPos);
  cudaDeviceSynchronize();
  delete retirementCount;
  delete [] minmax_ptr;
}

template<typename T>
static __global__ void countAtRootNode(
    const int n,
    __out int *octCounter,
    const Box<T> box,
    const Particle4<T> *ptclPos)
{
  int np_octant[8] = {0};
  const int beg = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = beg; i < n; i += gridDim.x * blockDim.x)
    if (i < n)
    {
      const Particle4<T> p = ptclPos[i];
      const Position<T> pos(p.x(), p.y(), p.z());
      const int octant = Octant(box.centre, pos);
      np_octant[0] += (octant == 0);
      np_octant[1] += (octant == 1);
      np_octant[2] += (octant == 2);
      np_octant[3] += (octant == 3);
      np_octant[4] += (octant == 4);
      np_octant[5] += (octant == 5);
      np_octant[6] += (octant == 6);
      np_octant[7] += (octant == 7);
    };

  const int laneIdx = threadIdx.x & (WARP_SIZE-1);
#pragma unroll
  for (int k = 0; k < 8; k++)
  {
    int np = np_octant[k];
#pragma unroll
    for (int i = 4; i >= 0; i--)
      np += __shfl_xor(np, 1<<i, WARP_SIZE);
    if (laneIdx == 0)
      atomicAdd(&octCounter[8+k],np);
  }

}

template<int NLEAF, typename T>
static __global__ void buildOctree(
    const int n,
    const Box<T> *domain,
    CellData *d_cellDataList,
    int *stack_memory_pool,
    Particle4<T> *ptcl,
    Particle4<T> *buff,
    Particle4<T> *d_ptclVel,
    int *ncells_return = NULL)
{
  cellDataList = d_cellDataList;
  ptclVel      = d_ptclVel;

  memPool = stack_memory_pool;
  printf("n=          %d\n", n);
  printf("d_node_max= %d\n", d_node_max);
  printf("d_cell_max= %d\n", d_cell_max);
  printf("GPU: box_centre= %g %g %g   hsize= %g\n",
      domain->centre.x,
      domain->centre.y,
      domain->centre.z,
      domain->hsize);

  int *octCounter = new int[8+8];
  for (int k = 0; k < 16; k++)
    octCounter[k] = 0;
  countAtRootNode<T><<<256, 256>>>(n, octCounter, *domain, ptcl);
  cudaDeviceSynchronize();

  int total = 0;
  for (int k = 8; k < 16; k++)
  {
    printf("octCounter[%d]= %d\n", k-8, octCounter[k]);
    total += octCounter[k];
  }
  printf("total= %d  n= %d\n", total, n);

  int *octCounterN = new int[8+8+8+64+8];
#pragma unroll
  for (int k = 0; k < 8; k++)
  {
    octCounterN[     k] = 0;
    octCounterN[8+   k] = octCounter[8+k  ];
    octCounterN[8+8 +k] = k == 0 ? 0 : octCounterN[8+8+k-1] + octCounterN[8+k-1];
    octCounterN[8+16+k] = 0;
  }
#pragma unroll
  for (int k = 8; k < 64; k++)
    octCounterN[8+16+k] = 0;

  for (int k = 0; k < 8; k++)
    printf("k= %d n = %d offset= %d \n",
        k, octCounterN[8+k], octCounterN[8+8+k]);

#ifdef IOCOUNT
  io_words = 0;
#endif
  nnodes = 0;
  nleaves = 0;
  nlevels = 0;
  ncells  = 0;
  nbodies_leaf = 0;

  octCounterN[1] = 0;
  octCounterN[2] = n;

  dim3 grid, block;
  computeGridAndBlockSize(grid, block, n);
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
#if 1
  buildOctant<NLEAF,T,true><<<grid, block,0,stream>>>
    (*domain, 0, 0, 0, octCounterN, ptcl, buff);
  cudaDeviceSynchronize();
#endif

  printf(" nptcl  = %d\n", n);
  printf(" nb_leaf= %d\n", nbodies_leaf);
  printf(" nnodes = %d\n", nnodes);
  printf(" nleaves= %d\n", nleaves);
  printf(" ncells=  %d\n",  ncells);
  printf(" nlevels= %d\n", nlevels);

  if (ncells_return != NULL)
    *ncells_return = ncells;

#ifdef IOCOUNT
  printf(" io= %g MB \n" ,io_words*4.0/1024.0/1024.0);
#endif
  delete [] octCounter;
  delete [] octCounterN;
}

/********************************/
/*****   DRIVING ROUTINES   *****/
/********************************/

  template<int NLEAF, typename T>
void testTree(const int n, const unsigned int seed)
{
  typedef typename vec<4,T>::type T4;

  /* prepare initial conditions */

  host_mem< Particle4<T> > h_ptclPos, h_ptclVel;
  h_ptclPos.alloc(n);
  h_ptclVel.alloc(n);

#ifdef PLUMMER
  const Plummer data(n, seed);
  for (int i = 0; i < n; i++)
  {
    h_ptclPos[i].x()    = data. pos[i].x;
    h_ptclPos[i].y()    = data. pos[i].y;
    h_ptclPos[i].z()    = data. pos[i].z;
    h_ptclVel[i].x()    = data. vel[i].x;
    h_ptclVel[i].y()    = data. vel[i].y;
    h_ptclVel[i].z()    = data. vel[i].z;
    h_ptclVel[i].mass() = i; //data.mass[i];
#ifdef PSHFL_SANITY_CHECK
    h_ptclPos[i].mass() = i;
    h_ptclVel[i].mass() = i;
#else
    h_ptclVel[i].mass() = data.mass[i];
    h_ptclPos[i].mass() = data.mass[i];
#endif
  }
#else
  for (int i = 0; i < n; i++)
  {
    h_ptclPos[i].x()    = drand48();
    h_ptclPos[i].y()    = drand48();
    h_ptclPos[i].z()    = drand48();
    h_ptclVel[i].x()    = 0.0;
    h_ptclVel[i].y()    = 0.0;
    h_ptclVel[i].z()    = 0.0;
#ifdef PSHFL_SANITY_CHECK
    h_ptclPos[i].mass() = i;
    h_ptclVel[i].mass() = i;
#else
    h_ptclVel[i].mass() = 1.0/T(n);
    h_ptclPos[i].mass() = 1.0/T(n);
#endif
  }
#endif

  /*  copy data to the device */

  cuda_mem< Particle4<T> > d_ptclPos, d_ptclVel, d_ptclPos_tmp;

  d_ptclPos    .alloc(n);
  d_ptclVel    .alloc(n);
  d_ptclPos_tmp.alloc(n);

  d_ptclPos.h2d(h_ptclPos);
  d_ptclVel.h2d(h_ptclVel);

  /* compute bounding box */

  cuda_mem< Box<T> > d_domain;
  d_domain.alloc(1);
  {
    cudaDeviceSynchronize();
    const double t0 = rtc();
    const int NBLOCKS  = 256;
    const int NTHREADS = 256;
    computeDomainSize<NTHREADS,NBLOCKS,T><<<1,1>>>(n, d_ptclPos, d_domain);
    kernelSuccess("cudaDomainSize");
    const double t1 = rtc();
    const double dt = t1 - t0;
    fprintf(stderr, " cudaDomainSize done in %g sec : %g Mptcl/sec\n",  dt, n/1e6/dt);
  }

  /****** build tree *****/

  /* allocate memory for the stack nodes */

  const int node_max = n/10;
  const int nstack   = (8+8+8+64+8)*node_max;
  fprintf(stderr, "nstack= %g MB \n", sizeof(int)*nstack/1024.0/1024.0);
  cuda_mem<int> d_stack_memory_pool;
  d_stack_memory_pool.alloc(nstack);
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_node_max, &node_max, sizeof(int), 0, cudaMemcpyHostToDevice));

  /* allocate memory for cell data */

  const int cell_max = n;
  fprintf(stderr, "celldata= %g MB \n", cell_max*sizeof(CellData)/1024.0/1024.0);
  cuda_mem<CellData> d_cellDataList;
  d_cellDataList.alloc(cell_max);
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_cell_max, &cell_max, sizeof(int), 0, cudaMemcpyHostToDevice));

  /* prefer shared memory for kernels */

  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&buildOctant      <NLEAF,T,true>,  cudaFuncCachePreferShared));
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&buildOctant      <NLEAF,T,false>, cudaFuncCachePreferShared));
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&buildOctantSingle<NLEAF,T>,       cudaFuncCachePreferShared));

  /**** launch tree building kernel ****/

  host_mem<int> h_ncells;
  cuda_mem<int> d_ncells;
  h_ncells.alloc(1);
  d_ncells.alloc(1);
  {
    CUDA_SAFE_CALL(cudaMemset(d_stack_memory_pool,0,nstack*sizeof(int)));
    cudaDeviceSynchronize();
    const double t0 = rtc();
    buildOctree<NLEAF,T><<<1,1>>>(
        n, d_domain, d_cellDataList, d_stack_memory_pool, d_ptclPos, d_ptclPos_tmp, d_ptclVel);
    kernelSuccess("buildOctree");
    const double t1 = rtc();
    const double dt = t1 - t0;
    fprintf(stderr, " buildOctree done in %g sec : %g Mptcl/sec\n",  dt, n/1e6/dt);
    std::swap(d_ptclPos_tmp.ptr, d_ptclVel.ptr);
  }


#ifdef PSHFL_SANITY_CHECK
  {
    d_ptclPos.d2h(h_ptclPos);
    d_ptclVel.d2h(h_ptclVel);
    for (int i = 0; i < n; i++)
      assert(h_ptclPos[i].mass() == h_ptclVel[i].mass());
  }
#endif


  /******** now particles are sorted, build tree again ******/

  {
    CUDA_SAFE_CALL(cudaMemset(d_stack_memory_pool,0,nstack*sizeof(int)));
    cudaDeviceSynchronize();
    const double t0 = rtc();
    buildOctree<NLEAF,T><<<1,1>>>(
        n, d_domain, d_cellDataList, d_stack_memory_pool, d_ptclPos, d_ptclPos_tmp, d_ptclVel, d_ncells);
    kernelSuccess("buildOctree");
    d_ncells.d2h(h_ncells);
    const double t1 = rtc();
    const double dt = t1 - t0;
    fprintf(stderr, " buildOctree done in %g sec : %g Mptcl/sec\n",  dt, n/1e6/dt);
  }

}

int main(int argc, char * argv [])
{
  int n = 4000000;
  if (argc > 1)
  {
    assert(argc > 1);
    n = atoi(argv[1]);
  }
  assert(n > 0);

  fprintf(stderr, " n= %d \n", n);

#ifdef FP64
  typedef double real;
#else
  typedef float real;
#endif

#ifndef NPERLEAF
  const int NLEAF = 16;
#else
  const int NLEAF = NPERLEAF;
#endif

  testTree<NLEAF, real>(n, argc > 2 ? atoi(argv[2]) : 19810614);

};
