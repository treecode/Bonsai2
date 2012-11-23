#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>
#include "rtc.h"
#include "plummer.h"
#include "cudamem.h"

#define __out 

#define WARP_SIZE2 5
#define WARP_SIZE  32

__device__ unsigned int nnodes = 0;
__device__ unsigned int nleaves = 0;
__device__ unsigned int nlevels = 0;
__device__ unsigned int nbodies_leaf = 0;


__device__   int *memPool;
__constant__ int d_node_max;
__device__ unsigned long long io_words;

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
struct __align__(4) ParticleLight
{
  Position<T> pos;
  float     idFlt;
  __host__ __device__ ParticleLight() {}
  __host__ ParticleLight(const Position<T> &_pos, const int _id) : pos(_pos), idFlt(*(float*)&_id) {}
  __device__ int d_id() const {return __float_as_int(idFlt); }
  __device__ void shfl(const ParticleLight &p, const int i) 
  {
    pos.      shfl(p.pos,   i);
    idFlt = myshfl(idFlt, p.idFlt, i);
  }

#if 1
  __device__ ParticleLight(const float4 v) :
    pos(v.x, v.y, v.z), idFlt(v.w) {}
  __device__ operator float4() const {return make_float4(pos.x, pos.y, pos.z, idFlt);}
#endif
#if 0
  __host__   int h_id() const {return __float_as_int(idFlt); }
#endif
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
  const int NTHREADS = 8 * WARP_SIZE;
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

static __device__ __forceinline__ int warpBinReduce(const bool p)
{
  const unsigned int b = __ballot(p);
  return __popc(b);
}

template<int NLEAF, typename T>
static __global__ void buildOctantSingle(
    Box<T> box,
    const int nOctants,
    const int octantMask,
    __out int *octCounterBase,
    ParticleLight<T> *ptcl,
    ParticleLight<T> *buff,
    const int level = 0)
{
  const int laneId = threadIdx.x & (WARP_SIZE-1);
  const int warpId = threadIdx.x >> WARP_SIZE2;

  const int octant2process = (octantMask >> (3*blockIdx.y)) & 0x7;

  int *octCounter = octCounterBase + blockIdx.y*(8+8+8+64+8);

  const int data  = octCounter[laneId];
  const int nCell = __shfl(data, 8+warpId, WARP_SIZE);
  const int nBeg  = __shfl(data, 1, WARP_SIZE);
  const int nEnd  = __shfl(data, 2, WARP_SIZE);

  int cellCounter = __shfl(data, 8+8+warpId, WARP_SIZE);

  /* each of the 8 warps are responsible for each of the octant */
  if (level > 0)
    box = ChildBox(box, octant2process);
  const Box<T> childBox = ChildBox(box, warpId);

  /* counter in shmem for each of the octant */
  int nChildren[8] = {0};

  assert(blockIdx.x == 0);

  /* process particle array */
  int addrBase = nBeg;
#pragma unroll
  for (int k = 0; k < 8; k++, addrBase += WARP_SIZE)  /* process particles in shared memory */
  {
    if (addrBase >= nEnd) break;
    const int  addr = addrBase + laneId;
    const bool mask = addr < nEnd;
#if 0
    const ParticleLight<T> p = ptcl[mask ? addr : nEnd-1];       /* stream loads */
#else
    const ParticleLight<T> p( ((float4*)ptcl)[mask ? addr : nEnd-1] );  /* float4 vector loads */
#endif
#if 1
    __syncthreads();  
#endif

#if 0          /* sanity check, check on the fly that tree structure is corrent */
    { 
      if (box.centre.x - box.hsize > p.pos.x ||
          box.centre.y - box.hsize > p.pos.y ||
          box.centre.z - box.hsize > p.pos.z ||
          box.centre.x + box.hsize < p.pos.x ||
          box.centre.y + box.hsize < p.pos.y ||
          box.centre.z + box.hsize < p.pos.z)
      {
        printf("CELL, level= %d  pos= %g %g %g   c= %g %g %g  hsize= %g\n", level,
            p.pos.x, p.pos.y,p.pos.z,
            box.centre.x, box.centre.y, box.centre.z, box.hsize);
        assert(0);
      }
    }
#endif

    /* use prefix sums to compute offset to where scatter particles */
    const int     use = mask && (Octant(box.centre, p.pos) == warpId);
    const int2 offset = warpBinExclusiveScan(use);  /* x is write offset, y is element count */

    if (offset.y > 0)
    {
      const int addrB = cellCounter;
      cellCounter += offset.y;

      int subOctant = -1;
      if (use)
      {
        ((float4*)buff)[addrB+offset.x] = p;         /* float4 vector stores   */
        if (nCell > NLEAF)
          subOctant = Octant(childBox.centre, p.pos);
      }

      if (nCell > NLEAF)
      {
#pragma unroll
        for (int k = 0; k < 8; k++)
          nChildren[k] += warpBinReduce(subOctant == k);
      }
    }
  }

  /* done processing particles, store counts atomically in gmem */
  __shared__ int nPtclChild[8][8];

  if (laneId == 0)
  {
#pragma unroll
    for (int k = 0; k < 8; k++)
      nPtclChild[warpId][k] = nChildren[k];
  }

  /* number of particles in each cell's subcells */
  const int nSubCell = laneId < 8 ? nPtclChild[warpId][laneId] : 0;

  /* last block finished, analysis of the data and schedule new kernel for children */

  __syncthreads();  /* must be present, otherwise race conditions occurs between parent & children */

  int *shmem = &nPtclChild[0][0];
  shmem[laneId] = 0;

  __syncthreads();

  if (threadIdx.x == 0)
    atomicCAS(&nlevels, level, level+1);

  const int nEnd1 = cellCounter;
  const int nBeg1 = nEnd1 - nCell;

  if (laneId == 0)
    shmem[warpId] = nCell;
  __syncthreads();

  const bool isNode = shmem[laneId] > NLEAF;
  const int   nNode = isNode ? shmem[laneId] : 0;

  const int2 nSubNodes = warpBinExclusiveScan(isNode);
  if (warpId)
    shmem[8+laneId] = nSubNodes.x;
  __syncthreads();

  int nCellmax = isNode ? nNode : 0;
#pragma unroll
  for (int i = 4; i >= 0; i--)
    nCellmax = max(nCellmax, __shfl_xor(nCellmax, 1<<i, WARP_SIZE));

  if (threadIdx.x == 0 && nSubNodes.y > 0)
  {
    shmem[8+8] = atomicAdd(&nnodes,nSubNodes.y);
#if 1   /* temp solution, a better one is to use RingBuffer */
    assert(shmem[8+8] < d_node_max);
#endif
  }
  __syncthreads();

  const int next_node = shmem[8+8];
  int *octCounterNbase = &memPool[next_node*(8+8+8+64+8)];

  if (nCell > NLEAF)
  {
    int *octCounterN = octCounterNbase + shmem[8+warpId]*(8+8+8+64+8);

    /* compute offsets */
    int cellOffset = nSubCell;
#pragma unroll
    for(int i = 0; i < 3; i++)  /* log2(8) steps */
      cellOffset = shfl_scan_add_step(cellOffset, 1 << i);
    cellOffset -= nSubCell;

    /* store offset in memory */

    cellOffset = __shfl_up(cellOffset, 8, WARP_SIZE);
    if (laneId < 8) cellOffset = nSubCell;
    else            cellOffset += nBeg1;
    cellOffset = __shfl_up(cellOffset, 8, WARP_SIZE);

    if (laneId <  8) cellOffset = 0;
    if (laneId == 1) cellOffset = nBeg1;
    if (laneId == 2) cellOffset = nEnd1;

    if (laneId < 24)
      octCounterN[laneId] = cellOffset;
  }

  /***************************/
  /*  launch  child  kernel  */
  /***************************/

  if (nSubNodes.y > 0 && warpId == 0)
  {
    int octant_mask = nNode > NLEAF ?  (laneId << (3*nSubNodes.x)) : 0;
#pragma unroll
    for (int i = 4; i >= 0; i--)
      octant_mask |= __shfl_xor(octant_mask, 1<<i, WARP_SIZE);

    if (threadIdx.x == 0)
    {
      dim3 grid, block;
      computeGridAndBlockSize(grid, block, nCellmax);
      cudaStream_t stream;
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

      grid.x = 1;
      grid.y = nSubNodes.y;
      buildOctantSingle<NLEAF,T><<<grid,block,0,stream>>>(box, nSubNodes.y, octant_mask, octCounterNbase, buff, ptcl, level+1);
    }
  }

  /******************/
  /* process leaves */
  /******************/

  if (nCell <= NLEAF && nCell > 0)
  {
    if (laneId == 0)
    {
      assert(nEnd1 - nBeg1 <= NLEAF);
      atomicAdd(&nleaves,1);
      atomicAdd(&nbodies_leaf, nEnd1-nBeg1);
    }
    if (!(level&1))
      for (int i = nBeg1+laneId; i < nEnd1; i += WARP_SIZE)
        if (i < nEnd1)
          ((float4*)ptcl)[i] = ((float4*)buff)[i];
  }
}

/****** this is the main functions that build the tree recursively *******/

template<int NLEAF, typename T>
static __global__ void buildOctant(
    Box<T> box,
    const int nOctants,
    const int octantMask,
    __out int *octCounterBase,
    ParticleLight<T> *ptcl,
    ParticleLight<T> *buff,
    const int level = 0)
{
  const int laneId = threadIdx.x & (WARP_SIZE-1);
  const int warpId = threadIdx.x >> WARP_SIZE2;

  const int octant2process = (octantMask >> (3*blockIdx.y)) & 0x7;

  int *octCounter = octCounterBase + blockIdx.y*(8+8+8+64+8);

  const int data  = octCounter[laneId];
  const int nCell = __shfl(data, 8+warpId, WARP_SIZE);
  const int nBeg  = __shfl(data, 1, WARP_SIZE);
  const int nEnd  = __shfl(data, 2, WARP_SIZE);

  /* each of the 8 warps are responsible for each of the octant */
  if (level > 0)
    box = ChildBox(box, octant2process);
  const Box<T> childBox = ChildBox(box, warpId);

  /* counter in shmem for each of the octant */
  int nChildren[8] = {0};

  /* process particle array */
  const int nBeg_block = nBeg + blockIdx.x * blockDim.x;
  for (int i = nBeg_block; i < nEnd; i += gridDim.x * blockDim.x)
  {
    if (threadIdx.x == 0)
      atomicAdd(&io_words, 8*WARP_SIZE*4);
#pragma unroll
    for (int k = 0; k < 8; k++)  /* process particles in shared memory */
    {
      const int  addr = i + (k<<WARP_SIZE2) + laneId;
      const bool mask = addr < nEnd;

#if 0
      const ParticleLight<T> p = ptcl[mask ? addr : nEnd-1];       /* stream loads */
#else
      const ParticleLight<T> p( ((float4*)ptcl)[mask ? addr : nEnd-1] );  /* float4 vector loads */
#endif
#if 1
      __syncthreads();    /* seems to help, probably because make more efficient use of L1 cache */
#endif

#if 0          /* sanity check, check on the fly that tree structure is corrent */
      { 
        if (box.centre.x - box.hsize > p.pos.x ||
            box.centre.y - box.hsize > p.pos.y ||
            box.centre.z - box.hsize > p.pos.z ||
            box.centre.x + box.hsize < p.pos.x ||
            box.centre.y + box.hsize < p.pos.y ||
            box.centre.z + box.hsize < p.pos.z)
        {
          printf("CELL, level= %d  pos= %g %g %g   c= %g %g %g  hsize= %g\n", level,
              p.pos.x, p.pos.y,p.pos.z,
              box.centre.x, box.centre.y, box.centre.z, box.hsize);
          assert(0);
        }
      }
#endif

      /* use prefix sums to compute offset to where scatter particles */
      const int     use = mask && (Octant(box.centre, p.pos) == warpId);
      const int2 offset = warpBinExclusiveScan(use);  /* x is write offset, y is element count */

      if (offset.y > 0)
      {
        const int addr0 = laneId == 0 ? atomicAdd(&octCounter[8+8+warpId], offset.y) : -1;
        const int addrB = __shfl(addr0, 0, WARP_SIZE);
        if (laneId == 0)
          atomicAdd(&io_words, offset.y*4);

        int subOctant = -1;
        if (use)
        {
          ((float4*)buff)[addrB+offset.x] = p;         /* float4 vector stores   */
          if (nCell > NLEAF)
            subOctant = Octant(childBox.centre, p.pos);
        }

        if (nCell > NLEAF)
        {
#pragma unroll
          for (int k = 0; k < 8; k++)
            nChildren[k] += warpBinReduce(subOctant == k);
        }
      }
    }
  }

  /* done processing particles, store counts atomically in gmem */

  __shared__ int nPtclChild[8][8];

  if (laneId == 0)
  {
#pragma unroll
    for (int k = 0; k < 8; k++)
      nPtclChild[warpId][k] = nChildren[k];
  }
  if (laneId < 8)
    if (nPtclChild[warpId][laneId] > 0)
      atomicAdd(&octCounter[8+16+warpId*8 + laneId], nPtclChild[warpId][laneId]);

  /* last block finished, analysis of the data and schedule new kernel for children */

  __syncthreads();  /* must be present, otherwise race conditions occurs between parent & children */

  int *shmem = &nPtclChild[0][0];
  shmem[laneId] = 0;

  __shared__ int lastBlock;
  if (threadIdx.x == 0)
  {
    const int ticket = atomicAdd(&octCounter[8+8+8+64+octant2process], 1);
    lastBlock = (ticket == gridDim.x-1);
  }
  __syncthreads();

  if (!lastBlock) return;

  if (threadIdx.x == 0)
    atomicCAS(&nlevels, level, level+1);

  const int nEnd1 = octCounter[8+8+warpId];
  const int nBeg1 = nEnd1 - nCell;

  if (laneId == 0)
    shmem[warpId] = nCell;
  __syncthreads();

  const bool isNode = shmem[laneId] > NLEAF;
  const int   nNode = isNode ? shmem[laneId] : 0;

  const int2 nSubNodes = warpBinExclusiveScan(isNode);
  if (warpId)
    shmem[8+laneId] = nSubNodes.x;
  __syncthreads();

  int nCellmax = isNode ? nNode : 0;
#pragma unroll
  for (int i = 4; i >= 0; i--)
    nCellmax = max(nCellmax, __shfl_xor(nCellmax, 1<<i, WARP_SIZE));

  if (threadIdx.x == 0 && nSubNodes.y > 0)
  {
    shmem[8+8] = atomicAdd(&nnodes,nSubNodes.y);
#if 1   /* temp solution, a better one is to use RingBuffer */
    assert(shmem[8+8] < d_node_max);
#endif
  }
  __syncthreads();

  const int next_node = shmem[8+8];
  int *octCounterNbase = &memPool[next_node*(8+8+8+64+8)];

  if (nCell > NLEAF)
  {
    int *octCounterN = octCounterNbase + shmem[8+warpId]*(8+8+8+64+8);

    /* number of particles in each cell's subcells */
    const int nSubCell = laneId < 8 ? octCounter[8+16+warpId*8 + laneId] : 0;

    /* compute offsets */
    int cellOffset = nSubCell;
#pragma unroll
    for(int i = 0; i < 3; i++)  /* log2(8) steps */
      cellOffset = shfl_scan_add_step(cellOffset, 1 << i);
    cellOffset -= nSubCell;

    /* store offset in memory */

    cellOffset = __shfl_up(cellOffset, 8, WARP_SIZE);
    if (laneId < 8) cellOffset = nSubCell;
    else            cellOffset += nBeg1;
    cellOffset = __shfl_up(cellOffset, 8, WARP_SIZE);

    if (laneId <  8) cellOffset = 0;
    if (laneId == 1) cellOffset = nBeg1;
    if (laneId == 2) cellOffset = nEnd1;

    if (laneId < 24)
      octCounterN[laneId] = cellOffset;
  }

  /***************************/
  /*  launch  child  kernel  */
  /***************************/

  if (nSubNodes.y > 0 && warpId == 0)
  {
    int octant_mask = nNode > NLEAF ?  (laneId << (3*nSubNodes.x)) : 0;
#pragma unroll
    for (int i = 4; i >= 0; i--)
      octant_mask |= __shfl_xor(octant_mask, 1<<i, WARP_SIZE);

    if (threadIdx.x == 0)
    {
      dim3 grid, block;
      computeGridAndBlockSize(grid, block, nCellmax);
      cudaStream_t stream;
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

      grid.y = nSubNodes.y;
      if (nCellmax <= block.x)
      {
        grid.x = 1;
        buildOctantSingle<NLEAF,T><<<grid,block,0,stream>>>(box, nSubNodes.y, octant_mask, octCounterNbase, buff, ptcl, level+1);
      }
      else
      {
        buildOctant<NLEAF,T><<<grid,block,0,stream>>>(box, nSubNodes.y, octant_mask, octCounterNbase, buff, ptcl, level+1);
      }
    }
  }

  /******************/
  /* process leaves */
  /******************/

  if (nCell <= NLEAF && nCell > 0)
  {
    if (laneId == 0)
    {
      assert(nEnd1 - nBeg1 <= NLEAF);
      atomicAdd(&nleaves,1);
      atomicAdd(&nbodies_leaf, nEnd1-nBeg1);
    }
    if (!(level&1))
      for (int i = nBeg1+laneId; i < nEnd1; i += WARP_SIZE)
        if (i < nEnd1)
          ((float4*)ptcl)[i] = ((float4*)buff)[i];
  }
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

template<typename T, const int NBLOCKS, const int NTHREADS>
static __global__ void computeBoundingBox(
    const int n,
    __out Position<T> *minmax_ptr,
    __out Box<T>      *box_ptr,
    __out int *retirementCount,
    const ParticleLight<T> *ptcl)
{
  __shared__ Position<T> shmin[NTHREADS], shmax[NTHREADS];

  const int gridSize = NTHREADS*NBLOCKS*2;
  int i = blockIdx.x*NTHREADS*2 + threadIdx.x;

  Position<T> bmin(T(+1e10)), bmax(T(-1e10));

  while (i < n)
  {
    const ParticleLight<T> p = ptcl[i];
    bmin = Position<T>::min(bmin, p.pos);
    bmax = Position<T>::max(bmax, p.pos);
    if (i + NTHREADS < n)
    {
      const ParticleLight<T> p = ptcl[i + NTHREADS];
      bmin = Position<T>::min(bmin, p.pos);
      bmax = Position<T>::max(bmax, p.pos);
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
    };
  }
}


template<typename T>
static __global__ void countAtRootNode(
    const int n,
    __out int *octCounter,
    const Box<T> box,
    const ParticleLight<T> *ptcl)
{
  const int beg = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = beg; i < n; i += gridDim.x * blockDim.x)
    if (i < n)
    {
      const ParticleLight<T> p = ptcl[i];
      const int octant = Octant(box.centre, p.pos);
      atomicAdd(&octCounter[8+octant],1);
    };

#if 0
  __shared__ bool lastBlock;
  __threadfence();
  if (threadIdx.x == 0)
  {
    const int ticket = atomicInc((unsigned int*)octCounter, gridDim.x);
    lastBlock = (ticket == gridDim.x-1);
  };

  __syncthreads();
#endif
}

template<int NLEAF, typename T>
static __global__ void buildOctree(
    const int n,
    int* memory_pool,
    __out ParticleLight<T> *ptcl,
    __out ParticleLight<T> *buff)
{
  memPool = memory_pool;
  printf("d_node_max= %d\n", d_node_max);
  const int NTHREADS = 256;
  const int NBLOCKS  = 256;
  Box<T> *box_ptr = new Box<T>();
  Position<T> *minmax_ptr = new Position<T>[2*NBLOCKS];
  int *retirementCount = new int;
  *retirementCount = 0;
  __threadfence();
  computeBoundingBox<T,NBLOCKS,NTHREADS><<<NBLOCKS,NTHREADS>>>(n, minmax_ptr, box_ptr, retirementCount, ptcl);
  cudaDeviceSynchronize();
  delete retirementCount;

  printf("GPU: box_centre= %g %g %g   hsize= %g\n",
      box_ptr->centre.x,
      box_ptr->centre.y,
      box_ptr->centre.z,
      box_ptr->hsize);

  int *octCounter = new int[8+8];
  for (int k = 0; k < 16; k++)
    octCounter[k] = 0;
  countAtRootNode<T><<<256, 256>>>(n, octCounter, *box_ptr, ptcl);
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

  io_words = 0;
  nnodes = 0;
  nleaves = 0;
  nlevels = 0;
  nbodies_leaf = 0;

  octCounterN[1] = 0;
  octCounterN[2] = n;



  dim3 grid, block;
  computeGridAndBlockSize(grid, block, n);
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  buildOctant<NLEAF,T><<<grid, block,0,stream>>>(box_ptr[0], 1, 0, octCounterN, ptcl, buff);
  cudaDeviceSynchronize();
  //  delete [] octCounterN;

  printf(" nptcl  = %d\n", n);
  printf(" nb_leaf= %d\n", nbodies_leaf);
  printf(" nnodes = %d\n", nnodes);
  printf(" nleaves= %d\n", nleaves);
  printf(" nlevels= %d\n", nlevels);

  printf(" io= %g MB \n" ,io_words*4.0/1024.0/1024.0);


  delete [] octCounter;
  delete [] minmax_ptr;
  delete box_ptr;
}

typedef float real;

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

  host_mem< ParticleLight<real> > h_ptcl;
  h_ptcl.alloc(n);
#ifdef PLUMMER
  const Plummer data(n, argc > 2 ? atoi(argv[2]) : 19810614);
  for (int i = 0; i < n; i++)
    h_ptcl[i] = ParticleLight<real>(Position<real>(data.pos[i].x, data.pos[i].y, data.pos[i].z), i);
#else
  for (int i = 0; i < n; i++)
    h_ptcl[i] = ParticleLight<real>(Position<real>(drand48(), drand48(), drand48()), i);
#endif
  Position<real> bmin(+1e10), bmax(-1e10);
  for (int i = 0; i < n; i++)
  {
    //    printf("%g %g %g \n", h_ptcl[i].pos.x, h_ptcl[i].pos.y, h_ptcl[i].pos.z);
    bmin = Position<real>::min(bmin, h_ptcl[i].pos);
    bmax = Position<real>::max(bmax, h_ptcl[i].pos);
  }
  //  exit(0);
  const Position<real> cvec((bmax.x+bmin.x)*(0.5), (bmax.y+bmin.y)*(0.5), (bmax.z+bmin.z)*(0.5));
  const Position<real> hvec((bmax.x-bmin.x)*(0.5), (bmax.y-bmin.y)*(0.5), (bmax.z-bmin.z)*(0.5));
  const real h = fmax(hvec.z, fmax(hvec.y, hvec.x));
  real hsize = (1.0);
  while (hsize > h) hsize *= (0.5);
  while (hsize < h) hsize *= (2.0);

  fprintf(stderr, "bmin= %g %g %g \n", bmin.x, bmin.y, bmin.z);
  fprintf(stderr, "bmax= %g %g %g \n", bmax.x, bmax.y, bmax.z);

  printf("box_centre= %g %g %g   hsize= %g\n",
      cvec.x,
      cvec.y,
      cvec.z,
      hsize);

  cuda_mem< ParticleLight<real> > d_ptcl1, d_ptcl2;
  d_ptcl1.alloc(n);
  d_ptcl2.alloc(n);
  d_ptcl1.h2d(h_ptcl);


  int node_max = n/10;

  cuda_mem<int> memory_pool;
  const unsigned long long nstack = (8+8+8+64+8)*node_max;
  fprintf(stderr, " nstack= %g MB \n", sizeof(int)*nstack/1024.0/1024.0);

  memory_pool.alloc(nstack);
  CUDA_SAFE_CALL(cudaMemset(memory_pool,0,nstack*sizeof(int)));

  CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_node_max, &node_max, sizeof(int), 0, cudaMemcpyHostToDevice));


#ifndef NPERLEAF
  const int NLEAF = 16;
#else
  const int NLEAF = NPERLEAF;
#endif

#if 1
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&buildOctant<NLEAF, real>, cudaFuncCachePreferL1));
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&buildOctantSingle<NLEAF, real>, cudaFuncCachePreferL1));
#elif 1
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&buildOctant<NLEAF, real>, cudaFuncCachePreferShared));
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&buildOctantSingle<NLEAF, real>, cudaFuncCachePreferShared));
#else
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&buildOctant<NLEAF, real>, cudaFuncCachePreferEqual));
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&buildOctantSingle<NLEAF, real>, cudaFuncCachePreferEqual));
#endif

  {
    const double t0 = rtc();
    buildOctree<NLEAF, real><<<1,1>>>(n, memory_pool, d_ptcl1, d_ptcl2);
    const int ret = (cudaDeviceSynchronize() != cudaSuccess);
    if (ret)
    {
      fprintf(stderr, "CNP tree launch failed: %s\n", cudaGetErrorString(cudaGetLastError()));
      assert(0);
    }

    const double t1 = rtc();
    const double dt = t1 - t0;

    fprintf(stderr, " done in %g sec : %g Mptcl/sec\n",
        dt, n/1e6/dt);
  }

  CUDA_SAFE_CALL(cudaMemset(memory_pool,0,nstack*sizeof(int)));

  {
    const double t0 = rtc();
    buildOctree<NLEAF, real><<<1,1>>>(n, memory_pool, d_ptcl1, d_ptcl2);
    const int ret = (cudaDeviceSynchronize() != cudaSuccess);
    if (ret)
      fprintf(stderr, "CNP tree launch failed: %s\n", cudaGetErrorString(cudaGetLastError()));

    const double t1 = rtc();
    const double dt = t1 - t0;

    fprintf(stderr, " done in %g sec : %g Mptcl/sec\n",
        dt, n/1e6/dt);
  }

};
