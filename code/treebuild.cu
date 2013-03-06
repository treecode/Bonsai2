#include "Treecode.h"

#define NWARPS_OCTREE2 3
#define NWARPS2 NWARPS_OCTREE2
#define NWARPS  (1<<NWARPS2)

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

namespace treeBuild
{
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
  static __device__ __forceinline__ int lanemask_lt()
  {
    int mask;
    asm("mov.u32 %0, %lanemask_lt;" : "=r" (mask));
    return mask;
  }
  static __device__ __forceinline__ int warpBinReduce(const bool p)
  {
    const unsigned int b = __ballot(p);
    return __popc(b);
  }
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

  static __forceinline__ __device__ void computeGridAndBlockSize(dim3 &grid, dim3 &block, const int np)
  {
    const int NTHREADS = (1<<NWARPS_OCTREE2) * WARP_SIZE;
    block = dim3(NTHREADS);
    assert(np > 0);
    grid = dim3(min(max(np/(NTHREADS*4),1), 512));
  }

  __device__ unsigned int retirementCount = 0;

  __constant__ int d_node_max;
  __constant__ int d_cell_max;

  __device__ unsigned int nnodes = 0;
  __device__ unsigned int nleaves = 0;
  __device__ unsigned int nlevels = 0;
  __device__ unsigned int nbodies_leaf = 0;
  __device__ unsigned int ncells = 0;

  __device__   int *memPool;
  __device__   CellData *cellDataList;
  __device__   void *ptclVel_tmp;

  template<int NTHREAD2>
    __device__ float2 minmax_block(float2 sum)
    {
      extern __shared__ float shdata[];
      float *shMin = shdata;
      float *shMax = shdata + (1<<NTHREAD2);

      const int tid = threadIdx.x;
      shMin[tid] = sum.x;
      shMax[tid] = sum.y;
      __syncthreads();

#pragma unroll    
      for (int i = NTHREAD2-1; i >= 6; i--)
      {
        const int offset = 1 << i;
        if (tid < offset)
        {
          shMin[tid] = sum.x = fminf(sum.x, shMin[tid + offset]);
          shMax[tid] = sum.y = fmaxf(sum.y, shMax[tid + offset]);
        }
        __syncthreads();
      }

      if (tid < 32)
      {
        volatile float *vshMin = shMin;
        volatile float *vshMax = shMax;
#pragma unroll
        for (int i = 5; i >= 0; i--)
        {
          const int offset = 1 << i;
          vshMin[tid] = sum.x = fminf(sum.x, vshMin[tid + offset]);
          vshMax[tid] = sum.y = fmaxf(sum.y, vshMax[tid + offset]);
        }
      }

      __syncthreads();

      return sum;
    }

  template<const int NTHREAD2, typename T>
    static __global__ void computeBoundingBox(
        const int n,
        __out Position<T> *minmax_ptr,
        __out Box<T>      *box_ptr,
        const Particle4<T> *ptclPos)
    {
      const int NTHREAD = 1<<NTHREAD2;
      const int NBLOCK  = NTHREAD;

      Position<T> bmin(T(+1e10)), bmax(T(-1e10));

      const int nbeg = blockIdx.x * NTHREAD + threadIdx.x;
      for (int i = nbeg; i < n; i += NBLOCK*NTHREAD)
        if (i < n)
        {
          const Particle4<T> p = ptclPos[i];
          const Position<T> pos(p.x(), p.y(), p.z());
          bmin = Position<T>::min(bmin, pos);
          bmax = Position<T>::max(bmax, pos);
        }   

      float2 res;
      res = minmax_block<NTHREAD2>(make_float2(bmin.x, bmax.x)); bmin.x = res.x; bmax.x = res.y;
      res = minmax_block<NTHREAD2>(make_float2(bmin.y, bmax.y)); bmin.y = res.x; bmax.y = res.y;
      res = minmax_block<NTHREAD2>(make_float2(bmin.z, bmax.z)); bmin.z = res.x; bmax.z = res.y;

      if (threadIdx.x == 0) 
      {
        minmax_ptr[blockIdx.x         ] = bmin;
        minmax_ptr[blockIdx.x + NBLOCK] = bmax;
      }

      __shared__ bool lastBlock;
      __threadfence();
      __syncthreads();

      if (threadIdx.x == 0)
      {
        const int ticket = atomicInc(&retirementCount, NBLOCK);
        lastBlock = (ticket == NBLOCK - 1);
      }

      __syncthreads();

      if (lastBlock)
      {

        bmin = minmax_ptr[threadIdx.x];
        bmax = minmax_ptr[threadIdx.x + NBLOCK];

        float2 res;
        res = minmax_block<NTHREAD2>(make_float2(bmin.x, bmax.x)); bmin.x = res.x; bmax.x = res.y;
        res = minmax_block<NTHREAD2>(make_float2(bmin.y, bmax.y)); bmin.y = res.x; bmax.y = res.y;
        res = minmax_block<NTHREAD2>(make_float2(bmin.z, bmax.z)); bmin.z = res.x; bmax.z = res.y;

        __syncthreads();

        if (threadIdx.x == 0)
        {
#if 1
          printf("bmin= %g %g %g \n", bmin.x, bmin.y, bmin.z);
          printf("bmax= %g %g %g \n", bmax.x, bmax.y, bmax.z);
#endif
          const Position<T> cvec((bmax.x+bmin.x)*T(0.5), (bmax.y+bmin.y)*T(0.5), (bmax.z+bmin.z)*T(0.5));
          const Position<T> hvec((bmax.x-bmin.x)*T(0.5), (bmax.y-bmin.y)*T(0.5), (bmax.z-bmin.z)*T(0.5));
          const T h = fmax(hvec.z, fmax(hvec.y, hvec.x));
          T hsize = T(1.0);
          while (hsize > h) hsize *= T(0.5);
          while (hsize < h) hsize *= T(2.0);

          const int NMAXLEVEL = 20;

          const T hquant = hsize / T(1<<NMAXLEVEL);
          const long long nx = (long long)(cvec.x/hquant);
          const long long ny = (long long)(cvec.y/hquant);
          const long long nz = (long long)(cvec.z/hquant);

          const Position<T> centre(hquant * T(nx), hquant * T(ny), hquant * T(nz));

          *box_ptr = Box<T>(centre, hsize);
          retirementCount = 0;
        }
      }
    }

  /*******************/

  template<int NLEAF, typename T, bool STOREIDX>
    static __global__ void 
    __launch_bounds__( 256, 8)
    buildOctant(
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
      for (int i = 0; i < 8*9*NWARPS; i += NWARPS*WARP_SIZE)
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

        /* compute suboctant of the octant into which particle will fall */
        if (p4octant < 8)
        {
          const int p4subOctant = Octant(shChildBox[p4octant].centre, Position<T>(p4.x(), p4.y(), p4.z()));
          p4.set_oct(p4subOctant);
        }

        /* compute number of particles in each of the octants that will be processed by thead block */
        int np = 0;
#pragma unroll
        for (int octant = 0; octant < 8; octant++)
        {
          const int sum = warpBinReduce(p4octant == octant);
          if (octant == laneIdx)
            np = sum;
        }

        /* increment atomic counters in a single instruction for thread-blocks to participated */
        int addrB0;
        if (laneIdx < 8)
          addrB0 = atomicAdd(&octCounter[8+8+laneIdx], np);

        /* compute addresses where to write data */
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

        /* write the data in a single instruction */ 
        if (addrW >= 0)
          buff[addrW] = p4;

        /* count how many particles in suboctants in each of the octants */
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
                  warpBinReduce(k+0 == subOctant),
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
      }
      __syncthreads();

      if (warpIdx >= 8) return;


#pragma unroll
      for (int k = 0; k < 8; k += 4)
      {
        int4 nSubOctant = laneIdx < NWARPS ? (*(int4*)&nShChildrenFine[laneIdx][warpIdx][k]) : make_int4(0,0,0,0);
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

      if (laneIdx < 8)
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
#if 0
          cudaStream_t stream;
          cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

          grid.y = nSubNodes.y;  /* each y-coordinate of the grid will be busy for each parent cell */
          atomicAdd(&n_scheduled,1);
          atomicAdd(&n_in_que, 1);
          atomicMax(&n_in_que_max, n_in_que);
#if defined(FASTMODE) && NWARPS==8
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
#else
          grid.y = nSubNodes.y;  /* each y-coordinate of the grid will be busy for each parent cell */
#if defined(FASTMODE) && NWARPS==8
          if (nCellmax <= block.x)
          {
            grid.x = 1;
            buildOctantSingle<NLEAF,T><<<grid,block>>>
              (box, cellIndexBase+blockIdx.y, cellFirstChildIndex,
               octant_mask, octCounterNbase, buff, ptcl, level+1);
          }
          else
#endif
            buildOctant<NLEAF,T,false><<<grid,block>>>
              (box, cellIndexBase+blockIdx.y, cellFirstChildIndex,
               octant_mask, octCounterNbase, buff, ptcl, level+1);
#endif
          const cudaError_t err = cudaGetLastError();
          if (err != cudaSuccess)
          {
            printf(" launch failed 1: %s  level= %d n =%d \n", cudaGetErrorString(err), level);
            assert(0);
          }
        }
      }

      /******************/
      /* process leaves */
      /******************/

      if (nCell <= NLEAF && nCell > 0)
      {
        if (laneIdx == 0)
        {
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
              Particle4<T> vel = ((Particle4<T>*)ptclVel_tmp)[pos.get_idx()];
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
              Particle4<T> vel = ((Particle4<T>*)ptclVel_tmp)[pos.get_idx()];
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
      ptclVel_tmp  = (void*)d_ptclVel;

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
      assert(cudaGetLastError() == cudaSuccess);
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
#if 1
      buildOctant<NLEAF,T,true><<<grid, block>>>
        (*domain, 0, 0, 0, octCounterN, ptcl, buff);
      assert(cudaDeviceSynchronize() == cudaSuccess);
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
}


  template<typename real_t, int NLEAF>
void Treecode<real_t, NLEAF>::buildTree()
{
  /* compute bounding box */

  cuda_mem< Box<real_t> >  d_domain;
  d_domain.alloc(1);
  {
    const int NTHREAD2 = 8;
    const int NTHREAD  = 1<<NTHREAD2;
    const int NBLOCK   = NTHREAD;

    cuda_mem< Position<real_t> > minmax;
    minmax.alloc(NBLOCK*2);

    cudaDeviceSynchronize();
    const double t0 = rtc();
    treeBuild::computeBoundingBox<NTHREAD2,real_t><<<NBLOCK,NTHREAD,NTHREAD*sizeof(float2)>>>
      (nPtcl, minmax, d_domain, d_ptclPos);
    kernelSuccess("cudaDomainSize");
    const double dt = rtc() - t0;
    fprintf(stderr, " cudaDomainSize done in %g sec : %g Mptcl/sec\n",  dt, nPtcl/1e6/dt);
  }
#if 0
  host_mem< Box<real_t> >  h_domain;
  h_domain.alloc(1);
  d_domain.d2h(h_domain);
  fprintf(stderr, "  %g %g %g  h= %g \n", 
      h_domain[0].centre.x,
      h_domain[0].centre.y,
      h_domain[0].centre.z,
      h_domain[0].hsize);
#endif


  /*** build tree ***/

  /*** allocate stack memory ***/
  const int node_max = nPtcl/10;
  const int nstack   = (8+8+8+64+8)*node_max;
  fprintf(stderr, "nstack= %g MB \n", sizeof(int)*nstack/1024.0/1024.0);
  cuda_mem<int> d_stack_memory_pool;
  d_stack_memory_pool.alloc(nstack);
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(treeBuild::d_node_max, &node_max, sizeof(int), 0, cudaMemcpyHostToDevice));

  /*** allocate cell memory ***/
  const int cell_max = nPtcl;
  fprintf(stderr, "celldata= %g MB \n", cell_max*sizeof(CellData)/1024.0/1024.0);
  cuda_mem<CellData> d_cellDataList;
  d_cellDataList.alloc(cell_max);
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(treeBuild::d_cell_max, &cell_max, sizeof(int), 0, cudaMemcpyHostToDevice));

  cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount,16384);

  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&treeBuild::buildOctant      <NLEAF,real_t,true>,  cudaFuncCachePreferShared));
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&treeBuild::buildOctant      <NLEAF,real_t,false>, cudaFuncCachePreferShared));
#if 0
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&treeBuild::buildOctantSingle<NLEAF,T>,       cudaFuncCachePreferShared));
#endif

  CUDA_SAFE_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));



  host_mem<int> h_ncells;
  cuda_mem<int> d_ncells;
  h_ncells.alloc(1);
  d_ncells.alloc(1);
  {
    CUDA_SAFE_CALL(cudaMemset(d_stack_memory_pool,0,nstack*sizeof(int)));
    cudaDeviceSynchronize();
    const double t0 = rtc();
    treeBuild::buildOctree<NLEAF,real_t><<<1,1>>>(
        nPtcl, d_domain, d_cellDataList, d_stack_memory_pool, d_ptclPos, d_ptclPos_tmp, d_ptclVel);
    kernelSuccess("buildOctree");
    const double t1 = rtc();
    const double dt = t1 - t0;
    fprintf(stderr, " buildOctree done in %g sec : %g Mptcl/sec\n",  dt, nPtcl/1e6/dt);
    std::swap(d_ptclPos_tmp.ptr, d_ptclVel.ptr);
  }


}

template struct Treecode<float, _NLEAF>;
