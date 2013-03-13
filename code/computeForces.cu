#include "Treecode.h"
#include <algorithm>

#include "cuda_primitives.h"

namespace computeForces
{

#define CELL_LIST_MEM_PER_WARP (4096*32)
  
  texture<uint4,  1, cudaReadModeElementType> texCellData;
  texture<float4, 1, cudaReadModeElementType> texCellSize;
  texture<float4, 1, cudaReadModeElementType> texCellMonopole;
  texture<float4, 1, cudaReadModeElementType> texCellQuad0;
  texture<float2, 1, cudaReadModeElementType> texCellQuad1;
  texture<float4, 1, cudaReadModeElementType> texPtcl;

  template<int SHIFT>
    __forceinline__ static __device__ int ringAddr(const int i)
    {
      return (i & ((CELL_LIST_MEM_PER_WARP<<SHIFT) - 1));
    }


  /*******************************/
  /****** Opening criterion ******/
  /*******************************/

  //Improved Barnes Hut criterium
  static __device__ bool split_node_grav_impbh(
      const float4 cellSize, 
      const float3 groupCenter, 
      const float3 groupSize)
  {
    //Compute the distance between the group and the cell
    float3 dr = make_float3(
        fabsf(groupCenter.x - cellSize.x) - (groupSize.x),
        fabsf(groupCenter.y - cellSize.y) - (groupSize.y),
        fabsf(groupCenter.z - cellSize.z) - (groupSize.z)
        );

    dr.x += fabsf(dr.x); dr.x *= 0.5f;
    dr.y += fabsf(dr.y); dr.y *= 0.5f;
    dr.z += fabsf(dr.z); dr.z *= 0.5f;

    //Distance squared, no need to do sqrt since opening criteria has been squared
    const float ds2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;

#if 1
    return (ds2 < fabsf(cellSize.w));
#else
    return true;
#endif
  }

  /******* force due to monopoles *********/

  template<typename real_t>
  static __device__ __forceinline__ typename vec<4,real_t>::type add_acc(
      typename vec<4,real_t>::type acc,  const float3 pos,
      const float massj, const float3 posj,
      const float eps2)
  {
    const float3 dr = make_float3(posj.x - pos.x, posj.y - pos.y, posj.z - pos.z);

    const float r2     = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z + eps2;
    const float rinv   = rsqrtf(r2);
    const float rinv2  = rinv*rinv;
    const float mrinv  = massj * rinv;
    const float mrinv3 = mrinv * rinv2;

    acc.w -= mrinv;
    acc.x += mrinv3 * dr.x;
    acc.y += mrinv3 * dr.y;
    acc.z += mrinv3 * dr.z;

    return acc;
  }


  /******* force due to quadrupoles *********/

  template<typename real_t>
  static __device__ __forceinline__ typename vec<4,real_t>::type add_acc(
      typename vec<4,real_t>::type acc, 
      const float3 pos,
      const float mass, const float3 com,
      const float4 Q0,  const float2 Q1, float eps2) 
  {
    const float3 dr = make_float3(pos.x - com.x, pos.y - com.y, pos.z - com.z);
    const float  r2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z + eps2;

    const float rinv  = rsqrtf(r2);
    const float rinv2 = rinv *rinv;
    const float mrinv  =  mass*rinv;
    const float mrinv3 = rinv2*mrinv;
    const float mrinv5 = rinv2*mrinv3; 
    const float mrinv7 = rinv2*mrinv5;   // 16

    float  D0  =  mrinv;
    float  D1  = -mrinv3;
    float  D2  =  mrinv5*(  3.0f);
    float  D3  =  mrinv7*(-15.0f); // 3

    const float q11 = Q0.x;
    const float q22 = Q0.y;
    const float q33 = Q0.z;
    const float q12 = Q0.w;
    const float q13 = Q1.x;
    const float q23 = Q1.y;

    const float  q  = q11 + q22 + q33;
    const float3 qR = make_float3(
        q11*dr.x + q12*dr.y + q13*dr.z,
        q12*dr.x + q22*dr.y + q23*dr.z,
        q13*dr.x + q23*dr.y + q33*dr.z);
    const float qRR = qR.x*dr.x + qR.y*dr.y + qR.z*dr.z;  // 22

    acc.w  -= D0 + 0.5f*(D1*q + D2*qRR);
    float C = D1 + 0.5f*(D2*q + D3*qRR);
    acc.x  += C*dr.x + D2*qR.x;
    acc.y  += C*dr.y + D2*qR.y;
    acc.z  += C*dr.z + D2*qR.z;               // 23

    // total: 16 + 3 + 22 + 23 = 64 flops 

    return acc;
  }


  /******* evalue forces from particles *******/
  template<int NI, bool FULL, typename real_t>
    static __device__ __forceinline__ void directAcc(
        typename vec<4,real_t>::type acc_i[NI], 
        const float3 pos_i[NI],
        const int ptclIdx,
        const float eps2)
    {
#if 1
#if 1
      const float4 M0 = (FULL || ptclIdx >= 0) ? tex1Dfetch(texPtcl, ptclIdx) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#else
      const float4 M0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#endif

//#pragma unroll
      for (int j = 0; j < WARP_SIZE; j++)
      {
        const float4 jM0 = make_float4(__shfl(M0.x, j), __shfl(M0.y, j), __shfl(M0.z, j), __shfl(M0.w,j));
        const float  jmass = jM0.w;
        const float3 jpos  = make_float3(jM0.x, jM0.y, jM0.z);
#pragma unroll
        for (int k = 0; k < NI; k++)
          acc_i[k] = add_acc<real_t>(acc_i[k], pos_i[k], jmass, jpos, eps2);
      }
#endif
    }

  /******* evalue forces from cells *******/
#ifdef  QUADRUPOLE
  template<int NI, bool FULL, typename real_t>
    static __device__ __forceinline__ void approxAcc(
        typename vec<4,real_t>::type acc_i[NI], 
        const float3 pos_i[NI],
        const int cellIdx,
        const float eps2)
    {
#if 1
      float4 M0, Q0;
      float2 Q1;
      if (FULL || cellIdx >= 0)
      {
        M0 = tex1Dfetch(texCellMonopole, cellIdx);
        const Quadrupole<float> Q(tex1Dfetch(texCellQuad0,cellIdx), tex1Dfetch(texCellQuad1,cellIdx));
        Q0 = Q.get_q0();
        Q1 = Q.get_q1();
      }
      else
      {
        M0 = Q0 =make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        Q1 = make_float2(0.0f, 0.0f);
      }

//#pragma unroll
      for (int j = 0; j < WARP_SIZE; j++)
      {
        const float4 jM0 = make_float4(__shfl(M0.x, j), __shfl(M0.y, j), __shfl(M0.z, j), __shfl(M0.w,j));
        const float4 jQ0 = make_float4(__shfl(Q0.x, j), __shfl(Q0.y, j), __shfl(Q0.z, j), __shfl(Q0.w,j));
        const float2 jQ1 = make_float2(__shfl(Q1.x, j), __shfl(Q1.y, j));
        const float  jmass = jM0.w;
        const float3 jpos  = make_float3(jM0.x, jM0.y, jM0.z);
#pragma unroll
        for (int k = 0; k < NI; k++)
          acc_i[k] = add_acc<real_t>(acc_i[k], pos_i[k], jmass, jpos, jQ0, jQ1, eps2);
      }
#endif
    }
#else
  template<int NI, bool FULL, typename real_t>
    static __device__ __forceinline__ void approxAcc(
        float4 acc_i[NI], 
        const float3 pos_i[NI],
        const int cellIdx,
        const float eps2)
    {
#if 1
#if 1
      const float4 M0 = (FULL || cellIdx >= 0) ? tex1Dfetch(texCellMonopole, cellIdx) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#else
      const float4 M0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#endif

//#pragma unroll
      for (int j = 0; j < WARP_SIZE; j++)
      {
        const float4 jM0 = make_float4(__shfl(M0.x, j), __shfl(M0.y, j), __shfl(M0.z, j), __shfl(M0.w,j));
        const float  jmass = jM0.w;
        const float3 jpos  = make_float3(jM0.x, jM0.y, jM0.z);
#pragma unroll
        for (int k = 0; k < NI; k++)
          acc_i[k] = add_acc<real_t>(acc_i[k], pos_i[k], jmass, jpos, eps2);
      }
#endif
    }
#endif



  template<int SHIFT, int BLOCKDIM2, int NI, typename real_t>
    static __device__ 
    uint2 treewalk_warp(
        typename vec<4,real_t>::type acc_i[NI],
        const float3 _pos_i[NI],
        const float3 groupCentre,
        const float3 groupSize,
        const float eps2,
        const int2 top_cells,
        int *shmem,
        int *cellList)
    {
      const int laneIdx = threadIdx.x & (WARP_SIZE-1);

      /* this helps to unload register pressure */
      float3 pos_i[NI];
#pragma unroll 1
      for (int i = 0; i < NI; i++)
        pos_i[i] = _pos_i[i];

      uint2 interactionCounters = {0,0}; /* # of approximate and exact force evaluations */

      volatile int *tmpList = shmem;

      int approxCellIdx, directPtclIdx;

      int directCounter = 0;
      int approxCounter = 0;


      for (int root_cell = top_cells.x; root_cell < top_cells.y; root_cell += WARP_SIZE)
        if (root_cell + laneIdx < top_cells.y)
          cellList[ringAddr<SHIFT>(root_cell - top_cells.x + laneIdx)] = root_cell + laneIdx;

      int nCells = top_cells.y - top_cells.x;

      int cellListBlock        = 0;
      int nextLevelCellCounter = 0;

      unsigned int cellListOffset = 0;

      /* process level with n_cells */
#if 1
      while (nCells > 0)
      {
        /* extract cell index from the current level cell list */
        const int cellListIdx = cellListBlock + laneIdx;
        const bool useCell    = cellListIdx < nCells;
        const int cellIdx     = cellList[ringAddr<SHIFT>(cellListOffset + cellListIdx)];
        cellListBlock += min(WARP_SIZE, nCells - cellListBlock);

        /* read from gmem cell's info */
        const float4   cellSize = tex1Dfetch(texCellSize, cellIdx);
        const CellData cellData = tex1Dfetch(texCellData, cellIdx);

        const bool splitCell = split_node_grav_impbh(cellSize, groupCentre, groupSize) ||
          (cellData.pend() - cellData.pbeg() < 3); /* force to open leaves with less than 3 particles */

        /**********************************************/
        /* split cells that satisfy opening condition */
        /**********************************************/

        const bool isNode = cellData.isNode();

        {
          const int firstChild = cellData.first();
          const int nChild= cellData.n();
          bool splitNode  = isNode && splitCell && useCell;

          /* use exclusive scan to compute scatter addresses for each of the child cells */
          const int2 childScatter = warpIntExclusiveScan(nChild & (-splitNode));

          /* make sure we still have available stack space */
          if (childScatter.y + nCells - cellListBlock > (CELL_LIST_MEM_PER_WARP<<SHIFT))
            return make_uint2(0xFFFFFFFF,0xFFFFFFFF);

#if 0
          /* if so populate next level stack in gmem */
          if (splitNode)
          {
            const int scatterIdx = cellListOffset + nCells + nextLevelCellCounter + childScatter.x;
            for (int i = 0; i < nChild; i++)
              cellList[ringAddr<SHIFT>(scatterIdx + i)] = firstChild + i;
          }
#else  /* use scan operation to accomplish steps above, doesn't bring performance benefit */
          int nChildren  = childScatter.y;
          int nProcessed = 0;
          int2 scanVal   = {0,0};
          const int offset = cellListOffset + nCells + nextLevelCellCounter;
          while (nChildren > 0)
          {
            tmpList[laneIdx] = 1;
            if (splitNode && (childScatter.x - nProcessed < WARP_SIZE))
            {
              splitNode = false;
              tmpList[childScatter.x - nProcessed] = -1-firstChild;
            }
            scanVal = inclusive_segscan_warp(tmpList[laneIdx], scanVal.y);
            if (laneIdx < nChildren)
              cellList[ringAddr<SHIFT>(offset + nProcessed + laneIdx)] = scanVal.x;
            nChildren  -= WARP_SIZE;
            nProcessed += WARP_SIZE;
          }
#endif
          nextLevelCellCounter += childScatter.y;  /* increment nextLevelCounter by total # of children */
        }

#if 1
        {
          /***********************************/
          /******       APPROX          ******/
          /***********************************/

          /* see which thread's cell can be used for approximate force calculation */
          const bool approxCell    = !splitCell && useCell;
          const int2 approxScatter = warpBinExclusiveScan(approxCell);

          /* store index of the cell */
          const int scatterIdx = approxCounter + approxScatter.x;
          tmpList[laneIdx] = approxCellIdx;
          if (approxCell && scatterIdx < WARP_SIZE)
            tmpList[scatterIdx] = cellIdx;

          approxCounter += approxScatter.y;

          /* compute approximate forces */
          if (approxCounter >= WARP_SIZE)
          {
            /* evalute cells stored in shmem */
            approxAcc<NI,true,real_t>(acc_i, pos_i, tmpList[laneIdx], eps2);

            approxCounter -= WARP_SIZE;
            const int scatterIdx = approxCounter + approxScatter.x - approxScatter.y;
            if (approxCell && scatterIdx >= 0)
              tmpList[scatterIdx] = cellIdx;
            interactionCounters.x += WARP_SIZE;
          }
          approxCellIdx = tmpList[laneIdx];
        }
#endif

#if 1
        {
          /***********************************/
          /******       DIRECT          ******/
          /***********************************/

          const bool isLeaf = !isNode;
          bool isDirect = splitCell && isLeaf && useCell;

          const int firstBody = cellData.pbeg();
          const int     nBody = cellData.pend() - cellData.pbeg();

          const int2 childScatter = warpIntExclusiveScan(nBody & (-isDirect));
          int nParticle  = childScatter.y;
          int nProcessed = 0;
          int2 scanVal   = {0,0};

          /* conduct segmented scan for all leaves that need to be expanded */
          while (nParticle > 0)
          {
            tmpList[laneIdx] = 1;
            if (isDirect && (childScatter.x - nProcessed < WARP_SIZE))
            {
              isDirect = false;
              tmpList[childScatter.x - nProcessed] = -1-firstBody;
            }
            scanVal = inclusive_segscan_warp(tmpList[laneIdx], scanVal.y);
            const int  ptclIdx = scanVal.x;

            if (nParticle >= WARP_SIZE)
            {
              directAcc<NI,true, real_t>(acc_i, pos_i, ptclIdx, eps2);
              nParticle  -= WARP_SIZE;
              nProcessed += WARP_SIZE;
              interactionCounters.y += WARP_SIZE;
            }
            else 
            {
              const int scatterIdx = directCounter + laneIdx;
              tmpList[laneIdx] = directPtclIdx;
              if (scatterIdx < WARP_SIZE)
                tmpList[scatterIdx] = ptclIdx;

              directCounter += nParticle;

              if (directCounter >= WARP_SIZE)
              {
                /* evalute cells stored in shmem */
                directAcc<NI,true, real_t>(acc_i, pos_i, tmpList[laneIdx], eps2);
                directCounter -= WARP_SIZE;
                const int scatterIdx = directCounter + laneIdx - nParticle;
                if (scatterIdx >= 0)
                  tmpList[scatterIdx] = ptclIdx;
                interactionCounters.y += WARP_SIZE;
              }
              directPtclIdx = tmpList[laneIdx];

              nParticle = 0;
            }
          }
        }
#endif

        /* if the current level is processed, schedule the next level */
        if (cellListBlock >= nCells)
        {
          cellListOffset += nCells;
          nCells = nextLevelCellCounter;
          cellListBlock = nextLevelCellCounter = 0;
        }

      }  /* level completed */
#endif

#if 1
      if (approxCounter > 0)
      {
        approxAcc<NI,false, real_t>(acc_i, pos_i, laneIdx < approxCounter ? approxCellIdx : -1, eps2);
        interactionCounters.x += approxCounter;
        approxCounter = 0;
      }
#endif

#if 1
      if (directCounter > 0)
      {
        directAcc<NI,false,real_t>(acc_i, pos_i, laneIdx < directCounter ? directPtclIdx : -1, eps2);
        interactionCounters.y += directCounter;
        directCounter = 0;
      }
#endif

      return interactionCounters;
    }

  __device__ unsigned int retired_groupCount = 0;

  __device__ unsigned long long g_direct_sum = 0;
  __device__ unsigned int       g_direct_max = 0;

  __device__ unsigned long long g_approx_sum = 0;
  __device__ unsigned int       g_approx_max = 0;

  __device__ double grav_potential = 0.0;

  template<int NTHREAD2, bool STATS, int NI>
    __launch_bounds__(1<<NTHREAD2, 1024/(1<<NTHREAD2))
    static __global__ 
    void treewalk(
        const int nGroups,
        const GroupData *groupList,
        const float eps2,
        const int start_level,
        const int2 *level_begIdx,
        const Particle4<float> *ptclPos,
        __out Particle4<float> *acc,
        __out int    *gmem_pool)
    {
      typedef float real_t;
      typedef typename vec<3,real_t>::type real3_t;
      typedef typename vec<4,real_t>::type real4_t;

#if 0
      typedef double real_acc;
#else
      typedef float real_acc;
#endif
      typedef typename vec<4,real_acc>::type real4_acc;

      const int NTHREAD = 1<<NTHREAD2;
      const int shMemSize = NTHREAD;
      __shared__ int shmem_pool[shMemSize];

      const int laneIdx = threadIdx.x & (WARP_SIZE-1);
      const int warpIdx = threadIdx.x >> WARP_SIZE2;

      const int NWARP2 = NTHREAD2 - WARP_SIZE2;
      const int sh_offs = (shMemSize >> NWARP2) * warpIdx;
      int *shmem = shmem_pool + sh_offs;
      int *gmem  =  gmem_pool + CELL_LIST_MEM_PER_WARP*((blockIdx.x<<NWARP2) + warpIdx);

      int2 top_cells = level_begIdx[start_level];
      top_cells.y++;

      while(1)
      {
        int groupIdx = 0;
        if (laneIdx == 0)
          groupIdx = atomicAdd(&retired_groupCount, 1);
        groupIdx = __shfl(groupIdx, 0, WARP_SIZE);

        if (groupIdx >= nGroups) 
          return;

        const GroupData group = groupList[groupIdx];
        const int pbeg = group.pbeg();
        const int np   = group.np();

        real3_t iPos[NI];
        real_t  iMass[NI];

#pragma unroll
        for (int i = 0; i < NI; i++)
        {
          const Particle4<real_t> ptcl = ptclPos[min(pbeg + i*WARP_SIZE+laneIdx, pbeg+np-1)];
          iPos [i] = make_float3(ptcl.x(), ptcl.y(), ptcl.z());
          iMass[i] = ptcl.mass();
        }

        real3_t rmin = {iPos[0].x, iPos[0].y, iPos[0].z};
        real3_t rmax = rmin; 

#pragma unroll
        for (int i = 0; i < NI; i++) 
          addBoxSize(rmin, rmax, Position<real_t>(iPos[i].x, iPos[i].y, iPos[i].z));

        rmin.x = __shfl(rmin.x,0);
        rmin.y = __shfl(rmin.y,0);
        rmin.z = __shfl(rmin.z,0);
        rmax.x = __shfl(rmax.x,0);
        rmax.y = __shfl(rmax.y,0);
        rmax.z = __shfl(rmax.z,0);

        const real_t half = static_cast<real_t>(0.5f);
        const real3_t cvec = {half*(rmax.x+rmin.x), half*(rmax.y+rmin.y), half*(rmax.z+rmin.z)};
        const real3_t hvec = {half*(rmax.x-rmin.x), half*(rmax.y-rmin.y), half*(rmax.z-rmin.z)};

        const int SHIFT = 0;

        real4_acc iAcc[NI] = {vec<4,real_acc>::null()};

        uint2 counters;
        counters =  treewalk_warp<SHIFT,NTHREAD2,NI,real_acc>
          (iAcc, iPos, cvec, hvec, eps2, top_cells, shmem, gmem);

        assert(!(counters.x == 0xFFFFFFFF && counters.y == 0xFFFFFFFF));

        const int pidx = pbeg + laneIdx;
        if (STATS)
        {
          int direct_max = counters.y;
          int direct_sum = 0;

          int approx_max = counters.x;
          int approx_sum = 0;

          real_acc gpot = static_cast<real_acc>(0.0f);

#pragma unroll
          for (int i = 0; i < NI; i++)
            if (i*WARP_SIZE + pidx < pbeg + np)
            {
              gpot       += iAcc[i].w*iMass[i];
              approx_sum += counters.x;
              direct_sum += counters.y;
            }

#pragma unroll
          for (int i = WARP_SIZE2-1; i >= 0; i--)
          {
            direct_max  = max(direct_max, __shfl_xor(direct_max, 1<<i));
            direct_sum += __shfl_xor(direct_sum, 1<<i);

            approx_max  = max(approx_max, __shfl_xor(approx_max, 1<<i));
            approx_sum += __shfl_xor(approx_sum, 1<<i);

            gpot += shfl_xor(gpot, 1<<i);
          }

          if (laneIdx == 0)
          {
            atomicMax(&g_direct_max,                     direct_max);
            atomicAdd(&g_direct_sum, (unsigned long long)direct_sum);

            atomicMax(&g_approx_max,                     approx_max);
            atomicAdd(&g_approx_sum, (unsigned long long)approx_sum);

            atomicAdd_double(&grav_potential, static_cast<real_acc>(0.5f)*gpot);
          }
        }

#pragma unroll
        for (int i = 0; i < NI; i++)
          if (pidx + i*WARP_SIZE< pbeg + np)
          {
            const real4_t iacc = {iAcc[i].x, iAcc[i].y, iAcc[i].z, iAcc[i].w};
            acc[i*WARP_SIZE + pidx] = iacc;
          }
      }
    }
}

  template<typename real_t>
double4 Treecode<real_t>::computeForces(const bool INTCOUNT)
{
  bindTexture(computeForces::texCellData,     (uint4* )d_cellDataList.ptr, nCells);
  bindTexture(computeForces::texCellSize,     d_cellSize.ptr,     nCells);
  bindTexture(computeForces::texCellMonopole, d_cellMonopole.ptr, nCells);
  bindTexture(computeForces::texCellQuad0,    d_cellQuad0.ptr,    nCells);
  bindTexture(computeForces::texCellQuad1,    d_cellQuad1.ptr,    nCells);
  bindTexture(computeForces::texPtcl,         d_ptclPos.ptr,      nPtcl);

  const int NTHREAD2 = 7;
  const int NTHREAD  = 1<<NTHREAD2;
  cuda_mem<int> d_gmem_pool;

  const int nblock = 8*13;
  printf("---1--\n");
  d_gmem_pool.alloc(CELL_LIST_MEM_PER_WARP*nblock*(NTHREAD/WARP_SIZE));
  printf("---2--\n");

#if 0
  CUDA_SAFE_CALL(cudaMemset(d_ptclAcc, 0, sizeof(Particle)*nPtcl));
#endif
  const int starting_level = 1;
  int value = 0;
  cudaDeviceSynchronize();
  const double t0 = rtc();
  unsigned long long lzero = 0;
  unsigned int       uzero = 0;
  double              fzero = 0.0;
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(computeForces::retired_groupCount, &value, sizeof(int)));
  if (INTCOUNT)
  {
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(computeForces::g_direct_sum, &lzero, sizeof(unsigned long long)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(computeForces::g_direct_max, &uzero, sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(computeForces::g_approx_sum, &lzero, sizeof(unsigned long long)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(computeForces::g_approx_max, &uzero, sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(computeForces::grav_potential, &fzero, sizeof(double)));
  }

  if (INTCOUNT)
  {
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&computeForces::treewalk<NTHREAD2,true,1>, cudaFuncCachePreferL1));
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&computeForces::treewalk<NTHREAD2,true,2>, cudaFuncCachePreferL1));
    if (nCrit <= WARP_SIZE)
      computeForces::treewalk<NTHREAD2,true,1><<<nblock,NTHREAD>>>(
          nGroups, d_groupList, eps2, starting_level, d_level_begIdx,
          d_ptclPos_tmp, d_ptclAcc,
          d_gmem_pool);
    else if (nCrit <= 2*WARP_SIZE)
      computeForces::treewalk<NTHREAD2,true,2><<<nblock,NTHREAD>>>(
          nGroups, d_groupList, eps2, starting_level, d_level_begIdx,
          d_ptclPos_tmp, d_ptclAcc,
          d_gmem_pool);
    else
      assert(0);
  }
  else
  {
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&computeForces::treewalk<NTHREAD2,false,1>, cudaFuncCachePreferL1));
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&computeForces::treewalk<NTHREAD2,false,2>, cudaFuncCachePreferL1));
    if (nCrit <= WARP_SIZE)
      computeForces::treewalk<NTHREAD2,false,1><<<nblock,NTHREAD>>>(
          nGroups, d_groupList, eps2, starting_level, d_level_begIdx,
          d_ptclPos_tmp, d_ptclAcc,
          d_gmem_pool);
    else if (nCrit <= 2*WARP_SIZE)
      computeForces::treewalk<NTHREAD2,false,2><<<nblock,NTHREAD>>>(
          nGroups, d_groupList, eps2, starting_level, d_level_begIdx,
          d_ptclPos_tmp, d_ptclAcc,
          d_gmem_pool);
    else
      assert(0);
  }
  kernelSuccess("treewalk");
  const double t1 = rtc();
  const double dt = t1 - t0;
  fprintf(stderr, " treewalk done in %g sec : %g Mptcl/sec\n",  dt, nPtcl/1e6/dt);


  double4 interactions = {0.0, 0.0, 0.0, 0.0};

  if (INTCOUNT)
  {
    unsigned long long direct_sum, approx_sum;
    unsigned int direct_max, approx_max;
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&direct_sum,     computeForces::g_direct_sum, sizeof(unsigned long long)));
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&direct_max,     computeForces::g_direct_max, sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&approx_sum,     computeForces::g_approx_sum, sizeof(unsigned long long)));
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&approx_max,     computeForces::g_approx_max, sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&grav_potential, computeForces::grav_potential, sizeof(double)));
    interactions.x = direct_sum*1.0/nPtcl;
    interactions.y = direct_max;
    interactions.z = approx_sum*1.0/nPtcl;
    interactions.w = approx_max;
    fprintf(stderr, " grav potential= %g \n", grav_potential);
  }

  unbindTexture(computeForces::texPtcl);
  unbindTexture(computeForces::texCellQuad1);
  unbindTexture(computeForces::texCellQuad0);
  unbindTexture(computeForces::texCellMonopole);
  unbindTexture(computeForces::texCellSize);
  unbindTexture(computeForces::texCellData);

  return interactions;
}

#include "TreecodeInstances.h"

