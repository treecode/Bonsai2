#include "Treecode.h"

namespace multipoles
{

  template<typename real_t>
    static __device__ __forceinline__ 
    void addBoxSize(typename vec<3,real_t>::type &rmin, typename vec<3,real_t>::type &rmax, const Particle4<real_t> ptcl);
  
  template<>
    static __device__ __forceinline__ 
    void addBoxSize<float>(float3 &rmin, float3 &rmax, const Particle4<float> ptcl)
  {
#pragma unroll
    for (int i = WARP_SIZE2-1; i >= 0; i--)
    {
      rmin.x = fminf(rmin.x, __shfl_xor(ptcl.x(), 1<<i, WARP_SIZE));
      rmax.x = fminf(rmax.x, __shfl_xor(ptcl.x(), 1<<i, WARP_SIZE));
      
      rmin.y = fminf(rmin.y, __shfl_xor(ptcl.y(), 1<<i, WARP_SIZE));
      rmax.y = fminf(rmax.y, __shfl_xor(ptcl.y(), 1<<i, WARP_SIZE));
      
      rmin.z = fminf(rmin.z, __shfl_xor(ptcl.z(), 1<<i, WARP_SIZE));
      rmax.z = fminf(rmax.z, __shfl_xor(ptcl.z(), 1<<i, WARP_SIZE));
    }
  }

  template<typename real_t>
    static __device__ __forceinline__
    void addMonopole(double4 &M0, const Particle4<real_t> ptcl)
    {
#pragma unroll
      for (int i = WARP_SIZE2-1; i >= 0; i--)
      {
        M0.x += __shfl_xor(ptcl.mass()*ptcl.x(), 1<<i, WARP_SIZE);
        M0.y += __shfl_xor(ptcl.mass()*ptcl.y(), 1<<i, WARP_SIZE);
        M0.z += __shfl_xor(ptcl.mass()*ptcl.z(), 1<<i, WARP_SIZE);
        M0.w += __shfl_xor(ptcl.mass()         , 1<<i, WARP_SIZE);
      }
    }

  template<typename real_t>
    static __device__ __forceinline__
    void addQuadrupole(double3 &Q0, double3 &Q1, const Particle4<real_t> ptcl)
    {
#pragma unroll
      for (int i = WARP_SIZE2-1; i >= 0; i--)
      {
        Q0.x += __shfl_xor(ptcl.mass()*ptcl.x()*ptcl.x(), 1<<i, WARP_SIZE);
        Q0.y += __shfl_xor(ptcl.mass()*ptcl.y()*ptcl.y(), 1<<i, WARP_SIZE);
        Q0.z += __shfl_xor(ptcl.mass()*ptcl.z()*ptcl.z(), 1<<i, WARP_SIZE);
        Q1.x += __shfl_xor(ptcl.mass()*ptcl.x()*ptcl.y(), 1<<i, WARP_SIZE);
        Q1.y += __shfl_xor(ptcl.mass()*ptcl.x()*ptcl.z(), 1<<i, WARP_SIZE);
        Q1.z += __shfl_xor(ptcl.mass()*ptcl.y()*ptcl.z(), 1<<i, WARP_SIZE);
      }
    }

#if 0
  template<int NTHREAD2, typename real_t>
    static __global__ void computeCellMultipoles(
        const int level, 
        const int2 *levelInfo,
        const CellData *cells,
        const Particle4<real_t> *ptclPos)
    {
      const int2 lvl_begendIdx = levelInfo[level];

      const int warpIdx = threadIdx.x >> WARP_SIZE2;
      const int laneIdx = threadIdx.x & (WARP_SIZE-1);

      const int NWARP2  = NTHREAD2 - WARP_SIZE2;
      const int NWARP   = 1 << NWARP2;

      const int cellIdx = lvl_begendIdx.x + blockIdx.x*NWARP + warpIdx;

      if (cellIdx > lvl_begendIdx.y) return;

      /* a warp compute properties of each cell */

      const CellData cell = cells[cellIdx];

      if (cell.isNode())  
      {     /* remember, 8x4 = 32 is a magic number */

        const int firstChild = cell.first();
        const int nChildren  = cell.n();

      }
      else
      { /* process leaf */
        const int firstBody = cell.pbeg();
        const int  lastBody = cell.pend();

        const vec<4,real_t> pnull = {0};
        vec<3,real_t> rmin(static_cast<real_t>(+1e10f));
        vec<3,real_t> rmax(static_cast<real_t>(-1e10f));
        double4 M0 = {0.0};
        double3 Q0 = {0.0};
        double3 Q1 = {0.0};

        for (int i = firstBody+laneIdx; i < lastBody; i += WARP_SIZE)
        {
          const Particle4<real_t> ptcl = i < lastBody ? ptclPos[i + laneIdx] : pnull;

          addBoxsize   (rmin, rmax, ptcl);
          addMonopole  (M0, ptcl);
          addQuadrupole(Q0, Q1, ptcl);
        }
      }
    }
#endif
  
  template<int NTHREAD2, typename real_t>
    static __global__ 
    __launch_bounds__(1<<NTHREAD2, 1024/(1<<NTHREAD2))
    void computeCellMultipoles(
        const int nCells,
        const CellData *cells,
        const Particle4<real_t> *ptclPos,
        __out typename vec<4,real_t>::type *sizeList,
        __out typename vec<4,real_t>::type *monopoleList,
        __out typename vec<3,real_t>::type *quadrpl0List,
        __out typename vec<3,real_t>::type *quadrpl1List)
    {
      const int warpIdx = threadIdx.x >> WARP_SIZE2;
      const int laneIdx = threadIdx.x & (WARP_SIZE-1);

      const int NWARP2  = NTHREAD2 - WARP_SIZE2;
      const int cellIdx = (blockDim.x<<NWARP2) + warpIdx;
      if (cellIdx > nCells) return;

      /* a warp compute properties of each cell */

      const CellData cell = cells[cellIdx];

      const int firstBody = cell.pbeg();
      const int  lastBody = cell.pend();

      const typename vec<4,real_t>::type pnull = {0};
      typename vec<3,real_t>::type rmin = {static_cast<real_t>(+1e10f)};
      typename vec<3,real_t>::type rmax = {static_cast<real_t>(-1e10f)};
      double4 M0 = {0.0};
      double3 Q0 = {0.0};
      double3 Q1 = {0.0};

      for (int i = firstBody+laneIdx; i < lastBody; i += WARP_SIZE)
      {
        const Particle4<real_t> ptcl = i < lastBody ? ptclPos[i] : pnull;

        addBoxSize(rmin, rmax, ptcl);
        addMonopole(M0, ptcl);
        addQuadrupole(Q0, Q1, ptcl);
      }

      if (threadIdx.x == 0)
      {
        sizeList[cellIdx] = make_float4(rmin.x, rmin.y, rmax.z, rmax.y);
        monopoleList[cellIdx] = make_float4(M0.x, M0.y, M0.z, M0.w);
        quadrpl0List[cellIdx] = make_float3(Q0.x, Q0.y, Q0.z);
        quadrpl1List[cellIdx] = make_float3(Q1.x, Q1.y, Q1.z);
      }
    }

};

  template<typename real_t, int NLEAF>
void Treecode<real_t, NLEAF>::computeMultipoles()
{
  cellSize    .realloc(nCells);
  cellMonopole.realloc(nCells);
  cellQuad0   .realloc(nCells);
  cellQuad1   .realloc(nCells);

  const int NTHREAD2 = 8;
  const int NTHREAD  = 1<< NTHREAD2;
  const int NWARP    = 1<<(NTHREAD2-WARP_SIZE2);
  const int nblock   = (nCells-1)/NWARP + 1;

  printf("nblock= %d \n", nblock);

  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&multipoles::computeCellMultipoles<NTHREAD2,real_t>,cudaFuncCachePreferL1));
  cudaDeviceSynchronize();
  const double t0 = rtc();
  multipoles::computeCellMultipoles<NTHREAD2,real_t><<<nblock,NTHREAD>>>(
      nCells, d_cellDataList, d_ptclPos,
      cellSize, cellMonopole, cellQuad0, cellQuad1);
  kernelSuccess("cellMultipole");
  const double dt = rtc() - t0;
  fprintf(stderr, " cellMultipole done in %g sec : %g Mptcl/sec  %g Mcell/sec\n",  dt, nPtcl/1e6/dt, nCells/1e6/dt);
}

#include "TreecodeInstances.h"

