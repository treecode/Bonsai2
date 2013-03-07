#include "Treecode.h"

namespace multipoles
{

#if 0  /* just a test case, but works for float only */
#define _USE_RESTRICT_ 
#endif

  template<typename real_t>
    static __device__ __forceinline__ 
    void addBoxSize(typename vec<3,real_t>::type &rmin, typename vec<3,real_t>::type &rmax, const Particle4<real_t> ptcl);
  template<typename real_t>
    static __device__ __forceinline__
    void addMonopole(double4 &M0, const Particle4<real_t> ptcl);
  template<typename real_t>
    static __device__ __forceinline__
    void addQuadrupole(double3 &Q0, double3 &Q1, const Particle4<real_t> ptcl);
  
  template<>
    static __device__ __forceinline__ 
    void addBoxSize<float>(float3 &_rmin, float3 &_rmax, const Particle4<float> ptcl)
  {
    float3 rmin = {ptcl.x(), ptcl.y(), ptcl.z()};
    float3 rmax = rmin;

#pragma unroll
    for (int i = WARP_SIZE2-1; i >= 0; i--)
    {
      rmin.x = fminf(rmin.x, __shfl_xor(rmin.x, 1<<i, WARP_SIZE));
      rmax.x = fminf(rmax.x, __shfl_xor(rmax.x, 1<<i, WARP_SIZE));
      
      rmin.y = fminf(rmin.y, __shfl_xor(rmin.y, 1<<i, WARP_SIZE));
      rmax.y = fminf(rmax.y, __shfl_xor(rmax.y, 1<<i, WARP_SIZE));
      
      rmin.z = fminf(rmin.z, __shfl_xor(rmin.z, 1<<i, WARP_SIZE));
      rmax.z = fminf(rmax.z, __shfl_xor(rmax.z, 1<<i, WARP_SIZE));
    }
    const int laneIdx = threadIdx.x & (WARP_SIZE-1);
    if (laneIdx == 0)
    {
      _rmin.x = fminf(_rmin.x, rmin.x);
      _rmin.y = fminf(_rmin.y, rmin.y);
      _rmin.z = fminf(_rmin.z, rmin.z);
      
      _rmax.x = fmaxf(_rmax.x, rmax.x);
      _rmax.y = fmaxf(_rmax.y, rmax.y);
      _rmax.z = fmaxf(_rmax.z, rmax.z);
    }
  }

  template<>
    static __device__ __forceinline__
    void addMonopole<float>(double4 &_M0, const Particle4<float> ptcl)
    {
      float4 M0 = {ptcl.mass()*ptcl.x(), ptcl.mass()*ptcl.y(), ptcl.mass()*ptcl.z(), ptcl.mass()};
#pragma unroll
      for (int i = WARP_SIZE2-1; i >= 0; i--)
      {
        M0.x += __shfl_xor(M0.x, 1<<i, WARP_SIZE);
        M0.y += __shfl_xor(M0.y, 1<<i, WARP_SIZE);
        M0.z += __shfl_xor(M0.z, 1<<i, WARP_SIZE);
        M0.w += __shfl_xor(M0.w, 1<<i, WARP_SIZE);
      }

      const int laneIdx = threadIdx.x & (WARP_SIZE-1);
      if (laneIdx == 0)
      {
        _M0.x += M0.x;
        _M0.y += M0.y;
        _M0.z += M0.z;
        _M0.w += M0.w;
      }
    }

  template<typename real_t>
    static __device__ __forceinline__
    void addQuadrupole(double3 &_Q0, double3 &_Q1, const Particle4<real_t> ptcl)
    {
      float3 Q0 = {ptcl.mass()*ptcl.x()*ptcl.x(), ptcl.mass()*ptcl.y()*ptcl.y(), ptcl.mass()*ptcl.z()*ptcl.z()};
      float3 Q1 = {ptcl.mass()*ptcl.x()*ptcl.y(), ptcl.mass()*ptcl.x()*ptcl.z(), ptcl.mass()*ptcl.y()*ptcl.z()};
#pragma unroll
      for (int i = WARP_SIZE2-1; i >= 0; i--)
      {
        Q0.x += __shfl_xor(Q0.x, 1<<i, WARP_SIZE);
        Q0.y += __shfl_xor(Q0.y, 1<<i, WARP_SIZE);
        Q0.z += __shfl_xor(Q0.z, 1<<i, WARP_SIZE);
        Q1.x += __shfl_xor(Q1.x, 1<<i, WARP_SIZE);
        Q1.y += __shfl_xor(Q1.y, 1<<i, WARP_SIZE);
        Q1.z += __shfl_xor(Q1.z, 1<<i, WARP_SIZE);
      }

      const int laneIdx = threadIdx.x & (WARP_SIZE-1);
      if (laneIdx == 0)
      {
        _Q0.x += Q0.x;
        _Q0.y += Q0.y;
        _Q0.z += Q0.z;
        _Q1.x += Q1.x;
        _Q1.y += Q1.y;
        _Q1.z += Q1.z;
      }
    }

#if 0
  template<int NTHREAD2, typename real_t>
    static __global__ void computeCellMultipoles(
        const int level, 
        const int2 *levelInfo,
        const CellData *cells,
        const Particle4<real_t> *ptclPos);
#endif

  __device__ unsigned int nflops = 0;

  template<int NTHREAD2, typename real_t>
    static __global__ 
    __launch_bounds__(1<<NTHREAD2, 1024/(1<<NTHREAD2))
    void computeCellMultipoles(
        const int nCells,
        const CellData *cells,
#ifdef _USE_RESTRICT_
        const float4* __restrict__ ptclPos,
#else
        const Particle4<real_t> *ptclPos,
#endif
        const real_t inv_theta,
        __out typename vec<4,real_t>::type *sizeList,
        __out typename vec<4,real_t>::type *monopoleList,
        __out typename vec<3,real_t>::type *quadrpl0List,
        __out typename vec<3,real_t>::type *quadrpl1List)
    {
      const int warpIdx = threadIdx.x >> WARP_SIZE2;
      const int laneIdx = threadIdx.x & (WARP_SIZE-1);

      const int NWARP2  = NTHREAD2 - WARP_SIZE2;
      const int cellIdx = (blockIdx.x<<NWARP2) + warpIdx;
      if (cellIdx > nCells) return;

      /* a warp compute properties of each cell */

      const CellData cell = cells[cellIdx];

      typename vec<3,real_t>::type rmin = {static_cast<real_t>(+1e10f)};
      typename vec<3,real_t>::type rmax = {static_cast<real_t>(-1e10f)};
      double4 M0 = {0.0};
      double3 Q0 = {0.0};
      double3 Q1 = {0.0};

      unsigned int nflop = 0;
#if 0
      if (cell.isNode())
      {
      }
      else
#endif
      {
        const int firstBody = cell.pbeg();
        const int  lastBody = cell.pend();

        const typename vec<4,real_t>::type pnull = {0};

        for (int i = firstBody+laneIdx; i < lastBody; i += WARP_SIZE)
        {
          nflop++;
#ifdef _USE_RESTRICT_
          const float4 _ptcl = i < lastBody ? ptclPos[i] : pnull;
          const Particle4<real_t> ptcl(_ptcl);
#else
          const Particle4<real_t> ptcl = i < lastBody ? ptclPos[i] : pnull;
#endif

          addBoxSize(rmin, rmax, ptcl);
          addMonopole(M0, ptcl);
          addQuadrupole(Q0, Q1, ptcl);
        }
      }


      if (laneIdx == 0)
      {
        const double inv_mass = 1.0/M0.w;
        M0.x *= inv_mass;
        M0.y *= inv_mass;
        M0.z *= inv_mass;
#if 0
        Q0.x = Q0.x*inv_mass - M0.x*M0.x;
        Q0.y = Q0.y*inv_mass - M0.y*M0.y;
        Q0.z = Q0.z*inv_mass - M0.z*M0.z;
        Q1.x = Q1.x*inv_mass - M0.x*M0.y;
        Q1.y = Q1.y*inv_mass - M0.x*M0.z;
        Q1.z = Q1.z*inv_mass - M0.y*M0.z;
#endif
#if 0
        {
          const double Qtmp = Q1.y;
          Q1.y = Q1.z;
          Q1.z = Qtmp;
        }
#endif

        const Position<real_t> cvec((rmax.x+rmin.x)*real_t(0.5f), (rmax.y+rmin.y)*real_t(0.5f), (rmax.z+rmin.z)*real_t(0.5f));
        const Position<real_t> hvec((rmax.x-rmin.x)*real_t(0.5f), (rmax.y-rmin.y)*real_t(0.5f), (rmax.z-rmin.z)*real_t(0.5f));
        const Position<real_t> com(M0.x, M0.y, M0.z);
        const real_t dx = cvec.x - com.x;
        const real_t dy = cvec.y - com.y;
        const real_t dz = cvec.z - com.z;
        const real_t  s = sqrt(dx*dy + dy*dz + dz);
        const real_t  l = max(static_cast<real_t>(2.0f)*max(hvec.x, max(hvec.y, hvec.z)), static_cast<real_t>(1.0e-6f));
        const real_t cellOp = (l*inv_theta) + s;
        const real_t cellOp2 = cellOp*cellOp;

        atomicAdd(&nflops, nflop);
        sizeList[cellIdx] = make_float4(com.x, com.y, com.z, cellOp2);
        monopoleList[cellIdx] = make_float4(M0.x, M0.y, M0.z, M0.w);
        quadrpl0List[cellIdx] = make_float3(Q0.x, Q0.y, Q0.z);
        quadrpl1List[cellIdx] = make_float3(Q1.x, Q1.y, Q1.z);
      }
    }

};

  template<typename real_t, int NLEAF>
void Treecode<real_t, NLEAF>::computeMultipoles()
{
  d_cellSize    .realloc(nCells);
  d_cellMonopole.realloc(nCells);
  d_cellQuad0   .realloc(nCells);
  d_cellQuad1   .realloc(nCells);

  const int NTHREAD2 = 8;
  const int NTHREAD  = 1<< NTHREAD2;
  const int NWARP    = 1<<(NTHREAD2-WARP_SIZE2);
  const int nblock   = (nCells-1)/NWARP + 1;

  printf("nblock= %d \n", nblock);

  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&multipoles::computeCellMultipoles<NTHREAD2,real_t>,cudaFuncCachePreferL1));
  cudaDeviceSynchronize();
  const double t0 = rtc();
#ifdef _USE_RESTRICT_
  multipoles::computeCellMultipoles<NTHREAD2,real_t><<<nblock,NTHREAD>>>(
      nCells, d_cellDataList, (float4*)d_ptclPos.ptr,
      1.0/theta,
      d_cellSize, d_cellMonopole, d_cellQuad0, d_cellQuad1);
#else
  multipoles::computeCellMultipoles<NTHREAD2,real_t><<<nblock,NTHREAD>>>(
      nCells, d_cellDataList, d_ptclPos,
      1.0/theta,
      d_cellSize, d_cellMonopole, d_cellQuad0, d_cellQuad1);
#endif
  kernelSuccess("cellMultipole");
  const double dt = rtc() - t0;
  fprintf(stderr, " cellMultipole done in %g sec : %g Mptcl/sec  %g Mcell/sec\n",  dt, nPtcl/1e6/dt, nCells/1e6/dt);

  unsigned int nflops;
  const double SCALEDP = WARP_SIZE*10.0*5;
  const double SCALESP = WARP_SIZE*(5*6.0 + 15.0);
  CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&nflops, multipoles::nflops, sizeof(int)));
  fprintf(stderr, "flop_rate: DP= %g, SP= %g, tot= %g GFLOP/s", 
      nflops*SCALEDP/1e9/dt,
      nflops*SCALESP/1e9/dt,
      nflops*(SCALESP+SCALEDP)/1e9/dt);

#if 0
  {
    std::vector<float4> cellSize(nCells), cellMonopole(nCells);
    std::vector<float3> cellQuad0(nCells), cellQuad1(nCells);
    d_cellSize.d2h(&cellSize[0]);
    d_cellMonopole.d2h(&cellMonopole[0]);
    d_cellQuad0.d2h(&cellQuad0[0]);
    d_cellQuad1.d2h(&cellQuad1[0]);

    float3 bmin = {+1e10f}, bmax = {-1e10f};
    double mtot = 0.0;
    float3 com = {0.0};
    float3 Q0 = {0.0};
    float3 Q1 = {0.0};
    for (int i = 0; i < 8; i++)
    {
      const float4 m = cellMonopole[i];
      const float4 c = cellSize    [i];
      const float3 q0 = cellQuad0[i];
      const float3 q1 = cellQuad1[i];
      bmin.x = std::min(bmin.x, c.x - c.w);
      bmin.y = std::min(bmin.y, c.y - c.w);
      bmin.z = std::min(bmin.z, c.z - c.w);
      bmax.x = std::max(bmax.x, c.x + c.w);
      bmax.y = std::max(bmax.y, c.y + c.w);
      bmax.z = std::max(bmax.z, c.z + c.w);
      mtot += m.w;
      com.x += m.x*m.w;
      com.y += m.y*m.w;
      com.z += m.z*m.w;
      Q0.x += q0.x;
      Q0.y += q0.y;
      Q0.z += q0.z;
      Q1.x += q1.x;
      Q1.y += q1.y;
      Q1.z += q1.z;
      fprintf(stderr," cell= %d  m= %g \n", i, m.w);
    }
    const float inv_mass = 1.0/mtot;
    com.x *= inv_mass;
    com.y *= inv_mass;
    com.z *= inv_mass;
    Q0.x = Q0.x*inv_mass - com.x*com.x;
    Q0.y = Q0.y*inv_mass - com.y*com.y;
    Q0.z = Q0.z*inv_mass - com.z*com.z;
    Q1.x = Q1.x*inv_mass - com.x*com.y;
    Q1.y = Q1.y*inv_mass - com.x*com.z;
    Q1.z = Q1.z*inv_mass - com.y*com.z;
    printf("bmin= %g %g %g \n", bmin.x, bmin.y, bmin.z);
    printf("bmax= %g %g %g \n", bmax.x, bmax.y, bmax.z);
    printf("mtot= %g\n", mtot);
    printf("com = %g %g %g\n", com.x, com.y, com.z);
    printf("Q0= %g %g %g  \n", Q0.x, Q0.y, Q0.z);
    printf("Q1= %g %g %g  \n", Q1.x, Q1.y, Q1.z);
  }
#endif
}

#include "TreecodeInstances.h"

