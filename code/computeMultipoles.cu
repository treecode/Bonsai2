#include "Treecode.h"
  

namespace multipoles
{

#if 0  /* just a test case, but works for float only */
#define _USE_RESTRICT_ 
#endif


  template<typename real_t>
    static __device__ __forceinline__ real_t shfl_xor(const real_t x, const int lane, const int warpSize = WARP_SIZE);

  template<>
    static __device__ __forceinline__ double shfl_xor<double>(const double x, const int lane, const int warpSize)
    {
      return __hiloint2double(
          __shfl_xor(__double2hiint(x), lane, warpSize),
          __shfl_xor(__double2loint(x), lane, warpSize));
    }
  template<>
    static __device__ __forceinline__ float shfl_xor<float>(const float x, const int lane, const int warpSize)
    {
      return __shfl_xor(x, lane, warpSize);
    }

  template<typename real_t>
    static __device__ __forceinline__ 
    void addBoxSize(typename vec<3,real_t>::type &_rmin, typename vec<3,real_t>::type &_rmax, const Particle4<real_t> ptcl)
    {
      typename vec<3,real_t>::type rmin = {ptcl.x(), ptcl.y(), ptcl.z()};
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

  template<typename treal, typename real_t>
    static __device__ __forceinline__
    void addMonopole(double4 &_M, const Particle4<real_t> ptcl)
    {
      const treal x = ptcl.x();
      const treal y = ptcl.y();
      const treal z = ptcl.z();
      const treal m = ptcl.mass();
      typename vec<4,treal>::type M = {m*x,m*y,m*z,m};
#pragma unroll
      for (int i = WARP_SIZE2-1; i >= 0; i--)
      {
        M.x += shfl_xor(M.x, 1<<i);
        M.y += shfl_xor(M.y, 1<<i);
        M.z += shfl_xor(M.z, 1<<i);
        M.w += shfl_xor(M.w, 1<<i);
      }

      _M.x += M.x;
      _M.y += M.y;
      _M.z += M.z;
      _M.w += M.w;
    }

  template<typename treal, typename real_t>
    static __device__ __forceinline__
    void addQuadrupole(Quadrupole<double> &_Q, const Particle4<real_t> ptcl)
    {
      const treal x = ptcl.x();
      const treal y = ptcl.y();
      const treal z = ptcl.z();
      const treal m = ptcl.mass();
      Quadrupole<treal> Q;
      Q.xx() = m * x*x;
      Q.yy() = m * y*y;
      Q.zz() = m * z*z;
      Q.xy() = m * x*y;
      Q.xz() = m * x*z;
      Q.yz() = m * y*z;
#pragma unroll
      for (int i = WARP_SIZE2-1; i >= 0; i--)
      {
        Q.xx() += shfl_xor(Q.xx(), 1<<i);
        Q.yy() += shfl_xor(Q.yy(), 1<<i);
        Q.zz() += shfl_xor(Q.zz(), 1<<i);
        Q.xy() += shfl_xor(Q.xy(), 1<<i);
        Q.xz() += shfl_xor(Q.xz(), 1<<i);
        Q.yz() += shfl_xor(Q.yz(), 1<<i);
      }

      _Q.xx() += Q.xx();
      _Q.yy() += Q.yy();
      _Q.zz() += Q.zz();
      _Q.xy() += Q.xy();
      _Q.xz() += Q.xz();
      _Q.yz() += Q.yz();
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
        const int nPtcl,
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
        __out typename vec<4,real_t>::type *quadrpl0List,
        __out typename vec<2,real_t>::type *quadrpl1List)
    {
      const int warpIdx = threadIdx.x >> WARP_SIZE2;
      const int laneIdx = threadIdx.x & (WARP_SIZE-1);

      const int NWARP2  = NTHREAD2 - WARP_SIZE2;
      const int cellIdx = (blockIdx.x<<NWARP2) + warpIdx;
      if (cellIdx >= nCells) return;

      /* a warp compute properties of each cell */

      const CellData cell = cells[cellIdx];

      typename vec<3,real_t>::type rmin = {static_cast<real_t>(+1e10f)};
      typename vec<3,real_t>::type rmax = {static_cast<real_t>(-1e10f)};
      double4 M = {0.0};
      Quadrupole<double> Q;

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

        for (int i = firstBody; i < lastBody; i += WARP_SIZE)
        {
          nflop++;
          const Particle4<real_t> ptcl(i + laneIdx < lastBody ? ptclPos[i+laneIdx] : pnull);

          addBoxSize(rmin, rmax, ptcl);
#if 0
          typedef double treal;
#else
          typedef float treal;
#endif
          addMonopole<treal>(M, ptcl);
          addQuadrupole<treal>(Q, ptcl);
        }
      }


      if (laneIdx == 0)
      {
        const double inv_mass = 1.0/M.w;
        M.x *= inv_mass;
        M.y *= inv_mass;
        M.z *= inv_mass;
#if 1
        Q.xx() = Q.xx()*inv_mass - M.x*M.x;
        Q.yy() = Q.yy()*inv_mass - M.y*M.y;
        Q.zz() = Q.zz()*inv_mass - M.z*M.z;
        Q.xy() = Q.xy()*inv_mass - M.x*M.y;
        Q.xz() = Q.xz()*inv_mass - M.x*M.z;
        Q.yz() = Q.yz()*inv_mass - M.y*M.z;
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
        const Position<real_t> com(M.x, M.y, M.z);
        const real_t dx = cvec.x - com.x;
        const real_t dy = cvec.y - com.y;
        const real_t dz = cvec.z - com.z;
        const real_t  s = sqrt(dx*dx + dy*dy + dz*dz);
        const real_t  l = max(static_cast<real_t>(2.0f)*max(hvec.x, max(hvec.y, hvec.z)), static_cast<real_t>(1.0e-6f));
        const real_t cellOp = l*inv_theta + s;
        const real_t cellOp2 = cellOp*cellOp;

        atomicAdd(&nflops, nflop);
        sizeList[cellIdx] = make_float4(com.x, com.y, com.z, cellOp2);

        typedef typename vec<4,real_t>::type real4_t;
        typedef typename vec<2,real_t>::type real2_t;
        monopoleList[cellIdx] = (real4_t){M.x, M.y, M.z, M.w};  
        const double4 q0 = Q.get_q0();
        const double2 q1 = Q.get_q1();
        quadrpl0List[cellIdx] = (real4_t){q0.x, q0.y, q0.z, q0.w};
        quadrpl1List[cellIdx] = (real2_t){q1.x, q1.y};
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
      nPtcl, nCells, d_cellDataList, (float4*)d_ptclPos.ptr,
      1.0/theta,
      d_cellSize, d_cellMonopole, d_cellQuad0, d_cellQuad1);
#else
  multipoles::computeCellMultipoles<NTHREAD2,real_t><<<nblock,NTHREAD>>>(
      nPtcl, nCells, d_cellDataList, d_ptclPos,
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
  fprintf(stderr, "flop_rate: DP= %g, SP= %g, tot= %g GFLOP/s \n", 
      nflops*SCALEDP/1e9/dt,
      nflops*SCALESP/1e9/dt,
      nflops*(SCALESP+SCALEDP)/1e9/dt);

#if 1
  {
    std::vector<float4> cellSize(nCells), cellMonopole(nCells);
    std::vector<float4> cellQuad0(nCells);
    std::vector<float2> cellQuad1(nCells);
    d_cellSize.d2h(&cellSize[0]);
    d_cellMonopole.d2h(&cellMonopole[0]);
    d_cellQuad0.d2h(&cellQuad0[0]);
    d_cellQuad1.d2h(&cellQuad1[0]);

#if 0
    for (int i = 0; i < nCells; i++)
    {
      printf("cell= %d:   size= %g %g %g | %g \n",
          i, cellSize[i].x, cellSize[i].y, cellSize[i].z, cellSize[i].w);
    }
    assert(0);
#endif

    float3 bmin = {+1e10f}, bmax = {-1e10f};
    double mtot = 0.0;
    double3 com = {0.0};
    Quadrupole<double> Q;
    for (int i = 0; i < 8; i++)
    {
      const float4 m = cellMonopole[i];
      const float4 c = cellSize    [i];
      const Quadrupole<real_t> q(cellQuad0[i], cellQuad1[i]);
      const float h = sqrt(c.w)*0.5;
      bmin.x = std::min(bmin.x, c.x - h);
      bmin.y = std::min(bmin.y, c.y - h);
      bmin.z = std::min(bmin.z, c.z - h);
      bmax.x = std::max(bmax.x, c.x + h);
      bmax.y = std::max(bmax.y, c.y + h);
      bmax.z = std::max(bmax.z, c.z + h);
      mtot += m.w;
      com.x += m.x*m.w;
      com.y += m.y*m.w;
      com.z += m.z*m.w;
      Q.xx() += q.xx();
      Q.yy() += q.yy();
      Q.zz() += q.zz();
      Q.xy() += q.xy();
      Q.xz() += q.xz();
      Q.yz() += q.yz();
      fprintf(stderr," cell= %d  m= %g \n", i, m.w);
    }
    const double inv_mass = 1.0/mtot;
    com.x *= inv_mass;
    com.y *= inv_mass;
    com.z *= inv_mass;
    Q.xx() = Q.xx()*inv_mass - com.x*com.x;
    Q.yy() = Q.yy()*inv_mass - com.x*com.x;
    Q.zz() = Q.zz()*inv_mass - com.x*com.x;
    Q.xy() = Q.xy()*inv_mass - com.x*com.x;
    Q.xz() = Q.xz()*inv_mass - com.x*com.x;
    Q.yz() = Q.yz()*inv_mass - com.x*com.x;
    printf("bmin= %g %g %g \n", bmin.x, bmin.y, bmin.z);
    printf("bmax= %g %g %g \n", bmax.x, bmax.y, bmax.z);
    printf("mtot= %g\n", mtot);
    printf("com = %g %g %g\n", com.x, com.y, com.z);
    printf("Q0= %g %g %g  \n", Q.xx(), Q.yy(), Q.zz());
    printf("Q1= %g %g %g  \n", Q.xy(), Q.xz(), Q.yz());
  }
#endif
}

#include "TreecodeInstances.h"

