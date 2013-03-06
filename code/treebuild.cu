#include "Treecode.h"

namespace treeBuild
{
  __device__ unsigned int retirementCount = 0;

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
#if 0
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
    treeBuild::computeBoundingBox<NTHREAD2,real_t><<<NBLOCK,NTHREAD,NTHREAD*sizeof(float2)>>>(nPtcl, minmax, d_domain, d_ptclPos);
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
}

template struct Treecode<float, _NLEAF>;
