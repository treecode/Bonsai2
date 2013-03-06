#include "Treecode.h"

namespace treeBuild
{
  __device__ unsigned int retirementCount = 0;
  template<typename T>
    static __forceinline__ __device__ Position<T> get_volatile(const volatile Position<T>  &v)
    {
      return Position<T>(v.x, v.y, v.z);
    };

  template<typename T, const int NTHREADS>
    __forceinline__ __device__ void reduceBlock(
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
        const int ticket = atomicInc(&retirementCount, NBLOCKS);
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

#if 1
          const int NMAXLEVEL = 20;

          const T hquant = hsize / T(1<<NMAXLEVEL);
          const long long nx = (long long)(cvec.x/hquant);
          const long long ny = (long long)(cvec.y/hquant);
          const long long nz = (long long)(cvec.z/hquant);

          const Position<T> centre(hquant * T(nx), hquant * T(ny), hquant * T(nz));
#else
          const Position<T> centre = cvec;
#endif

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
    const int NBLOCKS  = 256;
    const int NTHREADS = 256;
    cuda_mem< Position<real_t> > minmax;
    minmax.alloc(NBLOCKS*2);

    cudaDeviceSynchronize();
    const double t0 = rtc();
    treeBuild::computeBoundingBox<NTHREADS,NBLOCKS,real_t><<<NBLOCKS,NTHREADS>>>(nPtcl, minmax, d_domain, d_ptclPos);
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
