#include "Treecode.h"

#include <thrust/sort.h>

namespace makeGroups
{
  template<int NBITS>
  static __device__ unsigned long long get_key(int3 crd)
  {
    int i,xi, yi, zi;
    int mask;
    unsigned long long key;

    //0= 000, 1=001, 2=011, 3=010, 4=110, 5=111, 6=101, 7=100
    //000=0=0, 001=1=1, 011=3=2, 010=2=3, 110=6=4, 111=7=5, 101=5=6, 100=4=7
    const int C[8] = {0, 1, 7, 6, 3, 2, 4, 5};

    int temp;

    mask = 1 << (NBITS - 1);
    key  = 0;

#if 0
    uint4 key_new;
#endif

#pragma unroll
    for(i = 0; i < NBITS; i++, mask >>= 1)
    {
#if 0
      xi = (crd.x & mask) ? 1 : 0;
      yi = (crd.y & mask) ? 1 : 0;
      zi = (crd.z & mask) ? 1 : 0;        
#else
      xi = (crd.x & mask) & 1;
      yi = (crd.y & mask) & 1;
      zi = (crd.z & mask) & 1;
#endif

#if 0
      const int index = (xi << 2) + (yi << 1) + zi;
#else
      const int index = (xi+xi+xi+xi) + (yi+yi) + zi;
#endif

      int Cvalue;
      if(index == 0)
      {
        temp = crd.z; crd.z = crd.y; crd.y = temp;
        Cvalue = C[0];
      }
      else  if(index == 1 || index == 5)
      {
        temp = crd.x; crd.x = crd.y; crd.y = temp;
        if (index == 1) Cvalue = C[1];
        else            Cvalue = C[5];
      }
      else  if(index == 4 || index == 6)
      {
        crd.x = (crd.x) ^ (-1);
        crd.z = (crd.z) ^ (-1);
        if (index == 4) Cvalue = C[4];
        else            Cvalue = C[6];
      }
      else  if(index == 7 || index == 3)
      {
        temp  = (crd.x) ^ (-1);         
        crd.x = (crd.y) ^ (-1);
        crd.y = temp;
        if (index == 3) Cvalue = C[3];
        else            Cvalue = C[7];
      }
      else
      {
        temp = (crd.z) ^ (-1);         
        crd.z = (crd.y) ^ (-1);
        crd.y = temp;          
        Cvalue = C[2];
      }   

#if 0
      key = (key << 3) + C[index];
#else
      key = (key+key+key) + Cvalue;
#endif

#if 0
      if(i == 19)
      {
        key_new.y = key;
        key = 0;
      }
      if(i == 9)
      {
        key_new.x = key;
        key = 0;
      }
#endif
    } //end for

#if 0
    key_new.z = key;

    return key_new;
#else
    return key;
#endif
  }

  template<int NBINS, typename real_t>
    static __global__ 
    void computeKeys(
        const int n,
        const Box<real_t> *d_domain,
        const Particle4<real_t> *ptclPos,
        __out unsigned long long *keys,
        __out int *values)
    {
      const int idx = blockIdx.x*blockDim.x + threadIdx.x;
      if (idx >= n) return;

      const Particle4<real_t> ptcl = ptclPos[idx];

      const Box<real_t> domain = d_domain[0];
      const real_t inv_domain_size = static_cast<real_t>(0.5f)/domain.hsize;
      const Position<real_t> bmin(
          domain.centre.x - domain.hsize,
          domain.centre.y - domain.hsize,
          domain.centre.z - domain.hsize);

      const int xc = static_cast<int>((ptcl.x() - bmin.x) * inv_domain_size);
      const int yc = static_cast<int>((ptcl.y() - bmin.y) * inv_domain_size);
      const int zc = static_cast<int>((ptcl.z() - bmin.z) * inv_domain_size);


      keys  [idx] = get_key<NBINS>(make_int3(xc,yc,zc));
      values[idx] = idx;
    }

  template<typename real_t>
    static __global__
    void shuffle_ptcl(
        const int n,
        const int *map,
        const Particle4<real_t> *ptclIn,
        __out Particle4<real_t> *ptclOut)
    {
      const int idx = blockIdx.x*blockDim.x + threadIdx.x;
      if (idx >= n) return;

      const int mapIdx = map[idx];
      ptclOut[idx] = ptclIn[mapIdx];
    }

  __device__ unsigned int groupCounter = 0;

  template<int NGROUP2>
  static __global__
    void make_groups(const int n, GroupData *groupList)
    {
      const int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx >= n) return;

      const int NGROUP = 1<<NGROUP2;
#if 0
      const int groupIdx = idx >> NGROUP2;
#endif
      const int iptclIdx = idx & (NGROUP - 1);
      const int firstPtcl = idx & (-(NGROUP-1));

      if (iptclIdx == 0)
      {
        const int idx = atomicAdd(&groupCounter,1);
        groupList[idx] = GroupData(firstPtcl, min(NGROUP, n-firstPtcl));
      }
    }

};

  template<typename real_t, int NLEAF>
void Treecode<real_t, NLEAF>::makeGroups()
{
  const int nthread = 256;
  const int nblock  = (nPtcl-1)/nthread + 1;
  const int NBINS = 21; 

  d_key.realloc(2.0*nPtcl);
  d_value.realloc(nPtcl);
  d_groupList.realloc(nCells);

  unsigned long long *d_keys = (unsigned long long*)d_key.ptr;
  int *d_values = d_value.ptr;

  cudaDeviceSynchronize();
  const double t0 = rtc();
  nGroups = 0;
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(makeGroups::groupCounter, &nGroups, sizeof(int)));
  makeGroups::computeKeys<NBINS,real_t><<<nblock,nthread>>>(nPtcl, d_domain, d_ptclPos, d_keys, d_values);

  thrust::device_ptr<unsigned long long> keys_beg(d_keys);
  thrust::device_ptr<unsigned long long> keys_end(d_keys + nPtcl);
  thrust::device_ptr<int> vals_beg(d_values);
  thrust::sort_by_key(keys_beg, keys_end, vals_beg); 

  makeGroups::shuffle_ptcl<real_t><<<nblock,nthread>>>(nPtcl, d_values, d_ptclPos, d_ptclPos_tmp);
 
  const int NGROUP2 = 5;
  makeGroups::make_groups<NGROUP2><<<nblock,nthread>>>(nPtcl, d_groupList);

  kernelSuccess("makeGroups");
  const double t1 = rtc();
  const double dt = t1 - t0;
  fprintf(stderr, " makeGroups done in %g sec : %g Mptcl/sec\n",  dt, nPtcl/1e6/dt);
  CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&nGroups, makeGroups::groupCounter, sizeof(int)));


  fprintf(stderr, "nGroup= %d\n", nGroups);
}

#include "TreecodeInstances.h"

