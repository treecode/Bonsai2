#include "Treecode.h"

#include <thrust/sort.h>
#include <thrust/scan.h>

namespace makeGroups
{

  template<typename T>
    static __global__ void shuffle(const int n, const int *map, const T *in, T *out)
    {
      const int gidx = blockDim.x*blockIdx.x + threadIdx.x;
      if (gidx >= n) return;

      out[gidx] = in[map[gidx]];
    }

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
        xi = (crd.x & mask) ? 1 : 0;
        yi = (crd.y & mask) ? 1 : 0;
        zi = (crd.z & mask) ? 1 : 0;        

        const int index = (xi << 2) + (yi << 1) + zi;

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

        key = (key<<3) + Cvalue;

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

      const int xc = static_cast<int>((ptcl.x() - bmin.x) * inv_domain_size * (1<<NBINS));
      const int yc = static_cast<int>((ptcl.y() - bmin.y) * inv_domain_size * (1<<NBINS));
      const int zc = static_cast<int>((ptcl.z() - bmin.z) * inv_domain_size * (1<<NBINS));

      keys  [idx] = get_key<NBINS>(make_int3(xc,yc,zc));
      values[idx] = idx;
    }

  static __global__
    void mask_keys(
        const int n, 
        const unsigned long long mask,
        unsigned long long *keys,
        unsigned long long *keys_inv,
        int *ptclBegIdx,
        int *ptclEndIdx)
    {
      const int gidx = blockIdx.x*blockDim.x + threadIdx.x;
      if (gidx >= n) return;

#if 0
      if (gidx < 100)
      {
        printf("gidx= %d : keys= %llx  maks= %llx  res= %llx\n",
            gidx, keys[gidx], mask, keys[gidx] & mask);
      }

#endif
      keys[gidx] &= mask;
      keys_inv[n-gidx-1] = keys[gidx];

      extern __shared__ unsigned long long shKeys[];

      const int tid = threadIdx.x;
      shKeys[tid+1] = keys[gidx] & mask;

      int shIdx = 0;
      int gmIdx = max(blockIdx.x*blockDim.x-1,0);
      if (tid == 1)
      {
        shIdx = blockDim.x+1;
        gmIdx = min(blockIdx.x*blockDim.x + blockDim.x,n-1);
      }
      if (tid < 2)
        shKeys[shIdx] = keys[gmIdx] & mask;

      __syncthreads();

      const int idx = tid+1;
      const unsigned long long currKey = shKeys[idx  ];
      const unsigned long long prevKey = shKeys[idx-1];
      const unsigned long long nextKey = shKeys[idx+1];

      if (currKey != prevKey || gidx == 0)
        ptclBegIdx[gidx] = gidx;
      else
        ptclBegIdx[gidx] = 0;

      if (currKey != nextKey || gidx == n-1)
        ptclEndIdx[n-1-gidx] = gidx+1;
      else
        ptclEndIdx[n-1-gidx] = 0;

    }

  __device__ unsigned int groupCounter= 0;

  static __global__
    void make_groups(const int n, const int nCrit,
        const int *ptclBegIdx, 
        const int *ptclEndIdx,
        GroupData *groupList)
    {
      const int gidx = blockDim.x * blockIdx.x + threadIdx.x;
      if (gidx >= n) return;

      const int ptclBeg = ptclBegIdx[gidx];
      assert(gidx >= ptclBeg);

      const int igroup   = (gidx - ptclBeg)/nCrit;
      const int groupBeg = ptclBeg + igroup * nCrit;

#if 0
      if (gidx < 100)
        printf("gidx= %d  groupBeg =%d\n",gidx, groupBeg);
      return;
#endif

      if (gidx == groupBeg)
      {
        const int groupIdx = atomicAdd(&groupCounter,1);
        const int ptclEnd = ptclEndIdx[n-1-gidx];
        groupList[groupIdx] = GroupData(groupBeg, min(nCrit, ptclEnd - groupBeg));
      }
    }

  struct keyCompare
  {
    __host__ __device__
      bool operator()(const unsigned long long x, const unsigned long long y)
      {
        return x < y;
      }
  };

};

  template<typename real_t>
void Treecode<real_t>::makeGroups(int levelSplit, const int nCrit)
{
  this->nCrit = nCrit;
  const int nthread = 256;

  d_key.realloc(2.0*nPtcl);
  d_value.realloc(nPtcl);
  d_groupList.realloc(nPtcl); //nCells);

  unsigned long long *d_keys = (unsigned long long*)d_key.ptr;
  int *d_values = d_value.ptr;

  nGroups = 0;
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(makeGroups::groupCounter, &nGroups, sizeof(int)));

  const int nblock  = (nPtcl-1)/nthread + 1;
  const int NBINS = 21; 

  cudaDeviceSynchronize();
  const double t0 = rtc();
  makeGroups::computeKeys<NBINS,real_t><<<nblock,nthread>>>(nPtcl, d_domain, d_ptclPos, d_keys, d_values);

  levelSplit = std::max(1,levelSplit);  /* pick the coarse segment boundaries at the levelSplit */
  unsigned long long mask= 0;
  for (int i = 0; i < NBINS; i++)
  {
    mask <<= 3;
    if (i < levelSplit)
      mask |= 0x7;
  }
  printf("mask= %llx  \n", mask);

  /* sort particles by PH key */
  thrust::device_ptr<unsigned long long> keys_beg(d_keys);
  thrust::device_ptr<unsigned long long> keys_end(d_keys + nPtcl);
  thrust::device_ptr<int> vals_beg(d_value.ptr);
#if 1
  thrust::sort_by_key(keys_beg, keys_end, vals_beg); 
#else
  thrust::sort_by_key(keys_beg, keys_end, vals_beg, makeGroups::keyCompare());
#endif
  makeGroups::shuffle<Particle><<<nblock,nthread>>>(nPtcl, d_value, d_ptclPos, d_ptclPos_tmp);

  cuda_mem<int> d_ptclBegIdx, d_ptclEndIdx;
  cuda_mem<unsigned long long> d_keys_inv;
  d_ptclBegIdx.alloc(nPtcl);
  d_ptclEndIdx.alloc(nPtcl);
  d_keys_inv.alloc(nPtcl);
  makeGroups::mask_keys<<<nblock,nthread,(nthread+2)*sizeof(unsigned long long)>>>(nPtcl, mask, d_keys, d_keys_inv, d_ptclBegIdx, d_ptclEndIdx);

  thrust::device_ptr<int> valuesBeg(d_ptclBegIdx.ptr);
  thrust::device_ptr<int> valuesEnd(d_ptclEndIdx.ptr);
  thrust::inclusive_scan_by_key(keys_beg,     keys_end,    valuesBeg, valuesBeg);

  thrust::device_ptr<unsigned long long> keys_inv_beg(d_keys_inv.ptr);
  thrust::device_ptr<unsigned long long> keys_inv_end(d_keys_inv.ptr + nPtcl);
  thrust::inclusive_scan_by_key(keys_inv_beg, keys_inv_end, valuesEnd, valuesEnd);

#if 0
  std::vector<int> beg(nPtcl), end(nPtcl);
  std::vector<unsigned long long> h_keys(nPtcl);
  d_ptclBegIdx.d2h(&beg[0]);
  d_ptclEndIdx.d2h(&end[0]);
  d_key.d2h((int*)&h_keys[0],2*nPtcl);
  for (int i = 0; i < nPtcl; i++)
  {
    printf("i= %d : keys= %llx beg= %d  end= %d\n", i, h_keys[i], beg[i], end[nPtcl-1-i]);
  }
#endif

  makeGroups::make_groups<<<nblock,nthread>>>(nPtcl, nCrit, d_ptclBegIdx, d_ptclEndIdx, d_groupList);

  kernelSuccess("makeGroups");
  const double t1 = rtc();
  const double dt = t1 - t0;
  fprintf(stderr, " makeGroups done in %g sec : %g Mptcl/sec\n",  dt, nPtcl/1e6/dt);
  CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&nGroups, makeGroups::groupCounter, sizeof(int)));


  fprintf(stderr, "nGroup= %d <nCrit>= %g \n", nGroups, nPtcl*1.0/nGroups);
#if 0
  {
    std::vector<int2> groups(nGroups);
    d_groupList.d2h((GroupData*)&groups[0], nGroups);
    int np_in_group = 0;
    for (int i = 0 ;i < nGroups; i++)
    {
#if 0
      printf("groupdIdx= %d  :: pbeg= %d  np =%d \n", i, groups[i].x, groups[i].y);
#else
      np_in_group += groups[i].y;
#endif
    }
    printf("np_in_group= %d    np= %d\n", np_in_group, nPtcl);
    assert(0);
  }
#endif

}

#include "TreecodeInstances.h"

