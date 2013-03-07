#pragma once

#include "anyoption.h"
#include "read_tipsy.h"
#include "cudamem.h"
#include "plummer.h"
#include "Particle4.h"
#include "rtc.h"
#include <string>
#include <sstream>

#define _NLEAF 16
#define __out

static void kernelSuccess(const char kernel[] = "kernel")
{
  cudaDeviceSynchronize();
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "%s launch failed: %s\n", kernel, cudaGetErrorString(err));
    assert(0);
  }
}
struct CellData
{
  private:
    enum {NLEAF_SHIFT = 29};
    enum {NLEAF_MASK  = ~(0x7U << NLEAF_SHIFT)};
    enum {LEVEL_SHIFT = 27};
    enum {LEVEL_MASK  = ~(0x1FU << LEVEL_SHIFT)};
    uint4 packed_data;
  public:
    __device__ CellData(
        const int level,
        const unsigned int parentCell,
        const unsigned int nBeg,
        const unsigned int nEnd,
        const unsigned int first = 0xFFFFFFFF,
        const unsigned int n = 0xFFFFFFFF)
    {
      int packed_firstleaf_n = 0xFFFFFFFF;
      if (n != 0xFFFFFFFF)
        packed_firstleaf_n = first | ((unsigned int)n << NLEAF_SHIFT);
      packed_data = make_uint4(parentCell | (level << LEVEL_SHIFT), packed_firstleaf_n, nBeg, nEnd);
    }

    __host__ __device__ int n()      const {return packed_data.y >> NLEAF_SHIFT;}
    __host__ __device__ int first()  const {return packed_data.y  & NLEAF_MASK;}
    __host__ __device__ int parent() const {return packed_data.x  & LEVEL_MASK;}
    __host__ __device__ int level()  const {return packed_data.x >> LEVEL_SHIFT;}
    __host__ __device__ int pbeg()   const {return packed_data.z;}
    __host__ __device__ int pend()   const {return packed_data.w;}

    __host__ __device__ bool isLeaf() const {return packed_data.y == 0xFFFFFFFF;}
    __host__ __device__ bool isNode() const {return !isLeaf();}

    __host__ __device__ void update_first(const int first) 
    {
      packed_data.y = first | ((unsigned int)n() << NLEAF_SHIFT);
    }
};

template<typename real_t, int NLEAF>
struct Treecode
{
  typedef Particle4<real_t> Particle;

  int nPtcl;
  host_mem<Particle> h_ptclPos, h_ptclVel;
  cuda_mem<Particle> d_ptclPos, d_ptclVel, d_ptclPos_tmp;
  cuda_mem<Box<real_t> > d_domain;
  cuda_mem<Position<real_t> > d_minmax;
  cuda_mem<int2> d_level_begIdx;

  int node_max, cell_max, stack_size;
  cuda_mem<int>  d_stack_memory_pool;
  cuda_mem<CellData> d_cellDataList, d_cellDataList_tmp;
  cuda_mem<int>      d_key, d_value;

  Treecode() 
  {
    d_domain.alloc(1);
    d_minmax.alloc(2048);
    d_level_begIdx.alloc(32);  /* max 32 levels */
    CUDA_SAFE_CALL(cudaMemset(d_level_begIdx,0, 32*sizeof(int2)));
  }

  void alloc(const int nPtcl)
  {
    this->nPtcl = nPtcl;
    h_ptclPos.alloc(nPtcl);
    h_ptclVel.alloc(nPtcl);
    d_ptclPos.alloc(nPtcl);
    d_ptclVel.alloc(nPtcl);
    d_ptclPos_tmp.alloc(nPtcl);
 
    /* allocate stack memory */ 
    node_max = nPtcl/10;
    stack_size = (8+8+8+64+8)*node_max;
    fprintf(stderr, "stack_size= %g MB \n", sizeof(int)*stack_size/1024.0/1024.0);
    d_stack_memory_pool.alloc(stack_size);
  
    /* allocate celldata memory */
    cell_max = nPtcl;
    fprintf(stderr, "celldata= %g MB \n", cell_max*sizeof(CellData)/1024.0/1024.0);
    d_cellDataList.alloc(cell_max);
    d_cellDataList_tmp.alloc(cell_max);
    d_key.alloc(cell_max);
    d_value.alloc(cell_max);
  };

  void ptcl_d2h()
  {
    d_ptclPos.d2h(h_ptclPos);
    d_ptclVel.d2h(h_ptclVel);
  }


  void ptcl_h2d()
  {
    d_ptclPos.h2d(h_ptclPos);
    d_ptclVel.h2d(h_ptclVel);
  }

  void buildTree();
  void computeMultipoles();
  void computeForces();
  void moveParticles();
  void computeEnergies();

};


