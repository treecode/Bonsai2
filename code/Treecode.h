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

template<typename real_t, int NLEAF>
struct Treecode
{
  typedef Particle4<real_t> Particle;

  int nPtcl;
  host_mem<Particle> h_ptclPos, h_ptclVel;
  cuda_mem<Particle> d_ptclPos, d_ptclVel, d_ptclPos_tmp;

  Treecode() {}

  void alloc(const int nPtcl)
  {
    this->nPtcl = nPtcl;
    h_ptclPos.alloc(nPtcl);
    h_ptclVel.alloc(nPtcl);
    d_ptclPos.alloc(nPtcl);
    d_ptclVel.alloc(nPtcl);
    d_ptclPos_tmp.alloc(nPtcl);
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
};


