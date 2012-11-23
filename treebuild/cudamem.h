#pragma once

#include "cutil.h"

template<typename T>
struct cuda_mem
{
  size_t n;
  T *ptr;
  cuda_mem() : ptr(NULL), n(0) {}
#if 1
  ~cuda_mem() {if (ptr != NULL) free();}
#endif
  void alloc(const size_t size)
  {
    assert(ptr == NULL);
    n = size;
    CUDA_SAFE_CALL(cudaMalloc(&ptr, n * sizeof(T)));
  }
  void free()
  {
    if (ptr == NULL) return;
    CUDA_SAFE_CALL(cudaFree(ptr));
    ptr = NULL;
  };
  void realloc(const size_t size)
  {
    if (size > n && n > 0) free();
    if (ptr == NULL) alloc(size);
  };
  void h2d(const T *host_ptr)
  {
    assert(host_ptr != NULL);
    assert( ptr != NULL);
    CUDA_SAFE_CALL(cudaMemcpy(ptr, host_ptr, n * sizeof(T), cudaMemcpyHostToDevice));
  }
  void d2h(T *host_ptr) const
  {
    assert(host_ptr != NULL);
    assert( ptr != NULL);
    CUDA_SAFE_CALL(cudaMemcpy(host_ptr, ptr, n * sizeof(T), cudaMemcpyDeviceToHost));
  }
  operator T* ()
  {
    return ptr;
  }
  operator const T* () const
  {
    return ptr;
  }
};

/* simple PINNED host memory management */
template<typename T>
struct host_mem
{
  size_t n;
  T *ptr;
  host_mem() : ptr(NULL), n(0) {}
#if 1
  ~host_mem() {if (ptr != NULL) free();}
#endif
  void alloc(const size_t size)
  {
    assert(ptr == NULL);
    n = size;
    CUDA_SAFE_CALL(cudaMallocHost(&ptr, n * sizeof(T), cudaHostAllocMapped || cudaHostAllocWriteCombined));
  }
  void free()
  {
    if (ptr == NULL) return;
    CUDA_SAFE_CALL(cudaFreeHost(ptr));
    ptr = NULL;
  };
  void realloc(const size_t size)
  {
    if (size > n && n > 0) free();
    if (ptr == NULL) alloc(size);
  };
  operator T* ()
  {
    return ptr;
  }
  const T& operator[](const size_t i) const { return ptr[i]; }
  T& operator[](const size_t i) { return ptr[i]; }
};
