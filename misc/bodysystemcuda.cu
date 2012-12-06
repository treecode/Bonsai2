#include <cassert>
/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <helper_cuda.h>
#include <math.h>

#include <GL/glew.h>
#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA standard includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "bodysystem.h"

__constant__ float softeningSquared;
__constant__ double softeningSquared_fp64;

cudaError_t setSofteningSquared(float softeningSq)
{
    return cudaMemcpyToSymbol(softeningSquared,
                              &softeningSq,
                              sizeof(float), 0,
                              cudaMemcpyHostToDevice);
}

cudaError_t setSofteningSquared(double softeningSq)
{
    return cudaMemcpyToSymbol(softeningSquared_fp64,
                              &softeningSq,
                              sizeof(double), 0,
                              cudaMemcpyHostToDevice);
}


template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

template<typename T>
__device__ T rsqrt_T(T x)
{
    return rsqrt(x);
}

template<>
__device__ float rsqrt_T<float>(float x)
{
    return rsqrtf(x);
}


// Macros to simplify shared memory addressing
#define SX(i) sharedPos[i+blockDim.x*threadIdx.y]
// This macro is only used when multithreadBodies is true (below)
#define SX_SUM(i,j) sharedPos[i+blockDim.x*j]

template <typename T>
__device__ T getSofteningSquared()
{
    return softeningSquared;
}
template <>
__device__ double getSofteningSquared<double>()
{
    return softeningSquared_fp64;
}

template <typename T>
struct DeviceData
{
    T *dPos[2]; // mapped host pointers
    T *dVel;
    cudaEvent_t  event;
    unsigned int offset;
    unsigned int numBodies;
};


template <typename T>
__device__ typename vec3<T>::Type
bodyBodyInteraction(typename vec3<T>::Type ai,
                    typename vec4<T>::Type bi,
                    typename vec4<T>::Type bj)
{
    typename vec3<T>::Type r;
    
    asm("//bodyN1");

    // r_ij  [3 FLOPS]
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;
    
    asm("//bodyN1a");

    // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
    T distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
    distSqr += getSofteningSquared<T>();

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    T invDist = rsqrt_T(distSqr);
    T invDistCube =  invDist * invDist * invDist;

    // s = m_j * invDistCube [1 FLOP]
    T s = bj.w * invDistCube;

    // a_i =  a_i + s * r_ij [6 FLOPS]
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;
    asm("//bodyN2");

    return ai;
}

template <int NPERT, typename T>
__forceinline__ __device__ void
bodyBodyInteractionN(typename vec3<T>::Type ai[NPERT],
                     typename vec4<T>::Type bi[NPERT],
                     typename vec4<T>::Type bj)
{
    typename vec3<T>::Type r[NPERT];

    asm("//bodyN1");

    // r_ij  [3 FLOPS]
#pragma unroll
    for (int k = 0; k < NPERT; k++)
    {
      r[k].x = bj.x - bi[k].x;
      r[k].y = bj.y - bi[k].y;
      r[k].z = bj.z - bi[k].z;
    }
    
    asm("//bodyN1a");

    // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
    T distSqr[NPERT] = {T(0.0)};
#pragma unroll
    for (int k = 0; k < NPERT; k++)
      distSqr[k] += r[k].x * r[k].x;
#pragma unroll
    for (int k = 0; k < NPERT; k++)
      distSqr[k] += r[k].y * r[k].y;
#pragma unroll
    for (int k = 0; k < NPERT; k++)
      distSqr[k] += r[k].z * r[k].z;
#pragma unroll
    for (int k = 0; k < NPERT; k++)
      distSqr[k] += getSofteningSquared<T>();

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    T invDist[NPERT];
#pragma unroll
    for (int k = 0; k < NPERT; k++)
      invDist[k] = rsqrt_T(distSqr[k]);

    T invDistCube[NPERT];
#pragma unroll
    for (int k = 0; k < NPERT; k++)
      invDistCube[k] =  invDist[k] * invDist[k] * invDist[k];

    // s = m_j * invDistCube [1 FLOP]
    T s[NPERT];
#pragma unroll
    for (int k = 0; k < NPERT; k++)
      s[k] = bj.w * invDistCube[k];

    // a_i =  a_i + s * r_ij [6 FLOPS]

#pragma unroll
    for (int k = 0; k < NPERT; k++)
      ai[k].x += r[k].x * s[k];
#pragma unroll
    for (int k = 0; k < NPERT; k++)
      ai[k].y += r[k].y * s[k];
#pragma unroll
    for (int k = 0; k < NPERT; k++)
      ai[k].z += r[k].z * s[k];
    asm("//bodyN2");
}


// This is the "tile_calculation" function from the GPUG3 article.
template <int NPERT, typename T>
  __device__ void
gravitation(typename vec4<T>::Type iPos[NPERT],
    typename vec3<T>::Type accel[NPERT])
{
  typename vec4<T>::Type *sharedPos = SharedMemory<typename vec4<T>::Type>();

  // The CUDA 1.1 compiler cannot determine that i is not going to
  // overflow in the loop below.  Therefore if int is used on 64-bit linux
  // or windows (or long instead of long long on win64), the compiler
  // generates suboptimal code.  Therefore we use long long on win64 and
  // long on everything else. (Workaround for Bug ID 347697)
#ifdef _Win64
  unsigned long long j = 0;
#else
  unsigned long j = 0;
#endif

  // Here we unroll the loop to reduce bookkeeping instruction overhead
  // 32x unrolling seems to provide best performance

  // Note that having an unsigned int loop counter and an unsigned
  // long index helps the compiler generate efficient code on 64-bit
  // OSes.  The compiler can't assume the 64-bit index won't overflow
  // so it incurs extra integer operations.  This is a standard issue
  // in porting 32-bit code to 64-bit OSes.
  //
  //
#if 0
#pragma unroll 32
  for (unsigned int counter = 0; counter < blockDim.x; counter++)
  {
    const typename vec4<T>::Type jpos = SX(j++);
#pragma unroll
    for (int k = 0; k < NPERT; k++)
    {
      accel[k] = bodyBodyInteraction<T>(accel[k], iPos[k],jpos);
    }
  }
#else
  typename vec3<T>::Type ai[NPERT];
  typename vec4<T>::Type bi[NPERT];
#pragma unroll
    for (int k = 0; k < NPERT; k++)
    {
      ai[k] = accel[k];
      bi[k] = iPos [k];
    }
#pragma unroll 32
  for (unsigned int counter = 0; counter < blockDim.x; counter++)
  {
    const typename vec4<T>::Type bj = SX(j++);
    bodyBodyInteractionN<NPERT,T>(ai, bi, bj);
  }
#pragma unroll
    for (int k = 0; k < NPERT; k++)
    {
      accel[k] = ai[k];
      iPos [k] = bi[k];
    }
#endif
}

// WRAP is used to force each block to start working on a different
// chunk (and wrap around back to the beginning of the array) so that
// not all multiprocessors try to read the same memory locations at
// once.
#define WRAP(x,m) (((x)<(m))?(x):((x)-(m)))  // Mod without divide, works on values from 0 up to 2m

#if 0
template <typename T, bool multithreadBodies>
  __device__ typename vec3<T>::Type
computeBodyAccel(typename vec4<T>::Type bodyPos,
    typename vec4<T>::Type *positions,
    int numBodies)
{
  typename vec4<T>::Type *sharedPos = SharedMemory<typename vec4<T>::Type>();

  typename vec3<T>::Type acc = {0.0f, 0.0f, 0.0f};

  int p = blockDim.x;
  int q = blockDim.y;
  int n = numBodies;
  int numTiles = (n + p*q - 1) / (p * q);

  for (int tile = blockIdx.y; tile < numTiles + blockIdx.y; tile++)
  {
    int index = multithreadBodies ?
      WRAP(blockIdx.x + q * tile + threadIdx.y, gridDim.x) :
      WRAP(blockIdx.x + tile, gridDim.x-1);
    index =  index * p + threadIdx.x;

    if (index < numBodies)
      sharedPos[threadIdx.x+blockDim.x*threadIdx.y] = positions[index];
    else
      sharedPos[threadIdx.x+blockDim.x*threadIdx.y].w = 0;

    __syncthreads();

    // This is the "tile_calculation" function from the GPUG3 article.
    acc = gravitation<T>(bodyPos, acc);

    __syncthreads();
  }

  // When the numBodies / thread block size is < # multiprocessors (16 on G80), the GPU is
  // underutilized.  For example, with a 256 threads per block and 1024 bodies, there will only
  // be 4 thread blocks, so the GPU will only be 25% utilized. To improve this, we use multiple
  // threads per body.  We still can use blocks of 256 threads, but they are arranged in q rows
  // of p threads each.  Each thread processes 1/q of the forces that affect each body, and then
  // 1/q of the threads (those with threadIdx.y==0) add up the partial sums from the other
  // threads for that body.  To enable this, use the "--p=" and "--q=" command line options to
  // this example. e.g.: "nbody.exe --n=1024 --p=64 --q=4" will use 4 threads per body and 256
  // threads per block. There will be n/p = 16 blocks, so a G80 GPU will be 100% utilized.

  // We use a bool template parameter to specify when the number of threads per body is greater
  // than one, so that when it is not we don't have to execute the more complex code required!
  if (multithreadBodies)
  {
    SX_SUM(threadIdx.x, threadIdx.y).x = acc.x;
    SX_SUM(threadIdx.x, threadIdx.y).y = acc.y;
    SX_SUM(threadIdx.x, threadIdx.y).z = acc.z;

    __syncthreads();

    // Save the result in global memory for the integration step
    if (threadIdx.y == 0)
    {
      for (int i = 1; i < blockDim.y; i++)
      {
        acc.x += SX_SUM(threadIdx.x,i).x;
        acc.y += SX_SUM(threadIdx.x,i).y;
        acc.z += SX_SUM(threadIdx.x,i).z;
      }
    }
  }

  return acc;
}
#endif

template <int NPERT, typename T, bool multithreadBodies>
  __device__ void
computeBodyAccel(typename vec4<T>::Type bodyPos[NPERT],
    typename vec4<T>::Type *positions,
    int numBodies,
    typename vec3<T>::Type accel[NPERT])
{
  typename vec4<T>::Type *sharedPos = SharedMemory<typename vec4<T>::Type>();

  //  typename vec3<T>::Type acc = {0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int k = 0; k < NPERT; k++)
  {
    accel[k].x = T(0.0);
    accel[k].y = T(0.0);
    accel[k].z = T(0.0);
  }

  int p = blockDim.x;
  int q = blockDim.y;
  int n = numBodies;
  int numTiles = n / (p * q);

  typename vec3<T>::Type ai[NPERT];
  typename vec4<T>::Type bi[NPERT];
#pragma unroll
  for (int k = 0; k < NPERT; k++)
  {
    ai[k] = accel[k];
    bi[k] = bodyPos [k];
  }

  const T eps2 = getSofteningSquared<T>();

  for (int tile = blockIdx.y; tile < numTiles + blockIdx.y; tile++)
  {
    sharedPos[threadIdx.x+blockDim.x*threadIdx.y] =
      multithreadBodies ?
      positions[WRAP(blockIdx.x + q * tile + threadIdx.y, gridDim.x) * p + threadIdx.x] :
      positions[WRAP(blockIdx.x + tile,                   gridDim.x) * p + threadIdx.x];

    __syncthreads();

    // This is the "tile_calculation" function from the GPUG3 article.
#if 0
        gravitation<NPERT,T>(bodyPos, accel);
#else
    {
      unsigned long j = 0;
#pragma unroll 64
      for (unsigned int counter = 0; counter < blockDim.x; counter++)
      {
        const typename vec4<T>::Type bj = SX(j++);
#if 0
        bodyBodyInteractionN<NPERT,T>(ai, bi, bj);
#else
        {
          typename vec3<T>::Type r[NPERT];

          //  asm("//bodyN1");

          // r_ij  [3 FLOPS]
#pragma unroll
          for (int k = 0; k < NPERT; k++)
          {
            r[k].x = bj.x - bi[k].x;
            r[k].y = bj.y - bi[k].y;
            r[k].z = bj.z - bi[k].z;
          }

          // asm("//bodyN1a");

          // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
          T distSqr[NPERT] = {T(0.0)};
#pragma unroll
          for (int k = 0; k < NPERT; k++)
            distSqr[k] += r[k].x * r[k].x;
#pragma unroll
          for (int k = 0; k < NPERT; k++)
            distSqr[k] += r[k].y * r[k].y;
#pragma unroll
          for (int k = 0; k < NPERT; k++)
            distSqr[k] += r[k].z * r[k].z;
#pragma unroll
          for (int k = 0; k < NPERT; k++)
            distSqr[k] += eps2;

          // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
          T invDist[NPERT];
#pragma unroll
          for (int k = 0; k < NPERT; k++)
            invDist[k] = rsqrt_T(distSqr[k]);

          T invDistCube[NPERT];
#pragma unroll
          for (int k = 0; k < NPERT; k++)
            invDistCube[k] =  invDist[k] * invDist[k] * invDist[k];

          // s = m_j * invDistCube [1 FLOP]
          T s[NPERT];
#pragma unroll
          for (int k = 0; k < NPERT; k++)
            s[k] = bj.w * invDistCube[k];

          // a_i =  a_i + s * r_ij [6 FLOPS]

#pragma unroll
          for (int k = 0; k < NPERT; k++)
            ai[k].x += r[k].x * s[k];
#pragma unroll
          for (int k = 0; k < NPERT; k++)
            ai[k].y += r[k].y * s[k];
#pragma unroll
          for (int k = 0; k < NPERT; k++)
            ai[k].z += r[k].z * s[k];
          //asm("//bodyN2");
        }
#endif
      }
    }
#endif

    __syncthreads();
  }

#pragma unroll
  for (int k = 0; k < NPERT; k++)
  {
    accel[k] = ai[k];
    bodyPos[k] = bi[k];
  }

  // When the numBodies / thread block size is < # multiprocessors (16 on G80), the GPU is
  // underutilized.  For example, with a 256 threads per block and 1024 bodies, there will only
  // be 4 thread blocks, so the GPU will only be 25% utilized. To improve this, we use multiple
  // threads per body.  We still can use blocks of 256 threads, but they are arranged in q rows
  // of p threads each.  Each thread processes 1/q of the forces that affect each body, and then
  // 1/q of the threads (those with threadIdx.y==0) add up the partial sums from the other
  // threads for that body.  To enable this, use the "--p=" and "--q=" command line options to
  // this example. e.g.: "nbody.exe --n=1024 --p=64 --q=4" will use 4 threads per body and 256
  // threads per block. There will be n/p = 16 blocks, so a G80 GPU will be 100% utilized.

  // We use a bool template parameter to specify when the number of threads per body is greater
  // than one, so that when it is not we don't have to execute the more complex code required!
#if 0
  if (multithreadBodies)
  {
    SX_SUM(threadIdx.x, threadIdx.y).x = acc.x;
    SX_SUM(threadIdx.x, threadIdx.y).y = acc.y;
    SX_SUM(threadIdx.x, threadIdx.y).z = acc.z;

    __syncthreads();

    // Save the result in global memory for the integration step
    if (threadIdx.y == 0)
    {
      for (int i = 1; i < blockDim.y; i++)
      {
        acc.x += SX_SUM(threadIdx.x,i).x;
        acc.y += SX_SUM(threadIdx.x,i).y;
        acc.z += SX_SUM(threadIdx.x,i).z;
      }
    }
  }
#endif

}


template<int NPERT, typename T, bool multithreadBodies>
  __global__ void
integrateBodies(typename vec4<T>::Type *newPos,
    typename vec4<T>::Type *oldPos,
    typename vec4<T>::Type *vel,
    unsigned int deviceOffset, unsigned int deviceNumBodies,
    float deltaTime, float damping, int totalNumBodies)
{
  int index = NPERT*blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= deviceNumBodies)
  {
    return;
  }

#if 0
  typename vec4<T>::Type position = oldPos[deviceOffset + index];
#else
  typename vec4<T>::Type position[NPERT];
#pragma unroll
  for (int k = 0; k < NPERT; k++)
    position[k] = oldPos[deviceOffset + index + k*blockDim.x];
#endif

  typename vec3<T>::Type accel[NPERT];
  computeBodyAccel<NPERT, T, multithreadBodies>(position, oldPos, totalNumBodies, accel);


  if (!multithreadBodies || (threadIdx.y == 0))
  {
    // acceleration = force \ mass;
    // new velocity = old velocity + acceleration * deltaTime
    // note we factor out the body's mass from the equation, here and in bodyBodyInteraction
    // (because they cancel out).  Thus here force == acceleration
#pragma unroll
    for (int k = 0; k < NPERT; k++)
    {
      typename vec4<T>::Type velocity = vel[deviceOffset + index + k*blockDim.x];

      velocity.x += accel[k].x * deltaTime;
      velocity.y += accel[k].y * deltaTime;
      velocity.z += accel[k].z * deltaTime;

      velocity.x *= damping;
      velocity.y *= damping;
      velocity.z *= damping;

      // new position = old position + velocity * deltaTime
      position[k].x += velocity.x * deltaTime;
      position[k].y += velocity.y * deltaTime;
      position[k].z += velocity.z * deltaTime;

      // store new position and velocity
      newPos[deviceOffset + index + k*blockDim.x] = position[k];
      vel[deviceOffset + index + k*blockDim.x]    = velocity;
    }
  }
}

  template <typename T>
void integrateNbodySystem(DeviceData<T> *deviceData,
    cudaGraphicsResource **pgres,
    unsigned int currentRead,
    float deltaTime,
    float damping,
    unsigned int numBodies,
    unsigned int numDevices,
    int p,
    int q,
    bool bUsePBO)
{
  if (bUsePBO)
  {
    checkCudaErrors(cudaGraphicsResourceSetMapFlags(pgres[currentRead], cudaGraphicsMapFlagsReadOnly));
    checkCudaErrors(cudaGraphicsResourceSetMapFlags(pgres[1-currentRead], cudaGraphicsMapFlagsWriteDiscard));
    checkCudaErrors(cudaGraphicsMapResources(2, pgres, 0));
    size_t bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&(deviceData[0].dPos[currentRead]), &bytes, pgres[currentRead]));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&(deviceData[0].dPos[1-currentRead]), &bytes, pgres[1-currentRead]));
  }

  cudaDeviceProp props;

  for (unsigned int dev = 0; dev != numDevices; dev++)
  {
    if (numDevices > 1)
    {
      cudaSetDevice(dev);
    }

    checkCudaErrors(cudaGetDeviceProperties(&props, dev));

    while ((deviceData[dev].numBodies > 0) && p > 1 &&
        (deviceData[dev].numBodies / p < (unsigned)props.multiProcessorCount))
    {
      p /= 2;
      q *= 2;
    }
    const int NPERT = 2;
    assert(q == 1);

    fprintf(stderr, "p= %d  q= %d\n", p,q);
    dim3 threads(p,q,1);
    dim3 grid((deviceData[dev].numBodies-1)/NPERT/p + 1, 1, 1);

    // execute the kernel:

    // When the numBodies / thread block size is < # multiprocessors
    // (16 on G80), the GPU is underutilized. For example, with 256 threads per
    // block and 1024 bodies, there will only be 4 thread blocks, so the
    // GPU will only be 25% utilized.  To improve this, we use multiple threads
    // per body.  We still can use blocks of 256 threads, but they are arranged
    // in q rows of p threads each.  Each thread processes 1/q of the forces
    // that affect each body, and then 1/q of the threads (those with
    // threadIdx.y==0) add up the partial sums from the other threads for that
    // body.  To enable this, use the "--p=" and "--q=" command line options to
    // this example.  e.g.: "nbody.exe --n=1024 --p=64 --q=4" will use 4
    // threads per body and 256 threads per block. There will be n/p = 16
    // blocks, so a G80 GPU will be 100% utilized.

    // We use a bool template parameter to specify when the number of threads
    // per body is greater than one, so that when it is not we don't have to
    // execute the more complex code required!
    int sharedMemSize = p * q * 4 * sizeof(T); // 4 floats for pos


    if (grid.x > 0 && threads.y == 1)
    {
      integrateBodies<NPERT, T, false><<< grid, threads, sharedMemSize >>>
        ((typename vec4<T>::Type *)deviceData[dev].dPos[1-currentRead],
         (typename vec4<T>::Type *)deviceData[dev].dPos[currentRead],
         (typename vec4<T>::Type *)deviceData[dev].dVel,
         deviceData[dev].offset, deviceData[dev].numBodies,
         deltaTime, damping, numBodies);
    }
    else if (grid.x > 0)
    {
      assert(0);
#if 0
      integrateBodies<T, true><<< grid, threads, sharedMemSize >>>
        ((typename vec4<T>::Type *)deviceData[dev].dPos[1-currentRead],
         (typename vec4<T>::Type *)deviceData[dev].dPos[currentRead],
         (typename vec4<T>::Type *)deviceData[dev].dVel,
         deviceData[dev].offset, deviceData[dev].numBodies,
         deltaTime, damping, numBodies);
#endif
    }

    if (numDevices > 1)
    {
      checkCudaErrors(cudaEventRecord(deviceData[dev].event));
      // MJH: Hack on older driver versions to force kernel launches to flush!
      cudaStreamQuery(0);
    }

    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution failed");
  }

  if (numDevices > 1)
  {
    for (unsigned int dev = 0; dev < numDevices; dev++)
    {
      checkCudaErrors(cudaEventSynchronize(deviceData[dev].event));
    }
  }

  if (bUsePBO)
  {
    checkCudaErrors(cudaGraphicsUnmapResources(2, pgres, 0));
  }
}


// Explicit specializations needed to generate code
template void integrateNbodySystem<float>(DeviceData<float> *deviceData,
    cudaGraphicsResource **pgres,
    unsigned int currentRead,
    float deltaTime,
    float damping,
    unsigned int numBodies,
    unsigned int numDevices,
    int p, int q,
    bool bUsePBO);

#if 1
template void integrateNbodySystem<double>(DeviceData<double> *deviceData,
    cudaGraphicsResource **pgres,
    unsigned int currentRead,
    float deltaTime,
    float damping,
    unsigned int numBodies,
    unsigned int numDevices,
    int p, int q,
    bool bUsePBO);
#endif
