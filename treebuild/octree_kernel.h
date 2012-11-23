// Definitions for the octree
#ifndef OCTREE_KERNEL_H
#define OCTREE_KERNEL_H

// Things to play with for tuning
#define COUNT_ELEMS_PER_THREAD  16       // Elements per thread, for counting stage
#define PART_ELEMS_PER_THREAD   4       // Elements per thread, for partitioning stage
#define MAX_ELEMS_PER_LEAF      32      // Leaf node if <= this many elements
#define OCTREE_WARPS            8       // Warps per block for octree kernels
#define OCTREE_THREADS          (OCTREE_WARPS * 32)     // Threads per block for octree kernels
#define CONTROL_THREADS         8       // Threads per block for build control kernel

// Define this if you want the CNP version (we'll allow both, later on)
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 350)
//#define USE_CNP
#endif

// Each element is (x,y,z) and an ID to map it back to original data.
template <typename T>
struct ElemCoord {
    T x;
    T y;
    T z;
    T id;
};

template <typename T>
struct Elem {
    ElemCoord<T> p;
};

// The octree node contains the geometric corners and mid-point (e.g. dimension),
// the counts of elements in this node, and a pointer to all sub-nodes.
template <typename SIZE_T, typename T>
struct OctreeNode {
    ElemCoord<T> origin;                // Top-left corner of this cube
    ElemCoord<T> mid;                   // Mid-point of this cube
    unsigned int depth;                 // Depth of this level of the tree
    SIZE_T count;                       // Number of points in this entire cube
    SIZE_T sorted_count;                // Offset counter for populating this cube
    OctreeNode<SIZE_T, T> *parent;      // Parent octree node of this one (doubly-linked)
    OctreeNode<SIZE_T, T> *self;        // My own pointer, for easy debug
    OctreeNode<SIZE_T, T> *subcube;     // Pointer to the 8 sub-cube data structures
    Elem<T> *src;                       // Data which lies within this cube
    Elem<T> *dest;                      // Place we're storing the data for subcubes
};

// A cube consists of 8 subcubes. We allocate it as a block
// so that they can all easily be accessed.
template <typename SIZE_T, typename T>
struct Cube {
    OctreeNode<SIZE_T, T> subcube[8];
};

// Mapping for fast conversion of longlong to char
union charmap {
    unsigned long long lval;
    unsigned char count[8];
};

// Prototype of host entry function
template <typename SIZE_T, typename T>
int OctreeLaunch(ElemCoord<T> origin, ElemCoord<T> mid, Elem<T> *srcData, SIZE_T npoints, int usecnp);

// General printf macros (host and device side)
#define CPRINTF(fmt, ...)  printf(fmt, ##__VA_ARGS__)
#define CPRINTF1(fmt, ...) //printf(fmt, ##__VA_ARGS__)
#define CPRINTF2(fmt, ...) //printf(fmt, ##__VA_ARGS__)
#define CPRINTF3(fmt, ...) //printf(fmt, ##__VA_ARGS__)

// Debug macros (device side only)
#define PRINTF(fmt, ...)  printf("[%d, %3d] " fmt, blockIdx.x, threadIdx.x, ##__VA_ARGS__)
#define PRINTF1(fmt, ...) PRINTF(fmt, ##__VA_ARGS__)
#define PRINTF2(fmt, ...) //PRINTF(fmt, ##__VA_ARGS__)
#define PRINTF3(fmt, ...) //PRINTF(fmt, ##__VA_ARGS__)

#endif // OCTREE_KERNEL_H
