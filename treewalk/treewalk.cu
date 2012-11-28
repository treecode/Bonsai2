static __device__ __forceinline__ 
float4 addMonopole(
    float4 acc
    const float4 ipos,
    const float4 jpos,
    const float  eps2)
{
#if 1 
  const float3 dr = make_float3(jpos.x - ipos.x, jpos.y - ipos.y, jpos.z - ipos.z);

  const float r2     = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z + eps2;
  const float rinv   = rsqrtf(r2);
  const float rinv2  = rinv*rinv;
  const float jmass  = jpos.w;
  const float mrinv  = jmass * rinv;
  const float mrinv3 = mrinv * rinv2;

  acc.w -= mrinv;
  acc.x += mrinv3 * dr.x;
  acc.y += mrinv3 * dr.y;
  acc.z += mrinv3 * dr.z;
#endif

  return acc;
}

static __device__ __forceinline__ 
float4 addQuadrupole(
    float4 acc, 
    const float4 pos,
    const float4 mass,
    const float4 Q0, 
    const float4 Q1, 
    const float eps2) 
{
#if 1
  const float3 dr = make_float3(pos.x - com.x, pos.y - com.y, pos.z - com.z);
  const float  r2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z + eps2;

  const float rinv  = rsqrtf(r2);
  const float rinv2 = rinv *rinv;
  const float mrinv  =  mass*rinv;
  const float mrinv3 = rinv2*mrinv;
  const float mrinv5 = rinv2*mrinv3; 
  const float mrinv7 = rinv2*mrinv5;   // 16

  const float  D0  =  mrinv;
  const float  D1  = -mrinv3;
  const float  D2  =  mrinv5*( 3.0f);
  const float  D3  = -mrinv7*(15.0f); // 3

  const float q11 = Q0.x;
  const float q22 = Q0.y;
  const float q33 = Q0.z;
  const float q12 = Q1.x;
  const float q13 = Q1.y;
  const float q23 = Q1.z;

  const float  q  = q11 + q22 + q33;
  const float3 qR = make_float3(
      q11*dr.x + q12*dr.y + q13*dr.z,
      q12*dr.x + q22*dr.y + q23*dr.z,
      q13*dr.x + q23*dr.y + q33*dr.z);
  const float qRR = qR.x*dr.x + qR.y*dr.y + qR.z*dr.z;  // 22

  acc.w  -= D0 + 0.5f*(D1*q + D2*qRR);
  float C = D1 + 0.5f*(D2*q + D3*qRR);
  acc.x  += C*dr.x + D2*qR.x;
  acc.y  += C*dr.y + D2*qR.y;
  acc.z  += C*dr.z + D2*qR.z;               // 23
#endif  // total: 16 + 3 + 22 + 23 = 64 flops 

  return acc;
}

//Improved Barnes Hut criterium
static __device__ 
bool split_node_grav_impbh(
    const float4 nodeCOM, 
    const float4 groupCenter, 
    const float4 groupSize)
{
  //Compute the distance between the group and the cell
  float3 dr = make_float3(
      fabsf(groupCenter.x - nodeCOM.x) - (groupSize.x),
      fabsf(groupCenter.y - nodeCOM.y) - (groupSize.y),
      fabsf(groupCenter.z - nodeCOM.z) - (groupSize.z)
      );

  dr.x += fabsf(dr.x); dr.x *= 0.5f;
  dr.y += fabsf(dr.y); dr.y *= 0.5f;
  dr.z += fabsf(dr.z); dr.z *= 0.5f;

  //Distance squared, no need to do sqrt since opening criteria has been squared
  const float ds2    = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;

  return (ds2 <= fabsf(nodeCOM.w));
}
  
/************************************/
/********* SEGMENTED SCAN ***********/
/************************************/

static __device__ __forceinline__ int ShflSegScanStepB(
            int partial,
            uint distance,
            uint up_offset)
{
  asm(
      "{.reg .u32 r0;"
      ".reg .pred p;"
      "shfl.up.b32 r0, %1, %2, 0;"
      "setp.le.u32 p, %2, %3;"
      "@p add.u32 %1, r0, %1;"
      "mov.u32 %0, %1;}"
      : "=r"(partial) : "r"(partial), "r"(up_offset), "r"(distance));
  return partial;
}

  template<const int SIZE2>
static __device__ __forceinline__ int inclusive_segscan_warp_step(int value, const int distance)
{
#pragma unroll
  for (int i = 0; i < SIZE2; i++)
    value = ShflSegScanStepB(value, distance, 1<<i);
  return value;
}

static __device__ __forceinline__ int2 inclusive_segscan_warp(
    const int packed_value, int &dist_block, int &nseg)
{
  const int  flag = packed_value < 0;
  const int  mask = -flag;
  const int value = (mask & (-1-packed_value)) + (~mask & 1);

  const int flags = __ballot(flag);

  nseg      += __popc      (flags) ;
  dist_block = __clz(__brev(flags));

  const int distance = __clz(flags & lanemask_le()) + laneId - 31;
  const int val = inclusive_segscan_warp_step<WARP_SIZE2>(value, min(distance, laneId));
  return make_int2(val, __shfl(val, WARP_SIZE-1, WARP_SIZE));
}

static __device__ __forceinline__ int inclusive_segscan_array(int *data, const int n)
{
  const int laneId = threadIdx.x & (WARP_SIZE-1);
  int dist, nseg = 0;
  const int2 scanVal = inclusive_segscan_warp(data[laneId], dist, nseg);
  data[laneId] = scanVal.x;
  if (n <= WARP_SIZE) return nseg;

  int offset = scanVal.y;

  for (int p = WARP_SIZE; p < n; p += WARP_SIZE)
  {
    const int2 scanVal = inclusive_segscan_warp(data[p+laneId], dist, nseg);
    data[p+laneId] = scanVal.x + (offset & (-(laneId < dist)));
    offset = scanVal.y;
  }

  return nseg;
}

/******* walking kernel *******/


void walkLevel(..)
{
  const int laneIdx = threadIdx.x & (WARP_SIZE-1);
  const int warpIdx = threadIdx.x >> WARP_SIZE2;
  
  for (int i = 0; i < nCells; i += WARP_SIZE)
  {
    const bool useCell = i < nCells;

    const int      cellIdx  = in_cellList[i + laneIdx];
    const float4   cellCOM  = in_cellCOM [    cellIdx];
    const NodeData cellData = in_cellData[    cellIdx];   /* vector int4 load */

    const int firstChild = cellData.firstChild();
    const int  nChildren = cellData. nChildren();

    const bool toSplit = splitCell(cellCOM, groupCentre, groupSize);
    const bool isNode  = cellCOM.w < 0.0f;

    /* expand children nodes into stack */

    const int2 childOffset = warpIntExclusiveScan(nChildren & (-(useCell && isNode && toSplit)));
    nextLevelNodes[nextLevelCounter + childOffset.x] = firstChild;

    inclusive_segscan_array(nextLevelNodes+nextLevelCounter, childOffset.y);
    nextLevelCounter += childOffset.y;

    /* if stack is complete, lunch kernel for the next level */

    if (nextLevelCounter >= NEXT_LEVEL_THRESHOLD);
    {
      nextLevelCounter -= NEXT_LEVEL_THRESHOLD;
#pragma unroll
      for (int i = 0;  i < NEXT_LEVEL_THRESHOLD; i += WARP_SIZE)
        gmem_nextLevelNodes[i + laneId] = nextLevelNodes[nextLevelCounter + i + laneId];

      walkLevel<<<NEXT_LEVEL_THRESHOLD/NTHREADS,NTHREADS>>>(gmem_nextLevelNodes, NEXT_LEVEL_THRESHOLD);
    }

    /* store approximate interactions */

    const bool     isApprox = !toSplit && useCell;
    const int2 approxOffset = warpBinExclusiveScan(isApprox);
    if (isApprox)
      approxCellList[approxCellCounter + approxOffset.x] = cellIdx;
    approxCellCounter += approxOffset.y;

    if (approxCellCounter >= APPROX_THRESHOLD)
    {
      approxCellCounter -= APPROX_THRESHOLD;
#pragma unroll
      for (int i = 0; i < APPROX_THRESHOLD; i += WARP_SIZE)
        gmem_approxCellList[i + laneId] = approxCellList[approxCellCounter + i + laneId];

      approxEvaluate<<<APPROX_THRESHOLD/NTHREADS,NTHREADS>>>(gmem_approxCellList, APPROX_THRESHOLD);
    }

    /* store direct interactions */

    const bool       isLeaf = !isNode;
    const bool     isDirect = toSplit && isLeaf;
    const int2 directOffset = warpBinExclusiveScan(isDirect);
    if (isDirect)
      directLeafList[directLeafCounter + directOffset.x] = cellIdx;
    directLeafCounter += directOffset.y;
    
    if (directLeafCounter >= DIRECT_LEAF_THRESHOLD)
    {
      directLeafCounter -= DIRECT_LEAF_THRESHOLD;
#pragma unroll
      for (int i = 0; i < DIRECT_LEAF_THRESHOLD; i += WARP_SIZE)
        gmem_directLeafList[i + laneId] = directLeafList[directLeafCounter + i + laneId];

      directEvaluate<<<DIRECT_LEAF_THRESHOLD/NTHREADS,NTHREADS>>>(gmem_directLeafList, DIRECT_LEAF_THRESHOLD);
    }

  }


}


