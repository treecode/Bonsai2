static __device__ __forceinline__
Force evaluateDirect_warp(Force f, const int jptclIdx, const Particle iPtcl, const Particle *ptclList)
{
#if 1 
  const Particle jPtcl = ptclList[jptclIdx];
  const real eps2 = _eps2_const;
#pragma unroll
  for (int i = 0; i < WARP_SIZE; i++)
  {
    const Particle jp = jPtcl.shfl(i);
    force = addMonpole(force, ipos, jpos, eps2);
  }
  return force;
#endif
}

static __device__ __forceinline__ 
float4 addMonopole(
    float4 acc
    const float4 ipos,
    const float4 jpos,
    const float  eps2)
{
  const float3 dr = make_float3(jpos.x - ipos.x, jpos.y - ipos.y, jpos.z - ipos.z);  // 3

  const float r2     = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z + eps2;  // 6
  const float rinv   = rsqrtf(r2);    // 2
  const float rinv2  = rinv*rinv;     // 1
  const float jmass  = jpos.w;
  const float mrinv  = jmass * rinv;  // 1
  const float mrinv3 = mrinv * rinv2; // 1

  acc.w -= mrinv;                     // 1
  acc.x += mrinv3 * dr.x;             // 2
  acc.y += mrinv3 * dr.y;             // 2
  acc.z += mrinv3 * dr.z;             // 2

  // total 21 flops

  return acc;
}

  static __device__ __forceinline__
Force evaluateApprox_warp(Force f, const int jcellIdx, Particle iPtcl, const float4 *nodeMM, const float4 *nodeQM)
{
#if 1
  const float4 jcellMM = nodeMM[jcellIdx];
  const float4 jcellQM = nodeQM[jcellIdx];
  const real eps2  = _eps2_const;
#pragma unroll
  for (int i = 0; i < WARP_SIZE; i++)
  {
    const float4 jcMM = shfl_float4(jcMM, i);
    const float4 jcQM = shfl_float4(jcQM, i);
    force = addQuadrupole(force, ipos, jcMM, jcQM, eps2);
  }
  return force;
#endif
}

static __device__ __forceinline__ 
float4 addQuadrupole(
    float4 acc, 
    const float4 pos,
    const float4 Q0, 
    const float4 Q1, 
    const float eps2) 
{
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
  // total: 16 + 3 + 22 + 23 = 64 flops 

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

static __device__ __forceinline__ int2 exclusive_segscan_warp(
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
  return make_int2(val-value, __shfl(val, WARP_SIZE-1, WARP_SIZE));
}

static __device__ __forceinline__ int exclusive_segscan_array(int *data, const int n)
{
  const int laneId = threadIdx.x & (WARP_SIZE-1);
  int dist, nseg = 0;
  const int2 scanVal = exclusive_segscan_warp(data[laneId], dist, nseg);
  data[laneId] = scanVal.x;
  if (n <= WARP_SIZE) return nseg;

  int offset = scanVal.y;

  for (int p = WARP_SIZE; p < n; p += WARP_SIZE)
  {
    const int2 scanVal = exclusive_segscan_warp(data[p+laneId], dist, nseg);
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

    const int firstChild = cellData.firstChild();  /* can also be first particle, if it is a leaf */
    const int  nChildren = cellData. nChildren();  /* can be more than 8,         if it is a leaf */

    const bool toSplit = splitCell(cellCOM, groupCentre, groupSize);
    const bool isNode  = cellData.isNode();

    /* expand children nodes into stack */

    {
      const int isSplit = isNode && toSplit && useCell;
      const int2 childOffset = warpIntExclusiveScan(nChildren & (-isSplit));
      for (int i = 0 ; i < childOffset.y; i += WARP_SIZE)
        if (i + laneId < childOffset.y)
          nextLevelNodes[nextLevelCounter + childOffset.x] = 1;
      if (isSplit)
        nextLevelNodes[nextLevelCounter + childOffset.x] = -1-firstChild;  /* negative numbers mean beginning of the segment */

      exclusive_segscan_array(nextLevelNodes+nextLevelCounter, childOffset.y);
      nextLevelCounter += childOffset.y;

      /* if stack is complete, lunch kernel for the next level */

      if (nextLevelCounter >= NEXT_LEVEL_THRESHOLD);
      {
        nextLevelCounter -= NEXT_LEVEL_THRESHOLD;
#pragma unroll
        for (int i = 0;  i < NEXT_LEVEL_THRESHOLD; i += WARP_SIZE)
          gNextLevelNodes[i + laneId] = nextLevelNodes[nextLevelCounter + i + laneId];

        /* schedule next level walk */
        walkLevel<<<NEXT_LEVEL_THRESHOLD/NTHREADS,NTHREADS>>>(gNextLevelNodes, NEXT_LEVEL_THRESHOLD);
      }
    }

    /* store & evaluate approximate interactions */

    {
      const bool     isApprox = !toSplit && useCell;
      const int2 approxOffset = warpBinExclusiveScan(isApprox);

      /* store cell's index in shared memory to evaluate later */
      if (isApprox && approxCounter + approxOffset.x < WARP_SIZE)
        approxList[approxCounter + approxOffset.x] = cellIdx;

      if (approxCounter + approxOffset.y >= WARP_SIZE)
      {
        /* evaluation part, since it is warp-synchronious programming, no shmem is used */
        iForce = evaluateApprox_warp(iForce, approxList[laneId], iPtcl, nodeMM, nodeQM);
        approxCounter -= WARP_SIZE;
        if (isApprox && approxCounter + approxOffset.x >= 0)
          approxList[approxCounter + approxOffset.x] = cellIdx;
      }
    }

    /* store & evaluate direct interactions */

    {
      const bool       isLeaf = !isNode;
      const bool     isDirect = toSplit && isLeaf && useCell;
      const int2 childOffset  = warpIntExclusiveScan(nChildren & (-isDirect));
      const int     nptcl = childOffset.y;
      const int    offset = childOffset.x;
      const int firstPtcl = firstChild;
      for (int i = 0; i < nptcl; i += WARP_SIZE)
        if (i + laneId < nptcl)
          directList[i + laneId] = 1;
      if (isDirect)
        directList[directCounter + offset] = -1-firstPtcl;

      exclusive_segscan_array(directList + directCounter, nptcl);
      directCounter += nptcl;

      while (directCounter >= WARP_SIZE)
      {
        directCounter -= WARP_SIZE;
        iForce = evaluateDirect_warp(iForce, directList[directCounter + laneId], iPtcl, ptclList);
      }
    }

  }

}


