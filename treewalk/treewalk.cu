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


void walkLevel(..)
{
  const int laneId = threadIdx.x & (WARP_SIZE-1);
  const int warpId = threadIdx.x >> WARP_SIZE2;
}


