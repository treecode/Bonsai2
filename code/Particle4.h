#pragma once

template<int N, typename T> struct vec;
template<> struct vec<4,float>  { typedef float4  type; };
template<> struct vec<4,double> { typedef double4 type; };

template<> struct vec<3,float>  { typedef float3  type; };
template<> struct vec<3,double> { typedef double3 type; };

template<typename T>
struct Position
{
  T x, y, z;
  __host__ __device__ Position() {}
  __host__ __device__ Position(const T _x) : x(_x), y(_x), z(_x) {}

  __host__ __device__ Position(const T _x, const T _y, const T _z) : x(_x), y(_y), z(_z) {}
  static __host__ __device__ Position min(const Position &lhs, const Position &rhs) 
  {
    return Position( 
        fmin(lhs.x, rhs.x),
        fmin(lhs.y, rhs.y),
        fmin(lhs.z, rhs.z));
  }
  static __host__ __device__ Position max(const Position &lhs, const Position &rhs) 
  {
    return Position( 
        fmax(lhs.x, rhs.x),
        fmax(lhs.y, rhs.y),
        fmax(lhs.z, rhs.z));
  }
};

template<typename T> 
struct Particle4
{

  private:
  typename vec<4,T>::type packed_data;
  public:

  __host__ __device__ T x   ()  const { return packed_data.x;}
  __host__ __device__ T y   ()  const { return packed_data.y;}
  __host__ __device__ T z   ()  const { return packed_data.z;}
  __host__ __device__ T mass()  const { return packed_data.w;}
  __forceinline__ __device__ int get_idx() const;
  __forceinline__ __device__ int set_idx(const int);
  __forceinline__ __device__ int get_oct() const;
  __forceinline__ __device__ int set_oct(const int);

  __host__ __device__ T& x    () { return packed_data.x;}
  __host__ __device__ T& y    () { return packed_data.y;}
  __host__ __device__ T& z    () { return packed_data.z;}
  __host__ __device__ T& mass () { return packed_data.w;}
};

template<> __device__ __forceinline__ int Particle4<float>::get_idx() const
{
  return (__float_as_int(packed_data.w) >> 4) & 0xF0000000;
}
template<> __device__ __forceinline__ int Particle4<float>::get_oct() const
{
  return __float_as_int(packed_data.w) & 0xF;
}
template<> __device__ __forceinline__ int Particle4<float>::set_idx(const int idx)
{
  const int oct = get_oct();
  packed_data.w = __int_as_float((idx << 4) | oct);
  return idx;
}
template<> __device__ __forceinline__ int Particle4<float>::set_oct(const int oct)
{
  const int idx = get_idx();
  packed_data.w = __int_as_float((idx << 4) | oct);
  return oct;
}

template<> __device__ __forceinline__ int Particle4<double>::get_idx() const
{
  return ((unsigned long long)(packed_data.w) >> 4) & 0xF0000000;
}
template<> __device__ __forceinline__ int Particle4<double>::get_oct() const
{
  return (unsigned long long)(packed_data.w) & 0xF;
}
template<> __device__ __forceinline__ int Particle4<double>::set_idx(const int idx)
{
  const int oct = get_oct();
  packed_data.w = (unsigned long long)((idx << 4) | oct);
  return idx;
}
template<> __device__ __forceinline__ int Particle4<double>::set_oct(const int oct)
{
  const int idx = get_idx();
  packed_data.w = (unsigned long long)((idx << 4) | oct);
  return oct;
}