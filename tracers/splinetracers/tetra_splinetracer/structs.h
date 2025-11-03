#pragma once
#include "glm/glm.hpp"
#include <optix.h>

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

template <typename T> struct StructuredBuffer {
  T *data;
  size_t size;
};


struct GTransform {
    float3 scales;
    float3 mean;
    float4 quat;
    float height;
};

struct SplineState//((packed))
{
  float2 distortion_parts;
  float2 cum_sum;
  float3 padding;
  // Spline state
  float t;
  float4 drgb;

  // Volume Rendering State
  float logT;
  /*
  float d_spline;
  float3 avg_color;
  float area;
  */
  float3 C;
};

// Always on GPU
struct Primitives {
  float3 *means; 
  float3 *scales; 
  float4 *quats; 
  float *densities; 
  size_t num_prims;
  float *features; 
  size_t feature_size;

  glm::vec3 *vertices;
  glm::ivec3 *indices;
  size_t num_vertices;
  size_t num_indices;
};

