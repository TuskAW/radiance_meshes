#define TRI_PER_G 4
#define PT_PER_G 4
// #define TRI_PER_G 6
// #define PT_PER_G 5
#include "cuda_util.h"
#include "initialize_density.h"
#include "glm/glm.hpp"
#include "structs.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include "Forward.h"

__device__ static const float SH_C0 = 0.28209479177387814f;

__global__ void
kern_initialize_density_so(
    const glm::vec3 *means,
    const glm::vec3 *scales,
    const glm::vec4 *quats,
    const float *densities,
    const float *features,
    const size_t num_prims,
    const glm::vec3 *rayo,
    float *initial_drgb)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 0 || i >= num_prims)
    return;

  const glm::vec4 quat = glm::normalize(quats[i]);
  const float density = densities[i];
  const glm::vec3 center = means[i];
  const glm::vec3 size = scales[i];

  const float r = quat.x;
  const float x = quat.y;
  const float y = quat.z;
  const float z = quat.w;

  const glm::mat3 Rt = {
      1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - r * z),
      2.0 * (x * z + r * y),

      2.0 * (x * y + r * z),       1.0 - 2.0 * (x * x + z * z),
      2.0 * (y * z - r * x),

      2.0 * (x * z - r * y),       2.0 * (y * z + r * x),
      1.0 - 2.0 * (x * x + y * y)};
  const glm::mat3 R = glm::transpose(Rt);
  //
  const glm::mat3 invS = {
      1/size.x, 0, 0, 0, 1/size.y, 0, 0, 0, 1/size.z,
  };
  const glm::mat3 S = {
      size.x, 0, 0, 0, size.y, 0, 0, 0, size.z,
  };

  const glm::vec3 Trayo = (Rt * (rayo[0] - center)) / size;
  float dist = max(Trayo.x, 0.f) + max(Trayo.y, 0.f) + max(Trayo.z, 0.f);
  if (dist <= 1 && Trayo.x >= 0 && Trayo.y >= 0 && Trayo.z >= 0) {
    // printf("(%i: %f,%f) pos: (%f, %f, %f)\n", i, dist, density, center.x, center.y, center.z);
    glm::vec3 color = {
      max(features[i*3 + 0] * SH_C0 + 0.5, 0.f),
      max(features[i*3 + 1] * SH_C0 + 0.5, 0.f),
      max(features[i*3 + 2] * SH_C0 + 0.5, 0.f),
    };
    // printf("c: %f, %f, %f, %f\n", density, density*color.x, density*color.y, density*color.z);
    atomicAdd(initial_drgb+0, density);
    atomicAdd(initial_drgb+1, density*color.x);
    atomicAdd(initial_drgb+2, density*color.y);
    atomicAdd(initial_drgb+3, density*color.z);
  }
}

void initialize_density(Params *params) {
  const size_t block_size = 1024;
  int num_prims = params->means.size;

  size_t size = sizeof(float)*4;

  float *d_initial_drgb;
  float *initial_drgb = (float*)malloc(size);
  CUDA_CHECK(cudaMalloc(&d_initial_drgb, size));
  CUDA_CHECK(cudaMemset(d_initial_drgb, 0, size));

  // printf("i: %p\n", (glm::vec3 *)(params->means.data));

  kern_initialize_density_so<<<(num_prims + block_size - 1) / block_size, block_size>>>(
      (glm::vec3 *)(params->means.data),
      (glm::vec3 *)(params->scales.data),
      (glm::vec4 *)(params->quats.data),
      (float *)(params->densities.data),
      (float *)(params->features.data),
      num_prims,
      (glm::vec3 *)(params->ray_origins.data),
      d_initial_drgb);
  CUDA_SYNC_CHECK();

  CUDA_CHECK(cudaMemcpy(initial_drgb, d_initial_drgb, size, cudaMemcpyDeviceToHost));
  params->initial_drgb.x = initial_drgb[0];
  params->initial_drgb.y = initial_drgb[1];
  params->initial_drgb.z = initial_drgb[2];
  params->initial_drgb.w = initial_drgb[3];
  cudaFree(d_initial_drgb);
  free(initial_drgb);
}

void initialize_density_zero(Params *params) {
  params->initial_drgb.x = 0.f;
  params->initial_drgb.y = 0.f;
  params->initial_drgb.z = 0.f;
  params->initial_drgb.w = 0.f;
}
