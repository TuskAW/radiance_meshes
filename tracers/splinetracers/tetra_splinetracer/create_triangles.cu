#define TRI_PER_G 4
#define PT_PER_G 4
// #define TRI_PER_G 6
// #define PT_PER_G 5
#include "cuda_util.h"
#include "create_triangles.h"
#include "glm/glm.hpp"
#include "structs.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <stdexcept>

// __device__ static const glm::ivec3 base_indices[8] = {
//     {0, 2, 1}, {0, 3, 2}, {0, 4, 3}, {0, 1, 4},
//     {5, 1, 2}, {5, 2, 3}, {5, 3, 4}, {5, 4, 1},
// };
// __device__ static const glm::ivec3 base_indices[6] = {
//     {0, 2, 1}, {0, 3, 2}, {0, 1, 3}, {1, 2, 4},
//     {2, 3, 4}, {3, 1, 4},
// };
__device__ static const glm::ivec3 base_indices[4] = {
    {0, 2, 1}, {0, 3, 2}, {0, 1, 3}, {1, 2, 3}};

__global__ void
kern_create_triangles(const glm::vec3 *means, const glm::vec3 *scales,
                      const glm::vec4 *quats, const float *densities,
                      const size_t num_prims, glm::vec3 *vertices,
                      glm::ivec3 *indices) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 0 || i >= num_prims)
    return;

  const glm::vec4 quat = glm::normalize(quats[i]);
  // const float density = densities[i];
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
  const glm::mat3 S = {
      size.x, 0, 0, 0, size.y, 0, 0, 0, size.z,
  };
  const glm::mat3 T = R * S;
  glm::mat4x3 xfm;

  xfm[0] = T[0];
  xfm[1] = T[1];
  xfm[2] = T[2];
  xfm[3] = center;

  // vertices[i * PT_PER_G + 0] = glm::vec3(xfm * glm::vec4(0.f, 0.f, 1.f, 1.f));
  //
  // vertices[i * PT_PER_G + 1] = glm::vec3(xfm * glm::vec4(0.f, 1.f, 0.f, 1.f));
  // vertices[i * PT_PER_G + 2] = glm::vec3(xfm * glm::vec4(1.f, 0.f, 0.f, 1.f));
  // vertices[i * PT_PER_G + 3] = glm::vec3(xfm * glm::vec4(0.f, -1.f, 0.f, 1.f));
  // vertices[i * PT_PER_G + 4] = glm::vec3(xfm * glm::vec4(-1.f, 0.f, 0.f, 1.f));
  // vertices[i * PT_PER_G + 5] = glm::vec3(xfm * glm::vec4(0.f, 0.f, -1.f, 1.f));

  vertices[i * PT_PER_G + 0] = glm::vec3(xfm * glm::vec4(0.f, 0.f, 1.f, 1.f));
  vertices[i * PT_PER_G + 1] = glm::vec3(xfm * glm::vec4(0.f, 1.f, 0.f, 1.f));
  vertices[i * PT_PER_G + 2] = glm::vec3(xfm * glm::vec4(1.f, 0.f, 0.f, 1.f));
  vertices[i * PT_PER_G + 3] = glm::vec3(xfm * glm::vec4(0.f, 0.f, 0.f, 1.f));

  // #pragma unroll
  //   for (int j = 0; j < TRI_PER_G; j++) {
  //     indices[i * TRI_PER_G + j] = i * PT_PER_G + base_indices[j];
  //   }

  // vertices[i * PT_PER_G + 1] = glm::vec3(xfm * glm::vec4(0.f, 1.f, 0.f, 1.f));
  // vertices[i * PT_PER_G + 2] = glm::vec3(xfm * glm::vec4( 0.86602540378f, -0.5f, 0.f, 1.f));
  // vertices[i * PT_PER_G + 3] = glm::vec3(xfm * glm::vec4(-0.86602540378f, -0.5f, 0.f, 1.f));
  // vertices[i * PT_PER_G + 4] = glm::vec3(xfm * glm::vec4(0.f, 0.f, -1.f, 1.f));

  // vertices[i * PT_PER_G + 0] = glm::vec3(xfm * glm::vec4(0.f, 0.f, 0.5, 1.f));
  // vertices[i * PT_PER_G + 1] = glm::vec3(xfm * glm::vec4(0.f, 1.f, -0.5, 1.f));
  // vertices[i * PT_PER_G + 2] = glm::vec3(xfm * glm::vec4( 0.86602540378f, -0.5f, -0.5, 1.f));
  // vertices[i * PT_PER_G + 3] = glm::vec3(xfm * glm::vec4(-0.86602540378f, -0.5f, -0.5, 1.f));

#pragma unroll
  for (int j = 0; j < TRI_PER_G; j++) {
    indices[i * TRI_PER_G + j] = i * PT_PER_G + base_indices[j];
  }
}

void create_triangles(Primitives &prims) {
  const size_t block_size = 1024;
  if (!((prims.num_vertices >= prims.num_prims * PT_PER_G) &&
        (prims.num_indices >= prims.num_prims * TRI_PER_G))) {
    if (prims.num_vertices != 0) {
      CUDA_CHECK(cudaFree(reinterpret_cast<void *>(prims.vertices)));
    }
    if (prims.num_indices != 0) {
      CUDA_CHECK(cudaFree(reinterpret_cast<void *>(prims.indices)));
    }
    prims.num_vertices = prims.num_prims * PT_PER_G;
    prims.num_indices = prims.num_prims * TRI_PER_G;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&prims.vertices),
                          prims.num_vertices * sizeof(glm::vec3)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&prims.indices),
                          prims.num_indices * sizeof(glm::ivec3)));
  }
  prims.num_vertices = prims.num_prims * PT_PER_G;
  prims.num_indices = prims.num_prims * TRI_PER_G;

  kern_create_triangles<<<(prims.num_prims + block_size - 1) / block_size,
                          block_size>>>(
      (glm::vec3 *)prims.means, (glm::vec3 *)prims.scales,
      (glm::vec4 *)prims.quats, prims.densities, prims.num_prims,
      prims.vertices, prims.indices);
  CUDA_SYNC_CHECK();
}
