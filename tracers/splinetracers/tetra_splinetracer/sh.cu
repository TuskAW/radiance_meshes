#define GLM_FORCE_CUDA

#define TRI_PER_G 4
#define PT_PER_G 4
// #define TRI_PER_G 6
// #define PT_PER_G 5
#include "create_triangles.h"
#include "cuda_util.h"
#include "glm/glm.hpp"
#include "structs.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <stdexcept>

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {1.0925484305920792f, -1.0925484305920792f,
                                  0.31539156525252005f, -1.0925484305920792f,
                                  0.5462742152960396f};
__device__ const float SH_C3[] = {-0.5900435899266435f, 2.890611442640554f,
                                  -0.4570457994644658f, 0.3731763325901154f,
                                  -0.4570457994644658f, 1.445305721320277f,
                                  -0.5900435899266435f};

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int deg, int max_coeffs,
                                        const glm::vec3 pos, glm::vec3 origin,
                                        const float *sh) {
  // The implementation is loosely based on code for
  // "Differentiable Point-Based Radiance Fields for
  // Efficient View Synthesis" by Zhang et al. (2022)
  glm::vec3 dir = pos - origin;
  dir = dir / glm::length(dir);

  glm::vec3 result = SH_C0 * sh[0];

  if (deg > 0) {
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;
    result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

    if (deg > 1) {
      float xx = x * x, yy = y * y, zz = z * z;
      float xy = x * y, yz = y * z, xz = x * z;
      result = result + SH_C2[0] * xy * sh[4] + SH_C2[1] * yz * sh[5] +
               SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
               SH_C2[3] * xz * sh[7] + SH_C2[4] * (xx - yy) * sh[8];

      if (deg > 2) {
        result = result + SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
                 SH_C3[1] * xy * z * sh[10] +
                 SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
                 SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
                 SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
                 SH_C3[5] * z * (xx - yy) * sh[14] +
                 SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
      }
    }
  }
  result += 0.5f;
  return glm::max(result, 0.0f);
}

__global__ void kern_eval_sh(const glm::vec3 *means, const float *shs,
                             const glm::vec3 origin, float *colors, int deg,
                             int max_coeffs) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 0 || i >= num_prims)
    return;
  glm::vec3 *sh = ((glm::vec3 *)shs) + idx * max_coeffs;
  glm::vec3 mean = means[i];
  glm::vec3 color = computeColorFromSH(deg, max_coeffs, mean, origin, sh);
  // convert to SH deg 0
  colors[idx * 3 + 0] = (color.x - 0.5) / C0;
  colors[idx * 3 + 1] = (color.y - 0.5) / C0;
  colors[idx * 3 + 2] = (color.z - 0.5) / C0;
}

Primitives eval_sh(Primitives prims, int deg, int max_coeffs, glm::vec3 origin,
                   float *color_buffer) {
  Primitives return_prims = prims;
  colors;
  kern_eval_sh<<<(prims.num_prims + block_size - 1) / block_size, block_size>>>(
      (glm::vec3 *)prims.means, prims.features, origin, color_buffer, deg,
      max_coeffs);
  CUDA_SYNC_CHECK();
  return_prims.features = color_buffer;
  return_prims.feature_size = 3;
  return return_prims;
}
