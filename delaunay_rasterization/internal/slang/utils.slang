// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import spherical_harmonics;
import intersect;
import safe_math;

static const float FLT_MAX = 1e20;
static const float FLT_MIN = -1e20;
static const float eps = 1e-7;

struct Triangle2D {
    float2 a, b, c;
}


struct SplatTetrahedra : IDifferentiable
{
    Tetrahedra tet;
    float4 rgbs;
};

struct Gaussian_3D : IDifferentiable
{
    float3 xyz_ws;
    SpherHarmCoeffs sh_coeffs;
    float4 rotations;
    float3 scales;
};

struct Splat_2D_Vertex : IDifferentiable
{
    float3 xyz_vs;
    float3 rgb;
    float2x2 cov_vs;
};

struct Splat_2D_AlphaBlend : IDifferentiable
{
    float3 xyz_vs;
    float3 rgb;
    float opacity;
    float2x2 inv_cov_vs;
};

[Differentiable]
float read_t1_float(uint32_t idx, DiffTensorView t1)
{
    return t1[uint2(idx, 0)];
}

float3 read_t3_float3(uint32_t idx, TensorView<float> t3)
{
    return float3(t3[uint2(idx, 0)],
                  t3[uint2(idx, 1)],
                  t3[uint2(idx, 2)]);
}

[Differentiable]
float3 read_t3_float3(uint32_t idx, DiffTensorView t3)
{
    return float3(t3[uint2(idx, 0)],
                  t3[uint2(idx, 1)],
                  t3[uint2(idx, 2)]);
}

[Differentiable]
float4 read_t4_float4(uint32_t idx, DiffTensorView t4)
{
    return float4(t4[uint2(idx, 0)],
                  t4[uint2(idx, 1)],
                  t4[uint2(idx, 2)],
                  t4[uint2(idx, 3)]);
}

float4 read_t4_float4(uint32_t idx, TensorView<float> t4)
{
    return float4(t4[uint2(idx, 0)],
                  t4[uint2(idx, 1)],
                  t4[uint2(idx, 2)],
                  t4[uint2(idx, 3)]);
}

[Differentiable]
float4 read_t2x2_float4(uint32_t idx, DiffTensorView t22)
{
    return float4(t22[uint3(idx, 0, 0)],
                  t22[uint3(idx, 1, 0)],
                  t22[uint3(idx, 0, 1)],
                  t22[uint3(idx, 1, 1)]);
}

[Differentiable]
float2x2 read_t2x2_float2x2(uint32_t idx, DiffTensorView t2x2)
{
    return float2x2(t2x2[uint3(idx, 0, 0)],
                    t2x2[uint3(idx, 1, 0)],
                    t2x2[uint3(idx, 0, 1)],
                    t2x2[uint3(idx, 1, 1)]);
}


float clip(float val, float min_val, float max_val)
{
    return max(min_val, min(max_val, val));
}


[Differentiable]
float3 geom_transform_points(float3 point, float4x4 transf_matrix)
{
    float4 p_out = mul(transf_matrix, float4(point, 1.0));
    return p_out.xyz / (p_out.w + eps);
}

[Differentiable]
float3 geom_transform_points2(float3 point, float4x4 transf_matrix)
{
    float4 p_out = mul(transf_matrix, float4(point, 1.0));
    return p_out.xyz;
}

uint4 load_virtual_tetrahedra(int32_t g_idx,
                              TensorView<int32_t> indices)
{
    return { 
        indices[uint2(g_idx, 0)],
        indices[uint2(g_idx, 1)],
        indices[uint2(g_idx, 2)],
        indices[uint2(g_idx, 3)]
    };
}

[Differentiable]
Tetrahedra load_tetrahedra(DiffTensorView vertices,
                           uint4 virtual_tetra)
{
    float3[4] tetra = {
        read_t3_float3(virtual_tetra.x, vertices),
        read_t3_float3(virtual_tetra.y, vertices),
        read_t3_float3(virtual_tetra.z, vertices),
        read_t3_float3(virtual_tetra.w, vertices),
    };
    return { tetra };
}

Tetrahedra load_tetrahedra(TensorView<float> vertices,
                           uint4 virtual_tetra)
{
    float3[4] tetra = {
        read_t3_float3(virtual_tetra.x, vertices),
        read_t3_float3(virtual_tetra.y, vertices),
        read_t3_float3(virtual_tetra.z, vertices),
        read_t3_float3(virtual_tetra.w, vertices),
    };
    return { tetra };
}
[Differentiable]
float compute_det(float2x2 M) {
    return M[0][0] * M[1][1] - M[0][1] * M[1][0];
}


[Differentiable]
float3 SolveLinearSystem(float3x3 A, float3 b)
{
    // Calculate the determinant of A
    float det = det_3x3(A._11_21_31, A._12_22_32, A._13_23_33);
    
    // Check if the system has a unique solution
    if (abs(det) < 1e-20)
    {
        return float3(0, 0, 0); // Return zero vector if no unique solution
    }

    float a11 = A[0][0];
    float a12 = A[0][1];
    float a13 = A[0][2];
    float a21 = A[1][0];
    float a22 = A[1][1];
    float a23 = A[1][2];
    float a31 = A[2][0];
    float a32 = A[2][1];
    float a33 = A[2][2];
    
    // Calculate the inverse of A
    return {
        (a22 * a33 - a23 * a32) / det * b.x + -(a12 * a33 - a13 * a32) / det * b.y + (a12 * a23 - a13 * a22) / det * b.z,
        -(a21 * a33 - a23 * a31) / det * b.x + (a11 * a33 - a13 * a31) / det * b.y + -(a11 * a23 - a13 * a21) / det * b.z,
        (a21 * a32 - a22 * a31) / det * b.x + -(a11 * a32 - a12 * a31) / det * b.y + (a11 * a22 - a12 * a21) / det * b.z
    };
    /*

    float3x3 inv_A = 
    {
        (a22 * A._33 - A._23 * A._32) / det, -(A._12 * A._33 - A._13 * A._32) / det, (A._12 * A._23 - A._13 * a22) / det,
        -(A._21 * A._33 - A._23 * A._31) / det, (A._11 * A._33 - A._13 * A._31) / det, -(A._11 * A._23 - A._13 * A._21) / det,
        (A._21 * A._32 - a22 * A._31) / det, -(A._11 * A._32 - A._12 * A._31) / det, (A._11 * a22 - A._12 * A._21) / det
    };
    
    // Multiply inv_A with b to get x
    return mul(inv_A, b);
    */
}

// Compute the circumcenter of a tetrahedron
[Differentiable]
float3 calculate_circumcenter(Tetrahedra tet)
{
    // Compute vectors from v0 to other vertices
    float3 a = tet.verts[1] - tet.verts[0];
    float3 b = tet.verts[2] - tet.verts[0];
    float3 c = tet.verts[3] - tet.verts[0];
    
    // Compute squares of lengths
    float aa = dot(a, a);
    float bb = dot(b, b);
    float cc = dot(c, c);
    
    // Compute cross products
    float3 cross_bc = cross(b, c);
    float3 cross_ca = cross(c, a);
    float3 cross_ab = cross(a, b);
    
    // Compute denominator
    float denominator = 2.0f * dot(a, cross_bc);
    
    // Guard against division by zero
    /*
    if (abs(denominator) < 1e-6f)
    {
        // Return center of mass as fallback
        return tet.verts[0] + (a + b + c) / 4.0f;
    }
    
    // Compute circumcenter relative to verts[0]
    float3 relative_circumcenter = (
        aa * cross_bc +
        bb * cross_ca +
        cc * cross_ab
    ) / denominator;
    */
    float3 relative_circumcenter = safe_div(
        aa * cross_bc +
        bb * cross_ca +
        cc * cross_ab
    , denominator);
    
    // Return absolute position
    return tet.verts[0] + relative_circumcenter;
}

[Differentiable]
SplatTetrahedra load_tet_alphablend(int32_t g_idx,
                                    DiffTensorView vertices,
                                    TensorView<int32_t> indices,
                                    DiffTensorView rgbs)
{
    uint4 virtual_tet = load_virtual_tetrahedra(g_idx, indices);
    Tetrahedra tet = load_tetrahedra(vertices, virtual_tet);
    return {
        tet,
        read_t4_float4(g_idx, rgbs)
    };
}

SplatTetrahedra load_tet_alphablend(int32_t g_idx,
                                    TensorView<float> vertices,
                                    TensorView<int32_t> indices,
                                    TensorView<float> rgbs)
{
    uint4 virtual_tet = load_virtual_tetrahedra(g_idx, indices);
    Tetrahedra tet = load_tetrahedra(vertices, virtual_tet);
    return {
        tet,
        read_t4_float4(g_idx, rgbs)
    };
}

struct CtrlPt: IDifferentiable {
    float4 rgba;
    float2 dist;
    float density;
};

[Differentiable]
CtrlPt evaluate_tetra(in SplatTetrahedra g, 
                      in float3 ray_o,
                      in float3 ray_d)
{
    // float2 dist = ray_tetrahedron_intersect(ray_o, ray_d, g.tet);
    // float2 dist = ray_tetrahedron_intersect(ray_o, ray_d, g.tet);
    float2 dist;
    // bool hit = cyrusBeckTetrahedronShearScale(
    bool hit = ray_tetrahedron_intersect_fused(
        ray_o, ray_d, float4x3(g.tet.verts[0], g.tet.verts[1], g.tet.verts[2], g.tet.verts[3]), dist);
    float intersection_t = (dist.x + dist.y)/2.f;
    float dt = max(abs(dist.y - dist.x), 0.0f);
    // float2 dist = cyrusBeckTetrahedronShearScale(ray_o, ray_d, g.tet.verts[0], g.tet.verts[1], g.tet.verts[2], g.tet.verts[3]);
    if (!hit) {
        return {{0.f, 0.f, 0.f, 0.f}, dist, g.rgbs.w};
    } else {
        float alpha = safe_clip(1-safe_exp(-g.rgbs.w * dt), 0, 1);
        float4 rgba = {
            alpha * g.rgbs.x, alpha * g.rgbs.y, alpha * g.rgbs.z, alpha};
        return {rgba, dist, g.rgbs.w};
    }
}

[Differentiable]
float3x3 get_covariance_from_quat_scales(float4 q, float3 s) {
    float r = q[0], x = q[1], y = q[2], z = q[3];

    float3x3 rotation_matrix = float3x3(
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y));

    float3x3 scales_matrix = float3x3(s[0], 0, 0,
                                      0, s[1], 0,
                                      0, 0, s[2]);

    float3x3 L = mul(rotation_matrix, scales_matrix);

    return mul(L, transpose(L));
}


bool is_point_inside_triangle(float2 p, Triangle2D tri) {
    // Compute barycentric coordinates
    float2 v0 = tri.b - tri.a;
    float2 v1 = tri.c - tri.a;
    float2 v2 = p - tri.a;
    float d00 = dot(v0, v0);
    float d01 = dot(v0, v1);
    float d11 = dot(v1, v1);
    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);
    float denom = d00 * d11 - d01 * d01;
    float u = safe_div((d11 * d20 - d01 * d21), denom);
    float v = safe_div((d00 * d21 - d01 * d20), denom);
    return (u >= 0) && (v >= 0) && (u + v <= 1);
}
