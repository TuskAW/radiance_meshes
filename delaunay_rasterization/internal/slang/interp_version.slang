import intersect;
import utils;
import safe_math;

struct SplatTetrahedra_interp : IDifferentiable
{
    Tetrahedra tet;
    float density;          // Single scalar density for the entire tetrahedron
    float4x3 vertex_color; // Per-vertex color to be barycentrically interpolated
};

// Given three vertex-colors (c0, c1, c2) and the barycentric coords (u,v)
//   w0 = u, w1 = v, w2 = 1 - u - v
// returns the final interpolated color:
[Differentiable]
float3 interpolate_triangle_color(
    float3 c0, float3 c1, float3 c2,
    float u, float v)
{
    float w0 = u;
    float w1 = v;
    float w2 = 1.0f - w0 - w1;
    return c0 * w0 + c1 * w1 + c2 * w2;
}

[Differentiable]
float4 tet_barycentric(float3 p, float3 a, float3 b, float3 c, float3 d)
{
    // Construct matrix T from tetrahedron vertices
    float3x3 T = float3x3(a - d, b - d, c - d);

    // Solve for (u, v, w)
    float3 uvw = SolveLinearSystem(T, p - d);

    // Compute t
    float t = 1.0 - uvw.x - uvw.y - uvw.z;

    return float4(uvw, t);
}

[Differentiable]
bool ray_tetrahedron_intersect_interp(
    float3 orig,
    float3 dir,
    Tetrahedra g,
    float4x3 vertex_color,  // color array for the tetrahedron's 4 vertices
    out float2 dist,
    out float3 color_enter,
    out float3 color_exit)
{
    float3 v0 = g.verts[0];
    float3 v1 = g.verts[1];
    float3 v2 = g.verts[2];
    float3 v3 = g.verts[3];

    bool   any_hit  = false;
    float  t_enter  = FLT_MAX;
    float  t_exit   = -FLT_MAX;
    float3 c_enter  = (float3)(0,0,0);
    float3 c_exit   = (float3)(0,0,0);

    // Temporary for each face test
    float t_face, u, v;
    float3 c_face;

    // Face #1: (v0, v1, v2)
    if (ray_triangle_intersect(orig, dir, v0, v1, v2, t_face, u, v))
    {
        any_hit = true;
        c_face = interpolate_triangle_color(
            vertex_color[0], vertex_color[1], vertex_color[2],
            u, v);

        // Track the near/far hits
        if (t_face < t_enter) {
            t_enter = t_face;
            c_enter = c_face;
        }
        if (t_face > t_exit) {
            t_exit = t_face;
            c_exit = c_face;
        }
    }

    // Face #2: (v0, v1, v3)
    if (ray_triangle_intersect(orig, dir, v0, v1, v3, t_face, u, v))
    {
        any_hit = true;
        c_face = interpolate_triangle_color(
            vertex_color[0], vertex_color[1], vertex_color[3],
            u, v);

        if (t_face < t_enter) {
            t_enter = t_face;
            c_enter = c_face;
        }
        if (t_face > t_exit) {
            t_exit = t_face;
            c_exit = c_face;
        }
    }

    // Face #3: (v0, v2, v3)
    if (ray_triangle_intersect(orig, dir, v0, v2, v3, t_face, u, v))
    {
        any_hit = true;
        c_face = interpolate_triangle_color(
            vertex_color[0], vertex_color[2], vertex_color[3],
            u, v);

        if (t_face < t_enter) {
            t_enter = t_face;
            c_enter = c_face;
        }
        if (t_face > t_exit) {
            t_exit = t_face;
            c_exit = c_face;
        }
    }

    // Face #4: (v1, v2, v3)
    if (ray_triangle_intersect(orig, dir, v1, v2, v3, t_face, u, v))
    {
        any_hit = true;
        c_face = interpolate_triangle_color(
            vertex_color[1], vertex_color[2], vertex_color[3],
            u, v);

        if (t_face < t_enter) {
            t_enter = t_face;
            c_enter = c_face;
        }
        if (t_face > t_exit) {
            t_exit = t_face;
            c_exit = c_face;
        }
    }

    // If the ray starts inside the tetrahedron, treat that as t_enter=0
    if (point_in_tetrahedron(g, orig)) {
        t_enter = 0.0f;
        float4 coords = tet_barycentric(orig, v0, v1, v2, v3);
        c_enter = coords.x * vertex_color[0] +
                  coords.y * vertex_color[1] +
                  coords.z * vertex_color[2] +
                  coords.w * vertex_color[3];
        any_hit = true;
    }

    dist        = {t_enter, t_exit};
    color_enter = c_enter;
    color_exit  = c_exit;
    return any_hit && (t_exit >= t_enter);
}

[Differentiable]
float3 compute_integral(float3 c0, float3 c1, float d, float t0, float t1) {
    // Precompute exponentials
    float expT0 = safe_exp(d * t0);
    float expT1 = safe_exp(d * t1);

    // Precompute common terms
    float dT0 = d * t0;
    float dT1 = d * t1;

    // Compute X and Y using precomputed values
    float X = expT0 - expT1 - dT0 * expT1 + dT1 * expT1;
    float Y = -expT0 + expT1 + dT0 * expT0 - dT1 * expT0;

    // Compute numerator and denominator
    float3 numerator = X * c0 + Y * c1;
    float denominator = d * d * safe_exp(d * (t0 + t1)) * (t0 - t1) * (t0 - t1);

    // Final result
    return safe_div(numerator, denominator);
}

[Differentiable]
CtrlPt evaluate_tetra_interp(
    in SplatTetrahedra_interp g, 
    in float3 ray_o,
    in float3 ray_d)
{
    float2 dist;
    float3 c_enter, c_exit;

    bool hit = ray_tetrahedron_intersect_interp(
        ray_o, ray_d,
        // float4x3(g.tet.verts[0], g.tet.verts[1], g.tet.verts[2], g.tet.verts[3]),
        g.tet,
        g.vertex_color,
        dist,
        c_enter,
        c_exit);

    // Same idea as before: average intersection_t, thickness dt, etc.
    float intersection_t = 0.5f * (dist.x + dist.y);
    float dt = max(abs(dist.y - dist.x), 0.0f);

    if (!hit) {
        return { (float4)(0.f, 0.f, 0.f, 0.f), intersection_t };
    } 
    else {
        // Opacity from density
        float alpha = safe_clip(1.0f - safe_exp(-g.density * dt), 0.f, 1.f);
        float3 final_color = {0.f, 0.f, 0.f};//compute_integral(c_enter, c_exit, g.density, dist.x, dist.y);

        // For demonstration: we can average the front/back colors, multiply by alpha:
        float4 rgba = {final_color.x,
                       final_color.y,
                       final_color.z,
                       alpha};
        return { rgba, intersection_t };
    }
}

////////////////////////////////////////////////////////////////////////////////
// 5) Loading functions for separate density & vertex colors
//    We add "_interp" suffix to keep the old ones intact.
////////////////////////////////////////////////////////////////////////////////

[Differentiable]
SplatTetrahedra_interp load_tet_alphablend_interp(
    int32_t        g_idx,
    DiffTensorView vertices,
    TensorView<int32_t> indices,
    DiffTensorView tet_density,
    DiffTensorView vertex_color)
{
    uint4 virtual_tet = load_virtual_tetrahedra(g_idx, indices);
    Tetrahedra tet    = load_tetrahedra(vertices, virtual_tet);

    // Here we assume each tetrahedron has a single scalar density in tet_density:
    float density = read_t1_float(g_idx, tet_density);

    // For vertex colors, we read color from each vertex index in the tetrahedron:
    float3 c0 = read_t3_float3(virtual_tet.x, vertex_color);
    float3 c1 = read_t3_float3(virtual_tet.y, vertex_color);
    float3 c2 = read_t3_float3(virtual_tet.z, vertex_color);
    float3 c3 = read_t3_float3(virtual_tet.w, vertex_color);

    SplatTetrahedra_interp result;
    result.tet       = tet;
    result.density   = density;
    result.vertex_color[0] = c0;
    result.vertex_color[1] = c1;
    result.vertex_color[2] = c2;
    result.vertex_color[3] = c3;
    return result;
}

// Non-diff version (if needed)
SplatTetrahedra_interp load_tet_alphablend_interp(
    int32_t        g_idx,
    TensorView<float> vertices,
    TensorView<int32_t> indices,
    TensorView<float>  tet_density,
    TensorView<float>  vertex_color)
{
    uint4 virtual_tet = load_virtual_tetrahedra(g_idx, indices);
    Tetrahedra tet    = load_tetrahedra(vertices, virtual_tet);

    float density = tet_density[g_idx];  // or read_float(g_idx, tet_density) if you prefer

    // For each vertex index:
    float3 c0 = read_t3_float3(virtual_tet.x, vertex_color);
    float3 c1 = read_t3_float3(virtual_tet.y, vertex_color);
    float3 c2 = read_t3_float3(virtual_tet.z, vertex_color);
    float3 c3 = read_t3_float3(virtual_tet.w, vertex_color);

    SplatTetrahedra_interp result;
    result.tet       = tet;
    result.density   = density;
    result.vertex_color[0] = c0;
    result.vertex_color[1] = c1;
    result.vertex_color[2] = c2;
    result.vertex_color[3] = c3;
    return result;
}
