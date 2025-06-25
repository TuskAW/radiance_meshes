import intersect;
import utils;
import safe_math;

struct SplatTetrahedraLinear : IDifferentiable
{
    Tetrahedra tet;
    // float4x3 vertex_color; // Per-vertex color to be barycentrically interpolated
    float3 base_color;
    float3 grd;
    float density;          // Single scalar density for the entire tetrahedron
    float3 d_grd;
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
float4 tet_barycentric(float3 p, float4x3 vs)
{
    // Construct matrix T from tetrahedron vertices
    float3x3 T = float3x3(vs[0] - vs[3], vs[1] - vs[3], vs[2] - vs[3]);

    // Solve for (u, v, w)
    float3 uvw = SolveLinearSystem(T, p - vs[3]);

    // Compute t
    float t = 1.0 - uvw.x - uvw.y - uvw.z;

    return float4(uvw, t);
}

void print_verts(float4x3 verts) {
    printf("%f, %f, %f, %f\n", verts[0].x, verts[1].x, verts[2].x, verts[3].x);
}
void print_vec(float3 vec) {
    printf("(%f, %f, %f),", vec.x, vec.y, vec.z);
}

[Differentiable]
float3 compute_integral(float3 c0, float3 c1, float dt, float d) {
    float alpha = safe_exp(-d*dt);
    float X = (-d*dt*alpha + 1 - alpha);
    float Y = (d*dt-1) + alpha;
    return safe_div(X*c0+Y*c1, d*dt);
}

[Differentiable]
inline float3 mix_color(float4x3 vertex_color, float4 bary) {
    return max(vertex_color[0] * bary.x + vertex_color[1] * bary.y + vertex_color[2] * bary.z + vertex_color[3] * bary.w, 0.f);
}

// [Differentiable]
// inline float3 linear_color(float4x3 grad, float3 v)
// {
//     // return grad[0] + grad[1] * v.x + grad[2] * v.y + grad[3] * v.z;
//     return {
//         max(grad[0].r + dot(grad[1], v), 0.f),
//         max(grad[0].g + dot(grad[2], v), 0.f),
//         max(grad[0].b + dot(grad[3], v), 0.f),
//     };
// }

[Differentiable]
inline float3 linear_color(float3 grad, float3 v)
{
    // return grad[0] + grad[1] * v.x + grad[2] * v.y + grad[3] * v.z;
    float f = dot(grad, v);
    return {f, f, f};
}

[Differentiable]
float area_above_zero(float a, float b, float t) {
    if (a * b < 0.0) {
        // The line segment crosses the x-axis, forming two triangles.
        // We need to find the x-intercept to calculate their areas.
        float x_intercept = -a * t / (b - a);
        if (a > 0.0) {
            // The first triangle is above the axis.
            return 0.5 * a * x_intercept;
        } else {
            // The second triangle is above the axis.
            return 0.5 * b * (t - x_intercept);
        }
    } else if (a >= 0.0 && b >= 0.0) {
        // The entire trapezoid is above the x-axis.
        // The area of a trapezoid is 0.5 * (sum of parallel sides) * height.
        return 0.5 * (a + b) * t;
    } else {
        // The entire trapezoid is below the x-axis.
        return 0.0;
    }
}

[Differentiable]
float exp_area(float a, float b, float t) {
    float s = b - a;
    if (abs(s) > 1e-3) {
        return safe_div(safe_exp(t * a + (1-t) * b) - safe_exp(b), -s);
    } else {
        return safe_exp(a) * t;
    }
}


[Differentiable]
bool evaluate_tetra_linear(
    in SplatTetrahedraLinear g, 
    in float3 ray_o,
    in float3 ray_d,
    no_diff in float t_min,
    out CtrlPt ctrl_pt)
{
    float2 dist;

    float4x3 verts = float4x3(g.tet.verts[0], g.tet.verts[1], g.tet.verts[2], g.tet.verts[3]);

    bool hit = ray_tetrahedron_intersect_fused(
        ray_o, ray_d, verts, dist);

    float enter = dist.x;
    float exit = max(dist.y, enter);
    float dt = max(dist.y - enter, 0.0f);
    float intersection_t = 0.5*(exit+enter);

    if (!hit || dt < 1e-5f) {
        ctrl_pt = { (float4)(0.f, 0.f, 0.f, 0.f), {0.f, 0.f}, safe_exp(g.density) };
        return false;
    } else {
        /*
        float3x3 T = transpose(float3x3(verts[0] - verts[3], verts[1] - verts[3], verts[2] - verts[3]));
        float3 uvw_enter = SolveLinearSystem(T, enter * ray_d + ray_o - verts[3]);
        float3 uvw_exit = SolveLinearSystem(T, exit * ray_d + ray_o - verts[3]);

        float w_enter = safe_clip(1.0 - uvw_enter.x - uvw_enter.y - uvw_enter.z, 0, 1);
        float w_exit = safe_clip(1.0 - uvw_exit.x - uvw_exit.y - uvw_exit.z, 0, 1);

        float4 b_enter = float4(uvw_enter, w_enter);
        float4 b_exit = float4(uvw_exit, w_exit);

        float3 c_enter = mix_color(g.vertex_color, b_enter);
        float3 c_exit = mix_color(g.vertex_color, b_exit);
        */

        //*
        float3 v_enter = enter * ray_d + ray_o - verts[0];
        float3 v_exit = exit * ray_d + ray_o - verts[0];
        // float3 c_enter = max(linear_color(g.vertex_color, v_enter), 0.f);
        // float3 c_exit = max(linear_color(g.vertex_color, v_exit), 0.f);
        float3 c_enter = max(linear_color(g.grd, v_enter) + g.base_color, 0.f);
        float3 c_exit = max(linear_color(g.grd, v_exit) + g.base_color, 0.f);

        float d_enter = dot(g.d_grd, v_enter) + g.density;
        float d_exit = dot(g.d_grd, v_exit) + g.density;

        // float area = area_above_zero(d_enter, d_exit, dt);
        float area = exp_area(d_enter, d_exit, dt);
        //*/
        // float a_s = dot(g.d_grd, ray_d);
        // float x = dt;
        // if (a_s < -1e-3) {
        //     x = min(dt, -safe_div(g.density, a_s));
        // }
        // float area = x * (g.density + a_s / 2);

        // float alpha = safe_clip(1.0f - safe_exp(-g.density * dt), 0.f, 1.f);
        float alpha = safe_clip(1.0f - safe_exp(-area), 0.f, 1.f);
        float lb = safe_exp(min(d_enter, d_exit));
        float ub = safe_exp(max(d_enter, d_exit));
        float e_density = safe_clip(safe_div(area, dt), lb, ub);
        float3 final_color = compute_integral(c_exit, c_enter, dt, e_density);

        float4 rgba = {final_color.x,
                       final_color.y,
                       final_color.z,
                       alpha};
        ctrl_pt = { rgba, {enter, exit}, e_density };
        return true;
    }
}

////////////////////////////////////////////////////////////////////////////////
// 5) Loading functions for separate density & vertex colors
//    We add "_interp" suffix to keep the old ones intact.
////////////////////////////////////////////////////////////////////////////////

[Differentiable]
SplatTetrahedraLinear load_tet_alphablend_interp(
    int32_t        g_idx,
    DiffTensorView vertices,
    TensorView<int32_t> indices,
    DiffTensorView tet_density)
{
    no_diff uint4 virtual_tet = load_virtual_tetrahedra(g_idx, indices);
    Tetrahedra tet    = load_tetrahedra(vertices, virtual_tet);

    // Here we assume each tetrahedron has a single scalar density in tet_density:
    float density = float(tet_density[g_idx, 0]);
    float3 base_color = {
        float(tet_density[g_idx, 1]),
        float(tet_density[g_idx, 2]),
        float(tet_density[g_idx, 3]),
    };
    float3 grd = {
        float(tet_density[g_idx, 4]),
        float(tet_density[g_idx, 5]),
        float(tet_density[g_idx, 6]),
    };
    float3 d_grd = {
        float(tet_density[g_idx, 7]),
        float(tet_density[g_idx, 8]),
        float(tet_density[g_idx, 9]),
    };

    SplatTetrahedraLinear result = {
        tet, base_color, grd, density, d_grd
    };
    return result;
}

// Non-diff version (if needed)
SplatTetrahedraLinear load_tet_alphablend_interp(
    int32_t        g_idx,
    TensorView<float> vertices,
    TensorView<int32_t> indices,
    TensorView<float>  tet_density)
{
    uint4 virtual_tet = load_virtual_tetrahedra(g_idx, indices);
    Tetrahedra tet    = load_tetrahedra(vertices, virtual_tet);

    float density = float(tet_density[g_idx, 0]);
    SplatTetrahedraLinear result;
    result.tet       = tet;
    result.density   = density;
    result.base_color = {
        float(tet_density[g_idx, 1]),
        float(tet_density[g_idx, 2]),
        float(tet_density[g_idx, 3]),
    };
    result.grd = {
        float(tet_density[g_idx, 4]),
        float(tet_density[g_idx, 5]),
        float(tet_density[g_idx, 6]),
    };
    result.d_grd = {
        float(tet_density[g_idx, 7]),
        float(tet_density[g_idx, 8]),
        float(tet_density[g_idx, 9]),
    };
    return result;
}

