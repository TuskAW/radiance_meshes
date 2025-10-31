
import utils;
import camera;
import safe_math;
import intersect;
import interp_version;

struct DistortionState5 : IDifferentiable {
    float x;  // old_state.x
    float y;  // old_state.y
    float z;  // old_state.z
    float w;  // old_state.w
    float v;  // new 5th term, e.g., for w_i^2 * interval / 3, or anything else
};

[Differentiable]
float moment_0_int(float a, float b, float s1, float T) {
    return s1;
}
[Differentiable]
float moment_1_int(float a, float b, float s1, float T) {
    return s1 * (b+a)/2;
}
[Differentiable]
float self_dist(float a, float b, float s, float T) {
    float dt = max(b - a, 0);
    return s*s*dt/3;
}

[Differentiable]
DistortionState5 update_distortion_state(
    DistortionState5 old_state,
    float T,
    float density,
    float2 dist)
{
    // no_diff float start = tukey_power_ladder(500*dist.x, -0.1);
    // no_diff float end = tukey_power_ladder(500*dist.y, -0.1);
    // float start = (2*dist.x) / (2+dist.x);
    // float end = (2*dist.y) / (2+dist.y);
    float start = dist.x;
    float end = dist.y;
    float scaling = 1;
    float d = density;

    float m0 = moment_0_int(start, end, d, T); // just weight
    float m1 = moment_1_int(start, end, d, T);

    DistortionState5 new_state;
    new_state.x = old_state.x + 2.0f * m1 * old_state.z;
    new_state.y = old_state.y + 2.0f * m0 * old_state.w;
    new_state.z = old_state.z + m0;
    new_state.w = old_state.w + m1;
    // new_state.v = old_state.v + T * T * self_dist(start, end, d);
    new_state.v = old_state.v + self_dist(start, end, d, T);
    // printf("%f - %f, d: %f, v: %f T: %f, m0: %f, m1: %f\n", start, end, d, new_state.v, T, m0, m1);

    return new_state;
}

DistortionState5 undo_distortion_state(
    DistortionState5 new_state,
    float T,
    float density,
    float2 dist)
{
    float start = dist.x;
    float end = dist.y;
    float scaling = 1;
    float d = density;

    float m0 = moment_0_int(start, end, d, T);
    float m1 = moment_1_int(start, end, d, T);

    DistortionState5 old_state;
    old_state.z = new_state.z - m0;
    old_state.w = new_state.w - m1;
    old_state.x = new_state.x - 2.0f * m1 * old_state.z;
    old_state.y = new_state.y - 2.0f * m0 * old_state.w;
    old_state.v = new_state.v - self_dist(start, end, d, T);
    return old_state;
}

#define LOG_EPS 1e-6f

[Differentiable]
float4 update_pixel_state(float4 pixel_state_t_nm1, float4 gauss_rgba_t_n)
{
    float transmittance_t_nm1 = safe_exp(pixel_state_t_nm1.a);
    float3 color_t_n = pixel_state_t_nm1.rgb + gauss_rgba_t_n.rgb * transmittance_t_nm1;
    float log_transmittance_t_n = pixel_state_t_nm1.a - gauss_rgba_t_n.a;
    
    return float4(color_t_n, log_transmittance_t_n);
}

float4 undo_pixel_state(float4 pixel_state_t_n, float4 gauss_rgba_t_n)
{
    float log_transmittance_t_nm1 = pixel_state_t_n.a + gauss_rgba_t_n.a;

    float transmittance_t_nm1 = safe_exp(log_transmittance_t_nm1);
    
    float3 color_t_nm1 = pixel_state_t_n.rgb - gauss_rgba_t_n.rgb * transmittance_t_nm1;
    
    return float4(color_t_nm1, log_transmittance_t_nm1);
}

struct AlphaOut: IDifferentiable {
    float4 rgba;
    DistortionState5 distortion_state;
};

[Differentiable]
AlphaOut update_alpha_out(AlphaOut prev, CtrlPt ctrl) {
    float T = safe_exp(prev.rgba.a);
    return {
        update_pixel_state(prev.rgba, ctrl.rgba),
        update_distortion_state(prev.distortion_state, T, ctrl.density, ctrl.dist)
    };
}

AlphaOut undo_alpha_out(AlphaOut curr, CtrlPt ctrl) {
    float4 pix_state = undo_pixel_state(curr.rgba, ctrl.rgba);
    float T = safe_exp(pix_state.a);
    return {
        pix_state,
        undo_distortion_state(curr.distortion_state, T, ctrl.density, ctrl.dist)
    };
}

[BackwardDerivative(bwd_alpha_blend)] // Use a custom derivative so that we can hand-write the structure of the reverse loop
AlphaOut alpha_blend(
                   DiffTensorView rays,
                   TensorView<int32_t> indices,
                   DiffTensorView vertices,
                   DiffTensorView tet_density,
                   DiffTensorView final_pixel_state,
                   DiffTensorView final_distortion_state,
                   TensorView<int32_t> n_contributors,
                   TensorView<int32_t> tet_adjacency,
                   TensorView<int32_t> start_tet_ids,
                   TensorView<int32_t> ray_path_buffer,
                   no_diff float min_t,
                   int ray_index,
                int max_ray_steps)
{
    float4 curr_pixel_state = float4(0.f, 0.f, 0.f, 0.f);
    DistortionState5 curr_distortion_state = { 0.f, 0.f, 0.f, 0.f, 0.f };
    AlphaOut ao = { curr_pixel_state, curr_distortion_state };
    int step_count = 0;
    
    Ray ray = {
        {rays[uint2(ray_index, 0)], rays[uint2(ray_index, 1)], rays[uint2(ray_index, 2)]},
        {rays[uint2(ray_index, 3)], rays[uint2(ray_index, 4)], rays[uint2(ray_index, 5)]}
    };
    
    // Get the pixel's starting tet index
    int current_tet_idx = start_tet_ids[ray_index];
    float t_enter = min_t;
    float t_exit = min_t;
    int3 last_face;
    float3 enter_norm;
    float3 exit_norm;
    int out_face;

    for (int step = 0; step < max_ray_steps; ++step)
    {
        if (current_tet_idx < 0) // Hit boundary (adj index is -1)
            break;
            
        // --- Record the path for the backward pass ---
        // ray_path_buffer[uint2(ray_index, step)] = current_tet_idx;

        // Load this tet's data
        SplatTetrahedra_interp g = load_tet_alphablend_interp(
            current_tet_idx, vertices, indices, tet_density);


        ray_tetrahedron_intersect_exit(
            ray.o, ray.d,
            { g.tet.verts[0], g.tet.verts[1], g.tet.verts[2], g.tet.verts[3] },
            t_enter,
            t_exit,
            exit_norm,
            out_face);


        // Find intersection and exit face
        CtrlPt tetra_ctrl = integrate_tetra_segment(
            g,
            ray.o,
            ray.d,
            t_enter,
            t_exit);
        // Update pixel state (same as before)
        ao = update_alpha_out(ao, tetra_ctrl);
        // if (ray_index == 3) {
        //     printf("%f - %f, %i: (%f, %f, %f)\n", t_enter, t_enter+t_exit, current_tet_idx, g.base_color.x, g.base_color.y, g.base_color.z);
        // }

        // Update t_entry for the next step
        t_enter = t_exit + 1e-8;

        // Walk to the next tet using the adjacency map
        current_tet_idx = tet_adjacency[uint2(current_tet_idx, out_face)];
        // printf("%f - %f (%i: %i)\n", t_enter, t_exit, out_face, current_tet_idx);

        if (ao.rgba.a < -5.5) {
            break;
        }
    }
    
    n_contributors[uint2(ray_index, 0)] = step_count;
    return ao;
}

void bwd_alpha_blend(
                  DiffTensorView rays,
                  TensorView<int32_t> indices,
                  DiffTensorView vertices,
                  DiffTensorView tet_density,
                  DiffTensorView final_pixel_state,
                  DiffTensorView final_distortion_state,
                  TensorView<int32_t> n_contributors,
                  TensorView<int32_t> tet_adjacency,
                  TensorView<int32_t> start_tet_ids,
                  TensorView<int32_t> ray_path_buffer,
                  float min_t,
                  int ray_index,
                  int max_ray_steps,
                  AlphaOut d_ao) // Gradient input
{
    bool is_inside = ray_index < rays.size(0);

    float4 current_pixel_state;
    DistortionState5 curr_distortion_state;
    int32_t n_contrib_fwd = 0;

    if (is_inside) {
        // Load final states written by the forward pass
        curr_distortion_state = {
            final_distortion_state[uint2(ray_index, 0)],
            final_distortion_state[uint2(ray_index, 1)],
            final_distortion_state[uint2(ray_index, 2)],
            final_distortion_state[uint2(ray_index, 3)],
            final_distortion_state[uint2(ray_index, 4)]
        };
        current_pixel_state = float4(
            final_pixel_state[uint2(ray_index, 0)],
            final_pixel_state[uint2(ray_index, 1)],
            final_pixel_state[uint2(ray_index, 2)],
            final_pixel_state[uint2(ray_index, 3)]
        );
        n_contrib_fwd = n_contributors[uint2(ray_index, 0)];
    }
    
    AlphaOut ao = {current_pixel_state, curr_distortion_state};

    // Re-create the ray
    Ray ray = {
        {rays[ray_index, 0], rays[ray_index, 1], rays[ray_index, 2]},
        {rays[ray_index, 3], rays[ray_index, 4], rays[ray_index, 5]}
    };
    
    // Dummy var for bwd_diff call signature
    int exit_face; 

    // Init differential pairs for ray gradients
    DifferentialPair<float3> dp_ray_o = diffPair(ray.o);
    DifferentialPair<float3> dp_ray_d = diffPair(ray.d);

    if (is_inside) 
    {
        // Loop over the recorded path in REVERSE
        for (int i = n_contrib_fwd - 1; i >= 0; i--)
        {
            // Get tet index from the path buffer
            uint32_t g_idx = ray_path_buffer[uint2(ray_index, i)];

            // Re-load data for this tet
            SplatTetrahedra_interp g = load_tet_alphablend_interp(g_idx, vertices, indices, tet_density);

            // Re-evaluate to get the intermediate state
            CtrlPt tetra_ctrl;
            if (evaluate_tetra_interp(g, ray.o, ray.d, min_t, tetra_ctrl, exit_face) && tetra_ctrl.rgba.a > 0) 
            {
                // Undo state
                ao = undo_alpha_out(ao, tetra_ctrl);

                // Back-propagate
                DifferentialPair<SplatTetrahedra_interp> dp_g = diffPair(g);
                DifferentialPair<CtrlPt> dp_tetra_ctrl = diffPair(tetra_ctrl);
                DifferentialPair<AlphaOut> dp_ao = diffPair(ao);

                bwd_diff(update_alpha_out)(dp_ao, dp_tetra_ctrl, d_ao);
                d_ao = dp_ao.getDifferential();
                
                float grad_scale = 1.0f; // As in reference
                
                CtrlPt.Differential d_tetra_scale = {
                    grad_scale * dp_tetra_ctrl.d.rgba,
                    grad_scale * dp_tetra_ctrl.d.dist,
                    grad_scale * dp_tetra_ctrl.d.density,
                };

                // Propagate gradients to eval and load functions
                bwd_diff(evaluate_tetra_interp)(dp_g, dp_ray_o, dp_ray_d, min_t, d_tetra_scale);
                bwd_diff(load_tet_alphablend_interp)(g_idx, vertices, indices, tet_density, dp_g.d);
            }
        }

        // // Accumulate gradients for the ray origin and direction
        // rays.atomicAdd(ray_index, 0, dp_ray_o.d.x);
        // rays.atomicAdd(ray_index, 1, dp_ray_o.d.y);
        // rays.atomicAdd(ray_index, 2, dp_ray_o.d.z);
        // rays.atomicAdd(ray_index, 3, dp_ray_d.d.x);
        // rays.atomicAdd(ray_index, 4, dp_ray_d.d.y);
        // rays.atomicAdd(ray_index, 5, dp_ray_d.d.z);
    }
}

// ========================================================================
// Kernel Entry Point (NEW)
// ========================================================================

[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void trace_rays_kernel(
    DiffTensorView rays,                 // Input: [N_rays, 6] (o.x, o.y, o.z, d.x, d.y, d.z)
    TensorView<int32_t> indices,         // Input: [N_tets, 4]
    DiffTensorView vertices,             // Input: [N_verts, 3]
    DiffTensorView tet_density,          // Input: [N_tets, C] (or however features are stored)
    DiffTensorView output_img,           // Output: [N_rays, 4] (r, g, b, log_transmittance)
    DiffTensorView distortion_img,       // Output: [N_rays, 5]
    TensorView<int32_t> n_contributors,  // Output: [N_rays, 1]
    TensorView<int32_t> tet_adjacency,   // Input: [N_tets, 4]
    TensorView<int32_t> start_tet_ids,   // Input: [N_rays]
    TensorView<int32_t> ray_path_buffer, // Output: [N_rays, max_ray_steps]
    no_diff float min_t,
    int max_ray_steps)
{
    // 1D kernel launch over all rays
    uint32_t ray_index = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;

    // Bounds check
    if (ray_index >= rays.size(0)) {
        return;
    }

    // Call the forward pass
    AlphaOut pixel_states = alpha_blend(
        rays,
        indices,
        vertices,
        tet_density,
        output_img,       // Pass-through for bwd pass
        distortion_img, // Pass-through for bwd pass
        n_contributors,
        tet_adjacency,
        start_tet_ids,
        ray_path_buffer,
        min_t,
        int(ray_index),
        max_ray_steps);
        
    // Store the final results
    output_img.storeOnce(uint2(ray_index, 0), pixel_states.rgba.r);
    output_img.storeOnce(uint2(ray_index, 1), pixel_states.rgba.g);
    output_img.storeOnce(uint2(ray_index, 2), pixel_states.rgba.b);
    output_img.storeOnce(uint2(ray_index, 3), pixel_states.rgba.a);

    distortion_img.storeOnce(uint2(ray_index, 0), pixel_states.distortion_state.x);
    distortion_img.storeOnce(uint2(ray_index, 1), pixel_states.distortion_state.y);
    distortion_img.storeOnce(uint2(ray_index, 2), pixel_states.distortion_state.z);
    distortion_img.storeOnce(uint2(ray_index, 3), pixel_states.distortion_state.w);
    distortion_img.storeOnce(uint2(ray_index, 4), pixel_states.distortion_state.v);
}