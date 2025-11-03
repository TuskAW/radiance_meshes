#define RT_EPS 0
//#define tri_per_g 8
#define tri_per_g 4
import spline_machine;
import optix;
import optix_intrinsics;
import tri_intersect;
import safe_math;
import sh;

// LaunchParams
RWStructuredBuffer<uint>        image;
RWStructuredBuffer<float4>      fimage;
RWStructuredBuffer<uint>        iters;
RWStructuredBuffer<uint>        last_face;
RWStructuredBuffer<uint>        touch_count;
RWStructuredBuffer<float4>      last_dirac;
RWStructuredBuffer<SplineState> last_state;
RWStructuredBuffer<int> tri_collection;
StructuredBuffer<float3>      ray_origins;
StructuredBuffer<float3>      ray_directions;

RWStructuredBuffer<float3> means;
RWStructuredBuffer<float3> scales;
RWStructuredBuffer<float4> quats;
RWStructuredBuffer<float> heights;
RWStructuredBuffer<float> features;

size_t sh_degree;
size_t max_iters;
float clip_dist;
float tmax;
float4 initial_drgb;
float max_prim_size;
RaytracingAccelerationStructure traversable;

ControlPoint get_ctrl_pt(uint ind, float3 rayo, float3 rayd) {
    uint prim_ind = (uint)floor(ind / tri_per_g);
    uint face_id = mod(ind, tri_per_g);

    let mean = means[prim_ind];
    let scale = scales[prim_ind];
    let quat = quats[prim_ind];
    let height = heights[prim_ind];

    Features feat;
    SHFeatures sh_feats = {prim_ind, sh_degree, features};
    feat.f0 = get_sh(sh_feats, 0);
    float3 rayd = {0, 0, 1};
    float3 color = eval_sh_col0(rayd, feat);
    ControlPoint ctrl_pt = intersect(rayo, rayd, scale, mean, quat, color, height, face_id);
    return ctrl_pt;
}

[shader("closesthit")]
void ch(
    inout uint2 garbage : SV_RayPayload,
    in float2 barycentric,
)
{
    uint ind = PrimitiveIndex();
    uint prim_ind = (uint)floor(ind / tri_per_g);
    uint face_id = mod(ind, tri_per_g);
    float3 rayd = l2_normalize(WorldRayDirection());
    float3 rayo = WorldRayOrigin();
    //*

    let mean = means[prim_ind];
    let scale = scales[prim_ind];
    let quat = quats[prim_ind];
    let height = heights[prim_ind];

    Features feat;
    SHFeatures sh_feats = {prim_ind, sh_degree, features};
    feat.f0 = get_sh(sh_feats, 0);
    float3 color = eval_sh_col0(rayd, feat);


    ControlPoint ctrl_pt = intersect(rayo, rayd, scale, mean, quat, color, height, face_id);
    //*/
    float t = RayTCurrent();
    //float t = ctrl_pt.t;
    set_payload1(asuint(ctrl_pt.dirac.x));
    set_payload2(asuint(ctrl_pt.dirac.y));
    set_payload3(asuint(ctrl_pt.dirac.z));
    set_payload4(asuint(ctrl_pt.dirac.w));


    set_payload0(asuint(t));
    set_payload5(ind);

}

[shader("anyhit")]
void ah()
{
    uint old_tri_ind = PrimitiveIndex();
    uint tri_ind = get_payload5();

    if (old_tri_ind == tri_ind) {
        IgnoreHit();
    }
}

[shader("miss")]
void ms()
{
    set_payload0(asuint(1e10));
    set_payload1(asuint(0.0f));
    set_payload2(asuint(0.0f));
    set_payload3(asuint(0.0f));
    set_payload4(asuint(0.0f));
    set_payload5(255u);
}


[shader("raygeneration")]
void rg_float()
{
    let FAST_MODE = false;

    const uint3 idx = DispatchRaysIndex();
    const uint3 dim = DispatchRaysDimensions();
    const float offset = 0; // 
    float3 direction = l2_normalize(ray_directions[idx.x]);
    float3 origin = ray_origins[idx.x] + (clip_dist-offset) * direction;
    const float texit = tmax - offset;
    float tmin = 0.0;
    float tmax = texit;
    SplineState state = make_empty_state();
    state.t = -offset;
    state.drgb = initial_drgb;

    let start_id = idx.x * max_iters;
    
    ControlPoint ctrl_pt = {};
    uint last_tri = -1;
    float prev_t = state.t;
    uint next_tri;
    float next_t = 1e9;

    int iter = 0;
    while (state.logT < 4.6 && iter < max_iters)
    {
        uint p0 = asuint(ctrl_pt.t);
        uint p1 = asuint(ctrl_pt.dirac.x);
        uint p2 = asuint(ctrl_pt.dirac.y);
        uint p3 = asuint(ctrl_pt.dirac.z);
        uint p4 = asuint(ctrl_pt.dirac.w);

        float _temp0 = asfloat((uint) last_tri);
        float _temp1, _temp2;
        uint tri = asuint(_temp0);
        uint p6 = asuint(_temp1);
        uint p7 = asuint(_temp2);

        let start_t = (FAST_MODE) ? abs(state.t+offset) : abs(prev_t + offset);

        optixTrace8(
                traversable,
                origin,
                direction,
                start_t,
                tmax,
                p0, p1, p2, p3, p4, tri, p6, p7);

        ctrl_pt.t = asfloat(p0);
        if (ctrl_pt.t > 1e9) {
            break;
        }
        if (last_tri == tri) {
            prev_t = nextafter(prev_t, 1e20);
            state.t = nextafter(state.t, 1e20);
            continue;
        }
        last_tri = tri;
        ctrl_pt.dirac.x = asfloat(p1);
        ctrl_pt.dirac.y = asfloat(p2);
        ctrl_pt.dirac.z = asfloat(p3);
        ctrl_pt.dirac.w = asfloat(p4);

        prev_t = state.t;
        next_t = asfloat(p6);
        next_tri = p7;

        if (iter < max_iters) {
            state = update(state, ctrl_pt, tmin, tmax, max_prim_size);
            touch_count[tri / tri_per_g]++;
            tri_collection[idx.x + iter * dim.x] = tri;
        }
        iter++;
    }
    let output = extract_color(state);
    fimage[idx.x] = {output.C.x, output.C.y, output.C.z, output.opacity};
    let dual_state = to_dual(state, ctrl_pt);
    last_state[idx.x] = dual_state;
    last_dirac[idx.x] = ctrl_pt.dirac;
    last_face[idx.x] = last_tri;
    iters[idx.x] = iter;
}
