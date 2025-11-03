#define epsilon 1e-20
import spline_machine;
import safe_math;

struct GTransform {
    float3 scales;
    float3 mean;
    float4 quat;
    float height;
};

[Differentiable]
float3x3 quat2mat(float4 quat)
{
    let r = quat[0];
    let x = quat[1];
    let y = quat[2];
    let z = quat[3];
    float3x3 R = {
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - r * z),
        2.0 * (x * z + r * y),

        2.0 * (x * y + r * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - r * x),

        2.0 * (x * z - r * y),
        2.0 * (y * z + r * x),
        1.0 - 2.0 * (x * x + y * y)
    };
    return R;
}


[Differentiable]
float3x3 diag(float3 vec)
{
    return {
      vec.x, 0, 0,
      0, vec.y, 0,
      0, 0, vec.z,
    };
}

/*
static const float3 BASE_VERTICES[5] = {
  { 0.f, 0.f, 1.f},
  { 0.f, 1.f, 0.f},
  { 0.86602540378f, -0.5f, 0.f},
  {-0.86602540378f, -0.5f, 0.f},
  { 0.f, 0.f,-1.f},
};

static const uint3 INDICES[6] = {
    {0,2,1}, {0,3,2}, {0,1,3},
    {1,2,4}, {2,3,4}, {3,1,4},
};
*/

/*
static const float3 BASE_VERTICES[4] = {
  { 0.f, 0.f, 0.5},
  { 0.f, 1.f, -0.5},
  { 0.86602540378f, -0.5f, -0.5},
  {-0.86602540378f, -0.5f, -0.5},
};

static const uint3 INDICES[4] = {
    {0,2,1}, {0,3,2}, {0,1,3}, {1,2,3}
};
//*/

//*
static const float3 BASE_VERTICES[4] = {
  { 0.f, 0.f, 1.f},
  { 0.f, 1.f, 0.f},
  { 1.f, 0.f, 0.f},
  { 0.f, 0.f, 0.f},
};

static const uint3 INDICES[4] = {
  {0, 2, 1}, {0, 3, 2}, {0, 1, 3}, {1, 2, 3}
};
//*/


bool ray_intersects_triangle(in float3 ray_origin, 
                             in float3 ray_vector, 
                             in float3 tri_a,
                             in float3 tri_b,
                             in float3 tri_c,
                             out float t)
{
    t = -1;
    float3 edge1 = tri_b - tri_a;
    float3 edge2 = tri_c - tri_a;
    float3 ray_cross_e2 = cross(ray_vector, edge2);
    float det = dot(edge1, ray_cross_e2);

    if (det > -epsilon && det < epsilon)
        return false;    // This ray is parallel to this triangle.

    float inv_det = 1.0 / det;
    float3 s = ray_origin - tri_a;
    float u = inv_det * dot(s, ray_cross_e2);

    if (u < 0 || u > 1)
        return false;

    float3 s_cross_e1 = cross(s, edge1);
    float v = inv_det * dot(ray_vector, s_cross_e1);

    if (v < 0 || u + v > 1)
        return false;

    // At this stage we can compute t to find out where the intersection point is on the line.
    t = inv_det * dot(edge2, s_cross_e1);

    if (t > epsilon) // ray intersection
    {
        return true;
    }
    else // This means that there is a line intersection but not a ray intersection.
        return false;
}

[Differentiable]
float ray_intersect_plane(float3 ray_origin, 
                          float3 ray_vector, 
                          float3 normal,
                          float3 point)
{
    let denom = dot(normal, ray_vector);
    let numer = dot(normal, ray_origin) - dot(normal, point);

    let dist = safe_div(numer, denom);
    return abs(dist);
    //return (abs(denom) > 1e-20) ? abs(dist) : length(point - ray_origin);
}

[Differentiable]
ControlPoint intersect(
    float3 rayo, float3 rayd, float3 scales, float3 mean, float4 quat, float3 color, float density, uint face_id
) {
    uint3 inds = INDICES[face_id];
    let R = quat2mat(safe_div(quat, length(quat)));
    let Rt = transpose(R);
    float3 clip_scale = max(scales, 1e-8);

    float3 tri_a = no_diff BASE_VERTICES[inds.x];
    float3 tri_b = no_diff BASE_VERTICES[inds.y];
    float3 tri_c = no_diff BASE_VERTICES[inds.z];
    float3 edge1 = tri_b - tri_a;
    float3 edge2 = tri_c - tri_a;
    float3 normal = l2_normalize( mul(cross(edge1, edge2) / clip_scale, Rt) );
    
    //printf("normal: %f, %f, %f\n", normal.x, normal.y, normal.z);

    float3 point_on_tri = mul(BASE_VERTICES[inds.x] * scales, Rt) + mean;
    float t = ray_intersect_plane(rayo, rayd, normal, point_on_tri);

    let dotprod = dot(normal, rayd);
    float dirac_multi = (dotprod < 0) ? density : -density;

    /*
    let Trayo = safe_div(mul(rayo - mean, R), clip_scale);
    let in_bounds = max(Trayo.x, 0.f) + max(Trayo.y, 0.f) + max(Trayo.z, 0.f);
    dirac_multi = ((in_bounds <= 1 && Trayo.x >= 0 && Trayo.y >= 0 && Trayo.z >= 0)) ? dirac_multi : 0.f;
    //*/

    //printf("Length: %f\n", length(Trayo));
    //float dirac_multi = density * calc_dirac(Trayd, face_id);
    ControlPoint out = {
        t+1e-5, dirac_multi, dirac_multi*color.x, dirac_multi*color.y, dirac_multi*color.z 
    };
    return out;
}
