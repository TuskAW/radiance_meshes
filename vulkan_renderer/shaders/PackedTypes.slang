#pragma once

#include "DxgiFormatConvert.h"

[Differentiable]
float2 ToOctahedral(const float3 v) {
    // Project the sphere onto the octahedron, and then onto the xy plane
    const float2 p = v.xy * (1 / (abs(v.x) + abs(v.y) + abs(v.z)));
    // Reflect the folds of the lower hemisphere over the diagonals
    return (v.z <= 0) ? ((1 - abs(p.yx)) * lerp(-1, 1, float2(p >= 0))) : p;
}
[Differentiable]
float3 FromOctahedral(const float2 p) {
    float3 v = float3(p, 1 - dot(1, abs(p)));
    if (v.z < 0) v.xy = (1 - abs(v.yx)) * lerp(-1, 1, float2(v.xy >= 0));
    return normalize(v);
}