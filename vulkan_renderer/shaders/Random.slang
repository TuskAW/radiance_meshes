#pragma once

#include "Random.h"
import MathUtils;

namespace RoseEngine {

float  UnormToFloat(uint  x) { return asfloat(0x3f800000 | ((x) >> 9)) - 1; }
float2 UnormToFloat(uint2 x) { return asfloat(0x3f800000 | ((x) >> 9)) - 1; }
float3 UnormToFloat(uint3 x) { return asfloat(0x3f800000 | ((x) >> 9)) - 1; }
float4 UnormToFloat(uint4 x) { return asfloat(0x3f800000 | ((x) >> 9)) - 1; }

struct RandomSampler {
	uint4 mState;

	__init(const uint seed, const uint2 index) {
        mState = uint4(index, seed, 0);
        mState = pcg4d(mState);
	}

	[mutating]
	void Skip(const uint n = 1) {
		mState.w += n;
	}

	[mutating]
	uint4 Next() {
		mState.w++;
        return pcg4d(mState);
    }

    [mutating]
    float4 NextFloat() {
        return UnormToFloat(Next());
    }
};

float3 SampleUniformSphere(const float2 uv) {
    const float z = uv.y * 2 - 1;
    const float r = sqrt(max(1 - z * z, 0));
    const float phi = 2 * M_PI * uv.x;
    return float3(r * cos(phi), r * sin(phi), z);
}

float2 SampleTexel(Texture2D<float4> image, float2 rnd, out float pdf, const uint maxIterations = 10) {
    uint2 imageExtent;
    uint levelCount;
    image.GetDimensions(0, imageExtent.x, imageExtent.y, levelCount);

    pdf = 1;
    int2 coord = 0;
    uint2 lastExtent = 1;
    for (uint i = 1; i < min(maxIterations + 1, levelCount - 1); i++) {
        const uint level = levelCount - 1 - i;
        uint tmp;
        uint2 extent;
        image.GetDimensions(level, extent.x, extent.y, tmp);
        const float inv_h = 1 / (float)extent.y;

        coord *= int2(extent / lastExtent);

        float4 p = 0;
        if (extent.x - coord.x > 1) {
            const float sy = sin(M_PI * (coord.y + 0.5f) * inv_h);
            p[0] = luminance(image.Load(int3(coord + int2(0, 0), (int)level)).rgb) * sy;
            p[1] = luminance(image.Load(int3(coord + int2(1, 0), (int)level)).rgb) * sy;
        }
        if (extent.y - coord.y > 1) {
            const float sy = sin(M_PI * (coord.y + 1.5f) * inv_h);
            p[2] = luminance(image.Load(int3(coord + int2(0, 1), (int)level)).rgb) * sy;
            p[3] = luminance(image.Load(int3(coord + int2(1, 1), (int)level)).rgb) * sy;
        }
        const float sum = dot(p, 1);
        if (sum < 1e-6) continue;
        p /= sum;

        for (int j = 0; j < 4; j++) {
            if (j == 3 || rnd.x < p[j]) {
                coord += int2(j % 2, j / 2);
                pdf *= p[j];
                rnd.x /= p[j];
                break;
            }
            rnd.x -= p[j];
        }

        lastExtent = extent;
    }

    pdf *= lastExtent.x * lastExtent.y;

    return (float2(coord) + rnd) / float2(lastExtent);
}
float SampleTexelPdf(Texture2D<float4> image, const float2 uv, const uint maxIterations = 10) {
    uint2 imageExtent;
    uint levelCount;
    image.GetDimensions(0, imageExtent.x, imageExtent.y, levelCount);

    float pdf = 1;
    uint2 lastExtent = 1;
    for (uint i = 1; i < min(maxIterations + 1, levelCount - 1); i++) {
        const uint level = levelCount - 1 - i;
        uint tmp;
        uint2 size;
        image.GetDimensions(level, size.x, size.y, tmp);
        const float inv_h = 1 / (float)size.y;

        const int2 uvi = int2(float2(size) * uv);
        const int2 coord = (uvi / 2) * 2;
        const uint2 o = min(uvi - coord, 1);

        float4 p = 0;
        if (size.x - coord.x > 1) {
            const float sy = sin(M_PI * (coord.y + 0.5f) * inv_h);
            p[0] = luminance(image.Load(int3(coord + int2(0, 0), (int)level)).rgb) * sy;
            p[1] = luminance(image.Load(int3(coord + int2(1, 0), (int)level)).rgb) * sy;
        }
        if (size.y - coord.y > 1) {
            const float sy = sin(M_PI * (coord.y + 1.5f) * inv_h);
            p[2] = luminance(image.Load(int3(coord + int2(0, 1), (int)level)).rgb) * sy;
            p[3] = luminance(image.Load(int3(coord + int2(1, 1), (int)level)).rgb) * sy;
        }
        const float sum = dot(p, 1);
        if (sum < 1e-6) continue;
        p /= sum;

        pdf *= p[o.y * 2 + o.x];

        lastExtent = size;
    }

    pdf *= lastExtent.x * lastExtent.y;

    return pdf;
}

}