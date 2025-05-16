#pragma once

// https://gist.github.com/mattatz/40a91588d5fb38240403f198a938a593

import MathUtils;

typedef float4 quat;

// Quaternion multiplication
// http://mathworld.wolfram.com/Quaternion.html
quat qmul(quat q1, quat q2) {
    return quat(
        q2.xyz * q1.w + q1.xyz * q2.w + cross(q1.xyz, q2.xyz),
        q1.w * q2.w - dot(q1.xyz, q2.xyz));
}

quat q_conj(quat q) {
    return quat(-q.xyz, q.w);
}

// https://jp.mathworks.com/help/aeroblks/quaternioninverse.html
quat q_inverse(quat q) {
    return q_conj(q) / dot(q, q);
}

// Vector rotation with a quaternion
// http://mathworld.wolfram.com/Quaternion.html
float3 q_rotate_vector(quat q, float3 v) {
    return qmul(q, qmul(quat(v, 0), q_conj(q))).xyz;
}

// A given angle of rotation about a given axis
quat q_angle_axis(float angle, float3 axis) {
    float sn = sin(angle * 0.5);
    float cs = cos(angle * 0.5);
    return quat(axis * sn, cs);
}

quat q_slerp(quat a, quat b, float t) {
    // if either input is zero, return the other.
    bool az = all(a == 0);
    bool bz = all(b == 0);
    if (az && bz) return quat(0, 0, 0, 1);
    else if (az) return b;
    else if (bz) return a;

    float cosHalfAngle = a.w * b.w + dot(a.xyz, b.xyz);

    if (cosHalfAngle >= 1.0 || cosHalfAngle <= -1.0) {
        return a;
    } else if (cosHalfAngle < 0) {
        b.xyz = -b.xyz;
        b.w = -b.w;
        cosHalfAngle = -cosHalfAngle;
    }

    float blendA;
    float blendB;
    if (cosHalfAngle < 0.999) {
        // do proper slerp for big angles
        float halfAngle = acos(cosHalfAngle);
        float sinHalfAngle = sin(halfAngle);
        float oneOverSinHalfAngle = 1.0 / sinHalfAngle;
        blendA = sin(halfAngle * (1.0 - t)) * oneOverSinHalfAngle;
        blendB = sin(halfAngle * t) * oneOverSinHalfAngle;
    } else {
        // do lerp if angle is really small.
        blendA = 1.0 - t;
        blendB = t;
    }

    quat result = quat(blendA * a.xyz + blendB * b.xyz, blendA * a.w + blendB * b.w);
    if (length(result) > 0)
        return normalize(result);

    return quat(0,0,0,1);
}

float4x4 q_to_matrix(quat quat) {
    float4x4 m = float4x4(float4(0, 0, 0, 0), float4(0, 0, 0, 0), float4(0, 0, 0, 0), float4(0, 0, 0, 0));

    float x = quat.x, y = quat.y, z = quat.z, w = quat.w;
    float x2 = x + x, y2 = y + y, z2 = z + z;
    float xx = x * x2, xy = x * y2, xz = x * z2;
    float yy = y * y2, yz = y * z2, zz = z * z2;
    float wx = w * x2, wy = w * y2, wz = w * z2;

    m[0][0] = 1.0 - (yy + zz);
    m[0][1] = xy - wz;
    m[0][2] = xz + wy;

    m[1][0] = xy + wz;
    m[1][1] = 1.0 - (xx + zz);
    m[1][2] = yz - wx;

    m[2][0] = xz - wy;
    m[2][1] = yz + wx;
    m[2][2] = 1.0 - (xx + yy);

    m[3][3] = 1.0;

    return m;
}