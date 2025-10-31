// lookup_kernel.slang
// This module contains the main kernel for finding which tetrahedron contains a given point.
// It uses the pre-computed grid acceleration structure.

// Struct to hold the four vertices of a tetrahedron
struct Tetrahedra {
    float3 verts[4];
};

/**
 * Loads the four vertex positions of a single tetrahedron from global memory.
 */
Tetrahedra load_tetrahedra(TensorView<float> vertices, TensorView<int32_t> indices, uint32_t tet_id) {
    Tetrahedra tet;
    uint4 v_indices = {
        (uint)indices[uint2(tet_id, 0)],
        (uint)indices[uint2(tet_id, 1)],
        (uint)indices[uint2(tet_id, 2)],
        (uint)indices[uint2(tet_id, 3)]
    };
    tet.verts[0] = { vertices[uint2(v_indices.x, 0)], vertices[uint2(v_indices.x, 1)], vertices[uint2(v_indices.x, 2)] };
    tet.verts[1] = { vertices[uint2(v_indices.y, 0)], vertices[uint2(v_indices.y, 1)], vertices[uint2(v_indices.y, 2)] };
    tet.verts[2] = { vertices[uint2(v_indices.z, 0)], vertices[uint2(v_indices.z, 1)], vertices[uint2(v_indices.z, 2)] };
    tet.verts[3] = { vertices[uint2(v_indices.w, 0)], vertices[uint2(v_indices.w, 1)], vertices[uint2(v_indices.w, 2)] };
    return tet;
}

/**
 * Robust orientation test using double precision.
 * Determines on which side of the plane defined by triangle (a, b, c) the point p lies.
 * The sign of the result indicates the side.
 */
double orientation(double3 p, double3 a, double3 b, double3 c) {
    return dot(p - a, cross(b - a, c - a));
}

/**
 * Checks if a point is inside a tetrahedron.
 * It performs four orientation tests. The point is inside if it is on the same side
 * of all four faces.
 */
bool point_in_tetrahedron(float3 p_f, Tetrahedra tet) {
    double3 p = { (double)p_f.x, (double)p_f.y, (double)p_f.z };
    double3 v0 = { (double)tet.verts[0].x, (double)tet.verts[0].y, (double)tet.verts[0].z };
    double3 v1 = { (double)tet.verts[1].x, (double)tet.verts[1].y, (double)tet.verts[1].z };
    double3 v2 = { (double)tet.verts[2].x, (double)tet.verts[2].y, (double)tet.verts[2].z };
    double3 v3 = { (double)tet.verts[3].x, (double)tet.verts[3].y, (double)tet.verts[3].z };

    // The signs of these determinants must all be the same for the point to be inside.
    double o1 = orientation(p, v0, v1, v2);
    double o2 = orientation(p, v1, v3, v2);
    double o3 = orientation(p, v0, v2, v3);
    double o4 = orientation(p, v0, v3, v1);

    return (o1 >= 0 && o2 >= 0 && o3 >= 0 && o4 >= 0) || (o1 <= 0 && o2 <= 0 && o3 <= 0 && o4 <= 0);
}

[AutoPyBindCUDA]
[CUDAKernel]
void lookup_points(
    TensorView<float> query_points,         // Input: (N_queries, 3) MORTON-SORTED query points
    TensorView<int32_t> indices,            // Input: (N_tets, 4) tetrahedron vertex indices
    TensorView<float> vertices,             // Input: (N_verts, 3) vertex positions
    TensorView<int32_t> sorted_keys,         // Input: The sorted cell IDs from the splatting phase
    TensorView<int32_t> sorted_tet_idx,     // Input: The sorted tetrahedron IDs, matching the keys
    TensorView<int32_t> tile_ranges,        // Input: (grid_size, 2) start/end index for each cell
    float3 world_min,                       // Input: The minimum corner of the grid
    float3 cell_size,                       // Input: The size of a single grid cell
    uint3 grid_dim,                         // Input: The dimensions of the grid
    TensorView<int32_t> out_tet_ids)          // Output: (N_queries) the ID of the containing tet for each point
{
    uint32_t g_idx = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (g_idx >= query_points.size(0)) {
        return;
    }

    // Load the query point for this thread
    float3 p = { query_points[uint2(g_idx, 0)], query_points[uint2(g_idx, 1)], query_points[uint2(g_idx, 2)] };

    // Determine which grid cell the point falls into
    float3 grid_f = (p - world_min) / cell_size;
    int3 grid_coords = { (int)grid_f.x, (int)grid_f.y, (int)grid_f.z };

    // Check if the point is outside the grid bounds
    if (grid_coords.x < 0 || grid_coords.x >= grid_dim.x ||
        grid_coords.y < 0 || grid_coords.y >= grid_dim.y ||
        grid_coords.z < 0 || grid_coords.z >= grid_dim.z)
    {
        out_tet_ids[g_idx] = -1; // Outside the entire volume
        return;
    }

    uint32_t cell_id = grid_coords.z * grid_dim.x * grid_dim.y + grid_coords.y * grid_dim.x + grid_coords.x;

    // Get the range of tetrahedra that overlap this cell
    int32_t start_idx = tile_ranges[uint2(cell_id, 0)];
    int32_t end_idx = tile_ranges[uint2(cell_id, 1)];

    // If the cell is empty, no tetrahedra overlap it
    if (start_idx >= end_idx) {
        out_tet_ids[g_idx] = -1;
        return;
    }

    // Iterate through the candidate tetrahedra and perform the accurate containment test
    for (int32_t i = start_idx; i < end_idx; ++i) {
        int32_t tet_id = sorted_tet_idx[i];
        Tetrahedra tet = load_tetrahedra(vertices, indices, tet_id);
        if (point_in_tetrahedron(p, tet)) {
            out_tet_ids[g_idx] = tet_id;
            return; // Found it!
        }
    }

    // If the loop finishes, the point is in a hole in the mesh
    out_tet_ids[g_idx] = -1;
}
