
// Struct to hold the four vertices of a tetrahedron
struct Tetrahedra {
    float3 verts[4];
};

/**
 * Loads the four vertex positions of a single tetrahedron from global memory.
 *
 * @param vertices A tensor view of all vertex positions in the mesh.
 * @param indices A uint4 containing the indices of the four vertices for this tetrahedron.
 * @return A Tetrahedra struct populated with the vertex coordinates.
 */
Tetrahedra load_tetrahedra(TensorView<float> vertices, uint4 indices) {
    Tetrahedra tet;
    tet.verts[0] = { vertices[uint2(indices.x, 0)], vertices[uint2(indices.x, 1)], vertices[uint2(indices.x, 2)] };
    tet.verts[1] = { vertices[uint2(indices.y, 0)], vertices[uint2(indices.y, 1)], vertices[uint2(indices.y, 2)] };
    tet.verts[2] = { vertices[uint2(indices.z, 0)], vertices[uint2(indices.z, 1)], vertices[uint2(indices.z, 2)] };
    tet.verts[3] = { vertices[uint2(indices.w, 0)], vertices[uint2(indices.w, 1)], vertices[uint2(indices.w, 2)] };
    return tet;
}

/**
 * Computes the axis-aligned bounding box (AABB) of a tetrahedron.
 *
 * @param tet The tetrahedron to process.
 * @param[out] min_v The minimum corner of the bounding box.
 * @param[out] max_v The maximum corner of the bounding box.
 */
void find_extent(Tetrahedra tet, out float3 min_v, out float3 max_v) {
    min_v = float3(1e20, 1e20, 1e20);
    max_v = float3(-1e20, -1e20, -1e20);
    for (int i = 0; i < 4; ++i) {
        min_v.x = min(min_v.x, tet.verts[i].x);
        min_v.y = min(min_v.y, tet.verts[i].y);
        min_v.z = min(min_v.z, tet.verts[i].z);
        max_v.x = max(max_v.x, tet.verts[i].x);
        max_v.y = max(max_v.y, tet.verts[i].y);
        max_v.z = max(max_v.z, tet.verts[i].z);
    }
}

/**
 * Main kernel for splatting tetrahedra.
 * For each tetrahedron, it computes its AABB and determines the range of grid cells it overlaps.
 */
[AutoPyBindCUDA]
[CUDAKernel]
void splat_tetrahedra(
    TensorView<int32_t> indices,          // Input: (N_tets, 4) indices of tetrahedron vertices
    TensorView<float> vertices,           // Input: (N_verts, 3) vertex positions
    float3 world_min,                     // Input: The minimum corner of the entire grid in world space
    float3 cell_size,                     // Input: The size of a single grid cell
    uint3 grid_dim,                       // Input: The dimensions of the grid (width, height, depth)
    TensorView<int32_t> out_tiles_touched,  // Output: (N_tets) number of cells each tet overlaps
    TensorView<int32_t> out_cell_rect)      // Output: (N_tets, 6) the min/max cell indices [min_x, min_y, min_z, max_x, max_y, max_z]
{
    uint32_t g_idx = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (g_idx >= indices.size(0)) {
        return;
    }

    // Load tetrahedron vertices
    uint4 tet_indices = {
        (uint)indices[uint2(g_idx, 0)],
        (uint)indices[uint2(g_idx, 1)],
        (uint)indices[uint2(g_idx, 2)],
        (uint)indices[uint2(g_idx, 3)]
    };
    Tetrahedra tet = load_tetrahedra(vertices, tet_indices);

    // Compute world-space AABB
    float3 min_v, max_v;
    find_extent(tet, min_v, max_v);

    // Convert AABB to grid cell coordinates
    float3 min_grid_f = (min_v - world_min) / cell_size;
    float3 max_grid_f = (max_v - world_min) / cell_size;
    
    // Clamp to grid boundaries and convert to integer indices
    int32_t min_cell_x = (int32_t)floor(max(min_grid_f.x, 0.0f));
    int32_t min_cell_y = (int32_t)floor(max(min_grid_f.y, 0.0f));
    int32_t min_cell_z = (int32_t)floor(max(min_grid_f.z, 0.0f));

    int32_t max_cell_x = (int32_t)ceil(min(max_grid_f.x, (float)grid_dim.x));
    int32_t max_cell_y = (int32_t)ceil(min(max_grid_f.y, (float)grid_dim.y));
    int32_t max_cell_z = (int32_t)ceil(min(max_grid_f.z, (float)grid_dim.z));
    
    // Ensure min is not greater than max
    max_cell_x = max(min_cell_x, max_cell_x);
    max_cell_y = max(min_cell_y, max_cell_y);
    max_cell_z = max(min_cell_z, max_cell_z);

    // Calculate number of touched cells
    int32_t n_tiles = (max_cell_x - min_cell_x) * (max_cell_y - min_cell_y) * (max_cell_z - min_cell_z);
    
    if (n_tiles == 0) {
        out_tiles_touched[g_idx] = 0;
        return;
    }
    
    // Write outputs
    out_tiles_touched[g_idx] = n_tiles;
    out_cell_rect[uint2(g_idx, 0)] = min_cell_x;
    out_cell_rect[uint2(g_idx, 1)] = min_cell_y;
    out_cell_rect[uint2(g_idx, 2)] = min_cell_z;
    out_cell_rect[uint2(g_idx, 3)] = max_cell_x;
    out_cell_rect[uint2(g_idx, 4)] = max_cell_y;
    out_cell_rect[uint2(g_idx, 5)] = max_cell_z;
}
