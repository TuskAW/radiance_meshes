// key_generator.slang
// This module generates the actual (cell_id, tet_id) pairs and computes the
// final compacted ranges after the pairs have been sorted.

/**
 * Generates key-value pairs (cell_id, tet_id) for each cell a tetrahedron overlaps.
 * This is the "splatting" part, creating the unsorted list.
 */
[AutoPyBindCUDA]
[CUDAKernel]
void generate_keys(
    TensorView<int32_t> cell_rect,          // Input: (N_tets, 6) min/max cell indices from the splatting kernel
    TensorView<int32_t> index_buffer_offset,// Input: (N_tets) Prefix sum of tiles touched
    uint3 grid_dim,                         // Input: Grid dimensions
    TensorView<int32_t> out_unsorted_keys, // Output: Unsorted cell IDs
    TensorView<int32_t> out_unsorted_tet_idx) // Output: Unsorted tetrahedron IDs
{
    int32_t g_idx = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;

    if (g_idx >= cell_rect.size(0)) {
        return;
    }

    // Determine the starting offset in the output buffers for this tetrahedron
    int32_t offset = (g_idx == 0) ? 0 : index_buffer_offset[g_idx - 1];

    // Get the cell range for this tetrahedron
    int32_t min_x = cell_rect[uint2(g_idx, 0)];
    int32_t min_y = cell_rect[uint2(g_idx, 1)];
    int32_t min_z = cell_rect[uint2(g_idx, 2)];
    int32_t max_x = cell_rect[uint2(g_idx, 3)];
    int32_t max_y = cell_rect[uint2(g_idx, 4)];
    int32_t max_z = cell_rect[uint2(g_idx, 5)];

    // Iterate over the 3D block of cells and write a key-value pair for each one
    for (int32_t z = min_z; z < max_z; z++) {
        for (int32_t y = min_y; y < max_y; y++) {
            for (int32_t x = min_x; x < max_x; x++) {
                uint32_t cell_id = z * grid_dim.x * grid_dim.y + y * grid_dim.x + x;
                out_unsorted_keys[offset] = cell_id;
                out_unsorted_tet_idx[offset] = g_idx;
                offset++;
            }
        }
    }
}


/**
 * Computes the start and end indices for each tile's list of tetrahedra.
 * This kernel runs *after* the keys and tet_indices have been sorted by key.
 */
[AutoPyBindCUDA]
[CUDAKernel]
void compute_tile_ranges(
    TensorView<int32_t> sorted_keys, // Input: The sorted cell IDs
    TensorView<int32_t> out_tile_ranges) // Output: (grid_size, 2) storing [start, end) for each cell
{
    int32_t g_idx = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;

    if (g_idx >= sorted_keys.size(0)) {
        return;
    }

    // Each thread checks if it's at the boundary between two different cell IDs
    uint32_t current_key = sorted_keys[g_idx];

    // Handle the very first element
    if (g_idx == 0) {
        out_tile_ranges[uint2(current_key, 0)] = 0;
    } else {
        uint32_t prev_key = sorted_keys[g_idx - 1];
        if (current_key != prev_key) {
            // This thread is the start of a new cell's range
            out_tile_ranges[uint2(prev_key, 1)] = g_idx; // The previous range ends here
            out_tile_ranges[uint2(current_key, 0)] = g_idx; // The new range starts here
        }
    }

    // Handle the very last element
    if (g_idx == sorted_keys.size(0) - 1) {
        out_tile_ranges[uint2(current_key, 1)] = g_idx + 1;
    }
}
