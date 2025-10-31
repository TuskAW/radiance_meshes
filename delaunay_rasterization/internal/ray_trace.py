import torch
import delaunay_rasterization.internal.slang.slang_modules as slang_modules

class TetrahedralRayTrace(torch.autograd.Function):
    """
    torch.autograd.Function for differentiable tetrahedral ray tracing.
    
    This function takes a set of rays, a tetrahedral mesh, and its
    adjacency information, and traces the rays through the mesh using
    a custom Slang CUDA kernel.
    """
    
    @staticmethod
    def forward(ctx, 
                rays: torch.Tensor,
                indices: torch.Tensor,
                vertices: torch.Tensor,
                tet_density: torch.Tensor,
                tet_adjacency: torch.Tensor,
                start_tet_ids: torch.Tensor,
                min_t: float,
                max_ray_steps: int
               ) -> tuple[torch.Tensor, torch.Tensor]:
        
        device = rays.device
        n_rays = rays.shape[0]

        # Allocate output tensors (1D per-ray outputs)
        distortion_img = torch.zeros((n_rays, 5), dtype=vertices.dtype, device=device)
        output_img = torch.zeros((n_rays, 4), dtype=vertices.dtype, device=device)
        n_contributors = torch.zeros((n_rays, 1), dtype=torch.int32, device=device)
        
        # Allocate the buffer to store the ray path for the backward pass
        ray_path_buffer = torch.zeros((n_rays, max_ray_steps), dtype=torch.int32, device=device)

        trace_shader = slang_modules.trace_rays_kernel

        # --- Call the new kernel function ---
        trace_kernel_with_args = trace_shader.trace_rays_kernel(
            rays=rays,
            indices=indices,
            vertices=vertices,
            tet_density=tet_density,
            output_img=output_img,
            distortion_img=distortion_img,
            n_contributors=n_contributors,
            tet_adjacency=tet_adjacency,
            start_tet_ids=start_tet_ids,
            ray_path_buffer=ray_path_buffer,
            min_t=min_t,
            max_ray_steps=max_ray_steps
        )

        # --- 1D Launch Configuration ---
        block_size = 256
        grid_size = (n_rays + block_size - 1) // block_size
        
        trace_kernel_with_args.launchRaw(
            blockSize=(block_size, 1, 1),
            gridSize=(grid_size, 1, 1)
        )

        # --- Save tensors for backward pass ---
        tensors = [
            rays, indices, vertices, tet_density,
            tet_adjacency, start_tet_ids,
            output_img, distortion_img, n_contributors,
            ray_path_buffer
        ]
        ctx.save_for_backward(*tensors)
        # print(n_contributors, start_tet_ids, output_img, ray_path_buffer)
        
        # --- Save non-tensors for backward pass ---
        ctx.min_t = min_t
        ctx.max_ray_steps = max_ray_steps

        # Return the two primary outputs
        return output_img, distortion_img

    @staticmethod
    def backward(ctx, 
                 grad_output_img: torch.Tensor, 
                 grad_distortion_img: torch.Tensor
                ) -> tuple:
        
        # --- Load saved tensors from forward pass ---
        (rays, indices, vertices, tet_density,
         tet_adjacency, start_tet_ids,
         output_img, distortion_img, n_contributors,
         ray_path_buffer) = ctx.saved_tensors
        
        # --- Load non-tensors ---
        min_t = ctx.min_t
        max_ray_steps = ctx.max_ray_steps

        # --- Allocate gradient tensors ---
        rays_grad = torch.zeros_like(rays)
        vertices_grad = torch.zeros_like(vertices)
        tet_density_grad = torch.zeros_like(tet_density)

        # --- Get the Slang Kernel ---
        trace_shader = slang_modules.trace_rays_kernel
        
        # --- Call the backward kernel ---
        kernel_with_args = trace_shader.trace_rays_kernel.bwd(
            rays=(rays, rays_grad),
            indices=indices,
            vertices=(vertices, vertices_grad),
            tet_density=(tet_density, tet_density_grad),
            output_img=(output_img, grad_output_img.contiguous()),
            distortion_img=(distortion_img, grad_distortion_img.contiguous()),
            n_contributors=n_contributors,
            tet_adjacency=tet_adjacency,
            start_tet_ids=start_tet_ids,
            ray_path_buffer=ray_path_buffer,
            min_t=min_t,
            max_ray_steps=max_ray_steps
        )
        
        # --- Launch with the same 1D config as forward ---
        n_rays = rays.shape[0]
        block_size = 256
        grid_size = (n_rays + block_size - 1) // block_size

        kernel_with_args.launchRaw(
            blockSize=(block_size, 1, 1),
            gridSize=(grid_size, 1, 1)
        )
        
        # --- Return gradients in the order of forward inputs ---
        # (rays, indices, vertices, tet_density, tet_adjacency, 
        #  start_tet_ids, min_t, max_ray_steps)
        return (rays_grad, 
                None, 
                vertices_grad, 
                tet_density_grad, 
                None, 
                None, 
                None, 
                None)