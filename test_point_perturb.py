import torch
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.spatial import Delaunay
from utils import topo_utils
from utils.contraction import contraction_jacobian, contract_points
from utils.train_util import *

# Total optimization iterations (frames) and number of points.
M = 500  
N = 100  
device = 'cpu'

def calculate_circumcenters_torch(vertices: torch.Tensor):
    """
    Compute the circumcenter and circumradius of a tetrahedron using PyTorch.
    
    Args:
        vertices: Tensor of shape (..., 4, 3) containing the vertices of the tetrahedron(s).
    
    Returns:
        circumcenter: Tensor of shape (..., 3) containing the circumcenter coordinates.
        radius: Tensor of shape (...) containing the circumradius.
    """
    a = vertices[..., 1, :] - vertices[..., 0, :]  
    b = vertices[..., 2, :] - vertices[..., 0, :]  
    c = vertices[..., 3, :] - vertices[..., 0, :]  

    aa = torch.sum(a * a, dim=-1, keepdim=True)  
    bb = torch.sum(b * b, dim=-1, keepdim=True)  
    cc = torch.sum(c * c, dim=-1, keepdim=True)  

    cross_bc = torch.cross(b, c, dim=-1)
    cross_ca = torch.cross(c, a, dim=-1)
    cross_ab = torch.cross(a, b, dim=-1)

    denominator = 2.0 * torch.sum(a * cross_bc, dim=-1, keepdim=True)
    mask = torch.abs(denominator) < 1e-6

    relative_circumcenter = (
        aa * cross_bc +
        bb * cross_ca +
        cc * cross_ab
    ) / torch.where(mask, torch.ones_like(denominator), denominator)

    radius = torch.norm(a - relative_circumcenter, dim=-1)
    return vertices[..., 0, :] + relative_circumcenter, radius

def project_points_to_tetrahedra(points, tets):
    """
    Projects each point in `points` (shape (N, 3)) onto the corresponding tetrahedron in `tets` (shape (N, 4, 3))
    by clamping negative barycentrics to zero and renormalizing them so that they sum to 1.
    """
    N = points.shape[0]
    v0 = tets[:, 0, :]
    T = tets[:, 1:, :] - v0.unsqueeze(1)
    T = T.permute(0,2,1)

    p_minus_v0 = points - v0
    x = torch.linalg.solve(T, p_minus_v0.unsqueeze(2)).squeeze(2)

    w0 = 1 - x.sum(dim=1, keepdim=True)
    bary = torch.cat([w0, x], dim=1)
    bary = bary.clip(min=0)

    norm = (bary.sum(dim=1, keepdim=True)).clip(min=1e-8)
    mask = (norm > 1).reshape(-1)
    bary[mask] = bary[mask] / norm[mask]

    p_proj = (T * bary[:, 1:].unsqueeze(1)).sum(dim=2) + v0
    return p_proj

# --- Initialization of vertex positions ---
S = 5
# Start with centers randomly distributed
centers = 2*S * torch.randn((N, 3), device=device)

# For this version we define an initial offset (e.g., using a circular offset) that is then optimized.
# Compute circle offsets as before for an initial guess:
radii = S * (0.1 + 0.4 * torch.rand((N,), device=device))
normals = torch.randn((N, 3), device=device)
normals = normals / normals.norm(dim=1, keepdim=True)
arbitrary = torch.tensor([1, 0, 0], device=device, dtype=torch.float32).expand(N, 3).contiguous()
dot = (normals * arbitrary).sum(dim=1)
mask = dot.abs() > 0.99
if mask.any():
    arbitrary[mask] = torch.tensor([0, 1, 0], device=device, dtype=torch.float32).reshape(1, 3)
e1 = torch.cross(normals, arbitrary)
e1 = e1 / e1.norm(dim=1, keepdim=True)
e2 = torch.cross(normals, e1)
theta0 = 0
offset = radii.unsqueeze(1) * (math.cos(theta0) * e1 + math.sin(theta0) * e2)

# Create a trainable parameter for vertices
# (For instance, starting from centers + offset.)
vertices = torch.nn.Parameter(centers + offset)

# Define a placeholder regularizer.
def regularizer(v):
    # Example: an L2 regularization on the vertex positions.
    # Replace or modify this function as needed.
    return torch.sum(v ** 2)

# Choose a regularization strength.
lambda_reg = 0.1

# Set up the Adam optimizer.
optimizer = torch.optim.Adam([vertices], lr=1e-2)

frames = []

# Optimization loop.
for i in range(M):
    optimizer.zero_grad()
    
    # Update connectivity using Delaunay on the current vertices.
    # (Detaching so that Delaunay triangulation is not part of the gradient computation.)
    vertices_np = vertices.detach().cpu().numpy()
    delaunay = Delaunay(vertices_np)
    indices = torch.tensor(delaunay.simplices, device=vertices.device).int()
    
    # Gather tetrahedra (each row in indices gives 4 vertex indices).
    tets = vertices[indices]
    
    # Compute circumcenters (and optionally, you can project them)
    circumcenter, radius = calculate_circumcenters_torch(tets.double())
    clipped_circumcenter = circumcenter  # or use project_points_to_tetrahedra if desired
    
    # Compute the contracted circumcenters.
    cc = contract_points(clipped_circumcenter)
    
    # Define a loss.
    # In this example, we try to bring the vertex positions closer to the contracted circumcenters.
    # You may wish to design a different loss that better suits your needs.
    cc_sense, sensitivity = topo_utils.compute_vertex_sensitivity(indices.cuda(), vertices.cuda(), circumcenter.cuda())
    mask = torch.ones((indices.shape[0]), dtype=bool)
    reg_perturb = compute_perturbation(indices, vertices, cc, 0.1*torch.ones((indices.shape[0])),
                                   mask, cc_sense.cpu(),
                                   1e+1, k=100, t=(1-0.005))
    loss = reg_perturb
    
    loss.backward()
    optimizer.step()
    
    # Save current contracted circumcenters for visualization.
    frames.append(cc.detach().cpu().numpy())

# --- Plotting and Animation ---
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scat = ax.scatter([], [], [], s=5)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)

frames_np = frames

def update(frame_num):
    data = frames_np[frame_num]
    scat._offsets3d = (data[:, 0], data[:, 1], data[:, 2])
    ax.set_title(f"Frame {frame_num+1}/{M}")
    return scat,

ani = FuncAnimation(fig, update, frames=M, interval=1, blit=False)
ani.save('optimized_vertices.mp4', writer='ffmpeg', fps=24)
