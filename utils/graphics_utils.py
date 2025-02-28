import math
import torch
import numpy as np

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    z_sign = 1.0

    # P = torch.zeros(4, 4)

    # P[0, 0] = 2.0 * znear / (right - left)
    # P[1, 1] = 2.0 * znear / (top - bottom)
    # P[0, 2] = (right + left) / (right - left)
    # P[1, 2] = (top + bottom) / (top - bottom)
    # P[3, 2] = z_sign
    # P[2, 2] = z_sign * zfar / (zfar - znear)
    # P[2, 3] = -(zfar * znear) / (zfar - znear)
    P = torch.tensor([
       [2.0 * znear / (right - left),     0.0,                          (right + left) / (right - left), 0.0 ],
       [0.0,                              2.0 * znear / (top - bottom), (top + bottom) / (top - bottom), 0.0 ],
       [0.0,                              0.0,                          z_sign * zfar / (zfar - znear),  -(zfar * znear) / (zfar - znear) ],
       [0.0,                              0.0,                          z_sign,                          0.0 ]
    ])
    return P

def l2_normalize_th(x, eps:float=torch.finfo(torch.float32).eps, dim:int=-1):
    """Normalize x to unit length along last axis."""
    return x / torch.sqrt(
        torch.clip(torch.sum(x**2, dim=dim, keepdim=True), eps, None)
    )

def tetra_volumes6(vertices: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Computes the signed volume * 6 for each tetrahedron.
    For a tetrahedron with vertices A, B, C, D, the quantity is:
         det( [B-A, C-A, D-A] )
    
    Args:
        vertices: Tensor of shape (N, 3) with vertex coordinates.
        indices: Tensor of shape (M, 4) with indices into the vertices tensor.
        
    Returns:
        Tensor of shape (M,) containing the signed 6x volumes.
    """
    # Extract vertices for each tetrahedron
    A = vertices[indices[:, 0]]
    B = vertices[indices[:, 1]]
    C = vertices[indices[:, 2]]
    D = vertices[indices[:, 3]]
    
    # Compute edge vectors relative to A
    AB = B - A
    AC = C - A
    AD = D - A
    
    # Stack edge vectors to form a matrix for each tetrahedron.
    # Each matrix is of shape (3, 3) where the columns are AB, AC, and AD.
    M = torch.stack([AB, AC, AD], dim=2)  # shape: (M, 3, 3)
    
    # Compute the determinant of each matrix. This is 6 times the volume.
    vol6 = torch.linalg.det(M)
    return vol6

# Optionally, if you want the absolute volume for each tetrahedron:
def tetra_volume(vertices: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Computes the absolute volume of each tetrahedron.
    
    Args:
        vertices: Tensor of shape (N, 3) with vertex coordinates.
        indices: Tensor of shape (M, 4) with indices into the vertices tensor.
        
    Returns:
        Tensor of shape (M,) containing the volumes.
    """
    vol6 = tetra_volumes6(vertices, indices)
    volume = torch.abs(vol6) / 6.0
    return volume