import torch
import numpy as np
from scipy.spatial import  Delaunay
from utils.topo_utils import calculate_circumcenters_torch
from dtlookup import lookup_inds

class BallSelection:
    def __init__(self, center, radius):
        self.center = center.reshape(1, 3)
        self.radius = radius.reshape(1)

    def select_mask(self, vertices):
        return torch.linalg.norm(vertices - center, dim=-1) < self.radius

class BoxSelection:
    def __init__(self, center, radius):
        self.center = center.reshape(1, 3)
        self.radius = radius.reshape(1, 3)

    def select_mask(self, vertices):
        return torch.linalg.norm((vertices - center) / self.radius, dim=-1, ord=torch.inf) < 1

class Selection:
    def __init__(self):
        self.invert = False
        self.selections = []

    def expand(self, primitive):
        self.selections.append(primitive)

    def select_mask(self, vertices):
        mask = torch.zeros((vertices.shape[0]), dtype=bool)
        for primitive in self.selections:
            mask |= primitive.select_mask(vertices)
        return mask & self.invert

    def select(self, vertices):
        return vertices[self.select_mask(vertices)]

    def __invert__(self):
        self.invert = ~self.invert


def combine_attr(density, rgb, gradient, sh):
    return torch.cat([density, rgb, gradient, sh], dim=1)

def split_attr(attr):
    density = attr[..., :1]
    rgb = attr[..., 1:4]
    gradient = attr[..., 4:7]
    sh = attr[..., 7:]
    return density, rgb, gradient, sh

def lookup_attr(circumcenters, target):
    new_inds = lookup_inds(target.indices, target.vertices, circumcenters)
    attrs = combine_attr(target.density, target.rgb, target.gradient, target.sh)
    return attrs[target]

def raw_insert(source, target, source_selection, target_point):
    source_vertices = source_selection.select(source.vertices) + target_point
    target_vertices = (~source_selection).select(target.vertices)
    vertices = torch.cat([
        source_vertices, target_vertices
    ], dim=0)
    new_indices_np = Delaunay(vertices.detach().cpu().numpy()).simplices.astype(np.int32)
    new_indices = torch.as_tensor(new_indices_np)

    new_tets = vertices[new_indices]

    n_source_vertices = source_vertices.shape[0]
    source_tets_mask = new_tets.min(dim=1) < n_source_vertices
    target_tets_mask = new_tets.max(dim=1) >= n_source_vertices

    source_cc, source_r = calculate_circumcenters_torch(new_tets[source_tets_mask].double())
    target_cc, target_r = calculate_circumcenters_torch(new_tets[target_tets_mask].double())

    # this is my first idea for insertion
    # associate tetrahedra
    source_attr = lookup_attr(source_cc, source)
    target_attr = lookup_attr(target_cc, target)
    # the second idea is to use a contrained delaunay triangulation

    full_attr = torch.empty((new_tets.shape[0], source_attr.shape[1]))
    full_attr[source_tets_mask] = source_attr
    full_attr[target_tets_mask] = target_attr

    # source_t_attr = lookup_attr(source, target)
    # full_attr[source_tets_mask, 7:] = source_t_attr[..., 7:]

    inserted = Frozen(
        vertices,
        torch.empty((0, 3)),
        new_indices,
        *split_attr(full_attr),
        center=target.center,
        scene_scaling=target.scene_scaling,
        mask=torch.ones(()),
        full_indices=new_indices,
        max_sh_deg=target.max_sh_deg
    )
    return inserted, dict(
        n_source_verts = n_source_vertices,
        source_tets_mask = source_tets_mask,
        target_tets_mask = target_tets_mask,
    )
