import numpy as np
from collections import defaultdict, deque

def extract_meshes(verts, tets):
    # ---------- 1.  collect candidate faces ------------------------------------
    # Each tet contributes four faces, always in counter-clockwise order if you
    # keep the vertex you *omit* in first position.
    faces = np.stack([
        tets[:, [1, 2, 3]],
        tets[:, [0, 3, 2]],
        tets[:, [0, 1, 3]],
        tets[:, [0, 2, 1]],
    ], axis=1).reshape(-1, 3)                 # (4*M, 3)

    # ---------- 2.  locate boundary faces --------------------------------------
    faces_sorted = np.sort(faces, axis=1)     # canonical key, orientation lost
    faces_key = np.ascontiguousarray(faces_sorted).view(
        np.dtype((np.void, faces_sorted.dtype.itemsize * 3))
    )
    keys, inv, counts = np.unique(faces_key, return_inverse=True, return_counts=True)
    boundary_mask     = counts[inv] == 1
    boundary_faces    = faces[boundary_mask]      # keeps the original winding
    # Build vertex→face lookup for a fast flood-fill.
    v2f = defaultdict(list)
    for fi, tri in enumerate(boundary_faces):
        for v in tri:
            v2f[v].append(fi)

    # Flood-fill over triangles that share at least one vertex
    components, seen = [], set()
    for seed in range(len(boundary_faces)):
        if seed in seen:
            continue
        q, comp = deque([seed]), []
        while q:
            f = q.popleft()
            if f in seen:
                continue
            seen.add(f)
            comp.append(f)
            for v in boundary_faces[f]:
                q.extend(v2f[v])   # enqueue all faces that touch this vertex
        components.append(np.array(comp, dtype=np.int64))

    meshes = []
    for comp in components:
        tris = boundary_faces[comp]          # (F,3)
        unique_vs, new_idx = np.unique(tris, return_inverse=True)
        tris_reindexed = new_idx.reshape(-1, 3)
        meshes.append(
            dict(
                vertex = dict(
                    x = verts[unique_vs, 0].astype(np.float32),
                    y = verts[unique_vs, 1].astype(np.float32),
                    z = verts[unique_vs, 2].astype(np.float32)), 
                face    = dict(
                    vertex_indices=tris_reindexed.astype(np.int32)))
        )
    return meshes

