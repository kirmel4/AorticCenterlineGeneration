import trimesh

def get_marching_cubes_surface(masks):
    return trimesh.voxel.ops.matrix_to_marching_cubes(masks)
