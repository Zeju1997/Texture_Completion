import sys
from pathlib import Path
import trimesh
from tqdm import tqdm
import struct
import numpy as np
from intersections import slice_mesh_plane


def sparse_to_dense_np(locs, values, dimx, dimy, dimz, default_val):
    nf_values = 1 if len(values.shape) == 1 else values.shape[1]
    dense = np.zeros([dimz, dimy, dimx, nf_values], dtype=values.dtype)
    dense.fill(default_val)
    dense[locs[:, 0], locs[:, 1], locs[:, 2], :] = values
    if nf_values > 1:
        return dense.reshape([dimz, dimy, dimx, nf_values])
    return dense.reshape([dimz, dimy, dimx])

# Steps
# Chop base mesh head
# remove all vertices where SDF < -1
def crop_mesh(base_mesh_path, output_path):
    current = trimesh.load_mesh(base_mesh_path)
    box = trimesh.creation.box(extents=[96, 96, 100])
    box.apply_translation(np.array([48, 48, 50]))
    mesh_chunk = slice_mesh_plane(mesh=current, plane_normal=-box.facets_normal, plane_origin=box.facets_origin)
    mesh_chunk.export(output_path, "obj")


def load_sdf(file_path):
    fin = open(file_path, 'rb')
    dimx = struct.unpack('Q', fin.read(8))[0]
    dimy = struct.unpack('Q', fin.read(8))[0]
    dimz = struct.unpack('Q', fin.read(8))[0]
    voxelsize = struct.unpack('f', fin.read(4))[0]
    world2grid = struct.unpack('f'*4*4, fin.read(4*4*4))
    world2grid = np.asarray(world2grid, dtype=np.float32).reshape([4, 4])
    num = struct.unpack('Q', fin.read(8))[0]
    locs = struct.unpack('I'*num*3, fin.read(num*3*4))
    locs = np.asarray(locs, dtype=np.int32).reshape([num, 3])
    locs = np.flip(locs, 1).copy() # convert to zyx ordering
    sdf = struct.unpack('f'*num, fin.read(num*4))
    sdf = np.asarray(sdf, dtype=np.float32)
    sdf /= voxelsize
    num_known = struct.unpack('Q', fin.read(8))[0]
    assert num_known == dimx * dimy * dimz
    known = struct.unpack('B'*num_known, fin.read(num_known))
    known = np.asarray(known, dtype=np.uint8).reshape([dimz, dimy, dimx])
    mask = np.logical_and(sdf >= -1, sdf <= 1)
    known[locs[:,0][mask], locs[:,1][mask], locs[:,2][mask]] = 1
    mask = sdf > 1
    known[locs[:,0][mask], locs[:,1][mask], locs[:,2][mask]] = 0
    # sdf = sparse_to_dense_np(locs, sdf[:,np.newaxis], dimx, dimy, dimz, -float('inf'))
    return known


def prune_mesh(current_path, base_sdf_path, destination_path):
    current = trimesh.load_mesh(current_path)
    if not type(current) == trimesh.Trimesh:
        print("MeshTypeError: ", type(current), current_path)
        return False
    known = load_sdf(base_sdf_path)
    vertices_to_remove = []
    faces_to_keep = [True] * current.faces.shape[0]
    vertices_to_keep = [True] * current.vertices.shape[0]
    for vid in range(current.vertices.shape[0]):
        closest_voxel = [int(current.vertices[vid][i] - 0.01) for i in range(3)]
        if known[closest_voxel[2], closest_voxel[1], closest_voxel[0]] >= 2:
            vertices_to_remove.append(vid)
            vertices_to_keep[vid] = False
    for fid in range(current.faces.shape[0]):
        v_ids = current.faces[fid]
        if (not vertices_to_keep[v_ids[0]]) and (not vertices_to_keep[v_ids[1]]) and (not vertices_to_keep[v_ids[2]]):
            faces_to_keep[fid] = False
    current.update_faces(np.array(faces_to_keep))
    try:
        current.process(validate=True)
        current.export(destination_path, "obj")
    except:
        return False
    return True


if __name__ == "__main__":
    base_mesh_root = sys.argv[1]
    base_sdf_root = sys.argv[2]
    current_mesh_root = sys.argv[3]
    destination = Path(sys.argv[4])
    current_meshes = list(Path(current_mesh_root).iterdir())
    base_meshes = [Path(base_mesh_root) / x.name.split(".")[0] / "model_c.obj" for x in current_meshes]
    base_cropped_meshes = [Path(base_mesh_root) / x.name.split(".")[0] / "model_c_cropped.obj" for x in current_meshes]
    base_sdf_paths = [Path(base_sdf_root) / f'{x.name.split("_")[0]}_{x.name.split("_")[1]}__cmp__{x.name.split("_")[2].split(".")[0]}.sdf' for x in current_meshes]
    destination.mkdir(exist_ok=True)
    for i in tqdm(list(range(len(current_meshes)))):
        #if current_meshes[i].name == "2t7WUuJeko7_room0_0.obj":
        obj_name = current_meshes[i].name.split(".")[0] + ".obj"
        if prune_mesh(current_meshes[i], base_sdf_paths[i], destination / obj_name):
            crop_mesh(destination / obj_name, destination / obj_name)
