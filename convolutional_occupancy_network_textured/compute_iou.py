from pathlib import Path
import trimesh
import numpy as np
from intersections import slice_mesh_plane
from scipy.spatial import cKDTree as KDTree
import struct

def crop_mesh(base_mesh_path):
    current = trimesh.load_mesh(base_mesh_path)
    box = trimesh.creation.box(extents=[96, 96, 100])
    box.apply_translation(np.array([48, 48, 50]))
    mesh_chunk = slice_mesh_plane(mesh=current, plane_normal=-box.facets_normal, plane_origin=box.facets_origin)
    mesh_chunk.export(base_mesh_path, "obj")


def sparse_to_dense_np(locs, values, dimx, dimy, dimz, default_val):
    nf_values = 1 if len(values.shape) == 1 else values.shape[1]
    dense = np.zeros([dimz, dimy, dimx, nf_values], dtype=values.dtype)
    dense.fill(default_val)
    dense[locs[:, 0], locs[:, 1], locs[:, 2], :] = values
    if nf_values > 1:
        return dense.reshape([dimz, dimy, dimx, nf_values])
    return dense.reshape([dimz, dimy, dimx])


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
    sdf = sparse_to_dense_np(locs, sdf[:,np.newaxis], dimx, dimy, dimz, -float('inf'))
    return sdf, known


def prune_mesh(mesh_path):
    current = trimesh.load_mesh(mesh_path)
    if not type(current) == trimesh.Trimesh:
        print("MeshTypeError: ", type(current), current_path)
        return False
    name = Path(mesh_path).name.split('.')[0]
    sdf_path = f"/cluster_HDD/sorona/adai/data/matterport/completion_blocks_2cm_test/individual_96-96-160_s96/{name.split('_')[0]}_{name.split('_')[1]}__cmp__{name.split('_')[2]}.sdf"
    sdf, known = load_sdf(sdf_path)
    known[(sdf < -2.5) & (sdf > -20.0)] = 2
    known[known < 2] = 0
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
    current.process(validate=True)
    # try:
        # current.process(validate=True)
        # current.export("test.obj", "obj")
    # except:
        # return current
    return current




def compute_mesh_iou_voxels_mod(mesh_input, mesh_pred, mesh_target):
    mesh_in = trimesh.load_mesh(mesh_input)
    mesh_pred = trimesh.load_mesh(mesh_pred)
    mesh_target = trimesh.load_mesh(mesh_target)
    res = 1.1875
    v_in = mesh_in.voxelized(pitch=res)
    v_pred = mesh_pred.voxelized(pitch=res)
    v_target = mesh_target.voxelized(pitch=res)
    
    v_in.fill()
    v_pred.fill()
    v_target.fill()

    v_pred_filled = set(tuple(x) for x in v_pred.points)
    v_target_filled = set(tuple(x) for x in v_target.points)
    v_in_filled = set(tuple(x) for x in v_in.points)
    v_pred_filled = v_pred_filled - (v_pred_filled - v_pred_filled.intersection(v_target_filled))
    # v_in_filled = set(tuple(x) for x in v_in.points).union(v_pred_filled - v_target_filled) 
    iou0 = len(v_in_filled.intersection(v_target_filled)) / len(v_in_filled.union(v_target_filled))
    iou1 = len(v_pred_filled.intersection(v_target_filled)) / len(v_pred_filled.union(v_target_filled))
    return iou0, iou1

def compute_mesh_iou_voxels(mesh_pred, mesh_target):
    mesh_pred = prune_mesh(mesh_pred)
    mesh_target = trimesh.load_mesh(mesh_target)
    res = 1.1875
    v_pred = mesh_pred.voxelized(pitch=res)
    v_target = mesh_target.voxelized(pitch=res)
    
    v_pred.fill()
    v_target.fill()

    v_pred_filled = set(tuple(x) for x in v_pred.points)
    v_target_filled = set(tuple(x) for x in v_target.points)
    iou = len(v_pred_filled.intersection(v_target_filled)) / len(v_pred_filled.union(v_target_filled))
    return iou


def compute_mesh_iou_voxels_removed(mesh_pred, mesh_target):
    mesh_pred = trimesh.load_mesh(mesh_pred)
    mesh_target = trimesh.load_mesh(mesh_target)
    res = 1.1875
    v_pred = mesh_pred.voxelized(pitch=res)
    v_target = mesh_target.voxelized(pitch=res)
    
    v_pred.fill()
    v_target.fill()

    v_pred_filled = set(tuple(x) for x in v_pred.points)
    v_target_filled = set(tuple(x) for x in v_target.points)
    v_pred_filled = v_pred_filled - (v_pred_filled - v_pred_filled.intersection(v_target_filled))
    iou = len(v_pred_filled.intersection(v_target_filled)) / len(v_pred_filled.union(v_target_filled))
    return iou



def compute_trimesh_chamfer(mesh_pred, mesh_target, offset=0, scale=1, num_mesh_samples=30000):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.
    gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see
               compute_metrics.ply for more documentation)
    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)
    """
    mesh_pred = trimesh.load_mesh(mesh_pred)
    mesh_target = trimesh.load_mesh(mesh_target)

    gt_points_np = trimesh.sample.sample_surface(mesh_target, num_mesh_samples)[0]
    gen_points_sampled = trimesh.sample.sample_surface(mesh_pred, num_mesh_samples)[0]

    gen_points_sampled = gen_points_sampled / scale - offset

    # one direction
    gen_points_kd_tree = KDTree(gen_points_sampled)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_np)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points_np)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer + gen_to_gt_chamfer


def compute_metrics_chamfer_ours_mp(path_to_visualization):
    path_to_visualization = Path(path_to_visualization)
    with open("CD_MP_OURS.csv", "w") as fptr:
        for x in [y.name.split("input.obj")[0] for y in path_to_visualization.iterdir() if y.name.endswith('input.obj')]:
            try:
                mesh_a = path_to_visualization / (x + "input.obj")
                mesh_b = path_to_visualization / (x + "target.obj")
                cd0 = compute_trimesh_chamfer(mesh_a, mesh_b)
                mesh_a = path_to_visualization / (x + "pred.obj")
                mesh_b = path_to_visualization / (x + "target.obj")
                cd1 = compute_trimesh_chamfer(mesh_a, mesh_b)
                print(x, cd0, cd1)
                fptr.write(f"{x},{cd0},{cd1}\n")
            except Exception as e:
                print("ERROR: ", x, e)


def compute_metrics_iou_ours_mp_mod(path_to_visualization):
    path_to_visualization = Path(path_to_visualization)
    with open("IoU_input_target.csv", "w") as fptr:
        for x in [y.name.split("input.obj")[0] for y in path_to_visualization.iterdir() if y.name.endswith('input.obj')]:
            try:
                mesh_a = path_to_visualization / (x + "input.obj")
                mesh_b = path_to_visualization / (x + "pred.obj")
                mesh_c = path_to_visualization / (x + "target.obj")
                iou0, iou1 = compute_mesh_iou_voxels(mesh_a, mesh_b, mesh_c)
                print(x, iou0, iou1)
                fptr.write(f"{x},{iou0},{iou1}\n")
            except Exception as e:
                print("ERROR: ", x, e)


def compute_metrics_iou_ours_mp(path_to_visualization_0, path_to_visualization_1):
    path_to_visualization = Path(path_to_visualization_0)
    with open("IoU_ours_vs_target.csv", "w") as fptr:
        for x in [y.name for y in path_to_visualization.iterdir() if y.name.endswith('.obj')]:
            try:
                mesh_a = path_to_visualization_0 / x
                mesh_b = path_to_visualization_1 / x
                iou = compute_mesh_iou_voxels(mesh_a, mesh_b)
                print(x, iou)
                fptr.write(f"{x},{iou}\n")
            except Exception as e:
                print("ERROR: ", x, e)


def crop_all_input_meshes(path_to_visualization):
    path_to_visualization = Path(path_to_visualization)
    for x in [y for y in path_to_visualization.iterdir() if y.name.endswith('input.obj')]:
        crop_mesh(x)


if __name__ == "__main__":
    import sys
    path_to_vis_0 = Path(sys.argv[1])
    path_to_vis_1 = Path(sys.argv[2])
    # crop_all_input_meshes(path_to_vis)
    # compute_metrics_chamfer_ours_mp(path_to_vis)
    compute_metrics_iou_ours_mp(path_to_vis_0, path_to_vis_1)
    # prune_mesh("2t7WUuJeko7_room5_1.obj")
    # print(compute_mesh_iou_voxels("2t7WUuJeko7_room0_0_target.obj", "2t7WUuJeko7_room0_0_ours.obj"))
