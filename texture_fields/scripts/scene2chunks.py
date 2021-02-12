import struct
import numpy as np
import os
import trimesh
from tqdm import tqdm
from intersections import slice_mesh_plane

sdf_base = None
sdf_scene_base = None
scene_base = None
dest = None


def read_w2g(sdf_path):
    fin = open(sdf_path, 'rb')
    fin.read(28)
    world2grid = np.array(struct.unpack('f' * 4 * 4, fin.read(4 * 4 * 4))).reshape((4, 4)).astype(np.float32)
    world2grid[:3, :3] = np.eye(3)
    #grid2world = np.linalg.inv(world2grid)
    fin.close()
    #return grid2world[:3, 3]
    return world2grid[:3, 3]


def scene_to_chunk(part):
    all_chunks = sorted([x for x in os.listdir(sdf_base) if "__cmp__" in x])
    part_len = len(os.listdir(scene_base)) // 12 + 1
    for scene in tqdm(os.listdir(scene_base)[part * part_len : (part + 1) * part_len]):
        scene_path = os.path.join(scene_base, scene)
        mesh = trimesh.load(scene_path, process=True)
        # mesh.export(os.path.join(dest, scene + ".obj"), "obj")
        filtered_chunks = [c for c in all_chunks if scene.split("__")[0] == c.split("__")[0]]
        if len(filtered_chunks) > 250:
            continue
        for chunk in tqdm(filtered_chunks):
            if scene.split("__")[0] == chunk.split("__")[0]:
                chunk_b_path = os.path.join(sdf_scene_base, chunk.split("__")[0] + "__0__.sdf")
                chunk_c_path = os.path.join(sdf_base, chunk)
                if os.path.exists(chunk_b_path):
                    try:
                       translation_c = read_w2g(chunk_c_path)
                       translation_b = read_w2g(chunk_b_path)
                       # translation_b = np.zeros_like(translation_c)
                       mesh.apply_translation(translation_c-translation_b)
                       # mesh.export(os.path.join(os.path.join(dest, chunk+".obj")))
                       box = trimesh.creation.box(extents=[96, 96, 160])
                       box.apply_translation(np.array([48, 48, 80]))
                       mesh_chunk = slice_mesh_plane(mesh=mesh, plane_normal=-box.facets_normal, plane_origin=box.facets_origin)
                       mesh_chunk.export(os.path.join(dest, chunk.split("__")[0]+"_"+chunk.split("__")[2].split(".")[0] + ".obj"), "obj")
                       mesh.apply_translation(translation_b-translation_c)
                    except:
                       print("ERROR: ", chunk_c_path)


if __name__ == "__main__":
    import sys
    scene_base = sys.argv[1]
    sdf_base = sys.argv[2]
    sdf_scene_base = sys.argv[3]
    dest = sys.argv[4]
    part = int(sys.argv[5])
    os.makedirs(dest, exist_ok=True)
    scene_to_chunk(part)
