import trimesh
import pyrender
import numpy as np
import os
import math
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import struct

LIGHTING = None
read_camera = None
window_dims = None
fill_holes = False
threshold = 0.35

def create_raymond_lights():
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))
    return nodes


def make_rotate(rx, ry, rz):
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3,3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3,3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3,3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0
    r = np.eye(4, 4)
    R = np.matmul(np.matmul(Rz,Ry),Rx)
    r[:3, :3] = R[:3, :3]
    return r


def adjust_intrinsic(intrinsic, old_shape, new_shape):
    intrinsic[0, 0] *= (new_shape[1] / old_shape[1])
    intrinsic[1, 1] *= (new_shape[0] / old_shape[0])
    intrinsic[0, 2] *= ((new_shape[1] - 1) / (old_shape[1] - 1))
    intrinsic[1, 2] *= ((new_shape[0] - 1) / (old_shape[0] - 1))
    return intrinsic


def read_camera_matterport(root_path, model_name, index):
    def read_matrix_file(mat_path):
        with open(mat_path, "r") as fptr:
            lines = fptr.read().splitlines()
            lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
            pose = np.asarray(lines[:4], dtype=np.float32)
            intrinsic = np.asarray(lines[4:], dtype=np.float32)
            return pose, intrinsic

    def read_sdf(sdf_path):
        fin = open(sdf_path, 'rb')
        fin.read(28)
        world2grid = struct.unpack('f' * 4 * 4, fin.read(4 * 4 * 4))
        fin.close()
        return np.array(world2grid).reshape((4, 4)).astype(np.float32)

    path_to_sdf = os.path.join(root_path, "sdf", model_name.split("_")[0]+"_"+model_name.split("_")[1]+"__cmp__"+model_name.split("_")[2]+".sdf")
    path_to_cam = os.path.join(root_path, model_name, "camera", f"{index}.txt")
    world2grid = read_sdf(path_to_sdf)
    pose, intrinsics = read_matrix_file(path_to_cam)
    # cam_W = torch.FloatTensor(np.eye(4, dtype=np.float32)[:3, :4]).unsqueeze(0)
    cam_W = np.matmul(np.matmul(world2grid, pose), make_rotate(math.radians(180), 0, 0))
    cam_K = adjust_intrinsic(intrinsics, (256, 320), window_dims)
    return cam_W, cam_K


def read_camera_shapenet(root_path, model_name, index):

    def read_matrix_file(mat_path):
        with open(mat_path, "r") as fptr:
            lines = fptr.read().splitlines()
            lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
            mat = np.asarray(lines, dtype=np.float32)
            return mat

    path_to_sdf = os.path.join(root_path, "sdf", model_name + ".npz")
    path_to_pose = os.path.join(root_path, model_name, "pose",  f"{index:03d}.txt")
    path_to_intr = os.path.join(root_path, model_name, "intrinsic",  f"{index:03d}.txt")
    world2grid = np.load(path_to_sdf)['world2grid'].reshape(4, 4).astype(np.float32)
    pose = read_matrix_file(path_to_pose)
    intrinsics = adjust_intrinsic(read_matrix_file(path_to_intr), (256, 320), window_dims)
    cam_W = np.matmul(np.matmul(world2grid, pose), make_rotate(math.radians(180), 0, 0))
    cam_K = intrinsics
    return cam_W, cam_K


def get_mesh_path(mesh_root, model_name, method):
    if method == 'texturefields':
        return os.path.join(mesh_root, model_name+".obj")
    if method.startswith('pifu'):
        view_idx = int(method.split("_")[1])
        return os.path.join(mesh_root, model_name + f"_{view_idx:03d}_pred.obj")
    if method == 'ours':
        return os.path.join(mesh_root, model_name, "pred_mesh.ply")
    return os.path.join(mesh_root, model_name, "model_c.obj")


def get_mesh_list(mesh_root, method):
    mesh_list = []
    if method == 'pifu':
        mesh_list = [x.split("_000_pred.obj")[0] for x in os.listdir(mesh_root) if x.endswith("_000_pred.obj")]
    elif method == 'texturefields':
        mesh_list = [x.split(".")[0] for x in os.listdir(mesh_root)]
    elif method == 'ours':
        mesh_list = os.listdir(mesh_root)
    else:
        mesh_list = os.listdir(mesh_root)
        #list_of_test_meshes = Path("/home/yawar/GAC-private/data/shapenet-chairs-3dgen-colorfix/split/test.txt").read_text().split("\n")
        #mesh_list = [x for x in os.listdir(mesh_root) if x in list_of_test_meshes]
    return sorted(mesh_list)


def get_rendered_depth(method, mesh_root, param_root, model_name, camera_index):
    trimesh_obj = trimesh.load(get_mesh_path(mesh_root, model_name, method), process=True)
    mesh = pyrender.Mesh.from_trimesh(trimesh_obj)
    extrinsic, intrinsic = read_camera(param_root, model_name, camera_index)
    scene = None
    if LIGHTING == 'plain':
        scene = pyrender.Scene(ambient_light=[0.75, 0.75, 0.75])
    else:
        scene = pyrender.Scene()
    scene.add(mesh)
    camera_intrinsic = pyrender.IntrinsicsCamera(intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], zfar=6000)
    scene.add(camera_intrinsic, pose=extrinsic)
    if LIGHTING != 'plain':
        for n in create_raymond_lights():
            scene.add_node(n, scene.main_camera_node)
    # pyrender.Viewer(scene, viewport_size=window_dims[::-1])
    r = pyrender.OffscreenRenderer(window_dims[1], window_dims[0])
    color, depth = r.render(scene)
    return depth


def render_mesh_with_camera(method, mesh_root, param_root, model_name, camera_index):
    trimesh_obj = trimesh.load(get_mesh_path(mesh_root, model_name, method), process=True)
    if type(trimesh_obj) == trimesh.Trimesh:
        mesh = pyrender.Mesh.from_trimesh(trimesh_obj)
    else:
        print("MeshTypeError: ", type(trimesh_obj), model_name)
        return None
    extrinsic, intrinsic = read_camera(param_root, model_name, camera_index)
    scene = None
    if LIGHTING=='plain':
        scene = pyrender.Scene(ambient_light=[0.75, 0.75, 0.75])
    else:
        scene = pyrender.Scene()
    scene.add(mesh)
    camera_intrinsic = pyrender.IntrinsicsCamera(intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], zfar=6000)
    scene.add(camera_intrinsic, pose=extrinsic)
    if LIGHTING != 'plain':
       for n in create_raymond_lights():
           scene.add_node(n, scene.main_camera_node)
    # pyrender.Viewer(scene, viewport_size=window_dims[::-1])
    r = pyrender.OffscreenRenderer(window_dims[1], window_dims[0])
    color, depth = r.render(scene)
    mesh_root_base = os.path.basename(mesh_root)
    #base_depth = get_rendered_depth(method, mesh_root.replace(mesh_root_base, 'ours_chunks'), param_root, model_name, camera_index)
    #print(mesh_root.replace(mesh_root_base, "our_chunks"))
    if fill_holes:
        color_true = np.asarray(Image.open(os.path.join(param_root, model_name, "input_image_eval", f"{camera_index}.jpg")))
        mask = depth == 0
        perc_missing = mask.sum() / (window_dims[0] * window_dims[1])
        if perc_missing > threshold:
            return None
        #replace_mask = np.logical_and(mask == 1, (base_depth == 0))
        fixed_color = np.zeros_like(color)
        fixed_color[:] = color[:]
        fixed_color[mask, :] = color_true[mask, :]
        #fixed_color[replace_mask, :] = color_true[replace_mask, :]
        color = fixed_color
    return Image.fromarray(color)


def render_texturefields(mesh_root, param_root, dest_root):
    meshlist = get_mesh_list(mesh_root, "texturefields")
    for mesh in tqdm(meshlist):
        dest_dir = os.path.join(dest_root, mesh)
        if not os.path.exists(os.path.join(param_root, mesh)):
            print('not found:', os.path.join(param_root, mesh))
            continue
        if os.path.exists(os.path.join(param_root, mesh, "camera")):
            viewlist = [int(x.split(".")[0]) for x in os.listdir(os.path.join(param_root, mesh, "camera"))]
        else:
            viewlist = list(range(24))
        os.makedirs(dest_dir, exist_ok=True)
        for view_idx in viewlist:
            image = render_mesh_with_camera("texturefields", mesh_root, param_root, mesh, view_idx)
            if image is not None:
                image.save(os.path.join(dest_dir, f"{view_idx:03}.png"))


def render_matterport_gt(param_root, dest_root):
    for model_name in [x for x in os.listdir(param_root) if x != 'sdf' and not x.endswith(".lst")]:
        dest_dir = os.path.join(dest_root, model_name)
        os.makedirs(dest_dir, exist_ok=True)
        viewlist = [int(x.split(".")[0]) for x in os.listdir(os.path.join(param_root, model_name, "camera"))]
        for view in viewlist:
            Image.open(os.path.join(param_root, model_name, "input_image_eval", f"{view}.jpg")).save(os.path.join(dest_dir, f"{view:03}.png"))


def render_ours(mesh_root, param_root, dest_root):
    meshlist = get_mesh_list(mesh_root, "ours")
    viewlist = list(range(24))
    for mesh in tqdm(meshlist):
        dest_dir = os.path.join(dest_root, mesh)
        os.makedirs(dest_dir, exist_ok=True)
        for view_idx in viewlist:
            render_mesh_with_camera("ours", mesh_root, param_root, mesh, view_idx).save(os.path.join(dest_dir, f"{view_idx:03}.png"))



def render_gt(mesh_root, param_root, dest_root):
    meshlist = get_mesh_list(mesh_root, "gt")
    for mesh in tqdm(meshlist):
        dest_dir = os.path.join(dest_root, mesh)
        if os.path.exists(os.path.join(param_root, mesh, "camera")):
            viewlist = [int(x.split(".")[0]) for x in os.listdir(os.path.join(param_root, mesh, "camera"))]
        else:
            viewlist = list(range(24))
        os.makedirs(dest_dir, exist_ok=True)
        for view_idx in viewlist:
            render_mesh_with_camera("gt", mesh_root, param_root, mesh, view_idx).save(os.path.join(dest_dir, f"{view_idx:03}.png"))


def render_pifu(mesh_root, param_root, dest_root):
    for pifu_view in tqdm(range(8)):
        meshlist = get_mesh_list(mesh_root, "pifu")
        viewlist = list(range(24))
        for mesh in tqdm(meshlist):
            dest_dir = os.path.join(dest_root, f"{pifu_view:03d}", mesh)
            os.makedirs(os.path.join(dest_root, f"{pifu_view:03d}"), exist_ok=True)
            os.makedirs(dest_dir, exist_ok=True)
            for view_idx in viewlist:
                if not os.path.exists(os.path.join(dest_dir, f"{view_idx:03}.png")):
                    render_mesh_with_camera(f"pifu_{pifu_view}", mesh_root, param_root, mesh, view_idx).save(os.path.join(dest_dir, f"{view_idx:03}.png"))


if __name__ == '__main__':
    import sys
    mesh_root = sys.argv[1]
    param_root = sys.argv[2]
    dest_root = sys.argv[3]
    method = sys.argv[4]
    read_camera = read_camera_matterport
    window_dims = (256, 320)
    fill_holes = True
    LIGHTING = "plain"
    if method == "pifu":
        render_pifu(mesh_root, param_root, dest_root)
    elif method == "gt":
        render_gt(mesh_root, param_root, dest_root)
    elif method == "gtmp":
        render_matterport_gt(param_root, dest_root)
    elif method == 'texturefields':
        render_texturefields(mesh_root, param_root, dest_root)
    elif method == 'ours':
        render_ours(mesh_root, param_root, dest_root)
