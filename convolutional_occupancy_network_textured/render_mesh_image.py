import numpy as np
from pathlib import Path
import math
import trimesh
import pyrender
from scipy.spatial.transform import Rotation as R
# from moviepy.editor import *
from tqdm import tqdm
# from pygifsicle import optimize
# from util.misc import write_text_to_image
import os
import matplotlib.pyplot as plt
from PIL import Image
import common
import marching_cubes as mc


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


def load_mesh_points(path):
    return np.asarray([[float(y) for y in x.split(" ")[1:4]] for x in Path(path).read_text().splitlines()])

def visualize_folder(folder, output_path, meshes=False, condition=lambda x: True, make_gif_and_save_to_disk=True):
    w_img = 256
    l_img = 256
    num_frames = 8
    r = pyrender.OffscreenRenderer(w_img, l_img)
    mesh_path = folder
    image_side = 1
    # files_to_vis = [x for x in folder.iterdir() if condition(x.name)]
    # files_to_vis = sorted(files_to_vis, key=lambda x: x.name)
    # image_side = int(math.floor(math.sqrt(len(files_to_vis))))
    # image_buffer = [np.zeros((image_side * w_img, image_side * w_img, 3), dtype=np.uint8) for i in range(num_frames)]
    try:
        # _r = i // image_side
        # _c = i % image_side
        _r = 1
        _c = 1
        if meshes:
            # base_mesh = trimesh.load_mesh(mesh_path)
            base_mesh = trimesh.load(mesh_path)
        else:
            points_mesh = load_mesh_points(mesh_path)
            base_mesh = trimesh.voxel.ops.multibox(centers=points_mesh, pitch=1) #  trimesh.voxel.ops.points_to_marching_cubes(points=points_mesh, pitch=1)
        bbox = base_mesh.bounding_box.bounds
        loc = (bbox[0] + bbox[1]) / 2
        scale = (bbox[1] - bbox[0]).max()
        base_mesh.apply_translation(-loc)
        base_mesh.apply_scale(1 / scale)
        mesh = pyrender.Mesh.from_trimesh(base_mesh)
        for cam_idx in range(num_frames):
            camera_rotation = np.eye(4)
            camera_rotation[:3, :3] = R.from_euler('y', (360 / num_frames) * -cam_idx, degrees=True).as_matrix() @ R.from_euler('x', -45, degrees=True).as_matrix()
            camera_translation = np.eye(4)
            camera_translation[:3, 3] = np.array([0, 0, 1.5])
            camera_pose = camera_rotation @ camera_translation
            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
            scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0])
            scene.add(mesh)
            scene.add(camera, pose=camera_pose)

            light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                            innerConeAngle=np.pi/16.0,
                            outerConeAngle=np.pi/6.0)
            scene.add(light, pose=camera_pose)
            '''
            for n in create_raymond_lights():
                scene.add_node(n, scene.main_camera_node)
            '''
            color, depth = r.render(scene)
            output_file = output_path + '_{}_.png'.format(cam_idx)
            # print("output file", output_file)
            # print("color", color.shape)
            im = Image.fromarray(color)
            im.save(output_file)
        return color
    except Exception as e:
        print("Visualization failed", e)
        return None


'''
def visualize_folder(folder, output_path, meshes=False, condition=lambda x: True, make_gif_and_save_to_disk=True):
    w_img = 96
    num_frames = 8
    r = pyrender.OffscreenRenderer(w_img, w_img)
    files_to_vis = [x for x in folder.iterdir() if condition(x.name)]
    files_to_vis = sorted(files_to_vis, key=lambda x: x.name)
    image_side = int(math.floor(math.sqrt(len(files_to_vis))))
    image_buffer = [np.zeros((image_side * w_img, image_side * w_img, 3), dtype=np.uint8) for i in range(num_frames)]
    try:
        for i, mesh_path in enumerate(files_to_vis[:image_side * image_side]):
            _r = i // image_side
            _c = i % image_side
            if meshes:
                base_mesh = trimesh.load_mesh(mesh_path)
            else:
                points_mesh = load_mesh_points(mesh_path)
                base_mesh = trimesh.voxel.ops.multibox(centers=points_mesh, pitch=1) #  trimesh.voxel.ops.points_to_marching_cubes(points=points_mesh, pitch=1)
            bbox = base_mesh.bounding_box.bounds
            loc = (bbox[0] + bbox[1]) / 2
            scale = (bbox[1] - bbox[0]).max()
            base_mesh.apply_translation(-loc)
            base_mesh.apply_scale(1 / scale)
            mesh = pyrender.Mesh.from_trimesh(base_mesh)
            for cam_idx in range(num_frames):
                camera_rotation = np.eye(4)
                camera_rotation[:3, :3] = R.from_euler('y', (360 / num_frames) * -cam_idx, degrees=True).as_matrix() @ R.from_euler('x', -45, degrees=True).as_matrix()
                camera_translation = np.eye(4)
                camera_translation[:3, 3] = np.array([0, 0, 1.5])
                camera_pose = camera_rotation @ camera_translation
                camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
                scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0])
                scene.add(mesh)
                scene.add(camera, pose=camera_pose)
                for n in create_raymond_lights():
                    scene.add_node(n, scene.main_camera_node)
                color, depth = r.render(scene)
                image_buffer[cam_idx][_r * w_img: (_r + 1) * w_img, _c * w_img: (_c + 1) * w_img, :] = color
        for img_buffer_index in range(len(image_buffer)):
            image_buffer[img_buffer_index] = np.pad(image_buffer[img_buffer_index], ((2, 2), (2, 2), (0, 0)), mode='constant', constant_values=0)
        if make_gif_and_save_to_disk:
            clip = ImageSequenceClip(image_buffer, fps=4)
            clip.write_gif(output_path, verbose=False, logger=None)
            optimize(str(output_path), options=["--no-warnings"])
        return image_buffer

        return color
    except Exception as e:
        print("Visualization failed", e)
        return None

def single_mode(args):
    paths = [Path(x) for x in args.dirs]

    comparison = True if len(args.dirs) > 1 else False

    num_frames = 32
    r = pyrender.OffscreenRenderer(250, 250)

    base_path = paths[0]
    args.name = (args.name if args.name is not None else (base_path.name if not comparison else paths[1].name))

    output_dir = base_path.parents[0] / f"gif_{args.name}"
    output_dir.mkdir(exist_ok=True)
    meshes = list(set([x.name.split("_")[0] for x in base_path.iterdir()]))

    if comparison:
        for p in paths[1:]:
            meshes_other = list(set([x.name.split("_")[0] for x in p.iterdir()]))
            meshes = [x for x in meshes if x in meshes_other]

    for mesh_name in meshes:
        keys = [(base_path, args.cond), (base_path, args.pred), ]
        if comparison:
            for p in paths[1:]:
                keys.append((p, args.pred))
        keys.append((base_path, 'gt'))
        rendered_images = {k: [] for k in keys}
        for s in rendered_images.keys():
            mesh_path = s[0] / f"{mesh_name}_{s[1]}.obj"
            base_mesh = trimesh.load_mesh(mesh_path)
            if type(base_mesh) == trimesh.Scene:
                points_mesh = load_mesh_points(mesh_path)
                if points_mesh.shape[0] == 0:
                    for cam_idx in range(num_frames):
                        rendered_images[s].append(np.ones((250, 250, 3), dtype=np.uint8) * 255)
                    continue
                base_mesh = trimesh.voxel.ops.points_to_marching_cubes(points=points_mesh, pitch=1)
            bbox = base_mesh.bounding_box.bounds
            loc = (bbox[0] + bbox[1]) / 2
            scale = (bbox[1] - bbox[0]).max()
            base_mesh.apply_translation(-loc)
            base_mesh.apply_scale(1 / scale)
            mesh = pyrender.Mesh.from_trimesh(base_mesh)
            for cam_idx in range(num_frames):
                camera_rotation = np.eye(4)
                camera_rotation[:3, :3] = R.from_euler('y', (360 / num_frames) * -cam_idx, degrees=True).as_matrix() @ R.from_euler('x', -args.angle, degrees=True).as_matrix()
                camera_translation = np.eye(4)
                camera_translation[:3, 3] = np.array([0, 0, 1.5])
                camera_pose = camera_rotation @ camera_translation
                camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
                scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0])
                scene.add(mesh)
                scene.add(camera, pose=camera_pose)
                for n in create_raymond_lights():
                    scene.add_node(n, scene.main_camera_node)
                # pyrender.Viewer(scene, use_raymond_lighting=True)
                color, depth = r.render(scene)
                rendered_images[s].append(color)

        gif_images = []
        for img_idx in range(len(rendered_images[list(rendered_images.keys())[0]])):
            to_stack = []
            for s in rendered_images.keys():
                to_stack.append(rendered_images[s][img_idx])
            gif_images.append(np.hstack(np.asarray(to_stack)))

        clip = ImageSequenceClip(gif_images, fps=8)
        clip.write_gif(output_dir / f"{mesh_name}.gif")
        optimize(str(output_dir / f"{mesh_name}.gif"), options=["--no-warnings"])


def visualize_gan_extremes(args):
    image_buffers = []
    in_dir = Path(args.dirs[0])
    iter = in_dir.name
    output_vis_path_rr = in_dir.parents[1] / "real_best" / iter
    output_vis_path_rf = in_dir.parents[1] / "real_worst" / iter
    output_vis_path_fr = in_dir.parents[1] / "fake_worst" / iter
    output_vis_path_ff = in_dir.parents[1] / "fake_best" / iter
    for opath in tqdm([output_vis_path_rr, output_vis_path_rf, output_vis_path_fr, output_vis_path_ff]):
        image_buffers.append(visualize_folder(opath, None, make_gif_and_save_to_disk=False, meshes=not args.pts))
    image_buffer = []
    for i in range(len(image_buffers[0])):
        image_buffer.append(write_text_to_image(np.vstack([np.hstack([image_buffers[0][i], image_buffers[1][i]]), np.hstack([image_buffers[2][i], image_buffers[3][i]])])))
    clip = ImageSequenceClip(image_buffer, fps=4)
    output_path = in_dir.parents[1] / f"gan_matrix_{iter}.gif"
    clip.write_gif(output_path, verbose=False, logger=None)
    optimize(str(output_path), options=["--no-warnings"])


def visualize_recon_extremes(args):
    image_buffers = []
    in_dir = Path(args.dirs[0])
    iter = in_dir.name
    output_vis_path_rb = in_dir.parents[1] / "iou_best" / iter
    output_vis_path_rw = in_dir.parents[1] / "iou_worst" / iter
    for opath in tqdm([output_vis_path_rb, output_vis_path_rw]):
        image_buffers.append(visualize_folder(opath, None, make_gif_and_save_to_disk=False, meshes=not args.pts))
    image_buffer = []
    for i in range(len(image_buffers[0])):
        image_buffer.append(np.hstack([image_buffers[0][i], image_buffers[1][i]]))
    clip = ImageSequenceClip(image_buffer, fps=4)
    output_path = in_dir.parents[1] / f"iou_matrix_{iter}.gif"
    clip.write_gif(output_path, verbose=False, logger=None)
    optimize(str(output_path), options=["--no-warnings"])
'''

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dirs', type=str, nargs='+', default=None, help='gpus')
    parser.add_argument('--angle', type=float, default=30, help='angle')
    parser.add_argument('--name', type=str, default=None, help='name')
    parser.add_argument('--pred', type=str, default='pred', help='name')
    parser.add_argument('--cond', type=str, default='cond', help='name')
    parser.add_argument('--no_opt', action='store_true')
    parser.add_argument('--pts', action='store_true')
    parser.add_argument('--mode', type=str, choices=['grid', 'single', 'extr_g', 'extr_r'], default='single')
    args = parser.parse_args()
    '''
    if args.mode == 'single':
        single_mode(args)
    elif args.mode == 'grid':
        things_to_visualize = [(Path(args.dirs[0]), args.cond), (Path(args.dirs[0]), args.pred), ]
        for p in args.dirs[1:]:
            things_to_visualize += [(Path(p), args.pred)]
        things_to_visualize += [(Path(args.dirs[0]), 'gt')]
        image_buffers = []
        for ttv in tqdm(things_to_visualize):
            image_buffers.append(visualize_folder(ttv[0], None, condition=lambda x: x.split('.')[0].split('_')[1] == ttv[1], make_gif_and_save_to_disk=False, meshes=not args.pts))
        image_buffer = []
        for i in range(len(image_buffers[0])):
            image_buffer.append(np.hstack([image_buffers[j][i] for j in range(len(image_buffers))]))
        clip = ImageSequenceClip(image_buffer, fps=4)
        output_path = Path(args.dirs[0]).parents[0] / f"gif_{args.name}.gif"
        clip.write_gif(output_path, verbose=False, logger=None)
        optimize(str(output_path), options=["--no-warnings"])
    elif args.mode == 'extr_g':
        visualize_gan_extremes(args)
    elif args.mode == 'extr_r':
        visualize_recon_extremes(args)
    '''

    data_file = os.path.join(os.getcwd(), "data/test.lst")
    gt_data_folder = os.path.join(os.getcwd(), "data/my_dataset/03001627_testset_ply")
    gt_output_path = os.path.join(gt_data_folder, "rendered_images")
    Path(gt_output_path).mkdir(parents=True, exist_ok=True)

    tf_data_folder = os.path.join(os.getcwd(), "out/voxels/shapenet_grid32/generation/meshes/03001627")
    tf_output_path = os.path.join(tf_data_folder, "rendered_images")
    Path(tf_output_path).mkdir(parents=True, exist_ok=True)
    idx = 0
    with open(data_file, 'r') as f:
        models_c = f.read().split('\n')
        for sample in models_c:
            if sample != "":
                idx = idx + 1
                print("it", idx)
                gt_data_file = os.path.join(gt_data_folder, sample) + ".ply"
                output_path = os.path.join(gt_output_path, sample)
                visualize_folder(gt_data_file, output_path, condition=lambda x: x.split('.')[0].split('_')[1] == ttv[1], make_gif_and_save_to_disk=False, meshes=True)
                tf_data_file = os.path.join(tf_data_folder, sample) + ".ply"
                output_path = os.path.join(tf_output_path, sample)
                visualize_folder(tf_data_file, output_path, condition=lambda x: x.split('.')[0].split('_')[1] == ttv[1], make_gif_and_save_to_disk=False, meshes=True)

