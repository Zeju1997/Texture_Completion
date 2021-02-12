import argparse
import trimesh
import numpy as np
import os
import glob
import sys
from multiprocessing import Pool
from functools import partial
# TODO: do this better
sys.path.append('..')
from im2mesh.utils import binvox_rw, voxels
from im2mesh.utils.libmesh import check_mesh_contains
import shutil
import torch
import common
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
parser = argparse.ArgumentParser('Sample a watertight mesh.')
#parser.add_argument('in_folder', type=str,
#                    help='Path to input watertight meshes.')
parser.add_argument('--n_proc', type=int, default=0,
                    help='Number of processes to use.')

parser.add_argument('--resize', action='store_true',
                    help='When active, resizes the mesh to bounding box.')

parser.add_argument('--rotate_xz', type=float, default=0.,
                    help='Angle to rotate around y axis.')

parser.add_argument('--bbox_padding', type=float, default=0.,
                    help='Padding for bounding box')
parser.add_argument('--bbox_in_folder', type=str,
                    help='Path to other input folder to extract'
                         'bounding boxes.')

#parser.add_argument('--pointcloud_folder', type=str,
#                    help='Output path for point cloud.')
parser.add_argument('--pointcloud_size', type=int, default=12500,
                    help='Size of point cloud.')

parser.add_argument('--voxels_folder', type=str,
                    help='Output path for voxelization.')
parser.add_argument('--voxels_res', type=int, default=32,
                    help='Resolution for voxelization.')

#parser.add_argument('--points_folder', type=str,
#                    help='Output path for points.')
parser.add_argument('--points_size', type=int, default=12500,
                    help='Size of points.')
parser.add_argument('--points_uniform_ratio', type=float, default=1.,
                    help='Ratio of points to sample uniformly'
                         'in bounding box.')
parser.add_argument('--points_sigma', type=float, default=0.01,
                    help='Standard deviation of gaussian noise added to points'
                         'samples on the surfaces.')
parser.add_argument('--points_padding', type=float, default=0.1,
                    help='Additional padding applied to the uniformly'
                         'sampled points on both sides (in total).')

parser.add_argument('--mesh_folder', type=str,
                    help='Output path for mesh.')

parser.add_argument('--overwrite', action='store_true',
                    help='Whether to overwrite output.')
parser.add_argument('--float16', action='store_true',
                    help='Whether to use half precision.')
parser.add_argument('--packbits', action='store_true',
                help='Whether to save truth values as bit array.')
'''

def main(args, visualize):
    #input_files = glob.glob(os.path.join(args.in_folder, '*.off'))
    #print(os.path.join(args.in_folder, '*.off'))

    input_files = glob.glob(os.path.join(args.in_folder, '*.obj'))
    if args.n_proc != 0:
        with Pool(args.n_proc) as p:
            p.map(partial(process_path, args=args), input_files)
    else:
        for p in input_files:
            process_path(p, args, visualize)


def process_path(in_path, args, visualize):
    in_file = os.path.basename(in_path)
    modelname = os.path.splitext(in_file)[0]
    mesh = trimesh.load(in_path, process=False)

    # Determine bounding box
    if not args.resize:
        # Standard bounding boux
        loc = np.zeros(3)
        scale = 1.
    else:
        if args.bbox_in_folder is not None:
            in_path_tmp = os.path.join(args.bbox_in_folder, modelname + '.off')
            mesh_tmp = trimesh.load(in_path_tmp, process=False)
            bbox = mesh_tmp.bounding_box.bounds
        else:
            bbox = mesh.bounding_box.bounds

        # Compute location and scale
        loc = (bbox[0] + bbox[1]) / 2
        scale = (bbox[1] - bbox[0]).max() / (1 - args.bbox_padding)

        # Transform input mesh
        mesh.apply_translation(-loc)
        mesh.apply_scale(1 / scale)

        if args.rotate_xz != 0:
            angle = args.rotate_xz / 180 * np.pi
            R = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
            mesh.apply_transform(R)

    # Export various modalities
    if args.pointcloud_folder is not None:
        mask = "tgt_"
        if mask in in_path:
            export_pointcloud(mesh, modelname, loc, scale, args, visualize)

    if args.voxels_folder is not None:
        mask = "in_"
        if mask in in_path:
            export_voxels(mesh, modelname, loc, scale, args, visualize)

    if args.points_folder is not None:
        mask = "tgt_"
        if mask in in_path:
            export_points(mesh, modelname, loc, scale, args, visualize)

    if args.mesh_folder is not None:
        export_mesh(mesh, modelname, loc, scale, args, visualize)

def color_points(mesh, points, occ):
    mesh_colors = mesh.vertices[:, [3, 4, 5]]
    mesh_pos = mesh.vertices[:, [0, 1, 2]]
    color = np.ones(points.shape) * 255
    for i in range(len(points)):
        '''
        if occ[i]:
            idx = ((mesh_pos - points[i, :]) ** 2).sum(1).argmin()
            color[i, :] = mesh_colors[idx, :]
        '''
        idx = ((mesh_pos - points[i, :]) ** 2).sum(1).argmin()
        color[i, :] = mesh_colors[idx, :]
    return color

def color(mesh, points):
    mesh_colors = mesh.vertices[:, [3, 4, 5]]
    mesh_pos = mesh.vertices[:, [0, 1, 2]]
    color = np.empty(points.shape)
    for i in range(len(points)):
        idx = ((mesh_pos - points[i, :]) ** 2).sum(1).argmin()
        color[i, :] = mesh_colors[idx, :]
    return color

def export_pointcloud(mesh, modelname, loc, scale, args, visualize):
    #filename = os.path.join(args.pointcloud_folder, modelname + '.npz')
    filename = os.path.join(args.pointcloud_folder, 'pointcloud.npz')
    if not args.overwrite and os.path.exists(filename):
        print('Pointcloud already exist: %s' % filename)
        return

    points, face_idx = mesh.sample(args.pointcloud_size, return_index=True)
    normals = mesh.face_normals[face_idx]

    obj_file = glob.glob(os.path.join(os.path.abspath(os.path.join(args.in_folder, os.pardir)), "1_scaled/*.obj"))
    original_mesh = common.Mesh.from_obj(obj_file[0])
    colors = color(original_mesh, points)
    colors[colors > 255] = 255 - 10e-6

    # Compress
    if args.float16:
        dtype = np.float16
    else:
        dtype = np.float32

    points = points.astype(dtype)
    normals = normals.astype(dtype)

    print('Writing pointcloud: %s' % filename)
    np.savez(filename, points=points, normals=normals, loc=loc, scale=scale, colors=colors)

def make_3d_grid(bb_min, bb_max, shape):
    ''' Makes a 3D grid.

    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    '''
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p


def export_voxels(mesh, modelname, loc, scale, args, visualize):
    if not mesh.is_watertight:
        print('Warning: mesh %s is not watertight!'
              'Cannot create voxelization.' % modelname)
        return

    filename = os.path.join(args.voxels_folder, 'model.binvox')

    if not args.overwrite and os.path.exists(filename):
        print('Voxels already exist: %s' % filename)
        return

    res = args.voxels_res # 32
    voxels_occ = voxels.voxelize_ray(mesh, res)

    # print("voxel occ shape", voxels_occ.shape)
    # print(voxels_occ)
    # print("voxel occ occupied", np.count_nonzero(voxels_occ)/(res*res*res))

    voxels_out = binvox_rw.Voxels(voxels_occ, (res,) * 3,
                                  translate=loc, scale=scale,
                                  axis_order='xyz')
    print('Writing voxels: %s' % filename)
    with open(filename, 'bw') as f:
        voxels_out.write(f)

    shape = (res,) * 3
    bb_min = (0.5,) * 3
    bb_max = (res - 0.5,) * 3
    points = make_3d_grid(bb_min, bb_max, shape=shape).numpy()
    points = (points / res - 0.5) # (32768, 3) min: [- 0.5 + 0.5 / 16 / 2]

    obj_file = glob.glob(os.path.join(os.path.abspath(os.path.join(args.in_folder, os.pardir)), "1_scaled/*.obj"))
    original_mesh = common.Mesh.from_obj(obj_file[0])
    voxels_color = color_voxel(original_mesh, points, res) # (32, 32, 32, 3)
    voxels_color[voxels_color > 255] = 255 - 10e-6
    voxels_color[voxels_color < 0] = 0

    filename = os.path.join(args.voxels_folder, 'voxels.npz')
    print('Writing voxels: %s' % filename)
    np.savez(filename, voxels_color=voxels_color)

    if visualize:
        # plot voxeled chair
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.voxels(voxels_occ,
              facecolors=voxels_color/255,
              #edgecolors=np.clip(2*colors - 0.5, 0, 1),  # brighter
              edgecolor='k',
              linewidth=0.5)
        ax.set(xlabel='X', ylabel='Y', zlabel='Z')
        plt.show()


def color_voxel(mesh, points, res):
    offset = 0.5 / 16 / 2 # 0.015625
    threshold = offset * offset * 3
    mesh_colors = mesh.vertices[:, [3, 4, 5]] # 17470
    mesh_pos = mesh.vertices[:, [0, 1, 2]]
    colors = np.zeros((res, res, res, 3))
    index = 0
    for i in range(res):
        for j in range(res):
            for k in range(res):
                color = 0
                idx = ((mesh_pos - points[index, :]) ** 2).sum(1).argmin()
                color = color + mesh_colors[idx, :]
                idx = ((mesh_pos - (points[index, :] + np.array([offset, offset, offset]))) ** 2).sum(1).argmin()
                color = color + mesh_colors[idx, :]
                idx = ((mesh_pos - (points[index, :] + np.array([offset, offset, -offset]))) ** 2).sum(1).argmin()
                color = color + mesh_colors[idx, :]
                idx = ((mesh_pos - (points[index, :] + np.array([offset, -offset, offset]))) ** 2).sum(1).argmin()
                color = color + mesh_colors[idx, :]
                idx = ((mesh_pos - (points[index, :] + np.array([offset, -offset, -offset]))) ** 2).sum(1).argmin()
                color = color + mesh_colors[idx, :]
                idx = ((mesh_pos - (points[index, :] + np.array([-offset, offset, offset]))) ** 2).sum(1).argmin()
                color = color + mesh_colors[idx, :]
                idx = ((mesh_pos - (points[index, :] + np.array([-offset, offset, -offset]))) ** 2).sum(1).argmin()
                color = color + mesh_colors[idx, :]
                idx = ((mesh_pos - (points[index, :] + np.array([-offset, -offset, offset]))) ** 2).sum(1).argmin()
                color = color + mesh_colors[idx, :]
                idx = ((mesh_pos - (points[index, :] + np.array([-offset, -offset, -offset]))) ** 2).sum(1).argmin()
                color = color + mesh_colors[idx, :]
                colors[i, j, k, :] = color / 9
                '''
                # diff = ((mesh_pos - points[index, :]) ** 2).sum(1)
                # print("points", points[index, :])
                diff = np.absolute(mesh_pos - points[index, :]).max(1)
                #idx = np.where(diff.max(0) >= offset)[0]
                #idx_x = np.argwhere(diff[:, 0] < offset).shape
                #idx_y = np.argwhere(diff[:, 1] < offset).shape
                #idx_z = np.argwhere(diff[:, 2] < offset).shape
                #print("diff min", diff.min())
                idx = np.argwhere(diff < offset).squeeze(-1)
                if idx.size > 0: # "point not found"
                    print("point found")
                    color = 0    
                    for a in range(len(idx)):
                        color = mesh_colors[idx[a], :]
                        print("color shape", color.shape)
                        color = color + 1
                else:
                    colors[i, j, k, :] = [255, 255, 255]
                '''
                index = index + 1
    return colors


def midpoints(x):
    sl = ()
    for i in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return x



def export_points(mesh, modelname, loc, scale, args, visualize):
    if not mesh.is_watertight:
        print('Warning: mesh %s is not watertight!'
              'Cannot sample points.' % modelname)
        return

    #filename = os.path.join(args.points_folder, modelname + '.npz')
    filename = os.path.join(args.points_folder, 'points.npz')

    if not args.overwrite and os.path.exists(filename):
        print('Points already exist: %s' % filename)
        return

    n_points_uniform = int(args.points_size * args.points_uniform_ratio)
    n_points_surface = args.points_size - n_points_uniform

    boxsize = 1 + args.points_padding
    points_uniform = np.random.rand(n_points_uniform, 3)
    points_uniform = boxsize * (points_uniform - 0.5)
    points_surface = mesh.sample(n_points_surface)
    points_surface += args.points_sigma * np.random.randn(n_points_surface, 3)
    points = np.concatenate([points_uniform, points_surface], axis=0)

    occupancies = check_mesh_contains(mesh, points) # (12500, )

    obj_file = glob.glob(os.path.join(os.path.abspath(os.path.join(args.in_folder, os.pardir)), "1_scaled/*.obj"))
    original_mesh = common.Mesh.from_obj(obj_file[0])
    colors = color_points(original_mesh, points, occupancies) # (12500, 3)
    colors[colors > 255] = 255 - 10e-6
    colors[colors < 0] = 0

    # Compress
    if args.float16:
        dtype = np.float16
    else:
        dtype = np.float32

    points = points.astype(dtype)

    if visualize:
        colors = colors / 255
        idx = np.argwhere(occupancies > 0).squeeze(-1)
        print("idx", idx.shape)
        test = points[idx, :]
        test_color = colors[idx, :]
        # plot query points
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # For each set of style and range settings, plot n random points in the box
        # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
        # ax.scatter(points[:, 0], points[:, 1], points[:, 2], facecolors=colors, marker='o')
        ax.scatter(test[:, 0], test[:, 1], test[:, 2], facecolors=test_color, marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

        # plot query points
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # For each set of style and range settings, plot n random points in the box
        # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
        # ax.scatter(points[:, 0], points[:, 1], points[:, 2], facecolors=colors, marker='o')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], facecolors=colors, marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()



    if args.packbits:
        print("packbits")
        occupancies = np.packbits(occupancies)

    print('Writing points: %s' % filename)
    np.savez(filename, points=points, occupancies=occupancies, loc=loc, scale=scale, colors=colors)

def randrange(n, vmin, vmax):
    """
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    """
    return (vmax - vmin)*np.random.rand(n) + vmin


def export_mesh(mesh, modelname, loc, scale, args, visualize):
    filename = os.path.join(args.mesh_folder, modelname + '.off')    
    if not args.overwrite and os.path.exists(filename):
        print('Mesh already exist: %s' % filename)
        return
    print('Writing mesh: %s' % filename)
    mesh.export(filename)


if __name__ == '__main__':
    data_folder = os.path.join(os.getcwd(), "external/mesh-fusion/data")
    data_file = os.path.join(data_folder, "my_shapenet/data_1.lst")
    temp = 0
    with open(data_file, 'r') as f:
        models_c = f.read().split('\n')
        for sample in models_c:
            if sample != "":
                sample_folder = os.path.join(data_folder, "my_shapenet/03001627", sample)
                input_folder = os.path.join(sample_folder, "2_watertight")
                output_folder = sample_folder
                parser = argparse.ArgumentParser('Sample a watertight mesh.')
                parser.add_argument('--in_folder', type=str, default=input_folder, help='Path to input watertight meshes.')
                parser.add_argument('--pointcloud_folder', default=sample_folder, type=str, help='Output path for point cloud.')
                parser.add_argument('--points_folder', default=sample_folder, type=str, help='Output path for points.')
                parser.add_argument('--n_proc', type=int, default=0, help='Number of processes to use.')
                parser.add_argument('--resize', action='store_true', help='When active, resizes the mesh to bounding box.')
                parser.add_argument('--rotate_xz', type=float, default=0., help='Angle to rotate around y axis.')
                parser.add_argument('--bbox_padding', type=float, default=0., help='Padding for bounding box')
                parser.add_argument('--bbox_in_folder', type=str, help='Path to other input folder to extract bounding boxes.')
                parser.add_argument('--pointcloud_size', type=int, default=100, help='Size of point cloud.')
                parser.add_argument('--voxels_folder', default=sample_folder, type=str, help='Output path for voxelization.')
                parser.add_argument('--voxels_res', type=int, default=32, help='Resolution for voxelization.')
                parser.add_argument('--points_size', type=int, default=100000, help='Size of points.')
                parser.add_argument('--points_uniform_ratio', type=float, default=1., help='Ratio of points to sample uniformly in bounding box.')
                parser.add_argument('--points_sigma', type=float, default=0.01, help='Standard deviation of gaussian noise added to points samples on the surfaces.')
                parser.add_argument('--points_padding', type=float, default=0.1, help='Additional padding applied to the uniformly sampled points on both sides (in total).')
                parser.add_argument('--mesh_folder', type=str, help='Output path for mesh.')
                parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite output.', default=True)
                parser.add_argument('--float16', action='store_true', help='Whether to use half precision.')
                parser.add_argument('--packbits', action='store_true', help='Whether to save truth values as bit array.')
                args = parser.parse_args()
                visualize = False
                main(args, visualize)
                if os.path.exists(input_folder) and os.path.isdir(input_folder):
                    shutil.rmtree(input_folder)
                obj_folder = os.path.join(os.path.abspath(os.path.join(args.in_folder, os.pardir)), "1_scaled")
                if os.path.exists(obj_folder) and os.path.isdir(obj_folder):
                    shutil.rmtree(obj_folder)
                temp = temp + 1
                print(temp)
