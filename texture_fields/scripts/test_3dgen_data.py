import numpy as np
from pathlib import Path
import torch
from imageio import imread
from skimage.transform import resize
import struct


def points_to_obj(points, path):
    with open(path, 'w') as f:
        for i in range(points.shape[0]):
            v = points[i, :]
            if not (v[0] == 0 and v[1] == 0 and v[2] == 0):
                f.write('v %f %f %f\n' % (v[0], v[1], v[2]))


def read_points_occ(path):
    points_dict = np.load(path)
    points = points_dict['points']
    # Break symmetry if given in float16:
    if points.dtype == np.float16:
        points = points.astype(np.float32)
        points += 1e-4 * np.random.randn(*points.shape)
    else:
        points = points.astype(np.float32)
    return points


def depthmap_to_world(depth, cam_W, cam_K, shape):
    batch_size, _, N, M = depth.size()
    # Turn depth around. This also avoids problems with inplace operations
    # depth = -depth.permute(0, 1, 3, 2)
    zero_one_row = torch.tensor([[0., 0., 0., 1.]])
    zero_one_row = zero_one_row.expand(batch_size, 1, 4)

    # add row to world mat
    cam_W = torch.cat((cam_W, zero_one_row), dim=1)
    # clean depth image for mask
    mask = (depth.abs() != float("Inf")).float()
    depth[depth == float("Inf")] = 0
    depth[depth == -1 * float("Inf")] = 0

    # 4d array to 2d array k=N*M
    d = depth.reshape(batch_size, 1, N * M)

    # create pixel location tensor
    px, py = torch.meshgrid([torch.arange(0, N), torch.arange(0, M)])

    p = torch.cat((
        px.expand(batch_size, 1, px.size(0), px.size(1)),
        py.expand(batch_size, 1, py.size(0), py.size(1))
    ), dim=1)
    p = p.reshape(batch_size, 2, py.size(0) * py.size(1))
    # Y: comment this
    # p = (p.float() / M * 2)

    # create terms of mapping equation x = P^-1 * d*(qp - b)
    P = cam_K[:, :2, :2].float()
    q = cam_K[:, 2:3, 2:3].float()
    b = cam_K[:, :2, 2:3].expand(batch_size, 2, d.size(2))
    Inv_P = torch.inverse(P)

    rightside = (p.float() * q.float() - b.float()) * d.float()
    x_xy = torch.bmm(Inv_P, rightside)

    # add depth and ones to location in world coord system
    x_world = torch.cat((x_xy, d, torch.ones_like(d)), dim=1)

    # derive loactoion in object coord via loc3d = W^-1 * x_world
    loc3d = torch.bmm(
        cam_W.expand(batch_size, 4, 4),
        x_world
    ).reshape(batch_size, 4, N, M)
    loc3d = loc3d.squeeze(0).permute((1, 2, 0))
    loc3d = loc3d.reshape(loc3d.shape[0] * loc3d.shape[1], loc3d.shape[2]).numpy()
    loc3d[:, 0] /= shape[0]
    loc3d[:, 1] /= shape[1]
    loc3d[:, 2] /= shape[2]
    loc3d -= 0.5
    print(loc3d.max(axis=0), loc3d.min(axis=0))
    return loc3d


def depthmap_to_world2(depth, cam_W, cam_K):
    depth = depth.squeeze()
    depth_shape = depth.shape
    print(depth_shape)
    depth = depth.flatten()
    cam_W = cam_W.squeeze()
    cam_K = cam_K.squeeze()
    zero_one_row = torch.tensor([[0., 0., 0., 1.]])
    cam_W = torch.cat((cam_W, zero_one_row), dim=0)
    cam_K = torch.cat((cam_K, zero_one_row), dim=0)
    x = torch.arange(0, depth_shape[0])
    y = torch.arange(0, depth_shape[1])
    x, y = torch.meshgrid(x, y)
    x = x.flatten().float()
    y = y.flatten().float()
    I = torch.zeros(4, depth.shape[0])
    I[0, :] = x * depth
    I[1, :] = y * depth
    I[2, :] = depth
    I[3, :] = 1.0
    loc3d = torch.mm(cam_W, torch.mm(torch.inverse(cam_K), I)).T.numpy() / 96 - 0.5
    # loc3d /= loc3d[3, :]
    return loc3d


def resize_image(img, size, order):
    img_out = resize(img, size, order=order, clip=False, mode='constant', anti_aliasing=False)
    return img_out


def adjust_intrinsic(intrinsic, old_shape, new_shape):
    print(intrinsic)
    intrinsic[0, 0] *= (new_shape[1] / old_shape[1])
    intrinsic[1, 1] *= (new_shape[0] / old_shape[0])
    intrinsic[0, 2] *= ((new_shape[1] - 1) / (old_shape[1] - 1))
    intrinsic[1, 2] *= ((new_shape[0] - 1) / (old_shape[0] - 1))
    print(intrinsic)
    return intrinsic


def read_depth_and_cameras_as_tensors(path_to_sdf, path_to_render, index):
    def read_matrix_file(mat_path):
        with open(mat_path, "r") as fptr:
            lines = fptr.read().splitlines()
            lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
            mat = np.asarray(lines, dtype=np.float32)
            return mat

    world2grid = np.load(path_to_sdf)['world2grid'].reshape(4, 4).astype(np.float32)
    depth_image = (resize_image(imread(Path(path_to_render) / f"depth_{index}.png").astype(np.float32), (128, 128), 0)).T / 1000
    pose = read_matrix_file(Path(path_to_render) / f"pose_{index}.txt")
    intrinsics = adjust_intrinsic(read_matrix_file(Path(path_to_render) / f"intr_{index}.txt"), (240, 320), (128, 128))
    # cam_W = torch.FloatTensor(np.eye(4, dtype=np.float32)[:3, :4]).unsqueeze(0)
    cam_W = torch.FloatTensor(np.matmul(world2grid, pose)[:3, :4]).unsqueeze(0)
    cam_K = torch.FloatTensor(intrinsics[:3, :4]).unsqueeze(0)
    depth = torch.FloatTensor(depth_image).unsqueeze(0).unsqueeze(0)
    return depth, cam_W, cam_K


def read_depth_and_cameras_as_tensors_matterport(path_to_sdf, path_to_render, index):
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
        return np.array(world2grid).reshape((4, 4))

    world2grid = read_sdf(path_to_sdf)
    depth_image = resize_image(imread(Path(path_to_render) / "depth" / f"{index}.png").astype(np.float32), (128, 128), 0).T / 1000
    pose, intrinsics = read_matrix_file(Path(path_to_render) / "camera" / f"{index}.txt")
    # cam_W = torch.FloatTensor(np.eye(4, dtype=np.float32)[:3, :4]).unsqueeze(0)
    cam_W = torch.FloatTensor(np.matmul(world2grid, pose)[:3, :4]).unsqueeze(0)
    cam_K = torch.FloatTensor(adjust_intrinsic(intrinsics, (256, 320), (128, 128))[:3, :4]).unsqueeze(0)
    depth = torch.FloatTensor(depth_image).unsqueeze(0).unsqueeze(0)
    return depth, cam_W, cam_K


if __name__ == '__main__':
    points_to_obj(read_points_occ('test_data_matterport/mJXqzFtmKg4_room1_0/points.npz'), "test_data_matterport/mJXqzFtmKg4_room1_0/out.obj")
    depth, cam_W, cam_K = read_depth_and_cameras_as_tensors_matterport("test_data_matterport/sdf/mJXqzFtmKg4_room1__cmp__0.sdf", "test_data_matterport/mJXqzFtmKg4_room1_0", 459)
    points_to_obj(depthmap_to_world(depth, cam_W, cam_K, (96, 96, 96)), "test_data_matterport/mJXqzFtmKg4_room1_0/depth.obj")
