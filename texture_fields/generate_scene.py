import torch
import os
import argparse
from tqdm import tqdm
from mesh2tex import data
from mesh2tex import config
from mesh2tex.checkpoints import CheckpointIO
from mesh2tex import geometry
import trimesh
import numpy as np
from pathlib import Path
import struct

# Get arguments and Config
parser = argparse.ArgumentParser(
    description='Generate Color for given mesh.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')

# Define device
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

# Read config
out_dir = cfg['training']['out_dir']
vis_dir = cfg['test']['vis_dir']
split = cfg['test']['dataset_split']
if split != 'test_vis' and split != 'test_eval':
    print('Are you sure not using test data?')
batch_size = cfg['generation']['batch_size']
gen_mode = cfg['test']['generation_mode']
model_url = cfg['model']['model_url']

# Dataset
dataset = config.get_dataset(split, cfg, input_sampling=False)
if cfg['data']['shapes_multiclass']:
    datasets = dataset.datasets_classes
else:
    datasets = [dataset]

# Load Model
models = config.get_models(cfg, device=device, dataset=dataset)
model_g = models['generator']
checkpoint_io = CheckpointIO(out_dir, model_g=model_g)
if model_url is None:
    checkpoint_io.load(cfg['test']['model_file'])
else:
    checkpoint_io.load(cfg['model']['model_url'])

os.makedirs(cfg['test']['vis_dir'], exist_ok=True)


def read_w2g(sdf_path):
    fin = open(sdf_path, 'rb')
    fin.read(28)
    world2grid = np.array(struct.unpack('f' * 4 * 4, fin.read(4 * 4 * 4))).reshape((4, 4)).astype(np.float32)
    world2grid[:3, :3] = np.eye(3)
    #grid2world = np.linalg.inv(world2grid)
    fin.close()
    #return grid2world[:3, 3]
    return world2grid[:3, 3]

model_g.eval()
scene_base = os.path.join(os.environ["raid"], "tmp_results/ours")
sdf_scene_base = "/mnt/sorona_angela_raid/data/matterport/mp_sdf_vox_2cm_color_trunc32"
sdf_base = "/mnt/sorona_angela_raid/data/matterport/completion_blocks_2cm_test/individual_96-96-160_s32_all"
test_scenes = ["_".join(x.split("_")[:2]) for x in Path("data/matterport/test_32/test.lst").read_text().split()]

for dataset in datasets:

    for scene in tqdm([x for x in os.listdir(scene_base) if "_".join(x.split("_")[:2]) in test_scenes]):

        #if not scene == "ARNzJeq3xxb_room13__0__pred.ply":
        #    continue

        mesh = trimesh.load(os.path.join(scene_base, scene), process=True)
        vertex_id_to_colors = np.zeros_like(mesh.visual.vertex_colors).astype(np.float32)
        loc = np.array(cfg['data']['loc'])
        scale = cfg['data']['scale']
        with torch.no_grad():
            dloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False, collate_fn=data.collate_remove_none)
            for idx, batch in tqdm(enumerate(dloader)):
                chunk = "_".join(batch['model'][0].split("_")[:2]) + "__cmp__" + batch['model'][0].split("_")[-1]
                chunk_room = chunk.split("__")[0]
                if scene.split("__")[0] == chunk_room: #and batch['model'][0] in ("ARNzJeq3xxb_room13_0", "ARNzJeq3xxb_room13_1", "ARNzJeq3xxb_room13_2"):
                    # make a vertexid => color array map
                    # load the scene
                    # translate it to origin 96x96x160
                    # prediction
                    # save colors for these vertexid
                    # average for all vertexids
                    chunk_b_path = os.path.join(sdf_scene_base, chunk.split("__")[0] + "__0__.sdf")
                    chunk_c_path = os.path.join(sdf_base, chunk+".sdf")

                    translation_c = read_w2g(chunk_c_path)
                    translation_b = read_w2g(chunk_b_path)
                    # translation_b = np.zeros_like(translation_c)
                    mesh.apply_translation(translation_c - translation_b)
                    # mesh.export(os.path.join(os.path.join(dest, chunk+".obj")))
                    indices = np.zeros(mesh.vertices.shape[0]).astype(np.bool)
                    for vid in range(mesh.vertices.shape[0]):
                        if 0 <= mesh.vertices[vid, 0] < 96 and 0 <= mesh.vertices[vid, 1] < 96 and 0 <= mesh.vertices[vid, 2] < 160:
                            indices[vid] = True
                    if np.any(indices):
                        mesh.apply_translation(-loc)
                        mesh.apply_scale(1 / scale)
                        loc3d = torch.from_numpy(np.array(mesh.vertices[indices, :], dtype=np.float32).T[np.newaxis, :]).to(device)
                        mesh_repr = geometry.get_representation(batch, device)
                        condition = batch['condition'].to(device)
                        geom_descr = model_g.encode_geometry(mesh_repr)
                        z = model_g.encode(condition)
                        z = z.cuda()
                        x = model_g.decode(loc3d, geom_descr, z).squeeze().cpu().numpy().T
                        pred_colors = np.hstack(((x * 255).astype(np.uint8), 255 * np.ones((x.shape[0], 1), dtype=np.uint8)))
                        vertex_id_to_colors[indices, :] += pred_colors
                        mesh.apply_scale(scale)
                        mesh.apply_translation(loc)
                    mesh.apply_translation(translation_b - translation_c)

        for vid in range(mesh.vertices.shape[0]):
            vertex_id_to_colors[vid, :] = (vertex_id_to_colors[vid, :] / ((255 if vertex_id_to_colors[vid, 3] == 0 else vertex_id_to_colors[vid, 3]) / 255))

        dest_path = os.path.join(cfg['test']['vis_dir'], scene.split(".")[0]+".obj")
        test_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, face_normals=mesh.face_normals, vertex_normals=mesh.vertex_normals, vertex_colors=vertex_id_to_colors)
        test_mesh.process()
        #test_mesh.apply_scale(scale)
        #test_mesh.apply_translation(loc)
        test_mesh.export(dest_path, "obj")


