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


model_g.eval()

for dataset in datasets:
    with torch.no_grad():
        dloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=12, shuffle=False, collate_fn=data.collate_remove_none)
        for idx, batch in tqdm(enumerate(dloader)):
            dest_path = os.path.join(cfg['test']['vis_dir'], batch['model'][0] + '.obj')
            # mesh_path = os.path.join(cfg['data']['path_mesh'], batch['model'][0], 'model_c.obj')
            # mesh_path = os.path.join(cfg['data']['path_mesh'], "mesh_onet", batch['model'][0]+".off")
            mesh_path = os.path.join(cfg['data']['path_mesh'], batch['model'][0]+".obj")
            # mesh_path = os.path.join(cfg['data']['path_mesh'], batch['model'][0], "pred_mesh.ply")
            mesh = trimesh.load(mesh_path, process=True)
            loc = np.array(cfg['data']['loc'])
            scale = cfg['data']['scale']
            # Transform input mesh
            mesh.apply_translation(-loc)
            mesh.apply_scale(1 / scale)
            loc3d = torch.from_numpy(np.array(mesh.vertices, dtype=np.float32).T[np.newaxis, :]).to(device)

            mesh_repr = geometry.get_representation(batch, device)
            condition = batch['condition'].to(device)

            geom_descr = model_g.encode_geometry(mesh_repr)

            z = model_g.encode(condition)
            z = z.cuda()

            x = model_g.decode(loc3d, geom_descr, z).squeeze().cpu().numpy().T
            pred_colors = np.hstack(((x * 255).astype(np.uint8), 255 * np.ones((x.shape[0], 1), dtype=np.uint8)))
            test_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, face_normals=mesh.face_normals, vertex_normals=mesh.vertex_normals, vertex_colors=pred_colors)
            test_mesh.process()
            test_mesh.apply_scale(scale)
            test_mesh.apply_translation(loc)
            test_mesh.export(dest_path, "obj")


