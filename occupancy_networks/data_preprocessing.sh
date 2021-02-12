# chmod +x data_preprocessing.sh
# generate obj file
# python external/mesh-fusion/data/generate_dataset.py

# generate off file
python external/mesh-fusion/1_scale.py

# generate depth file
python external/mesh-fusion/2_fusion.py --mode=render

# generate watertight mesh
python external/mesh-fusion/2_fusion.py --mode=fuse

# simplify mesh
#python external/mesh-fusion/3_simplify.py
#python external/mesh-fusion/3_simplify.py --in_dir=examples/ --out_dir=examples/

# sample mesh, create pointcloud.npz / points.npz
python sample_mesh.py --float16 --packbits
