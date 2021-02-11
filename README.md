# Texture_Completion

This project is developed within the course "Advanced Deep Learning for Computer Vision" at the TU Munich. The supervisor for my project is Yawar Siddiqui. This project is based on three other projects:
* Occupancy Networks
* Convolutional Occupancy Networks
* Texture Fields

The modified scripts from Occupancy Networks are used for data-processing, 
The from Texture Field

## Usage

First, implement the three Github projects, you can find them in the following linkes: [Occupancy Network](https://github.com/autonomousvision/occupancy_networks), [Convolutional Occupancy Networks](https://github.com/autonomousvision/convolutional_occupancy_networks) and [Texture Fields](https://github.com/autonomousvision/texture_fields). <br>
Tips: I could not use the environment.yaml file to create the working environment. I found it works when creating the conda environment manually with the following lines allways work. Install the packages based on their respective channel:

```
conda create -n my_conv_occ python=3.7
conda install -c cython==0.29.2
conda install pytorch==1.0.0 torchvision==0.2.1 -c pytorch
conda install -c conda-forge matplotlib-base==3.0.3
```

```
python train.py configs/my_model.yaml

python sample_mesh.py data/ColoredSDF/output --pointcloud_folder=out/pointcloud --points_folder=out/points --float16 --packbits

python train.py configs/voxel/shapenet_grid32.yaml

% tensorboard
tensorboard --logdir=/home/zeju/Documents/convolutional_occupancy_networks-master/out/voxels/shapenet_3plane/logs


python render_mesh_image.py --dirs=examples/0_in
```

Use the tensorboard to visualize the training and validation results. The result is stored in the `out/voxels/shapenet_3plane/logs` folder.
```
tensorboard --logdir=/home/zeju/Documents/convolutional_occupancy_networks-master/out/voxels/shapenet_3plane/logs
```
To generate the use the following command to use our pre-trained model.
```
python generate.py --dirs=examples/0_in
```

To generate the **ssim**, **l1 feature** and **FID** value use the following command line.
```
python evaluate_metrics.py
```
