import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from src.encoder.unet import UNet
from src.encoder.unet3d import UNet3D
from src.common import coordinate2index, normalize_coordinate, normalize_3d_coordinate, normalize_color_value


class LocalVoxelEncoder(nn.Module):
    ''' 3D-convolutional encoder network for voxel input.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent code c
        hidden_dim (int): hidden dimension of the network
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature 
        plane_type (str): 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        kernel_size (int): kernel size for the first layer of CNN
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    
    '''

    def __init__(self, dim=3, c_dim=128, unet=False, unet_kwargs=None, unet3d=False, unet3d_kwargs=None, 
                 plane_resolution=512, grid_resolution=None, plane_type='xz', kernel_size=3, padding=0.1):
        super().__init__()
        self.actvn = F.relu
        if kernel_size == 1:
            self.conv_in = nn.Conv3d(1, c_dim, 1)
        else:
            self.conv_in = nn.Conv3d(1, c_dim, kernel_size, padding=1) # kernel size

        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None

        if unet3d:
            self.unet3d = UNet3D(**unet3d_kwargs)
        else:
            self.unet3d = None

        self.c_dim = c_dim

        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution

        self.plane_type = plane_type
        self.padding = padding

    def generate_plane_features(self, p, c, plane='xz'):
        # acquire indices of features in plane
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # torch.Size([64, 32768, 2])
        index = coordinate2index(xy, self.reso_plane) # torch.Size([64, 1, 32768])

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane**2) # fea_plane torch.Size([64, 32, 4096]) / fea_plane torch.Size([1, 32, 4096])

        c = c.permute(0, 2, 1) # torch.Size([64, 32, 32768]) / torch.Size([1, 32, 32768])
        # If two index are the same -> same projection on the plane: take the mean of the feature
        # scatter mean sort the index from small to big
        # Result: one 32-dim feature per grid element
        fea_plane = scatter_mean(c, index, out=fea_plane) # torch.Size([64, 32, 4096]) / torch.Size([1, 32, 4096])
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane, self.reso_plane) # torch.Size([64, 32, 64, 64]) / torch.Size([1, 32, 64, 64])

        # process the plane features with UNet
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)

        return fea_plane # torch.Size([64, 32, 64, 64]) / torch.Size([1, 32, 64, 64])

    def generate_grid_features(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type='3d')
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.c_dim, self.reso_grid**3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid)
        fea_grid = fea_grid.reshape(p.size(0), self.c_dim, self.reso_grid, self.reso_grid, self.reso_grid)

        if self.unet3d is not None:
            fea_grid = self.unet3d(fea_grid)

        return fea_grid


    def forward(self, x):
        batch_size = x.size(0) # torch.Size([64, 32, 32, 32]) / torch.Size([1, 32, 32, 32])
        device = x.device
        n_voxel = x.size(1) * x.size(2) * x.size(3)

        # voxel 3D coordintates
        coord1 = torch.linspace(-0.5, 0.5, x.size(1)).to(device) # torch.Size([32])
        coord2 = torch.linspace(-0.5, 0.5, x.size(2)).to(device) # torch.Size([32])
        coord3 = torch.linspace(-0.5, 0.5, x.size(3)).to(device) # torch.Size([32])

        coord1 = coord1.view(1, -1, 1, 1).expand_as(x) # torch.Size([64, 32, 32, 32]) / torch.Size([1, 32, 32, 32]) [:, coord1, -0.5000, -0.5000]
        coord2 = coord2.view(1, 1, -1, 1).expand_as(x) # torch.Size([64, 32, 32, 32]) / torch.Size([1, 32, 32, 32]) [:, -0.5000, coord2 -0.5000]
        coord3 = coord3.view(1, 1, 1, -1).expand_as(x) # torch.Size([64, 32, 32, 32]) / torch.Size([1, 32, 32, 32]) [:, -0.5000, -0.5000, coord3]
        p = torch.stack([coord1, coord2, coord3], dim=4) # torch.Size([64, 32, 32, 32, 3]) / torch.Size([1, 32, 32, 32, 3])
        p = p.view(batch_size, n_voxel, -1) # torch.Size([64, 32768, 3]) / torch.Size([1, 32768, 3]), [-0.5,-0.5,-0.5] -> [-0.5,-0.5,0.5] -> [-0.5,0.5,0.5] -> [0.5,0.5,0.5]

        # Acquire voxel-wise feature
        x = x.unsqueeze(1) # torch.Size([64, 1, 32, 32, 32]) / torch.Size([1, 1, 32, 32, 32])
        c = self.actvn(self.conv_in(x)).view(batch_size, self.c_dim, -1) # torch.Size([64, 32, 32768]) / torch.Size([1, 32, 32768])
        # print("conv_in", self.conv_in(x).shape)  # torch.Size([64, 32, 32, 32, 32]) / torch.Size([1, 32, 32, 32, 32])
        c = c.permute(0, 2, 1) # torch.Size([64, 32768, 32]) / torch.Size([1, 32768, 32])

        fea = {}
        if 'grid' in self.plane_type:
            fea['grid'] = self.generate_grid_features(p, c)
        else:
            if 'xz' in self.plane_type:
                fea['xz'] = self.generate_plane_features(p, c, plane='xz') # torch.Size([64, 32, 64, 64]) / torch.Size([1, 32, 64, 64])
            if 'xy' in self.plane_type:
                fea['xy'] = self.generate_plane_features(p, c, plane='xy') # torch.Size([64, 32, 64, 64]) / torch.Size([1, 32, 64, 64])
            if 'yz' in self.plane_type:
                fea['yz'] = self.generate_plane_features(p, c, plane='yz') # torch.Size([64, 32, 64, 64]) / torch.Size([1, 32, 64, 64])
        return fea

class MainVoxelEncoder(nn.Module):
    ''' 3D-convolutional encoder network for voxel input.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent code c
        hidden_dim (int): hidden dimension of the network
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature
        plane_type (str): 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        kernel_size (int): kernel size for the first layer of CNN
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]

    '''

    def __init__(self, dim=3, c_dim=128, unet=False, unet_kwargs=None, unet3d=False, unet3d_kwargs=None,
                 plane_resolution=512, grid_resolution=None, plane_type='xz', kernel_size=3, padding=0.1):
        super().__init__()

        c_dim = 32
        '''
        if unet:
            c_dim = 64
        else:
            c_dim = 32
        '''

        self.actvn = F.relu
        if kernel_size == 1:
            self.conv_in_p = nn.Conv3d(1, c_dim, 1)
            self.conv_in_r = nn.Conv3d(1, c_dim, 1)
            self.conv_in_g = nn.Conv3d(1, c_dim, 1)
            self.conv_in_b = nn.Conv3d(1, c_dim, 1)
        else:
            self.conv_in_p = nn.Conv3d(1, c_dim, kernel_size, padding=1) # kernel size
            self.conv_in_r = nn.Conv3d(1, c_dim, kernel_size, padding=1) # kernel size
            self.conv_in_g = nn.Conv3d(1, c_dim, kernel_size, padding=1) # kernel size
            self.conv_in_b = nn.Conv3d(1, c_dim, kernel_size, padding=1) # kernel size

        if unet:
            self.unet_p = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
            self.unet_r = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
            self.unet_g = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
            self.unet_b = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet_p = None
            self.unet_r = None
            self.unet_g = None
            self.unet_b = None

        if unet3d:
            self.unet3d_p = UNet3D(**unet3d_kwargs)
            self.unet3d_r = UNet3D(**unet3d_kwargs)
            self.unet3d_g = UNet3D(**unet3d_kwargs)
            self.unet3d_b = UNet3D(**unet3d_kwargs)
            # self.unet3d_rgb = UNet3D(**unet3d_kwargs)
        else:
            self.unet3d_p = None
            self.unet3d_r = None
            self.unet3d_g = None
            self.unet3d_b = None

        self.c_dim = c_dim

        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution

        self.plane_type = plane_type
        self.padding = padding

        self.unet = unet
        self.unet3d = unet3d

    def generate_plane_features(self, p, c, plane='xz'):
        # acquire indices of features in plane
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # torch.Size([64, 32768, 2])
        index = coordinate2index(xy, self.reso_plane) # torch.Size([64, 1, 32768])

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane**2) # torch.Size([64, 32, 4096]) / torch.Size([1, 32, 4096])

        c = c.permute(0, 2, 1) # torch.Size([64, 32, 32768]) / torch.Size([1, 32, 32768])
        # If two index are the same -> same projection on the plane: take the mean of the feature
        # scatter mean sort the index from small to big
        # Result: one 32-dim feature per grid element
        fea_plane = scatter_mean(c, index, out=fea_plane) # torch.Size([64, 32, 4096]) / torch.Size([1, 32, 4096])
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane, self.reso_plane) # torch.Size([64, 32, 64, 64]) / torch.Size([1, 32, 64, 64])

        # process the plane features with UNet
        if plane == 'xy' or plane == 'xz' or plane == 'yz':
            fea_plane = self.unet_p(fea_plane)
        if plane == 'xy_r' or plane == 'xz_r' or plane == 'yz_r':
            fea_plane = self.unet_r(fea_plane)
        if plane == 'xy_b' or plane == 'xz_b' or plane == 'yz_b':
            fea_plane = self.unet_b(fea_plane)
        if plane == 'xy_g' or plane == 'xz_g' or plane == 'yz_g':
            fea_plane = self.unet_g(fea_plane)

        return fea_plane # torch.Size([64, 32, 64, 64]) / torch.Size([1, 32, 64, 64])

    def generate_grid_features(self, p, c, grid='grid'): # p: torch.Size([64, 32768, 3]) / torch.Size([1, 32768, 64])
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type='3d') # torch.Size([1, 1, 32768])
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.c_dim, self.reso_grid**3) # torch.Size([1, 64, 32768])
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid)
        fea_grid = fea_grid.reshape(p.size(0), self.c_dim, self.reso_grid, self.reso_grid, self.reso_grid) # torch.Size([1, 32, 32, 32, 32])

        if self.unet3d is not None:
            if grid == 'grid':
                fea_grid = self.unet3d_p(fea_grid)
            if grid == 'grid_r':
                fea_grid = self.unet3d_r(fea_grid)
                # fea_grid = self.unet3d_rgb(fea_grid)
            if grid == 'grid_g':
                fea_grid = self.unet3d_g(fea_grid)
                # fea_grid = self.unet3d_rgb(fea_grid)
            if grid == 'grid_b':
                fea_grid = self.unet3d_b(fea_grid)
                # fea_grid = self.unet3d_rgb(fea_grid)
        return fea_grid

    def forward(self, x, col):
        # print("col", col.shape) # torch.Size([64, 32, 32, 32, 3])
        batch_size = x.size(0) # torch.Size([64, 32, 32, 32]) / torch.Size([1, 32, 32, 32])
        device = x.device
        n_voxel = x.size(1) * x.size(2) * x.size(3)

        # voxel 3D coordintates
        coord1 = torch.linspace(-0.5, 0.5, x.size(1)).to(device) # torch.Size([32])
        coord2 = torch.linspace(-0.5, 0.5, x.size(2)).to(device) # torch.Size([32])
        coord3 = torch.linspace(-0.5, 0.5, x.size(3)).to(device) # torch.Size([32])
        coord1 = coord1.view(1, -1, 1, 1).expand_as(x) # torch.Size([64, 32, 32, 32]) / torch.Size([1, 32, 32, 32]) [:, coord1, -0.5000, -0.5000]
        coord2 = coord2.view(1, 1, -1, 1).expand_as(x) # torch.Size([64, 32, 32, 32]) / torch.Size([1, 32, 32, 32]) [:, -0.5000, coord2 -0.5000]
        coord3 = coord3.view(1, 1, 1, -1).expand_as(x) # torch.Size([64, 32, 32, 32]) / torch.Size([1, 32, 32, 32]) [:, -0.5000, -0.5000, coord3]
        p = torch.stack([coord1, coord2, coord3], dim=4) # torch.Size([64, 32, 32, 32, 3]) / torch.Size([1, 32, 32, 32, 3])
        p = p.view(batch_size, n_voxel, -1) # torch.Size([64, 32768, 3]) / torch.Size([1, 32768, 3]), [-0.5,-0.5,-0.5] -> [-0.5,-0.5,0.5] -> [-0.5,0.5,0.5] -> [0.5,0.5,0.5]

        #col = col.view(batch_size, n_voxel, -1) # torch.Size([64, 32768, 3]) / torch.Size([1, 32768, 3])
        #col = normalize_color_value(col)

        # Acquire voxel-wise feature
        x = x.unsqueeze(1) # torch.Size([64, 1, 32, 32, 32]) / torch.Size([1, 1, 32, 32, 32])
        # print("conv_in", self.conv_in(x).shape)  # torch.Size([64, 32, 32, 32, 32]) / torch.Size([1, 32, 32, 32, 32])
        c = self.actvn(self.conv_in_p(x)).view(batch_size, self.c_dim, -1) # torch.Size([64, 32, 32768]) / torch.Size([1, 32, 32768])
        c = c.permute(0, 2, 1) # torch.Size([64, 32768, 32]) / torch.Size([1, 32768, 32])

        r = col[:, :, :, :, 0].unsqueeze(1).float() # torch.Size([64, 1, 32, 32, 32]) / torch.Size([1, 1, 32, 32, 32])
        g = col[:, :, :, :, 1].unsqueeze(1).float() # torch.Size([64, 1, 32, 32, 32]) / torch.Size([1, 1, 32, 32, 32])
        b = col[:, :, :, :, 2].unsqueeze(1).float() # torch.Size([64, 1, 32, 32, 32]) / torch.Size([1, 1, 32, 32, 32])
        c_r = self.actvn(self.conv_in_r(r)).view(batch_size, self.c_dim, -1) # torch.Size([64, 32, 32768]) / torch.Size([1, 32, 32768])
        c_g = self.actvn(self.conv_in_g(g)).view(batch_size, self.c_dim, -1) # torch.Size([64, 32, 32768]) / torch.Size([1, 32, 32768])
        c_b = self.actvn(self.conv_in_b(b)).view(batch_size, self.c_dim, -1) # torch.Size([64, 32, 32768]) / torch.Size([1, 32, 32768])
        c_r = c_r.permute(0, 2, 1) # torch.Size([64, 32768, 32]) / torch.Size([1, 32768, 32])
        c_g = c_g.permute(0, 2, 1) # torch.Size([64, 32768, 32]) / torch.Size([1, 32768, 32])
        c_b = c_b.permute(0, 2, 1) # torch.Size([64, 32768, 32]) / torch.Size([1, 32768, 32])

        fea = {}
        if 'grid' in self.plane_type:
            fea['grid'] = self.generate_grid_features(p, c, grid='grid')
            fea["grid_r"] = self.generate_grid_features(p, c_r, grid='grid_r')
            fea["grid_g"] = self.generate_grid_features(p, c_g, grid='grid_g')
            fea["grid_b"] = self.generate_grid_features(p, c_b, grid='grid_b')
        else:
            if 'xz' in self.plane_type:
                fea['xz'] = self.generate_plane_features(p, c, plane='xz') # torch.Size([64, 32, 64, 64]) / torch.Size([1, 32, 64, 64])
                fea['xz_r'] = self.generate_plane_features(p, c_r, plane='xz') # torch.Size([64, 32, 64, 64]) / torch.Size([1, 32, 64, 64])
                fea['xz_b'] = self.generate_plane_features(p, c_b, plane='xz') # torch.Size([64, 32, 64, 64]) / torch.Size([1, 32, 64, 64])
                fea['xz_g'] = self.generate_plane_features(p, c_g, plane='xz') # torch.Size([64, 32, 64, 64]) / torch.Size([1, 32, 64, 64])
            if 'xy' in self.plane_type:
                fea['xy'] = self.generate_plane_features(p, c, plane='xy') # torch.Size([64, 32, 64, 64]) / torch.Size([1, 32, 64, 64])
                fea['xy_r'] = self.generate_plane_features(p, c_r, plane='xy') # torch.Size([64, 32, 64, 64]) / torch.Size([1, 32, 64, 64])
                fea['xy_g'] = self.generate_plane_features(p, c_g, plane='xy') # torch.Size([64, 32, 64, 64]) / torch.Size([1, 32, 64, 64])
                fea['xy_b'] = self.generate_plane_features(p, c_b, plane='xy') # torch.Size([64, 32, 64, 64]) / torch.Size([1, 32, 64, 64])
            if 'yz' in self.plane_type:
                fea['yz'] = self.generate_plane_features(p, c, plane='yz') # torch.Size([64, 32, 64, 64]) / torch.Size([1, 32, 64, 64])
                fea['yz_r'] = self.generate_plane_features(p, c_r, plane='yz') # torch.Size([64, 32, 64, 64]) / torch.Size([1, 32, 64, 64])
                fea['yz_g'] = self.generate_plane_features(p, c_g, plane='yz') # torch.Size([64, 32, 64, 64]) / torch.Size([1, 32, 64, 64])
                fea['yz_b'] = self.generate_plane_features(p, c_b, plane='yz') # torch.Size([64, 32, 64, 64]) / torch.Size([1, 32, 64, 64])
        return fea

class VoxelEncoder(nn.Module):
    ''' 3D-convolutional encoder network for voxel input.

    Args:
        dim (int): input dimension
        c_dim (int): output dimension
    '''

    def __init__(self, dim=3, c_dim=128):
        super().__init__()
        self.actvn = F.relu

        self.conv_in = nn.Conv3d(1, 32, 3, padding=1)

        self.conv_0 = nn.Conv3d(32, 64, 3, padding=1, stride=2)
        self.conv_1 = nn.Conv3d(64, 128, 3, padding=1, stride=2)
        self.conv_2 = nn.Conv3d(128, 256, 3, padding=1, stride=2)
        self.conv_3 = nn.Conv3d(256, 512, 3, padding=1, stride=2)
        self.fc = nn.Linear(512 * 2 * 2 * 2, c_dim)

    def forward(self, x):
        batch_size = x.size(0)

        x = x.unsqueeze(1)
        net = self.conv_in(x)
        net = self.conv_0(self.actvn(net))
        net = self.conv_1(self.actvn(net))
        net = self.conv_2(self.actvn(net))
        net = self.conv_3(self.actvn(net))

        hidden = net.view(batch_size, 512 * 2 * 2 * 2)
        c = self.fc(self.actvn(hidden))

        return c
