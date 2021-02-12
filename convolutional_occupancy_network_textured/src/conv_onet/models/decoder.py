import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers import ResnetBlockFC
from src.common import normalize_coordinate, normalize_3d_coordinate, map2local, normalize_color_value

class LocalDecoder(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(self, dim=3, c_dim=128,
                 hidden_size=256, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding
    

    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1) torch.Size([64, 1024, 2]) / torch.Size([1, 35937, 2])
        xy = xy[:, :, None].float() # torch.Size([64, 1024, 1, 2])
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1) torch.Size([64, 1024, 1, 2]) / torch.Size([1, 1024, 1, 2])
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1) # c (original): torch.Size([64, 32, 64, 64])
        return c # torch.Size([64, 32, 1024]) / torch.Size([1, 32, 35937])

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c


    def forward(self, p, c_plane, **kwargs):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if 'grid' in plane_type:
                c += self.sample_grid_feature(p, c_plane['grid']) # torch.Size([16, 32, 2048])
            if 'xz' in plane_type:
                c += self.sample_plane_feature(p, c_plane['xz'], plane='xz') # torch.Size([64, 32, 1024]) / torch.Size([1, 32, 35937])
            if 'xy' in plane_type:
                c += self.sample_plane_feature(p, c_plane['xy'], plane='xy') # torch.Size([64, 32, 1024]) / torch.Size([1, 32, 35937])
            if 'yz' in plane_type:
                c += self.sample_plane_feature(p, c_plane['yz'], plane='yz') # torch.Size([64, 32, 1024]) / torch.Size([1, 32, 35937])
            c = c.transpose(1, 2) # torch.Size([64, 1024, 32]) / torch.Size([1, 35937, 32])

        p = p.float() # torch.Size([64, 2048, 3]) / torch.Size([1, 35937, 3])
        net = self.fc_p(p) # torch.Size([64, 1024, 32]) / torch.Size([1, 35937, 32])

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c) # torch.Size([64, 1024, 32]) / torch.Size([1, 35937, 32])
            net = self.blocks[i](net) # torch.Size([64, 1024, 32]) / torch.Size([1, 35937, 32])

        out = self.fc_out(self.actvn(net)) # torch.Size([64, 1024, 1]) / torch.Size([1, 35937, 1]) / torch.Size([1, 12500, 1])
        out = out.squeeze(-1) # torch.Size([64, 1024]) / squeezed torch.Size([1, 35937]) / torch.Size([1, 12500])

        return out

class MainLocalDecoder(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(self, dim=3, c_dim=128,
                 hidden_size=256, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1):
        super().__init__()

        c_dim = 32

        self.c_dim = c_dim
        self.n_blocks = n_blocks

        # self.fc_in_r = nn.Linear(c_dim*2, c_dim*2)
        # self.fc_in_g = nn.Linear(c_dim*2, c_dim*2)
        # self.fc_in_b = nn.Linear(c_dim*2, c_dim*2)

        self.fc_in_r = nn.Linear(c_dim*2, c_dim)
        self.fc_in_g = nn.Linear(c_dim*2, c_dim)
        self.fc_in_b = nn.Linear(c_dim*2, c_dim)

        if c_dim != 0:
            self.fc_c_p = nn.ModuleList([
                # nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])
        if c_dim != 0:
            self.fc_c_r = nn.ModuleList([
                # nn.Linear(c_dim*2, hidden_size) for i in range(n_blocks)
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])
        if c_dim != 0:
            self.fc_c_g = nn.ModuleList([
                # nn.Linear(c_dim*2, hidden_size) for i in range(n_blocks)
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])
        if c_dim != 0:
            self.fc_c_b = nn.ModuleList([
                # nn.Linear(c_dim*2, hidden_size) for i in range(n_blocks)
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        self.fc_p = nn.Linear(dim, hidden_size)
        self.fc_r = nn.Linear(dim, hidden_size)
        self.fc_g = nn.Linear(dim, hidden_size)
        self.fc_b = nn.Linear(dim, hidden_size)

        self.blocks_p = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])
        self.blocks_r = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])
        self.blocks_g = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])
        self.blocks_b = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out_p = nn.Linear(hidden_size, 1)
        self.fc_out_r = nn.Linear(hidden_size, 1)
        self.fc_out_g = nn.Linear(hidden_size, 1)
        self.fc_out_b = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding

    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c # torch[64, 32, 1024]

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1) # torch.Size([1, 1024, 1, 1, 3])
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1) # torch.Size([1, 64, 1024])
        return c

    def forward(self, p, c_plane, **kwargs):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c_p = 0
            c_r = 0
            c_g = 0
            c_b = 0
            output = 0
            p = p.float() # torch.Size([64, 1024, 3])
            if 'grid' in plane_type:
                c_p += self.sample_grid_feature(p, c_plane['grid']) # torch.Size([1, 64, 1024])
                c_p = c_p.transpose(1, 2) # torch.Size([64, 1024, 32])
                net = self.fc_p(p) # torch.Size([64, 1024, 32]) / torch.Size([1, 35937, 32])
                for i in range(self.n_blocks):
                    if self.c_dim != 0:
                        net = net + self.fc_c_p[i](c_p) # torch.Size([64, 1024, 32]) / torch.Size([1, 35937, 32])
                    net = self.blocks_p[i](net) # torch.Size([64, 1024, 32]) / torch.Size([1, 35937, 32])
                out = self.fc_out_p(self.actvn(net)) # torch.Size([64, 1024, 1]) / torch.Size([1, 35937, 1]) / torch.Size([1, 12500, 1])
                #out = self.actvn(out)
                output = out

                c_r += self.sample_grid_feature(p, c_plane['grid_r']) # torch.Size([1, 64, 1024])
                c_r = c_r.transpose(1, 2) # torch.Size([64, 1024, 64])
                c = self.fc_in_r(torch.cat((c_p, c_r), 2)) # torch.Size([64, 1024, 64])
                # c = torch.cat((c_p, c_r), 2) # torch.Size([64, 1024, 64])
                net = self.fc_r(p) # torch.Size([64, 1024, 32]) / torch.Size([1, 35937, 32])
                for i in range(self.n_blocks):
                    if self.c_dim != 0:
                        net = net + self.fc_c_r[i](c) # torch.Size([64, 1024, 32]) / torch.Size([1, 35937, 32])
                    net = self.blocks_r[i](net) # torch.Size([64, 1024, 32]) / torch.Size([1, 35937, 32])
                out = self.fc_out_r(self.actvn(net)) # torch.Size([64, 1024, 1]) / torch.Size([1, 35937, 1]) / torch.Size([1, 12500, 1])
                #out = self.actvn(out)
                output = torch.cat((output, out), 2)

                c_g += self.sample_grid_feature(p, c_plane['grid_g']) # torch.Size([1, 64, 1024])
                c_g = c_g.transpose(1, 2) # torch.Size([64, 1024, 32])
                c = self.fc_in_g(torch.cat((c_p, c_g), 2)) # torch.Size([64, 1024, 64])
                # c = torch.cat((c_p, c_g), 2) # torch.Size([64, 1024, 64])
                net = self.fc_g(p) # torch.Size([64, 1024, 32]) / torch.Size([1, 35937, 32])
                for i in range(self.n_blocks):
                    if self.c_dim != 0:
                        net = net + self.fc_c_g[i](c) # torch.Size([64, 1024, 32]) / torch.Size([1, 35937, 32])
                    net = self.blocks_g[i](net) # torch.Size([64, 1024, 32]) / torch.Size([1, 35937, 32])
                out = self.fc_out_g(self.actvn(net)) # torch.Size([64, 1024, 1]) / torch.Size([1, 35937, 1]) / torch.Size([1, 12500, 1])
                #out = self.actvn(out)
                output = torch.cat((output, out), 2)

                c_b += self.sample_grid_feature(p, c_plane['grid_b']) # torch.Size([1, 64, 1024])
                c_b = c_b.transpose(1, 2) # torch.Size([64, 1024, 32])
                c = self.fc_in_b(torch.cat((c_p, c_b), 2)) # torch.Size([64, 1024, 64])
                # c = torch.cat((c_p, c_b), 2) # torch.Size([64, 1024, 64])
                net = self.fc_b(p) # torch.Size([64, 1024, 32]) / torch.Size([1, 35937, 32])
                for i in range(self.n_blocks):
                    if self.c_dim != 0:
                        net = net + self.fc_c_b[i](c) # torch.Size([64, 1024, 32]) / torch.Size([1, 35937, 32])
                    net = self.blocks_b[i](net) # torch.Size([64, 1024, 32]) / torch.Size([1, 35937, 32])
                out = self.fc_out_b(self.actvn(net)) # torch.Size([64, 1024, 1]) / torch.Size([1, 35937, 1]) / torch.Size([1, 12500, 1])
                #out = self.actvn(out)
                output = torch.cat((output, out), 2)

            if 'xz' in plane_type:
                c_p += self.sample_plane_feature(p, c_plane['xz'], plane='xz') # torch.Size([64, 32, 1024]) / torch.Size([1, 32, 1024])
                c_p += self.sample_plane_feature(p, c_plane['xy'], plane='xy')
                c_p += self.sample_plane_feature(p, c_plane['yz'], plane='yz')
                c_p = c_p.transpose(1, 2) # torch.Size([64, 1024, 32])
                net = self.fc_p(p) # torch.Size([64, 1024, 32]) / torch.Size([1, 35937, 32])
                for i in range(self.n_blocks):
                    if self.c_dim != 0:
                        net = net + self.fc_c_p[i](c_p) # torch.Size([64, 1024, 32]) / torch.Size([1, 35937, 32])
                    net = self.blocks_p[i](net) # torch.Size([64, 1024, 32]) / torch.Size([1, 35937, 32])
                out = self.fc_out_p(self.actvn(net)) # torch.Size([64, 1024, 1]) / torch.Size([1, 35937, 1]) / torch.Size([1, 12500, 1])
                output = out
            if 'xz_r' in plane_type:
                c_r += self.sample_plane_feature(p, c_plane['xz_r'], plane='xz')
                c_r += self.sample_plane_feature(p, c_plane['xy_r'], plane='xz')
                c_r += self.sample_plane_feature(p, c_plane['yz_r'], plane='xz')
                c_r = c_r.transpose(1, 2) # torch.Size([64, 1024, 32])
                c = self.fc_in_r(torch.cat((c_p, c_r), 2)) # torch.Size([64, 1024, 64])
                net = self.fc_r(p) # torch.Size([64, 1024, 32])
                for i in range(self.n_blocks):
                    if self.c_dim != 0:
                        net = net + self.fc_c_r[i](c)
                    net = self.blocks_r[i](net) # torch.Size([64, 1024, 32])
                out = self.fc_out_r(self.actvn(net)) # torch.Size([64, 1024, 1])
                #out = self.actvn(out) # guarantee positive output
                output = torch.cat((output, out), 2)
            if 'xz_g' in plane_type:
                c_g += self.sample_plane_feature(p, c_plane['xz_g'], plane='yz')
                c_g += self.sample_plane_feature(p, c_plane['xy_g'], plane='xz')
                c_g += self.sample_plane_feature(p, c_plane['yz_g'], plane='xz')
                c_g = c_g.transpose(1, 2) # torch.Size([64, 1024, 32])
                c = self.fc_in_g(torch.cat((c_p, c_g), 2)) # torch.Size([64, 1024, 64])
                net = self.fc_g(p) # torch.Size([64, 1024, 32])
                for i in range(self.n_blocks):
                    if self.c_dim != 0:
                        net = net + self.fc_c_g[i](c)
                    net = self.blocks_g[i](net) # torch.Size([64, 1024, 32])
                out = self.fc_out_g(self.actvn(net)) # torch.Size([64, 1024, 1])
                #out = self.actvn(out) # guarantee positive output
                output = torch.cat((output, out), 2) # torch.Size([64, 1024, 3])
            if 'xz_b' in plane_type:
                c_b += self.sample_plane_feature(p, c_plane['xz_b'], plane='xy')
                c_b += self.sample_plane_feature(p, c_plane['xy_b'], plane='xz')
                c_b += self.sample_plane_feature(p, c_plane['yz_b'], plane='xz')
                c_b = c_b.transpose(1, 2) # torch.Size([64, 1024, 32])
                c = self.fc_in_b(torch.cat((c_p, c_b), 2)) # torch.Size([64, 1024, 64])
                net = self.fc_b(p) # torch.Size([64, 1024, 32])
                for i in range(self.n_blocks):
                    if self.c_dim != 0:
                        net = net + self.fc_c_b[i](c)
                    net = self.blocks_b[i](net) # torch.Size([64, 1024, 32])
                out = self.fc_out_b(self.actvn(net)) # torch.Size([64, 1024, 1])
                #out = self.actvn(out) # guarantee positive output
                output = torch.cat((output, out), 2) # # torch.Size([64, 1024, 4])
        '''
        c_col = (c_r + c_b + c_g) / 3
        col = normalize_color_value(col) # torch.Size([64, 1024, 3])
        col = col.float() # torch.Size([64, 1024, 3])
        net_col = self.fc_p(col) # torch.Size([64, 1024, 32])
        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net_col = net_col + self.fc_c[i](c_r)
            net_col = self.blocks[i](net_col) # torch.Size([16, 2048, 32])
        out_col = self.fc_out(self.actvn(net_col)) # torch.Size([16, 2048, 1])
        out_col = out.squeeze(-1) # torch.Size([16, 2048])
        '''
        return output # torch.Size([64, 1024, 4])


class PatchLocalDecoder(nn.Module):
    ''' Decoder adapted for crop training.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        local_coord (bool): whether to use local coordinate
        unit_size (float): defined voxel unit size for local system
        pos_encoding (str): method for the positional encoding, linear|sin_cos
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]

    '''

    def __init__(self, dim=3, c_dim=128,
                 hidden_size=256, leaky=False, n_blocks=5, sample_mode='bilinear', local_coord=False, pos_encoding='linear', unit_size=0.1, padding=0.1):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        #self.fc_p = nn.Linear(dim, hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode

        if local_coord:
            self.map2local = map2local(unit_size, pos_encoding=pos_encoding)
        else:
            self.map2local = None

        if pos_encoding == 'sin_cos':
            self.fc_p = nn.Linear(60, hidden_size)
        else:
            self.fc_p = nn.Linear(dim, hidden_size)
    
    def sample_feature(self, xy, c, fea_type='2d'):
        if fea_type == '2d':
            xy = xy[:, :, None].float()
            vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
            c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        else:
            xy = xy[:, :, None, None].float()
            vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
            c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    def forward(self, p, c_plane, **kwargs):
        p_n = p['p_n']
        p = p['p']

        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if 'grid' in plane_type:
                c += self.sample_feature(p_n['grid'], c_plane['grid'], fea_type='3d')
            if 'xz' in plane_type:
                c += self.sample_feature(p_n['xz'], c_plane['xz'])
            if 'xy' in plane_type:
                c += self.sample_feature(p_n['xy'], c_plane['xy'])
            if 'yz' in plane_type:
                c += self.sample_feature(p_n['yz'], c_plane['yz'])
            c = c.transpose(1, 2)

        p = p.float()
        if self.map2local:
            p = self.map2local(p)
        
        net = self.fc_p(p)
        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)
            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out

class LocalPointDecoder(nn.Module):
    ''' Decoder for PointConv Baseline.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        n_blocks (int): number of blocks ResNetBlockFC layers
        sample_mode (str): sampling mode  for points
    '''

    def __init__(self, dim=3, c_dim=128,
                 hidden_size=256, leaky=False, n_blocks=5, sample_mode='gaussian', **kwargs):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])


        self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        if sample_mode == 'gaussian':
            self.var = kwargs['gaussian_val']**2

    def sample_point_feature(self, q, p, fea):
        # q: B x M x 3
        # p: B x N x 3
        # fea: B x N x c_dim
        #p, fea = c
        if self.sample_mode == 'gaussian':
            # distance betweeen each query point to the point cloud
            dist = -((p.unsqueeze(1).expand(-1, q.size(1), -1, -1) - q.unsqueeze(2)).norm(dim=3)+10e-6)**2
            weight = (dist/self.var).exp() # Guassian kernel
        else:
            weight = 1/((p.unsqueeze(1).expand(-1, q.size(1), -1, -1) - q.unsqueeze(2)).norm(dim=3)+10e-6)

        #weight normalization
        weight = weight/weight.sum(dim=2).unsqueeze(-1)

        c_out = weight @ fea # B x M x c_dim

        return c_out

    def forward(self, p, c, **kwargs):
        n_points = p.shape[1]

        if n_points >= 30000:
            pp, fea = c
            c_list = []
            for p_split in torch.split(p, 10000, dim=1):
                if self.c_dim != 0:
                    c_list.append(self.sample_point_feature(p_split, pp, fea))
            c = torch.cat(c_list, dim=1)

        else:
           if self.c_dim != 0:
                pp, fea = c
                c = self.sample_point_feature(p, pp, fea)

        p = p.float()
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out
