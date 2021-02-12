import os
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist
from src.common import (
    compute_iou, make_3d_grid, add_key,
)
from src.utils import visualize as vis
from src.training import BaseTrainer
import numpy as np

from torch.cuda.amp import GradScaler, autocast


class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples

    '''

    def __init__(self, model, optimizer, device=None, input_type='pointcloud',
                 vis_dir=None, threshold=0.5, eval_sample=False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample

        self.scaler = GradScaler()

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data, idx, epoch_it, gradient_accumulations):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''

        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss, loss_p, loss_rgb = self.compute_loss(data, epoch_it)
        loss.backward()
        self.optimizer.step()
        '''

        # v1
        self.model.train()
        loss, loss_p, loss_rgb = self.compute_loss(data, epoch_it)
        (loss / gradient_accumulations).backward()

        if (idx + 1) % gradient_accumulations == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        # v2
        '''
        self.model.train()
        with autocast():
            loss, loss_p, loss_rgb = self.compute_loss(data, epoch_it)
        self.scaler.scale(loss / gradient_accumulations).backward()

        if (idx + 1) % gradient_accumulations == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.model.zero_grad()
        '''

        return loss.item(), loss_p.item(), loss_rgb.item()
    
    def eval_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device
        threshold = self.threshold
        eval_dict = {}

        points = data.get('points').to(device) # torch.Size([1, 1024, 3])
        occ = data.get('points.occ').to(device) # torch.Size([1, 1024])
        p_colors = data.get('points.colors').to(device) # torch.Size([1, 1024, 3])

        inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device) # torch.Size([1, 32, 32, 32])
        voxels_occ = data.get('voxels') # torch.Size([1, 32, 32, 32])
        inputs_colors = data.get('inputs.voxels_color').to(device) # torch.Size([1, 32, 32, 32, 3])

        points_iou = data.get('points_iou').to(device) # torch.Size([1, 100000, 3])
        occ_iou = data.get('points_iou.occ').to(device) # torch.Size([1, 100000])

        batch_size = points.size(0)

        kwargs = {}
        
        # add pre-computed index
        inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
        # add pre-computed normalized coordinates
        points = add_key(points, data.get('points.normalized'), 'p', 'p_n', device=device)
        points_iou = add_key(points_iou, data.get('points_iou.normalized'), 'p', 'p_n', device=device)

        # Compute iou
        with torch.no_grad():
            p_out = self.model(points_iou, inputs, inputs_colors, sample=self.eval_sample, **kwargs)

        occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
        occ_iou_hat_np = (p_out.probs >= threshold).cpu().numpy()

        iou = compute_iou(occ_iou_np, occ_iou_hat_np[:, :, 0]).mean()
        eval_dict['iou'] = iou

        # Estimate voxel iou
        if voxels_occ is not None:
            voxels_occ = voxels_occ.to(device)
            points_voxels = make_3d_grid(
                (-0.5 + 1/64,) * 3, (0.5 - 1/64,) * 3, voxels_occ.shape[1:])
            points_voxels = points_voxels.expand(
                batch_size, *points_voxels.size())
            points_voxels = points_voxels.to(device)
            with torch.no_grad():
                p_out = self.model(points_voxels, inputs, inputs_colors, sample=self.eval_sample, **kwargs)

            voxels_occ_np = (voxels_occ >= 0.5).cpu().numpy()
            occ_hat_np = (p_out.probs >= threshold).cpu().numpy()
            iou_voxels = compute_iou(voxels_occ_np, occ_hat_np[:, :, 0]).mean()

            eval_dict['iou_voxels'] = iou_voxels

        return eval_dict

    def compute_loss(self, data, epoch_it):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        p = data.get('points').to(device) #torch.Size([64, 1024, 3])
        occ = data.get('points.occ').to(device) #torch.Size([64, 1024])
        p_colors = data.get('points.colors').to(device) #torch.Size([64, 1024, 3])
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device) #torch.Size([64, 32, 32, 32])
        inputs_colors = data.get('inputs.voxels_color').to(device) # torch.Size([64, 32, 32, 32, 3]) range[0,1]
        loss = 0

        if 'pointcloud_crop' in data.keys():
            # add pre-computed index
            inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
            inputs['mask'] = data.get('inputs.mask').to(device)
            # add pre-computed normalized coordinates
            p = add_key(p, data.get('points.normalized'), 'p', 'p_n', device=device)

        '''
        if epoch_it > 12351235:
            c = self.model.encode_inputs(inputs) # xy: torch.Size([64, 32, 64, 64])
            kwargs = {}
            # General points
            logits = self.model.decode(p, c, **kwargs).logits #torch.Size([64, 1024])
            loss_i = F.binary_cross_entropy_with_logits(
                logits, occ, reduction='none')
            loss = loss_i.sum(-1).mean()
        '''

        if epoch_it > 0:
            c_main = self.model.main_encode_inputs(inputs, inputs_colors) # xy: torch.Size([64, 32, 64, 64])
            kwargs = {}
            # General points
            logits = self.model.main_decode(p, c_main, **kwargs).logits #torch.Size([64, 1024, 4])
            # logits[:, :, [1, 2, 3]] = logits[:, :, [1, 2, 3]] * 255

            loss_l1 = torch.nn.L1Loss()
            loss_MSE = torch.nn.MSELoss()
            loss_r = loss_l1(logits[:, :, 1].double(), p_colors[:, :, 0].double())
            loss_g = loss_l1(logits[:, :, 2].double(), p_colors[:, :, 1].double())
            loss_b = loss_l1(logits[:, :, 3].double(), p_colors[:, :, 2].double())
            # loss_r = loss_MSE(logits[:, :, 1].double(), p_colors[:, :, 0].double())
            # loss_g = loss_MSE(logits[:, :, 2].double(), p_colors[:, :, 1].double())
            # loss_b = loss_MSE(logits[:, :, 3].double(), p_colors[:, :, 2].double())

            loss_bce_p = F.binary_cross_entropy_with_logits(logits[:, :, 0], occ, reduction='none')
            loss_p = loss_bce_p.sum(-1).mean()
            # loss = loss_p + (loss_r + loss_g + loss_b) / 3
            loss_rgb = loss_r + loss_g + loss_b
            # print("loss rgb", loss_rgb)
            loss = loss_p + loss_rgb
            # loss = loss_p
        '''
        unit_m = torch.ones(p_colors.shape[0], p_colors.shape[1], p_colors.shape[2])
        print("one", unit_m.shape)
        diff = torch.abs(loss_l1(logits[:, :, [1, 2, 3]]))
        print("diff", diff.shape)
        loss_rgb = torch.log(unit_m - diff).sum(-1).mean()
        print(loss_rgb.shape)
        #print(torch.abs(torch.abs(logits[:, :, [1, 2, 3]]) - torch.abs(p_colors)))
        #loss_rgb = torch.log(1 - (torch.abs(logits[:, :, [1, 2, 3]] - p_colors)).sum(-1).mean)
        '''
        #print("loss rgb portion", loss_rgb / loss_p)


        return loss, loss_p, loss_rgb
