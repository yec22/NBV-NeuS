import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_efficient_distloss import flatten_eff_distloss
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug
import models
from models.utils import cleanup
from models.ray_utils import get_rays
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR, binary_cross_entropy, ScaleAndShiftInvariantLoss
import cv2, random
from math import sqrt
from utils.misc import config_to_primitive

@systems.register('neus-system')
class NeuSSystem(BaseSystem):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def prepare(self):
        self.criterions = {
            'psnr': PSNR()
        }
        self.train_num_samples = self.config.model.train_num_rays * (self.config.model.num_samples_per_ray + self.config.model.get('num_samples_per_ray_bg', 0))
        self.train_num_rays = self.config.model.train_num_rays

    def forward(self, batch):
        return self.model(batch['rays'])
    
    def preprocess_data(self, batch, stage):
        # determine index
        if 'index' in batch: # test / validation
            index = batch['index']
        else: # train
            if self.config.model.batch_image_sampling:
                index = torch.randint(0, len(self.dataset.all_images_train), size=(self.train_num_rays,), dtype=torch.int64, device=self.dataset.all_images_train.device)
            else:
                index = torch.randint(0, len(self.dataset.all_images_train), size=(1,), dtype=torch.int64, device=self.dataset.all_images_train.device)
        
        # generate data pairs
        if stage in ['train']: # train
            c2w = self.dataset.all_c2w_train[index]
            x = torch.randint(
                0, self.dataset.w, size=(self.train_num_rays,), device=self.dataset.all_images_train.device
            )
            y = torch.randint(
                0, self.dataset.h, size=(self.train_num_rays,), device=self.dataset.all_images_train.device
            )
            if self.dataset.directions_train.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions_train[y, x]
            elif self.dataset.directions_train.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions_train[index, y, x]
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images_train[index, y, x].view(-1, self.dataset.all_images_train.shape[-1]).to(self.rank)
            fg_mask = self.dataset.all_fg_masks_train[index, y, x].view(-1).to(self.rank)
            pred_depth = self.dataset.pred_depths_train[index, y, x].view(-1).to(self.rank)
        else: # validation / test
            c2w = self.dataset.all_c2w[index][0]
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index][0]
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            fg_mask = self.dataset.all_fg_masks[index].view(-1).to(self.rank)
            pred_depth = self.dataset.pred_depths[index].view(-1).to(self.rank)

        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)

        if stage in ['train']:
            if self.config.model.background_color == 'white':
                self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
            elif self.config.model.background_color == 'random':
                self.model.background_color = torch.rand((3,), dtype=torch.float32, device=self.rank)
            else:
                raise NotImplementedError
        else:
            self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
        
        if self.dataset.apply_mask:
            rgb = rgb * fg_mask[...,None] + self.model.background_color * (1 - fg_mask[...,None])

        batch.update({
            'rays': rays,
            'rgb': rgb,
            'fg_mask': fg_mask,
            'pred_depth': pred_depth
        })
    
    def training_step(self, batch, batch_idx):
        out = self(batch)
        batch_size = batch['fg_mask'].shape[0]
        reshape_batch_size = int(sqrt(batch_size)) * int(sqrt(batch_size))

        loss = 0.

        # update train_num_rays
        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(self.train_num_rays * (self.train_num_samples / out['num_samples_full'].sum().item()))        
            self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)

        loss_rgb_mse = F.mse_loss(out['comp_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]])
        self.log('train/loss_rgb_mse', loss_rgb_mse)
        loss += loss_rgb_mse * self.C(self.config.system.loss.lambda_rgb_mse)

        loss_rgb_l1 = F.l1_loss(out['comp_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]])
        self.log('train/loss_rgb', loss_rgb_l1)
        loss += loss_rgb_l1 * self.C(self.config.system.loss.lambda_rgb_l1)        

        loss_eikonal = ((torch.linalg.norm(out['sdf_grad_samples'], ord=2, dim=-1) - 1.)**2).mean()
        self.log('train/loss_eikonal', loss_eikonal)
        loss += loss_eikonal * self.C(self.config.system.loss.lambda_eikonal)

        if self.C2(self.config.system.loss.get('lambda_depth', 0)) > 0:
            depth_render = out['depth'][:reshape_batch_size,...].squeeze().reshape(int(sqrt(reshape_batch_size)), int(sqrt(reshape_batch_size)), 1).permute(2, 0, 1)
            depth_pred = batch['pred_depth'][:reshape_batch_size,...].reshape(int(sqrt(reshape_batch_size)), int(sqrt(reshape_batch_size)), 1).permute(2, 0, 1)
            depth_mask = (batch['fg_mask'] > 0.5) & (out['depth'].squeeze() > 1e-3)
            depth_mask = depth_mask[:reshape_batch_size,...].reshape(int(sqrt(reshape_batch_size)), int(sqrt(reshape_batch_size)), 1).permute(2, 0, 1)
            depth_render[depth_mask] = 1. / depth_render[depth_mask]
            loss_depth = ScaleAndShiftInvariantLoss()(depth_render, depth_pred, depth_mask)
            self.log('train/loss_depth', loss_depth)
            loss += loss_depth * self.C2(self.config.system.loss.get('lambda_depth', 0))
        
        lbd = self.config.system.loss.get('lambda_consis', 0)
        if isinstance(lbd, int) or isinstance(lbd, float):
            pass
        else:
            lbd = config_to_primitive(lbd)
        if isinstance(lbd, list) and len(lbd) == 2:
            lbd = lbd + [self.config.trainer.max_steps]
        if self.C(lbd) > 0:
            loss_consis = torch.mean(out['consis_constraint'])
            loss += loss_consis * self.C(lbd)

        opacity = torch.clamp(out['opacity'].squeeze(-1), 1.e-3, 1.-1.e-3)
        loss_mask = binary_cross_entropy(opacity, batch['fg_mask'].float())
        self.log('train/loss_mask', loss_mask)
        loss += loss_mask * (self.C(self.config.system.loss.lambda_mask) if self.dataset.has_mask else 0.0)

        loss_opaque = binary_cross_entropy(opacity, opacity)
        self.log('train/loss_opaque', loss_opaque)
        loss += loss_opaque * self.C(self.config.system.loss.lambda_opaque)

        loss_sparsity = torch.exp(-self.config.system.loss.sparsity_scale * out['sdf_samples'].abs()).mean()
        self.log('train/loss_sparsity', loss_sparsity)
        loss += loss_sparsity * self.C(self.config.system.loss.lambda_sparsity)

        if self.C(self.config.system.loss.lambda_curvature) > 0:
            assert 'sdf_laplace_samples' in out, "Need geometry.grad_type='finite_difference' to get SDF Laplace samples"
            loss_curvature = out['sdf_laplace_samples'].abs().mean()
            self.log('train/loss_curvature', loss_curvature)
            loss += loss_curvature * self.C(self.config.system.loss.lambda_curvature)

        # distortion loss proposed in MipNeRF360
        # an efficient implementation from https://github.com/sunset1995/torch_efficient_distloss
        if self.C(self.config.system.loss.lambda_distortion) > 0:
            loss_distortion = flatten_eff_distloss(out['weights'], out['points'], out['intervals'], out['ray_indices'])
            self.log('train/loss_distortion', loss_distortion)
            loss += loss_distortion * self.C(self.config.system.loss.lambda_distortion)    

        if self.config.model.learned_background and self.C(self.config.system.loss.lambda_distortion_bg) > 0:
            loss_distortion_bg = flatten_eff_distloss(out['weights_bg'], out['points_bg'], out['intervals_bg'], out['ray_indices_bg'])
            self.log('train/loss_distortion_bg', loss_distortion_bg)
            loss += loss_distortion_bg * self.C(self.config.system.loss.lambda_distortion_bg)        

        losses_model_reg = self.model.regularizations(out)
        for name, value in losses_model_reg.items():
            self.log(f'train/loss_{name}', value)
            loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            loss += loss_
        
        self.log('train/inv_s', out['inv_s'], prog_bar=True)
        self.log('train/num_rays', float(self.train_num_rays), prog_bar=True)

        return {
            'loss': loss
        }
    
    """
    # aggregate outputs from different devices (DP)
    def training_step_end(self, out):
        pass
    """
    
    """
    # aggregate outputs from different iterations
    def training_epoch_end(self, out):
        pass
    """
    
    def validation_step(self, batch, batch_idx):
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        
        opa_unc = torch.sum(torch.abs(out['opacity'][out['opacity'] < 0.5] - 0)) +\
                      torch.sum(torch.abs(out['opacity'][out['opacity'] >= 0.5] - 1))
        self.opacity_uncertainty.append(opa_unc)

        eik_unc = torch.sum(out['eik_unc'][out['rays_valid_full'][...,0]]) / torch.sum(out['rays_valid_full'][...,0])
        self.eikonal_uncertainty.append(eik_unc)

        W, H = self.dataset.img_wh
        self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
        ] + ([
            {'type': 'rgb', 'img': out['comp_rgb_bg'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        ] if self.config.model.learned_background else []) + [
            {'type': 'grayscale', 'img': out['eik_unc'].view(H, W), 'kwargs': {'cmap': 'jet'}},
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {'cmap': 'jet'}},
            {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
        ])
        return {
            'psnr': psnr,
            'index': batch['index']
        }
          
    
    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """
    
    def validation_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('val/psnr', psnr, prog_bar=True, rank_zero_only=True, sync_dist=True)
        
        if self.global_step < self.config.trainer.max_steps:
            if self.initial_view:
                ### IDLE: random sample from all candidate views
                # select_view = random.sample(self.candidate_views, 1)[0]

                ### Option 1: maximize opacity uncertainty
                # self.opacity_uncertainty = self.opacity_uncertainty[:len(self.candidate_views)]
                # max_uncertainty = max(self.opacity_uncertainty)
                # select_view = self.candidate_views[self.opacity_uncertainty.index(max_uncertainty)]
                
                ### Option 2: maximize eikonal uncertainty
                self.eikonal_uncertainty = self.eikonal_uncertainty[:len(self.candidate_views)]
                max_uncertainty = max(self.eikonal_uncertainty)
                select_view = self.candidate_views[self.eikonal_uncertainty.index(max_uncertainty)]

                # update
                self.candidate_views.remove(select_view)
                self.initial_view.append(select_view)
                self.dataset.update(self.initial_view, self.candidate_views)
                # clear cache
                self.opacity_uncertainty = []
                self.eikonal_uncertainty = []
                print(self.initial_view)
        else:
            if self.initial_view:
                with open(self.get_save_path("select_views.txt"), "w") as f:
                    f.write(str(self.initial_view))


    def test_step(self, batch, batch_idx):
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        W, H = self.dataset.img_wh
        self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
        ] + ([
            {'type': 'rgb', 'img': out['comp_rgb_bg'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        ] if self.config.model.learned_background else []) + [
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
        ])
        return {
            'psnr': psnr,
            'index': batch['index']
        }      
    
    def test_epoch_end(self, out):
        """
        Synchronize devices.
        Generate image sequence using test outputs.
        """
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('test/psnr', psnr, prog_bar=True, rank_zero_only=True, sync_dist=True)    

            self.save_img_sequence(
                f"it{self.global_step}-test",
                f"it{self.global_step}-test",
                '(\d+)\.png',
                save_format='mp4',
                fps=30
            )
            
            self.export()
    
    def export(self):
        mesh = self.model.export(self.config.export)
        self.save_mesh(
            f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.obj",
            **mesh
        )        
