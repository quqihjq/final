import pdb
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import sys
import torch.distributed as dist
sys.path.append("./wrapper/bilateralfilter/build/lib.linux-x86_64-3.8")
# from bilateralfilter import bilateralfilter, bilateralfilter_batch

class PCL_Loss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07, margin=0.3):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.margin = margin  

    def forward(self, output_q, prototypes, flags):
        """
        output_q: (B, D)
        prototypes: (C, D)
        flags: (B,) → 样本所属类别 (1~C)，其中-1或0视为背景或无效样本
        """
        device = output_q.device
        num_cls = prototypes.size(0)
        valid_mask = (flags > 0)

        output_q = output_q[valid_mask]
        flags = flags[valid_mask]

        if output_q.size(0) == 0:
            return torch.tensor(0.0).to(device)

        logits = torch.matmul(output_q, prototypes.T) / self.temperature  # (B, C)

        pos_mask = torch.zeros_like(logits).to(device)
        for i, cls in enumerate(flags):
            pos_mask[i, cls - 1] = 1 

        neg_mask = 1 - pos_mask 

       
        pos_logits = logits * pos_mask
        exp_logits = torch.exp(logits - logits.max(dim=1, keepdim=True)[0])
        log_prob = pos_logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

       
        pos_loss = -(log_prob * pos_mask).sum() / (pos_mask.sum() + 1e-6)

       
        neg_logits = logits * neg_mask
        neg_max_margin = F.relu(neg_logits.max(dim=1)[0] - self.margin).mean()

      
        loss = pos_loss + neg_max_margin

        return loss

class LCL_Loss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def cal_pos_logit(self, flags, q_flags, logits, n_iter):
        mask_pos = flags == q_flags.T

        idx_m1, _ = torch.where(flags == -1)
        mask_pos[idx_m1] = 0

        logits[idx_m1] = 0  #
        pos = logits * mask_pos

        # filter wrong labeled pos pairs
        if n_iter >= 0:  # for stable determination

            mean_sim = pos.mean(dim=1, keepdim=True)
            _pos_index = pos >= mean_sim
            _wrong_pos = pos < mean_sim
            if (_wrong_pos.sum(dim=1) / (_pos_index.sum(dim=1) + 1e-6)).max() > 10:
                flags = flags.new_zeros(flags.size())
                pos = pos.new_zeros(pos.size())

        queue_index = torch.where(q_flags == -1)[0]
        logits[:, queue_index] = 0.0

        return pos, logits, flags

    def forward(self, output_q, q_all, flags, q_flags, n_iter):
        b = output_q.shape[0]  # batch size

        anchor_dot_contrast = torch.div(
            torch.matmul(output_q, q_all[b:].T), self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits_all = anchor_dot_contrast - logits_max.detach()

        pos, logits_all, flags = self.cal_pos_logit(flags, q_flags[b:], logits_all, n_iter)

        #  log_prob
        exp_logits = torch.exp(logits_all)
        exp_logits = exp_logits * (exp_logits != 1.0)
        log_prob = pos - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        loss = torch.tensor(0.0, device=output_q.device)

        nonzero_pos = pos != 0

        valid_mask = nonzero_pos.any(dim=1)  # [48]

        valid_log_prob = log_prob[valid_mask]
        valid_exp_logits = exp_logits.sum(dim=1, keepdim=True)[valid_mask]

        if valid_log_prob.size(0) > 0:
            loss = (valid_log_prob.sum() / valid_exp_logits.size(0))

        loss = - (self.temperature / self.base_temperature) * loss

        return loss, flags


def get_seg_loss(pred, label, ignore_index=255):
    ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    bg_label = label.clone()
    bg_label[label!=0] = ignore_index
    bg_sum = (bg_label != ignore_index).long().sum()
    # bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    bg_loss = ce(pred,bg_label.type(torch.long)).sum()/(bg_sum + 1e-6)
    fg_label = label.clone()
    fg_label[label==0] = ignore_index
    fg_sum = (fg_label != ignore_index).long().sum()
    # fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)
    fg_loss = ce(pred,fg_label.type(torch.long)).sum()/(fg_sum + 1e-6)

    return (bg_loss + fg_loss) * 0.5

def get_masked_ptc_loss(inputs, mask):
    b, c, h, w = inputs.shape
    
    inputs = inputs.reshape(b, c, h*w)

    def cos_sim(x):
        x = F.normalize(x, p=2, dim=1, eps=1e-8)
        cos_sim = torch.matmul(x.transpose(1,2), x)
        return torch.abs(cos_sim)

    inputs_cos = cos_sim(inputs)

    pos_mask = mask == 1
    neg_mask = mask == 0
    loss = 0.5*(1 - torch.sum(pos_mask * inputs_cos) / (pos_mask.sum()+1)) + 0.5 * torch.sum(neg_mask * inputs_cos) / (neg_mask.sum()+1)
    return loss

def get_seg_loss(pred, label, ignore_index=255):
    ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
    bg_label = label.clone()
    bg_label[label!=0] = ignore_index
    bg_sum = (bg_label != ignore_index).long().sum()
    # bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    bg_loss = ce(pred,bg_label.type(torch.long)).sum()/(bg_sum + 1e-6)
    fg_label = label.clone()
    fg_label[label==0] = ignore_index
    fg_sum = (fg_label != ignore_index).long().sum()
    # fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)
    fg_loss = ce(pred,fg_label.type(torch.long)).sum()/(fg_sum + 1e-6)

    return (bg_loss + fg_loss) * 0.5

def get_energy_loss(img, logit, label, img_box, loss_layer, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    pred_prob = F.softmax(logit, dim=1)

    if img_box is not None:
        crop_mask = torch.zeros_like(pred_prob[:, 0, ...])
        for idx, coord in enumerate(img_box):
            crop_mask[idx, coord[0]:coord[1], coord[2]:coord[3]] = 1
    else:
        crop_mask = torch.ones_like(pred_prob[:, 0, ...])

    _img = torch.zeros_like(img)
    _img[:,0,:,:] = img[:,0,:,:] * std[0] + mean[0]
    _img[:,1,:,:] = img[:,1,:,:] * std[1] + mean[1]
    _img[:,2,:,:] = img[:,2,:,:] * std[2] + mean[2]

    loss = loss_layer(_img, pred_prob, crop_mask, label.type(torch.uint8).unsqueeze(1), )

    return loss.cuda()

class DenseEnergyLoss(nn.Module):
    def __init__(self, weight, sigma_rgb, sigma_xy, scale_factor):
        super(DenseEnergyLoss, self).__init__()
        self.weight = weight
        self.sigma_rgb = sigma_rgb
        self.sigma_xy = sigma_xy
        self.scale_factor = scale_factor
    
    def forward(self, images, segmentations, ROIs, seg_label):
        """ scale imag by scale_factor """
        scaled_images = F.interpolate(images,scale_factor=self.scale_factor, recompute_scale_factor=True) 
        scaled_segs = F.interpolate(segmentations,scale_factor=self.scale_factor,mode='bilinear',align_corners=False, recompute_scale_factor=True)
        scaled_ROIs = F.interpolate(ROIs.unsqueeze(1),scale_factor=self.scale_factor, recompute_scale_factor=True).squeeze(1)
        scaled_seg_label = F.interpolate(seg_label,scale_factor=self.scale_factor,mode='nearest', recompute_scale_factor=True)
        unlabel_region = (scaled_seg_label.long() == 255).squeeze(1)

        return self.weight*DenseEnergyLossFunction.apply(
                scaled_images, scaled_segs, self.sigma_rgb, self.sigma_xy*self.scale_factor, scaled_ROIs, unlabel_region)
    
    def extra_repr(self):
        return 'sigma_rgb={}, sigma_xy={}, weight={}, scale_factor={}'.format(
            self.sigma_rgb, self.sigma_xy, self.weight, self.scale_factor
        )

class DenseEnergyLossFunction(Function):
    
    @staticmethod
    def forward(ctx, images, segmentations, sigma_rgb, sigma_xy, ROIs, unlabel_region):
        ctx.save_for_backward(segmentations)
        ctx.N, ctx.K, ctx.H, ctx.W = segmentations.shape
        Gate = ROIs.clone().to(ROIs.device)

        ROIs = ROIs.unsqueeze_(1).repeat(1,ctx.K,1,1)

        seg_max = torch.max(segmentations, dim=1)[0]
        Gate = Gate - seg_max
        Gate[unlabel_region] = 1
        Gate[Gate < 0] = 0
        Gate = Gate.unsqueeze_(1).repeat(1, ctx.K, 1, 1)

        segmentations = torch.mul(segmentations.cuda(), ROIs.cuda())
        ctx.ROIs = ROIs
        
        densecrf_loss = 0.0
        images = images.cpu().numpy().flatten()
        segmentations = segmentations.cpu().numpy().flatten()
        AS = np.zeros(segmentations.shape, dtype=np.float32)
        # bilateralfilter_batch(images, segmentations, AS, ctx.N, ctx.K, ctx.H, ctx.W, sigma_rgb, sigma_xy)
        Gate = Gate.cpu().numpy().flatten()
        AS = np.multiply(AS, Gate)
        densecrf_loss -= np.dot(segmentations, AS)
    
        # averaged by the number of images
        densecrf_loss /= ctx.N
        
        ctx.AS = np.reshape(AS, (ctx.N, ctx.K, ctx.H, ctx.W))
        return Variable(torch.tensor([densecrf_loss]), requires_grad=True)
        
    @staticmethod
    def backward(ctx, grad_output):
        grad_segmentation = -2*grad_output*torch.from_numpy(ctx.AS)/ctx.N
        grad_segmentation = grad_segmentation.cuda()
        grad_segmentation = torch.mul(grad_segmentation, ctx.ROIs.cuda())
        return None, grad_segmentation, None, None, None, None
    
