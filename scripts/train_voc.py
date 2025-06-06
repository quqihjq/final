import argparse
import datetime
import logging
import os
import random
import sys
sys.path.append("")
import numpy as np
import torch
# import torch.distributed as dist
import torch.nn.functional as F
from datasets import voc
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from model.losses import (get_masked_ptc_loss, get_seg_loss, get_energy_loss,
                          PCL_Loss, LCL_Loss, DenseEnergyLoss)
from torch.nn.parallel import DistributedDataParallel
from model.PAR import PAR
from utils import imutils,evaluate
from utils.camutils import (cam_to_label, multi_scale_cam2, label_to_aff_mask, 
                            refine_cams_with_bkg_v2, assign_tags, cam_to_roi_mask)
from utils.pyutils import AverageMeter, cal_eta, setup_logger
from engine import CPDGL, build_optimizer, build_validation
parser = argparse.ArgumentParser()
torch.hub.set_dir("./pretrained")

### loss weight
parser.add_argument("--w_ptc", default=0.3, type=float, help="w_ptc")
parser.add_argument("--w_lil", default=0.5, type=float, help="w_lil")
parser.add_argument("--w_lig", default=0.5, type=float, help="w_lig")
parser.add_argument("--w_seg", default=0.12, type=float, help="w_seg")
parser.add_argument("--w_reg", default=0.05, type=float, help="w_reg")

### training utils
parser.add_argument("--max_iters", default=20000, type=int, help="max training iters")
parser.add_argument("--log_iters", default=200, type=int, help=" logging iters")
parser.add_argument("--eval_iters", default=2000, type=int, help="validation iters")
parser.add_argument("--warmup_iters", default=1500, type=int, help="warmup_iters")


parser.add_argument("--update_prototype", default=600, type=int, help="begin to update prototypes")
parser.add_argument("--cam2mask", default=10000, type=int, help="use mask from last layer")
parser.add_argument("--backbone", default='vit_base_patch16_224', type=str, help="vit_base_patch16_224")

### cam utils
parser.add_argument("--high_thre", default=0.7, type=float, help="high_bkg_score")
parser.add_argument("--low_thre", default=0.25, type=float, help="low_bkg_score")
parser.add_argument("--bkg_thre", default=0.5, type=float, help="bkg_score")
parser.add_argument("--tag_threshold", default=0.2, type=int, help="filter cls tags")
parser.add_argument("--cam_scales", default=(1.0, 0.5, 0.75, 1.5), help="multi_scales for cam")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")
parser.add_argument("--pooling", default='gmp', type=str, help="pooling choice for patch tokens")

### knowledge extraction
parser.add_argument('--proto_m', default=0.9, type=float, help='momentum for computing the momving average of prototypes')
parser.add_argument("--temp_lil", default=0.08, type=float, help="temp")
parser.add_argument("--base_temp_lil", default=0.08, type=float, help="temp")
parser.add_argument("--temp_lig", default=0.1, type=float, help="temp")
parser.add_argument("--base_temp_lig", default=0.1, type=float, help="temp")
parser.add_argument("--momentum", default=0.999, type=float, help="momentum")
parser.add_argument('--ctc-dim', default=768, type=int, help='embedding dimension')
parser.add_argument('--moco_queue', default=4608, type=int, help='queue size; number of negative samples')
parser.add_argument("--aux_layer", default=-3, type=int, help="aux_layer")

### log utils
parser.add_argument("--pretrained", default=True, type=bool, help="use imagenet pretrained weights")
parser.add_argument("--save_ckpt", default=True, action="store_true", help="save_ckpt")
parser.add_argument("--seed", default=0, type=int, help="fix random seed")
parser.add_argument("--work_dir", default="w_outputs", type=str, help="w_outputs")
parser.add_argument("--log_tag", default="train_voc", type=str, help="train_voc")

### dataset utils
parser.add_argument("--data_folder", default='/datasets/voc/VOCdevkit2012/VOC2012', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='/datasets/voc', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=21, type=int, help="number of classes")
parser.add_argument("--crop_size", default=448, type=int, help="crop_size in training")
parser.add_argument("--local_crop_size", default=64, type=int, help="crop_size for local view")
parser.add_argument('--ncrops', default=12, type=int, help='number of crops')
parser.add_argument("--train_set", default="train_aug", type=str, help="training split")
parser.add_argument("--val_set", default="val", type=str, help="validation split")
parser.add_argument("--spg", default=4, type=int, help="samples_per_gpu")
parser.add_argument("--scales", default=(0.5, 2), help="random rescale in training")

### optimizer utils
parser.add_argument("--optimizer", default='PolyWarmupAdamW', type=str, help="optimizer")
parser.add_argument("--lr", default=6e-5, type=float, help="learning rate")
parser.add_argument("--warmup_lr", default=1e-6, type=float, help="warmup_lr")
parser.add_argument("--wt_decay", default=1e-2, type=float, help="weights decay")
parser.add_argument("--betas", default=(0.9, 0.999), help="betas for Adam")
parser.add_argument("--power", default=0.9, type=float, help="poweer factor for poly scheduler")

parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
parser.add_argument("--num_workers", default=10, type=int, help="num_workers")
parser.add_argument('--backend', default='nccl')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def focal_loss(logits, labels, alpha=0.25, gamma=2.0):
    prob = torch.sigmoid(logits)
    pt = (1 - prob) * labels + prob * (1 - labels)
    focal_weight = (alpha * (1 - pt) ** gamma).detach()  # 关键：detach()
    return F.binary_cross_entropy_with_logits(logits, labels, weight=focal_weight)


def train(args=None):

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend, )
    logging.info("Total gpus: %d, samples per gpu: %d..."%(dist.get_world_size(), args.spg))

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)
    device = torch.device(args.local_rank)

    ### build model 
    model, param_groups = CPDGL(args)
    model.to(device)
    model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    ### build dataloader 
    train_dataset = voc.VOC12ClsDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.train_set,
        stage='train',
        aug=True,
        rescale_range=args.scales,
        crop_size=args.crop_size,
        img_fliplr=True,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    val_dataset = voc.VOC12SegDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.val_set,
        stage='val',
        aug=False,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.spg,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
        sampler=train_sampler,
        prefetch_factor=4)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=False,
                            drop_last=False)
    # train_sampler.set_epoch(np.random.randint(args.max_iters))
    train_loader_iter = iter(train_loader)
    avg_meter = AverageMeter()

    ### build optimizer 
    optim = build_optimizer(args,param_groups)
    logging.info('\nOptimizer: \n%s' % optim)

    loss_layer = DenseEnergyLoss(weight=1e-7, sigma_rgb=15, sigma_xy=100, scale_factor=0.5)
    par = PAR(num_iter=10, dilations=[1,2,4,8,12,24]).cuda()

    for n_iter in range(args.max_iters):
        try:
            img_name, inputs, cls_label, label_idx, img_box, raw_image, w_image, s_image = next(train_loader_iter)
        except:
            # train_sampler.set_epoch(np.random.randint(args.max_iters))
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_label, label_idx, img_box, raw_image, w_image, s_image = next(train_loader_iter)

        crops = [raw_image, w_image, s_image]
        inputs = inputs.to(device, non_blocking=True)
        inputs_denorm = imutils.denormalize_img2(inputs.clone())
        cls_label = cls_label.to(device, non_blocking=True)

        cams, cams_aux = multi_scale_cam2(model, inputs=inputs, scales=args.cam_scales)
        valid_cam, _ = cam_to_label(cams.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
        # # **新增这行代码，使得伪标签更平滑**
        # valid_cam = torch.sigmoid(valid_cam * 10)
        refined_pseudo_label = refine_cams_with_bkg_v2(par, inputs_denorm, cams=valid_cam, cls_labels=cls_label,  high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index, img_box=img_box, )

        ### descompose image to remove bias
        roi_mask_source = cams if n_iter >= args.cam2mask else cams_aux #2*20*448*448
        roi_mask = cam_to_roi_mask(roi_mask_source.detach(), cls_label=cls_label, img_box=img_box,ignore_mid=True,bkg_thre=args.bkg_thre,low_thre=args.low_thre, high_thre=args.high_thre,ignore_index=args.ignore_index)
        k, _ = assign_tags(images=crops[2], roi_mask=roi_mask, crop_num=args.ncrops-2, crop_size=args.local_crop_size, threshold=args.tag_threshold)
        q, u= assign_tags(images=crops[0], roi_mask=roi_mask, crop_num=args.ncrops-2, crop_size=args.local_crop_size,threshold=args.tag_threshold)
        roi_crops = crops[:2] + q + k #(2+10+10)
        u = u.reshape(-1,1).cuda() #24*1

        cls, segs, fmap, cls_aux, out_q, q_feats, q_flags, pro = model(inputs,label_idx=label_idx,crops=roi_crops,cls_flags_local=u, n_iter=n_iter)
        #2*20 2*21*28*28 2*768*28*28 2*20 24*768 4656*768 4656 20*768
        ## cls loss & aux cls loss
        cls_loss = F.multilabel_soft_margin_loss(cls, cls_label)
        cls_loss_aux = F.multilabel_soft_margin_loss(cls_aux, cls_label)

        # ## cls loss & aux cls loss (使用 Focal Loss)
        # cls_loss = focal_loss(cls, cls_label)
        # cls_loss_aux = focal_loss(cls_aux, cls_label)

        # 平滑更新 queue_feats_all
        ema_momentum = 0.99
        # 关闭梯度计算，确保 in-place 操作不会影响计算图
        with torch.no_grad():
            q_feats[:out_q.shape[0]] = (
                    ema_momentum * q_feats[:out_q.shape[0]] + (1 - ema_momentum) * out_q
            )
            q_feats[:out_q.shape[0]] = F.normalize(q_feats[:out_q.shape[0]], p=2, dim=1)

        ### tag-guided contrastive losses
        get_pcl = PCL_Loss(temperature=args.temp_lig, base_temperature=args.base_temp_lig).cuda()
        get_lcl = LCL_Loss(temperature=args.temp_lil, base_temperature=args.base_temp_lil).cuda()
        pcl_loss, cls_flags_local_revised = get_pcl(out_q, q_feats, u,q_flags,n_iter) #1 24
        lcl_loss = get_lcl(out_q, pro, cls_flags_local_revised)

        ### seg_loss & reg_loss
        segs = F.interpolate(segs, size=refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)
        seg_loss = get_seg_loss(segs, refined_pseudo_label.type(torch.long), ignore_index=args.ignore_index)
        reg_loss = get_energy_loss(img=inputs, logit=segs, label=refined_pseudo_label, img_box=img_box, loss_layer=loss_layer)

        ### aff loss from ToCo, https://github.com/rulixiang/ToCo
        resized_cams_aux = F.interpolate(cams_aux, size=fmap.shape[2:], mode="bilinear", align_corners=False)
        _, pseudo_label_aux = cam_to_label(resized_cams_aux.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
        aff_mask = label_to_aff_mask(pseudo_label_aux)
        ptc_loss = get_masked_ptc_loss(fmap, aff_mask)

        # warmup
        if n_iter <= 1000:
            loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + args.w_ptc * ptc_loss + 0.0 * pcl_loss + 0.0 * lcl_loss + 0.0 * seg_loss + 0.0 * reg_loss
        elif n_iter <= 2000:
            loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + args.w_ptc * ptc_loss + args.w_lig * pcl_loss + args.w_lil * lcl_loss + 0.0 * seg_loss + 0.0 * reg_loss
        else:
            # 计算自适应损失权重
            alpha = min(1.0, n_iter / 5000)
            w_lig = alpha * args.w_lig
            w_lil = alpha * args.w_lil
            w_seg = alpha * args.w_seg
            w_reg = alpha * args.w_reg
            loss = (
                    1.0 * cls_loss + 1.0 * cls_loss_aux +
                    args.w_ptc * ptc_loss + w_lig * pcl_loss +
                    w_lil * lcl_loss + w_seg * seg_loss +
                    w_reg * reg_loss
            )
            # loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + args.w_ptc * ptc_loss + args.w_lig * lig_loss + args.w_lil * lil_loss + args.w_seg * seg_loss + args.w_reg * reg_loss

        cls_pred = (cls > 0).type(torch.int16)
        cls_score = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])

        avg_meter.add({
            'cls_loss': cls_loss.item(),
            'ptc_loss': ptc_loss.item(),
            'cls_loss_aux': cls_loss_aux.item(),
            'seg_loss': seg_loss.item(),
            'cls_score': cls_score.item(),
            'pcl_loss': pcl_loss.item(),
            'lcl_loss': lcl_loss.item()
        })

        optim.zero_grad()
        loss.backward()
        optim.step()

        if (n_iter + 1) % args.log_iters == 0:

            delta, eta = cal_eta(time0, n_iter + 1, args.max_iters)
            cur_lr = optim.param_groups[0]['lr']

            if args.local_rank == 0:
                logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.4f, cls_loss_aux: %.4f, ptc_loss: %.4f, pcl_loss: %.4f, lcl_loss: %.4f, seg_loss: %.4f..." % (n_iter + 1, delta, eta, cur_lr, \
                                                        avg_meter.pop('cls_loss'), avg_meter.pop('cls_loss_aux'), avg_meter.pop('ptc_loss'), avg_meter.pop('pcl_loss'),avg_meter.pop('lcl_loss'), avg_meter.pop('seg_loss')))

        if (n_iter + 1) % args.eval_iters == 0 and (n_iter + 1) >= 1:
            ckpt_name = os.path.join(args.ckpt_dir, "model_iter_%d.pth" % (n_iter + 1))
            if args.local_rank == 0:
                logging.info('Validating...')
                if args.save_ckpt:
                    torch.save(model.state_dict(), ckpt_name)
            val_cls_score, tab_results = build_validation(model=model, data_loader=val_loader, args=args)
            if args.local_rank == 0:
                logging.info("val cls score: %.6f" % (val_cls_score))
                logging.info("\n"+tab_results)

    return True


if __name__ == "__main__":

    args = parser.parse_args()
    timestamp_1 = "{0:%Y-%m}".format(datetime.datetime.now())
    timestamp_2 = "{0:%d-%H-%M-%S}".format(datetime.datetime.now())
    exp_tag = f'{args.log_tag}_{timestamp_2}'
    args.work_dir = os.path.join(args.work_dir, timestamp_1, exp_tag)
    args.ckpt_dir = os.path.join(args.work_dir, "checkpoints")
    args.pred_dir = os.path.join(args.work_dir, "predictions")

    if args.local_rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.pred_dir, exist_ok=True)

        setup_logger(filename=os.path.join(args.work_dir, 'train_pro_queue_feats_wloss_700.log'))
        logging.info('Pytorch version: %s' % torch.__version__)
        logging.info("GPU type: %s"%(torch.cuda.get_device_name(0)))
        logging.info('\nargs: %s' % args)

    ## fix random seed
    setup_seed(args.seed)
    train(args=args)
