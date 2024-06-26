import os
import sys
import json
import numpy as np
import time
from collections import defaultdict
from tqdm import tqdm
from easydict import EasyDict
import pprint

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.logger import LOGGER, TB_LOGGER, AverageMeter, RunningMeter, add_log_to_file
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, set_random_seed, set_cuda, wrap_model
from utils.distributed import all_gather

from optim import get_lr_sched_decay_rate
from optim.misc import build_optimizer

from parser import load_parser, parse_with_config

from data.eslabel_dataset import ESLabelDataset, eslabel_collate_fn
from data.espcd_dataset import ESPcdDataset, espcd_collate_fn

from model.referit3d_net import ReferIt3DNet



def build_gtlabel_datasets(data_cfg):
    trn_dataset = ESLabelDataset(
        es_info_file="/mnt/hwfile/OpenRobotLab/lvruiyuan/embodiedscan_infos/embodiedscan_infos_train.pkl",
        # vg_raw_data_file="/mnt/hwfile/OpenRobotLab/lvruiyuan/es_gen_text/VG.json",
        vg_raw_data_file='/mnt/hwfile/OpenRobotLab/lvruiyuan/es_gen_text/vg_full/VG_train_10Percent_flattened_token_positive.json',
        cat2vec_file='../datasets/referit3d/annotations/meta_data/cat2vec.json',
        processed_scan_dir="../datasets/referit3d/scan_data_new"
    )
    val_dataset = ESLabelDataset(
        es_info_file="/mnt/hwfile/OpenRobotLab/lvruiyuan/embodiedscan_infos/embodiedscan_infos_val.pkl",
        # vg_raw_data_file="/mnt/hwfile/OpenRobotLab/lvruiyuan/es_gen_text/VG.json",
        vg_raw_data_file='/mnt/hwfile/OpenRobotLab/lvruiyuan/es_gen_text/vg_full/VG_val_5Percent_flattened_token_positive.json',
        cat2vec_file='../datasets/referit3d/annotations/meta_data/cat2vec.json',
        processed_scan_dir="../datasets/referit3d/scan_data_new"
    )
    return trn_dataset, val_dataset

# def build_gtpcd_datasets(data_cfg):
#     trn_dataset = ESPcdDataset(
#         data_cfg.trn_scan_split, data_cfg.anno_file, 
#         data_cfg.scan_dir, data_cfg.category_file,
#         cat2vec_file=data_cfg.cat2vec_file, 
#         random_rotate=data_cfg.random_rotate,
#         max_txt_len=data_cfg.max_txt_len, max_obj_len=data_cfg.max_obj_len,
#         keep_background=data_cfg.keep_background,
#         num_points=data_cfg.num_points, in_memory=True,
#     )
#     val_dataset = ESPcdDataset(
#         data_cfg.val_scan_split, data_cfg.anno_file, 
#         data_cfg.scan_dir, data_cfg.category_file,
#         cat2vec_file=data_cfg.cat2vec_file,
#         max_txt_len=None, max_obj_len=None, random_rotate=False,
#         keep_background=data_cfg.keep_background,
#         num_points=data_cfg.num_points, in_memory=True,
#     )
#     return trn_dataset, val_dataset


def main(opts):
    default_gpu, n_gpu, device = set_cuda(opts)

    if default_gpu:
        LOGGER.info(
            'device: {} n_gpu: {}, distributed training: {}'.format(
                device, n_gpu, bool(opts.local_rank != -1)
            )
        )
 
    seed = opts.seed
    if opts.local_rank != -1:
        seed += opts.rank
    set_random_seed(seed)

    if default_gpu:
        if not opts.test:
            save_training_meta(opts)
            TB_LOGGER.create(os.path.join(opts.output_dir, 'logs'))
            model_saver = ModelSaver(os.path.join(opts.output_dir, 'ckpts'))
            add_log_to_file(os.path.join(opts.output_dir, 'logs', 'log.txt'))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()

    # Prepare model
    model_config = EasyDict(opts.model)
    model = ReferIt3DNet(model_config, device)

    num_weights, num_trainable_weights = 0, 0
    for p in model.parameters():
        psize = np.prod(p.size())
        num_weights += psize
        if p.requires_grad:
            num_trainable_weights += psize 
    LOGGER.info('#weights: %d, #trainable weights: %d', num_weights, num_trainable_weights)

    if opts.resume_files:
        checkpoint = {}
        for resume_file in opts.resume_files:
            new_checkpoints = torch.load(resume_file, map_location=lambda storage, loc: storage)
            for k, v in new_checkpoints.items():
                if k not in checkpoint:
                    checkpoint[k] = v
        print('resume #params:', len(checkpoint), len([n for n in checkpoint.keys() if n in model.state_dict()]))
        for n in checkpoint.keys():
            if n not in model.state_dict():
                print(n, checkpoint[n].size())
        model.load_state_dict(checkpoint, strict=False)
    
    model_cfg = model.config
    model = wrap_model(model, device, opts.local_rank)

    # load data training set
    data_cfg = EasyDict(opts.dataset)
    if model_config.model_type == 'gtlabel':
        trn_dataset, val_dataset = build_gtlabel_datasets(data_cfg)
        collate_fn = eslabel_collate_fn
    elif model_config.model_type == 'gtpcd':
        trn_dataset, val_dataset = build_gtpcd_datasets(data_cfg)
        collate_fn = espcd_collate_fn
    LOGGER.info('train #scans %d, #data %d' % (len(trn_dataset.scan_ids), len(trn_dataset)))
    LOGGER.info('val #scans %d, #data %d' % (len(val_dataset.scan_ids), len(val_dataset)))

    # Build data loaders
    if opts.local_rank == -1:
        trn_sampler = None
        pre_epoch = lambda e: None
        real_batch_size = opts.batch_size
    else:
        size = dist.get_world_size()
        trn_sampler = DistributedSampler(
            trn_dataset, num_replicas=size, rank=dist.get_rank(), shuffle=True
        )
        pre_epoch = trn_sampler.set_epoch
        real_batch_size = opts.batch_size * size

    trn_dataloader = DataLoader(
        trn_dataset, batch_size=opts.batch_size, shuffle=True if trn_sampler is None else False, 
        num_workers=opts.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=False, prefetch_factor=1,
        sampler=trn_sampler
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=opts.batch_size, shuffle=False, 
        num_workers=opts.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=False, prefetch_factor=1,
    )
    opts.num_train_steps = len(trn_dataloader) * opts.num_epoch

    if opts.test:
        val_log, out_preds = validate(model, model_cfg, val_dataloader, return_preds=True)
        pred_dir = os.path.join(opts.output_dir, 'preds')
        os.makedirs(pred_dir, exist_ok=True)
        json.dump(out_preds, open(os.path.join(pred_dir, 'val_outs.json'), 'w'))
        return

    # Prepare optimizer
    optimizer, init_lrs = build_optimizer(model, opts)
    if opts.resume_optimizer is not None:
        optimizer_state = torch.load(opts.resume_optimizer)
        print('load optimizer epoch: %d, weights: %d' % (
            optimizer_state['epoch'], len(optimizer_state['optimizer']))
        )
        optimizer.load_state_dict(optimizer_state['optimizer'])

    LOGGER.info(f"***** Running training with {opts.world_size} GPUs *****")
    LOGGER.info("  Batch size = %d", opts.batch_size if opts.local_rank == -1 else opts.batch_size * opts.world_size)
    LOGGER.info("  Num epoch = %d, num steps = %d", opts.num_epoch, opts.num_train_steps)

    # to compute training statistics
    avg_metrics = defaultdict(AverageMeter)

    global_step = 0

    model.train()
    optimizer.zero_grad()
    optimizer.step()

    if default_gpu:
        val_log = validate(model, model_cfg, val_dataloader)
    
    val_best_scores =  {'epoch': -1, 'acc/og3d': -float('inf')}
    epoch_iter = range(opts.num_epoch)
    if default_gpu:
        epoch_iter = tqdm(epoch_iter)
    for epoch in epoch_iter:
        pre_epoch(epoch)    # for distributed

        start_time = time.time()
        batch_iter = trn_dataloader
        if default_gpu:
            batch_iter = tqdm(batch_iter)
        for batch in batch_iter:
            item_ids = batch['item_ids'] # list ['nr3d_012277', ...], len=128
            scan_ids = batch['scan_ids'] # list ['scene0520_00', ...], len=128
            txt_ids = batch['txt_ids'] # torch.int64([128, 28])
            txt_lens = batch['txt_lens'] # torch.int64([128])
            obj_fts = batch['obj_fts'] # torch.float32([128, 69, 1024, 6])
            obj_locs = batch['obj_locs'] # torch.float32([128, 69, 6])
            obj_colors = batch['obj_colors'] # torch.float32([128, 69, 3, 4])
            obj_lens = batch['obj_lens'] # torch.int64([128])
            obj_classes = batch['obj_classes'] # torch.int64([128, 69])
            tgt_obj_idxs = batch['tgt_obj_idxs'] # torch.int64([128])
            tgt_obj_classes = batch['tgt_obj_classes'] # torch.int64([128])
            obj_ids = batch['obj_ids'] # list[ list['0', '2', '7', ...] ], len=128
            txt_masks = batch['txt_masks'] # torch.bool([128, 28])
            obj_masks = batch['obj_masks'] # torch.bool([128, 69])
            batch_size = len(batch['scan_ids'])
            result, losses = model(batch, compute_loss=True)
            losses['total'].backward()

            # optimizer update and logging
            global_step += 1
            # learning rate scheduling:
            lr_decay_rate = get_lr_sched_decay_rate(global_step, opts)
            for kp, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = lr_this_step = init_lrs[kp] * lr_decay_rate
            TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

            # log loss
            # NOTE: not gathered across GPUs for efficiency
            loss_dict = {'loss/%s'%lk: lv.data.item() for lk, lv in losses.items()}
            for lk, lv in loss_dict.items():
                avg_metrics[lk].update(lv, n=batch_size)
            TB_LOGGER.log_scalar_dict(loss_dict)
            TB_LOGGER.step()

            # update model params
            if opts.grad_norm != -1:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), opts.grad_norm
                )
                TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
            optimizer.step()
            optimizer.zero_grad()
            # break
            
        LOGGER.info(
            'Epoch %d, lr: %.6f, %s', epoch+1,  
            optimizer.param_groups[-1]['lr'],
            ', '.join(['%s: %.4f'%(lk, lv.avg) for lk, lv in avg_metrics.items()])
        )
        if default_gpu and (epoch+1) % opts.val_every_epoch == 0:
            LOGGER.info(f'------Epoch {epoch+1}: start validation (val)------')
            val_log = validate(model, model_cfg, val_dataloader)
            TB_LOGGER.log_scalar_dict(
                {f'valid/{k}': v.avg for k, v in val_log.items()}
            )
            output_model_file = model_saver.save(
                model, epoch+1, optimizer=optimizer, save_latest_optim=True
            )
            if val_log['acc/og3d'].avg > val_best_scores['acc/og3d']:
                val_best_scores['acc/og3d'] = val_log['acc/og3d'].avg
                val_best_scores['epoch'] = epoch + 1
                model_saver.remove_previous_models(epoch+1)
            else:
                os.remove(output_model_file)    
    
    LOGGER.info('Finished training!')
    LOGGER.info(
        'best epoch: %d, best acc/og3d %.4f', val_best_scores['epoch'], val_best_scores['acc/og3d']
    )

@torch.no_grad()
def validate(model, model_cfg, val_dataloader, niters=None, return_preds=False):
    model.eval()

    output_attentions = True
    output_attentions = False
        
    avg_metrics = defaultdict(AverageMeter)
    out_preds = {}
    for ib, batch in enumerate(val_dataloader):
        batch_size = len(batch['scan_ids'])
        result, losses = model(
            batch, compute_loss=True, is_test=True,
            output_attentions=output_attentions, 
            output_hidden_states=False,
        )

        loss_dict = {'loss/%s'%lk: lv.data.item() for lk, lv in losses.items()}
        for lk, lv in loss_dict.items():
            avg_metrics[lk].update(lv, n=batch_size)

        # og3d_preds = torch.argmax(result['og3d_logits'], dim=1).cpu()
        og3d_preds = (result['og3d_logits'] > 0).cpu().numpy()
        tgt_mask = np.zeros_like(og3d_preds)
        for i, instance_idxs in enumerate(batch['tgt_obj_idxs']):
            for idx in instance_idxs:
                if idx >= 0:  
                    tgt_mask[i, idx] = 1    
        successful_preds = (og3d_preds == tgt_mask).sum(axis=-1)
        obj_lens = batch['obj_lens'].cpu().numpy()
        max_len = obj_lens.max()
        successful_preds -= max_len - obj_lens

        avg_metrics['acc/og3d'].update(
            successful_preds.sum(),
            n=obj_lens.sum()
        )
        # avg_metrics['acc/og3d_class'].update(
        #     torch.mean((batch['obj_classes'].gather(1, og3d_preds.unsqueeze(1)).squeeze(1) == batch['tgt_obj_classes']).float()).item(),
        #     n=batch_size
        # )
        if model_cfg.losses.obj3d_clf > 0:
            obj3d_clf_preds = torch.argmax(result['obj3d_clf_logits'], dim=2).cpu()
            avg_metrics['acc/obj3d_clf'].update(
                (obj3d_clf_preds[batch['obj_masks']] == batch['obj_classes'][batch['obj_masks']]).float().mean().item(),
                n=batch['obj_masks'].sum().item()
            )
        if model_cfg.losses.obj3d_clf_pre > 0:
            obj3d_clf_preds = torch.argmax(result['obj3d_clf_pre_logits'], dim=2).cpu()
            avg_metrics['acc/obj3d_clf_pre'].update(
                (obj3d_clf_preds[batch['obj_masks']] == batch['obj_classes'][batch['obj_masks']]).float().mean().item(),
                n=batch['obj_masks'].sum().item()
            )
        # if model_cfg.losses.txt_clf > 0:
        #     txt_clf_preds = torch.argmax(result['txt_clf_logits'], dim=1).cpu()
        #     avg_metrics['acc/txt_clf'].update(
        #         (txt_clf_preds == batch['tgt_obj_classes']).float().mean().item(),
        #         n=batch_size
        #     )
            
        if return_preds:
            for ib in range(batch_size):
                out_preds[batch['item_ids'][ib]] = {
                    'obj_ids': batch['obj_ids'][ib],
                    'obj_logits': result['og3d_logits'][ib, :batch['obj_lens'][ib]].data.cpu().numpy().tolist(),
                }
                if output_attentions:
                    out_preds[batch['item_ids'][ib]].update({
                        'all_self_attns': [x[:, ib, :batch['obj_lens'][ib], :batch['obj_lens'][ib]].data.cpu().numpy().tolist() for x in result['all_self_attns']],
                        'all_cross_attns': [x[ib, :batch['obj_lens'][ib], :batch['txt_lens'][ib]].data.cpu().numpy().tolist() for x in result['all_cross_attns']],
                    })
        if niters is not None and ib >= niters:
            break
    LOGGER.info(', '.join(['%s: %.4f'%(lk, lv.avg) for lk, lv in avg_metrics.items()]))

    model.train()
    if return_preds:
        return avg_metrics, out_preds
    return avg_metrics


def build_args():
    parser = load_parser()
    opts = parse_with_config(parser)

    if os.path.exists(opts.output_dir) and os.listdir(opts.output_dir):
        LOGGER.warning(
            "Output directory ({}) already exists and is not empty.".format(
                opts.output_dir
            )
        )

    return opts

if __name__ == '__main__':
    args = build_args()
    pprint.pprint(args)
    main(args)
