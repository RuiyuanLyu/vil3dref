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

from data.eslabelpcd_dataset import ESLabelPcdDataset, eslabelpcd_collate_fn
from data.eslabelpcd_dataset_eval import ESLabelPcdDatasetEval

from model.referit3d_net_mix import ReferIt3DNetMix



def build_datasets(data_cfg):
    trn_dataset = ESLabelPcdDataset(
        es_info_file="/mnt/hwfile/OpenRobotLab/lvruiyuan/embodiedscan_infos/embodiedscan_infos_train.pkl",
        vg_raw_data_file='/mnt/hwfile/OpenRobotLab/lvruiyuan/es_gen_text/vg_full/VG_train_20Percent_flattened_token_positive.json',
        cat2vec_file='../datasets/referit3d/annotations/meta_data/cat2vec.json',
        processed_scan_dir="../datasets/referit3d/scan_data_new"
    )
    val_dataset = ESLabelPcdDatasetEval(
        es_info_file="/mnt/hwfile/OpenRobotLab/lvruiyuan/embodiedscan_infos/embodiedscan_infos_val.pkl", 
        vg_raw_data_file='/mnt/hwfile/OpenRobotLab/lvruiyuan/es_gen_text/vg_full/VG_val_1Percent_flattened_token_positive.json',
        cat2vec_file='../datasets/referit3d/annotations/meta_data/cat2vec.json',
        processed_scan_dir="../datasets/referit3d/scan_data_new"
    )
    return trn_dataset, val_dataset


def main(opts):
    # torch.autograd.set_detect_anomaly(True)

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
    LOGGER.info("preparing model...")
    model = ReferIt3DNetMix(model_config, device)

    num_weights, num_trainable_weights = 0, 0
    LOGGER.info("computing model statistics...")
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
        if len(opts.resume_files) == 1:
            model.load_state_dict(checkpoint)
        print(
            'resume #params:', len(checkpoint), 
            len([n for n in checkpoint.keys() if n in model.teacher_model.state_dict()]),
            len([n for n in checkpoint.keys() if n in model.student_model.state_dict()]),
        )
        
        # TODO: the current state dict do not match due to 9 dof feature.        
        model.teacher_model.load_state_dict(checkpoint, strict=False)
        if opts.resume_student:
            model.student_model.load_state_dict(checkpoint, strict=False)
        else:
            student_checkpoint = torch.load(
                opts.resume_files[0], map_location=lambda storage, loc: storage
            )
            print('resume_student', len(student_checkpoint))
            model.student_model.load_state_dict(student_checkpoint, strict=False)


    model_cfg = model.config
    model = wrap_model(model, device, opts.local_rank)

    # load data training set
    data_cfg = EasyDict(opts.dataset)
    trn_dataset, val_dataset = build_datasets(data_cfg)
    collate_fn = eslabelpcd_collate_fn
    LOGGER.info('train #scans %d, #data %d' % (len(trn_dataset.scan_ids), len(trn_dataset)))
    LOGGER.info('val #scans %d, #data %d' % (len(val_dataset.scan_ids), len(val_dataset)))

    # Build data loaders
    if opts.local_rank == -1:
        trn_sampler = None
        pre_epoch = lambda e: None
    else:
        size = dist.get_world_size()
        trn_sampler = DistributedSampler(
            trn_dataset, num_replicas=size, rank=dist.get_rank(), shuffle=True
        )
        pre_epoch = trn_sampler.set_epoch

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
        # np.save(os.path.join(pred_dir, 'val_outs.npy'), out_preds)
        json.dump(out_preds, open(os.path.join(pred_dir, 'val_outs.json'), 'w'))
        return

    # Prepare optimizer
    optimizer, init_lrs = build_optimizer(model, opts)

    LOGGER.info(f"***** Running training with {opts.world_size} GPUs *****")
    LOGGER.info("  Batch size = %d", opts.batch_size if opts.local_rank == -1 else opts.batch_size * opts.world_size)
    LOGGER.info("  Num epoch = %d, num steps = %d", opts.num_epoch, opts.num_train_steps)

    # to compute training statistics
    avg_metrics = defaultdict(AverageMeter)

    global_step = 0

    model.train()
    optimizer.zero_grad()
    optimizer.step()

    print("start validation?")
    if default_gpu:
        val_log, out_preds = validate(model, model_cfg, val_dataloader, return_preds=True)
        # out_preds["esvg_11740"] has key 'obj_ids', 'obj_logits'
        # both lists, length can be 53.
        # import pdb; pdb.set_trace()
    
    val_best_scores =  {'epoch': -1, 'acc/og3d': 0}
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
            item_ids = batch['item_ids'] # list ['nr3d_012277', ...], len=64
            scan_ids = batch['scan_ids'] # list ['scene0520_00', ...], len=64
            txt_ids = batch['txt_ids'] # torch.int64([64, 28])
            txt_lens = batch['txt_lens'] # torch.int64([64])
            obj_gt_fts = batch['obj_gt_fts'] # torch.float32([64, 69, 300])
            obj_fts = batch['obj_fts'] # torch.float32([64, 69, 1024, 6])
            obj_locs = batch['obj_locs'] # torch.float32([64, 69, 6])
            obj_colors = batch['obj_colors'] # torch.float32([64, 69, 3, 4])
            obj_lens = batch['obj_lens'] # torch.int64([64])
            obj_classes = batch['obj_classes'] # torch.int64([64, 69])
            tgt_obj_idxs = batch['tgt_obj_idxs'] # torch.int64([64])
            tgt_obj_classes = batch['tgt_obj_classes'] # torch.int64([64])
            obj_ids = batch['obj_ids'] # list[ list['0', '2', '7', ...] ], len=64
            txt_masks = batch['txt_masks'] # torch.bool([64, 28])
            obj_masks = batch['obj_masks'] # torch.bool([64, 69])
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
            val_log, out_preds = validate(model, model_cfg, val_dataloader, return_preds=True)
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
        
    avg_metrics = defaultdict(AverageMeter)
    out_preds = {}
    gt_keeps = {}
    for ib, batch in enumerate(val_dataloader):
        batch_size = len(batch['scan_ids'])
        # print(batch['item_ids'][0])
        result, losses = model(batch, compute_loss=True, is_test=True)
        # for k, v in losses.items():
        #     loss = v.clone()  # 确保损失是可写的
        #     dist.all_reduce(loss, op=dist.ReduceOp.SUM)  # 求和
        #     loss /= dist.get_world_size()  # world_size是进程总数
        #     losses[k] = loss
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
        #     torch.mean((batch['obj_classes'].gather(1, og3d_preds.unsqueeze(1)).squeeze(1) == batch['tgt_obj_classes'][:, 0]).float()).item(),
        #     n=batch_size
        # )
        if model_cfg.losses.obj3d_clf:
            obj3d_clf_preds = torch.argmax(result['obj3d_clf_logits'], dim=2).cpu()
            avg_metrics['acc/obj3d_clf'].update(
                (obj3d_clf_preds[batch['obj_masks']] == batch['obj_classes'][batch['obj_masks']]).float().mean().item(),
                n=batch['obj_masks'].sum().item()
            )
        if model_cfg.losses.obj3d_clf_pre:
            obj3d_clf_preds = torch.argmax(result['obj3d_clf_pre_logits'], dim=2).cpu()
            avg_metrics['acc/obj3d_clf_pre'].update(
                (obj3d_clf_preds[batch['obj_masks']] == batch['obj_classes'][batch['obj_masks']]).float().mean().item(),
                n=batch['obj_masks'].sum().item()
            )
        if model_cfg.losses.txt_clf:
            txt_clf_preds = torch.argmax(result['txt_clf_logits'], dim=1).cpu()
            avg_metrics['acc/txt_clf'].update(
                (txt_clf_preds == batch['tgt_obj_classes'][:, 0]).float().mean().item(),
                n=batch_size
            )
        if model_cfg.losses.get('rot_clf', False):
            for il in model_cfg.mm_encoder.rot_layers:
                # rot_clf_preds = torch.argmax(result['all_rot_logits'][il], dim=1)
                rot_clf_preds = result['all_rot_preds'][il].cpu()
                gt_views = batch['target_views']
                gt_view_mask = gt_views != -100
                avg_metrics['acc/rot_clf_%d'%il].update(
                    (rot_clf_preds == gt_views)[gt_view_mask].float().mean().item(),
                    n=torch.sum(gt_view_mask)
                )
        if model_cfg.losses.get('txt_contrast', False):
            txt_ctr_preds = result['txt_pos_sims'] > torch.max(result['txt_neg_sims'], 1)[0]
            txt_ctr_preds = txt_ctr_preds.cpu().float()
            avg_metrics['acc/txt_contrast'].update(
                txt_ctr_preds.mean().item(),
                n=batch_size
            )
        # if return_preds:
        if 1: #HACK: always return preds
            for ib in range(batch_size):
                out_preds[batch['item_ids'][ib]] = {
                    'obj_ids': batch['obj_ids'][ib],
                    'obj_logits_gpu': result['og3d_logits'][ib, :batch['obj_lens'][ib]].data,
                    'obj_logits': result['og3d_logits'][ib, :batch['obj_lens'][ib]].data.cpu().numpy().tolist(),
                    'bboxes_3d': batch['obj_locs'][ib][:batch['obj_lens'][ib]],
                }
                tgt_num = int(batch['tgt_num'][ib])
                gt_keeps[batch['item_ids'][ib]] = {
                    'sub_class': batch['sub_class'][ib],
                    'gt_bboxes_3d': batch['tgt_obj_boxes'][ib][:tgt_num],
                }
        if niters is not None and ib >= niters:
            break

    from utils.grounding_metric import ground_eval
    gt_anno_list = []
    det_anno_list = []
    for k, v in out_preds.items():
        gt_anno_list.append(gt_keeps[k])
        det_anno_list.append({
            'target_scores_3d': v['obj_logits_gpu'],
            'bboxes_3d': v['bboxes_3d'],
        })
    eval_result = ground_eval(gt_anno_list, det_anno_list, LOGGER)
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

import subprocess
if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = build_args()
    pprint.pprint(args)
    main(args)
