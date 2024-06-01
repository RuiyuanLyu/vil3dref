import os
import json
import numpy as np
import random

from transformers import AutoTokenizer

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
try:
    from .common import pad_tensors, gen_seq_masks
except:
    from common import pad_tensors, gen_seq_masks


# target: to make batches.

# The batch is a dictionary containing the following keys:
# item_ids = batch['item_ids'] # list ['nr3d_012277', ...], len=64
# scan_ids = batch['scan_ids'] # list ['scene0520_00', ...], len=64
# txt_ids = batch['txt_ids'] # torch.int64([64, 28])
# txt_lens = batch['txt_lens'] # torch.int64([64])
# obj_fts = batch['obj_fts'] # torch.float32([64, 69, 1024, 6])
# obj_locs = batch['obj_locs'] # torch.float32([64, 69, 6])
# obj_colors = batch['obj_colors'] # torch.float32([64, 69, 3, 4])
# obj_lens = batch['obj_lens'] # torch.int64([64])
# obj_classes = batch['obj_classes'] # torch.int64([64, 69])
# tgt_obj_idxs = batch['tgt_obj_idxs'] # torch.int64([64])
# tgt_obj_classes = batch['tgt_obj_classes'] # torch.int64([64])
# obj_ids = batch['obj_ids'] # list[ list['0', '2', '7', ...] ], len=64
# txt_masks = batch['txt_masks'] # torch.bool([64, 28])
# obj_masks = batch['obj_masks'] # torch.bool([64, 69])
from utils.utils_read import read_es_infos, apply_mapping_to_keys, RAW2NUM_3RSCAN, RAW2NUM_MP3D, NUM2RAW_3RSCAN, NUM2RAW_MP3D, to_sample_idx, to_scene_id, to_list_of_int, to_list_of_str

class ESLabelDataset(Dataset):
    def __init__(self, es_info_file, vg_raw_data_file, cat2vec_file, processed_scan_dir):
        super().__init__()
        # from data.glove_embedding import get_glove_word2vec
        # self.cat2vec = get_glove_word2vec(glove_embedding_file)
        self.cat2vec = json.load(open(cat2vec_file, 'r'))
        self.cat2vec["steps"] = self.cat2vec["step"]
        self.word2vec_vocab = list(self.cat2vec.keys())
        count_type_from_zero = True
        self.es_info = read_es_infos(es_info_file, count_type_from_zero=count_type_from_zero)
        self.es_info = apply_mapping_to_keys(self.es_info, NUM2RAW_3RSCAN)
        self.prev_es_info = read_es_infos("/mnt/hwfile/OpenRobotLab/lvruiyuan/embodiedscan_infos/embodiedscan_infos_train_full.pkl", count_type_from_zero=count_type_from_zero)
        self.prev_es_info = apply_mapping_to_keys(self.es_info, NUM2RAW_3RSCAN)
        self.type2int = np.load(es_info_file, allow_pickle=True)["metainfo"]["categories"]
        if count_type_from_zero:
            self.type2int = {k:v-1 for k, v in self.type2int.items()}
        # self.es_info[scene_id] = {
        #     "bboxes": bboxes,
        #     "object_ids": object_ids,
        #     "object_types": object_types,
        #     "visible_view_object_dict": visible_view_object_dict,
        #     "extrinsics_c2w": extrinsics_c2w,
        #     "axis_align_matrix": axis_align_matrix,
        #     "intrinsics": intrinsics,
        #     "depth_intrinsics": depth_intrinsics,
        #     "image_paths": image_paths,
        # }
        self.vg_raw_data = json.load(open(vg_raw_data_file, 'r'))
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.scan_ids = list(self.es_info.keys())
        self.num_scans = len(self.scan_ids)
        self.num_points = 1024 # number of points to sample from each object
        # 需要维护两份数据，一份是关于bbox和pcd的，一份是关于vg txt的
        # vg txt的一个dict的例子：{'scan_id': 'scene0000_00', 'target_id': ['30'], 'distractor_ids': [], 'text': 'The X is used for sitting at the table.Please find the X.', 'target': ['stool'], 'anchors': [], 'anchor_ids': [], 'tokens_positive': [[33, 38], [4, 5], [55, 56]]}
        self.scan_dir = processed_scan_dir
        self.prepare_inst_colors()
        self.process_vg_raw_data()

    def process_vg_raw_data(self):
        self.vg_data = []
        num_drops = 0
        for i, item in enumerate(self.vg_raw_data):
            item_id = "es" + str(i)
            scan_id = item['scan_id']
            scan_id = to_scene_id(scan_id)
            scan_id = NUM2RAW_3RSCAN.get(scan_id, scan_id)
            es_info = self.es_info.get(scan_id)
            if es_info is None:
                num_drops += 1 
                continue
            if scan_id not in self.inst_colors:
                num_drops += 1
                continue
            obj_ids = [int(x) for x in es_info['object_ids']] 
            types = es_info['object_types']
            txt = item['text']
            txt_ids = self.tokenizer.encode(txt) 
            txt_len = len(txt_ids)
            target_ids = to_list_of_int(item.get("target_id", []))
            tgt_num = len(target_ids)
            try:
                target_idxs = [obj_ids.index(x) for x in target_ids]
            except Exception as e:
                num_drops += 1
                continue
            target_types = [types[_] for _ in target_idxs]
            tgt_obj_class = [self.type2int[x] for x in target_types]

            vg_item = {
                "item_id": item_id,
                "scan_id": scan_id,
                "txt_ids": txt_ids,
                "txt_len": txt_len,
                "tgt_obj_idx": target_idxs,
                "tgt_obj_class": tgt_obj_class,
                "tgt_num": tgt_num,
            }
            self.vg_data.append(vg_item)
        del self.vg_raw_data

    def prepare_inst_colors(self):
        self.inst_colors = {}
        num_dropped = 0
        for scan_id in self.scan_ids:
            fname = os.path.join(self.scan_dir, 'instance_id_to_gmm_color', f'{scan_id}.json')
            if not os.path.exists(fname):
                num_dropped += 1
                continue
            with open(fname, 'r') as f:
                inst_colors = json.load(f)
            inst_colors = [np.concatenate(
                [np.array(x['weights'])[:, None], np.array(x['means'])],
                axis=1
            ).astype(np.float32) for x in inst_colors]
            inst_colors = np.array(inst_colors)
            obj_ids = self.es_info[scan_id]["object_ids"]
            prev_obj_ids = self.prev_es_info[scan_id]["object_ids"]
            index_list = np.where(obj_ids == prev_obj_ids[:, np.newaxis])[0]
            if not len(index_list) == len(obj_ids):
                num_dropped
                continue
            inst_colors = inst_colors[index_list]
            self.inst_colors[scan_id] = inst_colors
        print(f"colors: dropped {num_dropped} scenes and keeped {len(list(self.inst_colors.keys()))}")

    def get_inst_colors(self, scan_id):
        return self.inst_colors[scan_id]

    def get_scan_gt_pcd_data(self, scan_id):
        """
            returns pcd_data and obj_pcds
        """
        pcd_data_path = os.path.join(self.scan_dir, 'pcd_with_global_alignment', f'{scan_id}.pth')
        if not os.path.exists(pcd_data_path):
            print(f"Error: {pcd_data_path} does not exist.")
            return None
        data = torch.load(pcd_data_path)
        pc, colors, label, instance_ids = data
        pcd_data = np.concatenate([pc, colors], 1)
        obj_pcds = []
        for obj_id in self.es_info[scan_id]['object_ids']:
            obj_id = int(obj_id)
            mask = instance_ids == obj_id
            obj_pcd = pcd_data[mask]
            obj_pcds.append(obj_pcd)
        return pcd_data, obj_pcds
    
    def __len__(self):
        return len(self.vg_data)
    
    def _get_obj_inputs(self, scan_id):
    # obj_ids = [str(x) for x in self.es_info[scan_id]['object_ids']] 
        obj_ids = [str(x) for x in range(len(self.es_info[scan_id]['object_ids']))]
        # print(obj_ids)
        # NOTE: the object id is not compact. Do not use this obj_id to look up for es annos
        obj_classes = self.es_info[scan_id]['object_type_ints'] 
        obj_labels = self.es_info[scan_id]['object_types']
        # obj_gt_fts = [self.do_word2vec(obj_label) for obj_label in obj_labels]
        obj_locs = self.es_info[scan_id]['bboxes']
        obj_colors = self.get_inst_colors(scan_id)
        scan_pcd, obj_pcds = self.get_scan_gt_pcd_data(scan_id)
        obj_item = {
            "obj_ids": obj_ids,
            "obj_locs": obj_locs,
            "obj_colors": obj_colors,
            "obj_classes": obj_classes,
            "obj_labels": obj_labels,
        }
        return obj_item


    def __getitem__(self, idx):
        item = self.vg_data[idx]
        obj_item = self._get_obj_inputs(item["scan_id"])
        item_ids = item["item_id"] # str
        scan_ids = item["scan_id"] # str
        txt_ids = torch.LongTensor(item["txt_ids"])  # torch.int64 ([max_len_txt])
        txt_lens = torch.LongTensor([item["txt_len"]])  # torch.int64 ([])
        # obj_gt_fts = torch.FloatTensor(obj_item["obj_gt_fts"])  # torch.float32 ([max_num_obj, 300])
        obj_locs = torch.FloatTensor(obj_item["obj_locs"])  # torch.float32 ([max_num_obj, 6/9])
        obj_colors = torch.FloatTensor(obj_item["obj_colors"])  # torch.float32 ([max_num_obj, 3, 4])
        # obj_lens = torch.LongTensor([item["obj_len"]])  # torch.int64 ([])
        obj_classes = torch.LongTensor(obj_item["obj_classes"])  # torch.int64 ([max_num_obj])
        tgt_obj_idxs = torch.LongTensor(item["tgt_obj_idx"])  # torch.int64 ([num targets]) new!
        tgt_obj_classes = torch.LongTensor(item["tgt_obj_class"])  # torch.int64 (num targets) new!
        tgt_num = torch.LongTensor([item["tgt_num"]])  # torch.int64 ([])
        obj_ids = obj_item["obj_ids"]  # list['0', '2', '7', ...], all available object ids in the scan, not the target object ids
        # txt_masks = ... # torch.bool ([max_len_txt])
        # obj_masks = ... # torch.bool ([max_num_obj])
        aug_obj_labels = obj_item["obj_labels"]
        aug_obj_classes = torch.LongTensor([self.type2int[x] for x in aug_obj_labels])
        if self.cat2vec is None:
            aug_obj_fts = aug_obj_classes
        else:
            aug_obj_fts = torch.FloatTensor([self.cat2vec[x] for x in aug_obj_labels])
        
        outs = {
            "item_ids": item_ids,
            "scan_ids": scan_ids,
            "txt_ids": txt_ids,
            "txt_lens": txt_lens,
            "obj_fts": aug_obj_fts,
            "obj_locs": obj_locs,
            "obj_colors": obj_colors,
            "obj_lens": len(aug_obj_fts),
            "obj_classes": obj_classes,
            "tgt_obj_idxs": tgt_obj_idxs,
            "tgt_obj_classes": tgt_obj_classes,
            "obj_ids": obj_ids,
            "tgt_num": tgt_num,
        }
        return outs

def eslabel_collate_fn(data):
    # input: list of dicts, each dict is the output of __getitem__
    outs = {}
    for key in data[0].keys():
        outs[key] = [x[key] for x in data]
    outs['txt_ids'] = pad_sequence(outs['txt_ids'], batch_first=True)
    outs['txt_lens'] = torch.LongTensor(outs['txt_lens'])
    outs['txt_masks'] = gen_seq_masks(outs['txt_lens'])

    # outs['obj_gt_fts'] = pad_tensors(outs['obj_gt_fts'], lens=outs['obj_lens'])
    if len(outs['obj_fts'][0].size()) == 1:
        outs['obj_fts'] = pad_sequence(outs['obj_fts'], batch_first=True)
    else:
        outs['obj_fts'] = pad_tensors(outs['obj_fts'], lens=outs['obj_lens'])
    outs['obj_locs'] = pad_tensors(outs['obj_locs'], lens=outs['obj_lens'], pad=0)
    outs['obj_colors'] = pad_tensors(outs['obj_colors'], lens=outs['obj_lens'], pad=0)
    outs['obj_lens'] = torch.LongTensor(outs['obj_lens'])
    outs['obj_masks'] = gen_seq_masks(outs['obj_lens'])

    outs['obj_classes'] = pad_sequence(
        outs['obj_classes'], batch_first=True, padding_value=-100
    )
    outs['tgt_num'] = torch.LongTensor(outs['tgt_num'])
    outs['tgt_obj_idxs'] = pad_tensors(outs['tgt_obj_idxs'], lens=outs['tgt_num'], pad=-1) # torch.Size([64, 26])
    outs['tgt_obj_classes'] = pad_tensors(outs['tgt_obj_classes'], lens=outs['tgt_num'], pad=-100)
    outs['tgt_masks'] = gen_seq_masks(outs['tgt_num'])

    return outs
