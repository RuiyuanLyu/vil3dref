import os
import json
import numpy as np
import random

from transformers import AutoTokenizer
from sklearn.mixture import GaussianMixture

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
try:
    from .common import pad_tensors, gen_seq_masks
except:
    from common import pad_tensors, gen_seq_masks
MAX_NUM_OBJ = 256


# target: to make batches.

# The batch is a dictionary containing the following keys:
# item_ids = batch['item_ids'] # list ['nr3d_012277', ...], len=64
# scan_ids = batch['scan_ids'] # list ['scene0520_00', ...], len=64
# txt_ids = batch['txt_ids'] # torch.int64([64, 28])
# txt_lens = batch['txt_lens'] # torch.int64([64])
# obj_gt_fts = batch['obj_gt_fts'] # torch.float32([64, 69, 300])
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
from utils.utils_3d import compute_bbox_from_points_list
class ESLabelPcdDatasetEval(Dataset):
    def __init__(self, es_info_file, vg_raw_data_file, cat2vec_file, processed_scan_dir):
        super().__init__()
        # from data.glove_embedding import get_glove_word2vec
        # self.word2vec = get_glove_word2vec(glove_embedding_file)
        self.word2vec = json.load(open(cat2vec_file, 'r'))
        self.word2vec_vocab = list(self.word2vec.keys())
        count_type_from_zero = True
        self.es_info = read_es_infos(es_info_file, count_type_from_zero=count_type_from_zero)
        self.es_info = apply_mapping_to_keys(self.es_info, NUM2RAW_3RSCAN)
        # NOTE: prev es_info is used to fix the bug in gmm color loading.
        self.type2int = np.load(es_info_file, allow_pickle=True)["metainfo"]["categories"]
        if count_type_from_zero:
            self.type2int = {k:v-1 for k, v in self.type2int.items()}
        self.vg_raw_data = json.load(open(vg_raw_data_file, 'r'))
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.scan_ids = list(self.es_info.keys())
        self.num_scans = len(self.scan_ids)
        self.num_points = 1024 # number of points to sample from each object
        self.scan_dir = processed_scan_dir
        self.scan_gt_pcd_data = {}
        self.inst_colors = {}
        self.process_vg_raw_data()

    def process_vg_raw_data(self):
        self.vg_data = []
        num_drops = 0
        for i, item in enumerate(self.vg_raw_data):
            item_id = item["sub_class"] + str(i)
            scan_id = item['scan_id']
            scan_id = to_scene_id(scan_id)
            scan_id = NUM2RAW_3RSCAN.get(scan_id, scan_id)
            es_info = self.es_info.get(scan_id)
            if es_info is None:
                num_drops += 1 
                continue
            obj_ids = [int(x) for x in es_info['object_ids']] 
            bboxes = es_info['bboxes']
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
            target_bboxes = bboxes[target_idxs]
            target_types = [types[_] for _ in target_idxs]
            tgt_obj_class = [self.type2int[x] for x in target_types]

            # HACK: in this ESLabelPcdDatasetEval, the tgt_obj_idx should not take effect in the eval method we care. Only the target bboxes are important.
            # ONLY the tgt_obj_box is used.
            sub_class = item.get("sub_class", "").lower()
            vg_item = {
                "item_id": item_id,
                "scan_id": scan_id,
                "txt_ids": txt_ids,
                "txt_len": txt_len,
                "tgt_obj_idx": target_idxs,
                "tgt_obj_box": target_bboxes,
                "tgt_obj_class": tgt_obj_class,
                "tgt_num": tgt_num,
                "sub_class": sub_class,
            }
            self.vg_data.append(vg_item)
        print(f"dropped {num_drops}, keeping {len(self.vg_data)}")
        del self.vg_raw_data

    def do_word2vec(self, text):
        # for ood vocab (multiword): take average.
        if text in self.word2vec:
            return self.word2vec[text]
        else:
            key = random.choice(self.word2vec_vocab)
            return self.word2vec[key]

    def get_obj_inputs(self, scan_id):
        obj_colors = self.get_inst_colors(scan_id)
        scan_pcd, obj_pcds, obj_locs = self.get_scan_gt_pcd_data(scan_id) #This do not provide gt, but predicted by other methods. 
        num_objs = len(obj_pcds)
        obj_labels = ["object" for _ in range(num_objs)] #HACK: for ESLabelPcdDatasetEval ONLY. The model should predict this label in the grounding phase.
        obj_classes = [self.type2int[x] for x in obj_labels]
        obj_gt_fts = [self.do_word2vec(obj_label) for obj_label in obj_labels]

        obj_fts = self.get_obj_fts(obj_pcds)
        obj_item = {
            "obj_ids": [x for x in range(num_objs)],
            "obj_gt_fts": obj_gt_fts,
            "obj_fts": obj_fts,
            "obj_locs": np.array(obj_locs), # should be [num_objs, 9], TODO: check this.
            "obj_colors": obj_colors,
            "obj_classes": obj_classes,
            "obj_labels": obj_labels,
        }
        return obj_item

    def get_inst_colors(self, scan_id):
        fname = os.path.join('/mnt/petrelfs/lvruiyuan/repos/vil3dref/datasets/referit3d/scan_data_for_es_pred', 'instance_id_to_gmm_color', f'{scan_id}.npy')
        if scan_id in self.inst_colors:
            return self.inst_colors[scan_id]
        if os.path.exists(fname):
            obj_colors = np.load(fname)
            if MAX_NUM_OBJ <= len(obj_colors):
                self.inst_colors[scan_id] = obj_colors[:MAX_NUM_OBJ]
                return self.inst_colors[scan_id]
            # In the saved file, the max num obj might be different
        obj_pcds = self.get_scan_gt_pcd_data(scan_id)[1]
        num_objs = len(obj_pcds)
        obj_colors = []
        for i in range(num_objs):
            color = obj_pcds[i][:, 3:6]
            if len(color) > 10:
                gm = GaussianMixture(n_components=3, covariance_type='full', random_state=0).fit(color)
                weights = gm.weights_
                means = gm.means_
                obj_colors.append(np.concatenate([weights[:, None], means], 1))
            else:
                obj_colors.append(np.zeros((3,4)))
            # assert obj_colors[-1].shape == (3, 4), f"Error: {obj_colors[-1].shape}"
        obj_colors = np.array(obj_colors)
        self.inst_colors[scan_id] = obj_colors
        np.save(fname, obj_colors)
        return self.inst_colors[scan_id]

    def get_scan_gt_pcd_data(self, scan_id):
        """
            returns pcd_data and obj_pcds
        """
        if scan_id in self.scan_gt_pcd_data:
            return self.scan_gt_pcd_data[scan_id]
        pcd_data_path = os.path.join(self.scan_dir, 'pcd_with_global_alignment', f'{scan_id}.pth')
        if not os.path.exists(pcd_data_path):
            print(f"Error: {pcd_data_path} does not exist.")
            return None
        data = torch.load(pcd_data_path)
        pc, colors, label, instance_ids = data
        pcd_data = np.concatenate([pc, colors], 1)
        
        mask_scan_id = RAW2NUM_3RSCAN.get(scan_id, scan_id)
        mask_data_path = os.path.join("/mnt/hwfile/OpenRobotLab/lvruiyuan/pcd_data/embodiedscan_pred_mask_by_box", f"{mask_scan_id}.npz")
        data = np.load(mask_data_path)
        
        masks = data['arr_0'] # shape num points, num_objs
        label = data['arr_1'] # unused
        score = data['arr_2'] 
        # print(masks.shape, label.shape, score.shape)
        inds = np.argsort(score)[::-1][:MAX_NUM_OBJ]
        masks = masks[inds]
        score = score[inds]
        num_obj = len(inds)

        obj_pcds = []
        obj_locs = []
        from utils.utils_3d import compute_bbox_from_points, matrix_to_euler_angles
        for obj_id in range(num_obj):
            mask = masks[obj_id]
            obj_pcd = pcd_data[mask]
            obj_pcds.append(obj_pcd)

            if len(obj_pcd) > 10:
                center, size, rotmat = compute_bbox_from_points(obj_pcd[:, :3])
                euler = matrix_to_euler_angles(rotmat, "ZXY")
                obj_loc = np.concatenate([center, size, euler]).reshape(9)
            else:
                obj_loc = np.zeros(9)
            obj_locs.append(obj_loc)
        self.scan_gt_pcd_data[scan_id] = (pcd_data, obj_pcds, obj_locs)
        return pcd_data, obj_pcds, obj_locs
    
    def get_obj_fts(self, obj_pcds):
        obj_fts = []
        for obj_pcd in obj_pcds:
            if len(obj_pcd) > 0:
                pcd_idxs = np.random.choice(len(obj_pcd), size=self.num_points, replace=(len(obj_pcd) < self.num_points))
                obj_pcd = obj_pcd[pcd_idxs]
                obj_pcd[:, :3] = obj_pcd[:, :3] - obj_pcd[:, :3].mean(0)
                max_dist = np.max(np.sqrt(np.sum(obj_pcd[:, :3]**2, 1)))
                if max_dist < 1e-6: # take care of tiny point-clouds, i.e., padding
                    max_dist = 1
                obj_pcd[:, :3] = obj_pcd[:, :3] / max_dist
            else:
                obj_pcd = np.zeros((self.num_points, 6))
            obj_fts.append(obj_pcd)
        obj_fts = np.array(obj_fts)
        return obj_fts

    def __len__(self):
        return len(self.vg_data)

    def __getitem__(self, idx):
        item = self.vg_data[idx]
        obj_item = self.get_obj_inputs(item["scan_id"])
        item_ids = item["item_id"] # str
        scan_ids = item["scan_id"] # str
        txt_ids = torch.LongTensor(item["txt_ids"])  # torch.int64 ([max_len_txt])
        txt_lens = torch.LongTensor([item["txt_len"]])  # torch.int64 ([])
        obj_gt_fts = torch.FloatTensor(obj_item["obj_gt_fts"])  # torch.float32 ([max_num_obj, 300])
        obj_fts = torch.FloatTensor(obj_item["obj_fts"])  # torch.float32 ([max_num_obj, 1024, 6])
        obj_locs = torch.FloatTensor(obj_item["obj_locs"])  # torch.float32 ([max_num_obj, 6/9])
        obj_colors = torch.FloatTensor(obj_item["obj_colors"])  # torch.float32 ([max_num_obj, 3, 4])
        # obj_lens = torch.LongTensor([item["obj_len"]])  # torch.int64 ([])
        obj_classes = torch.LongTensor(obj_item["obj_classes"])  # torch.int64 ([max_num_obj])
        tgt_obj_idxs = torch.LongTensor(item["tgt_obj_idx"])  # torch.int64 ([num targets]) new!
        tgt_obj_boxes = torch.FloatTensor(item["tgt_obj_box"]) # torch.float (num targets, 9) new!
        tgt_obj_classes = torch.LongTensor(item["tgt_obj_class"])  # torch.int64 (num targets) new!
        tgt_num = torch.LongTensor([item["tgt_num"]])  # torch.int64 ([])
        obj_ids = obj_item["obj_ids"]  # list['0', '2', '7', ...], all available object ids in the scan, not the target object ids
        # txt_masks = ... # torch.bool ([max_len_txt])
        # obj_masks = ... # torch.bool ([max_num_obj])
                #         "space": space,
                # "direct": direct,
                # "multi": multi,
                # "hard": hard,
        outs = {
            "item_ids": item_ids,
            "scan_ids": scan_ids,
            "txt_ids": txt_ids,
            "txt_lens": txt_lens,
            "obj_gt_fts": obj_gt_fts,
            "obj_fts": obj_fts,
            "obj_locs": obj_locs,
            "obj_colors": obj_colors,
            "obj_lens": len(obj_fts),
            "obj_classes": obj_classes,
            "tgt_obj_idxs": tgt_obj_idxs,
            "tgt_obj_boxes": tgt_obj_boxes,
            "tgt_obj_classes": tgt_obj_classes,
            "tgt_num": tgt_num,
            "obj_ids": obj_ids,
            "sub_class": item["sub_class"],
        }
        return outs

def eslabelpcd_collate_fn(data):
    # input: list of dicts, each dict is the output of __getitem__
    outs = {}
    for key in data[0].keys():
        outs[key] = [x[key] for x in data]
    outs['txt_ids'] = pad_sequence(outs['txt_ids'], batch_first=True)
    outs['txt_lens'] = torch.LongTensor(outs['txt_lens'])
    outs['txt_masks'] = gen_seq_masks(outs['txt_lens'])

    outs['obj_gt_fts'] = pad_tensors(outs['obj_gt_fts'], lens=outs['obj_lens'])
    outs['obj_fts'] = pad_tensors(outs['obj_fts'], lens=outs['obj_lens'], pad_ori_data=True)
    outs['obj_locs'] = pad_tensors(outs['obj_locs'], lens=outs['obj_lens'], pad=0)
    outs['obj_colors'] = pad_tensors(outs['obj_colors'], lens=outs['obj_lens'], pad=0)
    outs['obj_lens'] = torch.LongTensor(outs['obj_lens'])
    outs['obj_masks'] = gen_seq_masks(outs['obj_lens'])

    outs['obj_classes'] = pad_sequence(
        outs['obj_classes'], batch_first=True, padding_value=-100
    )
    outs['tgt_num'] = torch.LongTensor(outs['tgt_num'])
    outs['tgt_obj_idxs'] = pad_tensors(outs['tgt_obj_idxs'], lens=outs['tgt_num'], pad=-1) # torch.Size([64, 26])
    outs['tgt_obj_boxes'] = pad_tensors(outs['tgt_obj_boxes'], lens=outs['tgt_num'], pad=0)
    outs['tgt_obj_classes'] = pad_tensors(outs['tgt_obj_classes'], lens=outs['tgt_num'], pad=-100)
    outs['tgt_masks'] = gen_seq_masks(outs['tgt_num'])
    return outs
