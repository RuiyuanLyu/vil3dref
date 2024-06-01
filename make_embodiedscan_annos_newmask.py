import os
import numpy as np
import torch
import mmengine
import json
from og3d_src.utils.utils_read import read_es_infos, to_scene_id, NUM2RAW_3RSCAN, RAW2NUM_3RSCAN
from utils.utils_3d import is_inside_box, euler_angles_to_matrix
from utils.decorators import mmengine_track_func

dataroot_mp3d = "/mnt/hwfile/OpenRobotLab/lvruiyuan/pcd_data/matterport3d/scans"
dataroot_3rscan = "/mnt/hwfile/OpenRobotLab/lvruiyuan/pcd_data/3rscan/scans"
dataroot_scannet = "/mnt/hwfile/OpenRobotLab/lvruiyuan/pcd_data/scannet/scans"

box_pred_dir = "/mnt/hwfile/OpenRobotLab/lvruiyuan/pcd_data/embodiedscan_pred_boxes"
save_dir = "/mnt/hwfile/OpenRobotLab/lvruiyuan/pcd_data/embodiedscan_pred_mask_by_box"

def load_pcd_data(scene):
    scene = NUM2RAW_3RSCAN.get(scene, scene)
    for d in ['mp3d', '3rscan', 'scannet']:
        d_root = eval(f'dataroot_{d}')
        pcd_file = os.path.join(d_root, scene, "pc_infos.npy")
        if os.path.exists(pcd_file):
            break
    pc_infos = np.load(pcd_file)
    nan_mask = np.isnan(pc_infos).any(axis=1)
    pc_infos = pc_infos[~nan_mask]
    pc = pc_infos[:, :3]
    color = pc_infos[:, 3:6]
    label = pc_infos[:, 6].astype(np.uint16) # this do not matter in the current code
    # semantic_ids = pc_infos[:, 7].astype(np.uint16)
    return pc, color, label

def create_scene_mask(scene, overwrite=False):

    box_pred_file = os.path.join(box_pred_dir, f"{scene}.npy.npz")
    es_anno = np.load(box_pred_file)

    out_file_name = os.path.join(save_dir, f"{scene}.npz")
    if os.path.exists(out_file_name) and not overwrite:
        return True
    pc, color, label = load_pcd_data(scene)
    label = np.ones_like(label) * -100
    if np.isnan(pc).any() or np.isnan(color).any():
        print(f"nan detected in {scene}")
    instance_ids = np.ones(pc.shape[0], dtype=np.int16) * (-100)
    
    bboxes =  es_anno["boxes"].reshape(-1, 9)
    bboxes[:, 3:6] = np.clip(bboxes[:, 3:6], a_min=1e-2, a_max=None)
    scores = es_anno["scores"]
    object_types = es_anno["labels"] # int

    masks = np.zeros((len(bboxes), pc.shape[0]), dtype=bool)
    for i, box in enumerate(bboxes):
        center, size, euler = box[:3], box[3:6], box[6:]
        R = euler_angles_to_matrix(euler, convention="ZXY")
        R = R.reshape((3,3))
        box_pc_mask = is_inside_box(pc, center, size, R)
        masks[i] = box_pc_mask
    
    print(masks.shape)
    print(object_types.shape)
    print(scores.shape)

    np.savez_compressed(out_file_name, masks, object_types, scores)
    return True



if __name__ == "__main__":
    tasks = os.listdir(box_pred_dir)
    tasks = [x.split('.')[0] for x in tasks]
    mmengine.track_parallel_progress(create_scene_mask, tasks, nproc=8)
    # for x in tasks:
    #     create_scene_mask(x)