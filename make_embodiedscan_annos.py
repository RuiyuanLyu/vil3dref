import os
import numpy as np
import torch
import mmengine
import json
from og3d_src.utils.utils_read import read_annotation_pickles
from utils.utils_3d import is_inside_box, euler_angles_to_matrix
from utils.decorators import mmengine_track_func

dataroot_mp3d = "/mnt/petrelfs/share_data/lvruiyuan/pcd_data/matterport3d/scans"
dataroot_3rscan = "/mnt/petrelfs/share_data/lvruiyuan/pcd_data/3rscan/scans"
dataroot_scannet = "/mnt/petrelfs/share_data/lvruiyuan/pcd_data/scannet/scans"
output_dir = "/mnt/petrelfs/lvruiyuan/repos/vil3dref/datasets/referit3d/scan_data_new"
es_info_file = "/mnt/petrelfs/lvruiyuan/repos/vil3dref/embodiedscan_infos/embodiedscan_infos_val_full.pkl"

TYPE2INT = np.load(es_info_file, allow_pickle=True)["metainfo"]["categories"] # str2int

def load_pcd_data(scene):
    pcd_file = os.path.join(DATAROOT, scene, "pc_infos.npy")
    pc_infos = np.load(pcd_file)
    nan_mask = np.isnan(pc_infos).any(axis=1)
    pc_infos = pc_infos[~nan_mask]
    pc = pc_infos[:, :3]
    color = pc_infos[:, 3:6]
    label = pc_infos[:, 6].astype(np.uint16) # this do not matter in the current code
    # semantic_ids = pc_infos[:, 7].astype(np.uint16)
    return pc, color, label

def create_scene_pcd(scene, es_anno, overwrite=False):
    if es_anno is None:
        return None
    out_file_name = os.path.join(output_dir,"pcd_with_global_alignment", f"{scene}.pth")
    if os.path.exists(out_file_name) and not overwrite:
        return True
    pc, color, label = load_pcd_data(scene)
    label = np.ones_like(label) * -100
    if np.isnan(pc).any() or np.isnan(color).any():
        print(f"nan detected in {scene}")
    instance_ids = np.ones(pc.shape[0], dtype=np.int16) * (-100)
    bboxes =  es_anno["bboxes"]
    bboxes[:, 3:6] = np.clip(bboxes[:, 3:6], a_min=1e-2, a_max=None)
    object_ids = es_anno["object_ids"]
    object_types = es_anno["object_types"] # str
    sorted_indices = sorted(enumerate(bboxes), key=lambda x: -np.prod(x[1][3:6])) # the larger the box, the smaller the index
    sorted_indices_list = [index for index, value in sorted_indices]

    bboxes = [bboxes[index] for index in sorted_indices_list]
    object_ids = [object_ids[index] for index in sorted_indices_list]
    object_types = [object_types[index] for index in sorted_indices_list]

    for box, obj_id, obj_type in zip(bboxes, object_ids, object_types):
        obj_type_id = TYPE2INT.get(obj_type, -1)
        center, size, euler = box[:3], box[3:6], box[6:]
        R = euler_angles_to_matrix(euler, convention="ZXY")
        R = R.reshape((3,3))
        box_pc_mask = is_inside_box(pc, center, size, R)
        num_points_in_box = np.sum(box_pc_mask)
        # if num_points_in_box == 0:
        #     print(f"do not contain points: {obj_type}, {obj_id}")
        instance_ids[box_pc_mask] = obj_id
        label[box_pc_mask] = obj_type_id

    out_data = (pc, color, label, instance_ids)
    os.makedirs(os.path.dirname(out_file_name), exist_ok=True)
    torch.save(out_data, out_file_name)
    return True

def create_instance_id_mapping(scene, es_anno, overwrite=False):
    if es_anno is None:
        return None
    out_file_name = os.path.join(output_dir,"instance_id_to_name", f"{scene}.json")
    if os.path.exists(out_file_name) and not overwrite:
        return True
    object_ids = es_anno["object_ids"]
    object_types = es_anno["object_types"] # str
    out_list = []
    if len(object_ids) > 0:
        for obj_type in object_types:
            out_list.append(obj_type)
    os.makedirs(os.path.dirname(out_file_name), exist_ok=True)
    with open(out_file_name, "w") as f:
        json.dump(out_list, f)
    return True

def create_instance_id_loc(scene, es_anno, overwrite=False):
    if es_anno is None:
        return None
    out_file_name = os.path.join(output_dir,"instance_id_to_loc", f"{scene}.npy")
    if os.path.exists(out_file_name) and not overwrite:
        return True
    object_bboxes = es_anno["bboxes"] # 9 dof np array
    object_ids = es_anno["object_ids"]
    out_array = np.zeros((0, 9))
    if len(object_ids) > 0:
        out_array = np.zeros((len(object_bboxes), 9))
        for i, bbox in enumerate(object_bboxes):
            out_array[i] = bbox
    os.makedirs(os.path.dirname(out_file_name), exist_ok=True)
    np.save(out_file_name, out_array)
    return True

from sklearn.mixture import GaussianMixture

def create_instance_colors(scene, es_anno, overwrite=False):
    # copied from preprocess/scannetv2/precompute_instance_colors.py
    out_file_name = os.path.join(output_dir, 'instance_id_to_gmm_color', '%s.json'%scene)
    if os.path.exists(out_file_name) and not overwrite:
        return

    scan_file = os.path.join(output_dir,"pcd_with_global_alignment", f"{scene}.pth")
    data = torch.load(scan_file) # xyz, rgb, obj_type, instance_labels
    _, colors, _, instance_labels = data

    # normalize
    # color might be [0, 255], or [0, 1], or [-1, 1]. need to normalize to [-1, 1]
    if np.all((colors >= 0) & (colors <= 1)):
        colors = colors * 2 - 1
    elif np.all((colors >= 0) & (colors <= 255)):
        colors = colors / 127.5 - 1
    if not np.all((colors >= -1) & (colors <= 1)):
        print(f"color max {colors.max()}, min {colors.min()}")

    clustered_colors = []
    instance_ids = es_anno["object_ids"]
    instance_ids = [x for x in instance_ids if x >= 0]
    for i in instance_ids:
        mask = instance_labels == i     # time consuming
        obj_colors = colors[mask]
        if len(obj_colors) < 10:
            clustered_colors.append({
                'weights': [0.0,0.0,0.0],
                'means': [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]
            })
            continue
        gm = GaussianMixture(n_components=3, covariance_type='full', random_state=0).fit(obj_colors)
        clustered_colors.append({
            'weights': gm.weights_.tolist(),
            'means': gm.means_.tolist(),
        })
    os.makedirs(os.path.dirname(out_file_name), exist_ok=True)
    json.dump(
        clustered_colors,
        open(out_file_name, 'w')
    )

@mmengine_track_func
def create_all(scene, es_anno):
    create_scene_pcd(scene, es_anno, overwrite=True)
    create_instance_colors(scene, es_anno, overwrite=True)
    create_instance_id_mapping(scene, es_anno, overwrite=True)
    create_instance_id_loc(scene, es_anno, overwrite=True)


if __name__ == "__main__":
    MODE = "scannet"
    assert MODE in ["mp3d", "3rscan", "scannet"]
    DATAROOT = eval(f"dataroot_{MODE}")
    scene_list = os.listdir(DATAROOT)
    scene_list = scene_list[:50]
    scene_list = ["scene0000_00"]
    embodiedscan_annotation_files = [
        "/mnt/petrelfs/lvruiyuan/repos/vil3dref/embodiedscan_infos/embodiedscan_infos_train_full.pkl",
        "/mnt/petrelfs/lvruiyuan/repos/vil3dref/embodiedscan_infos/embodiedscan_infos_val_full.pkl"
    ]
    train_split_file = "/mnt/petrelfs/lvruiyuan/repos/vil3dref/datasets/referit3d/annotations/splits/es_train.txt"
    val_split_file = "/mnt/petrelfs/lvruiyuan/repos/vil3dref/datasets/referit3d/annotations/splits/es_val.txt"
    anno_train = read_annotation_pickles(embodiedscan_annotation_files[0])
    anno_val = read_annotation_pickles(embodiedscan_annotation_files[1])
    with open(f"embodiedscan_infos/3rscan_mapping.json", 'r') as f:
        scene_mappings = json.load(f)
    ####################################################################
    # save splits
    mini_scenes = set(os.listdir(dataroot_mp3d)[:50] + os.listdir(dataroot_3rscan)[:50] + os.listdir(dataroot_scannet)[:50])
    reverse_mapping = {v:k for k,v in scene_mappings.items()}
    with open(train_split_file, 'w') as f:
        for key in anno_train:
            key = reverse_mapping.get(key, key)
            if key in mini_scenes:
                f.write(key + '\n')
    with open(val_split_file, 'w') as f:
        for key in anno_val:
            key = reverse_mapping.get(key, key)
            if key in mini_scenes:
                f.write(key + '\n')
    ####################################################################
    es_annos = {**anno_train, **anno_val}
    tasks = []
    for scene in scene_list:
        # only 3rscan needs mapping. mp3d do not.
        es_anno = es_annos.get(scene_mappings.get(scene, scene), None)
        if es_anno:
            tasks.append((scene, es_anno))
    mmengine.track_parallel_progress(create_all, tasks, nproc=8)
