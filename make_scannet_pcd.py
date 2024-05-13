from plyfile import PlyData
import numpy as np
import os.path as osp
import json
import os
import mmengine
from utils.decorators import mmengine_track_func

axis_alignment_info_file = '/mnt/petrelfs/share_data/lvruiyuan/pcd_data/scannet/scans_axis_alignment_matrices.json'
with open(axis_alignment_info_file, 'r') as f:
    scans_axis_alignment_matrices = json.load(f)

def align_to_axes(point_cloud, scan_id):
    """
    Align the scan to xyz axes using the alignment matrix found in scannet.
    """
    # Get the axis alignment matrix
    alignment_matrix = scans_axis_alignment_matrices[scan_id]
    alignment_matrix = np.array(alignment_matrix, dtype=np.float32).reshape(4, 4)

    # Transform the points
    pts = np.ones((point_cloud.shape[0], 4), dtype=point_cloud.dtype)
    pts[:, 0:3] = point_cloud
    point_cloud = np.dot(pts, alignment_matrix.transpose())[:, :3]  # Nx4

    # Make sure no nans are introduced after conversion
    assert (np.sum(np.isnan(point_cloud)) == 0)

    return point_cloud


def load_point_cloud_with_meta_data(top_scan_dir, scan_id, load_semantic_label=True, load_color=True, apply_global_alignment=True):
    """
    :param load_semantic_label:
    :param load_color:
    :param apply_global_alignment: rotation/translation of scan according to Scannet meta-data.
    :return:
    """
    scan_ply_suffix = '_vh_clean_2.labels.ply'
    mesh_ply_suffix = '_vh_clean_2.ply'

    scan_data_file = osp.join(top_scan_dir, scan_id, scan_id + scan_ply_suffix)
    data = PlyData.read(scan_data_file)
    x = np.asarray(data.elements[0].data['x'])
    y = np.asarray(data.elements[0].data['y'])
    z = np.asarray(data.elements[0].data['z'])
    pc = np.stack([x, y, z], axis=1)

    label = None
    if load_semantic_label:
        label = np.asarray(data.elements[0].data['label'])

    color = None
    if load_color:
        scan_data_file = osp.join(top_scan_dir, scan_id, scan_id + mesh_ply_suffix)
        data = PlyData.read(scan_data_file)
        r = np.asarray(data.elements[0].data['red'])
        g = np.asarray(data.elements[0].data['green'])
        b = np.asarray(data.elements[0].data['blue'])
        color = (np.stack([r, g, b], axis=1) / 256.0).astype(np.float32)

    # Global alignment of the scan
    if apply_global_alignment:
        pc = align_to_axes(pc, scan_id)
    return pc, color, label

@mmengine_track_func
def load_and_save(top_scan_dir, scan_id, save_dir):
    pc, color, label = load_point_cloud_with_meta_data(top_scan_dir, scan_id)
    out = np.concatenate([pc, color, label.reshape(-1, 1)], axis=1)
    assert out.shape[1] == 7
    output_file = osp.join(save_dir, scan_id, "pc_infos.npy")
    os.makedirs(osp.dirname(output_file), exist_ok=True)
    np.save(output_file, out)

if __name__ == '__main__':
    top_scan_dir = '/mnt/petrelfs/share_data/maoxiaohan/ScanNet_v2/scans'
    scan_ids = os.listdir(top_scan_dir)
    scan_ids = [scan_id for scan_id in scan_ids if scan_id.startswith('scene')]

    save_dir = "/mnt/petrelfs/share_data/lvruiyuan/pcd_data/scannet/scans"
    tasks = [(top_scan_dir, scan_id, save_dir) for scan_id in scan_ids]
    mmengine.utils.track_parallel_progress(load_and_save, tasks, nproc=10)
