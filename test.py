import torch

fname = '/mnt/petrelfs/lvruiyuan/repos/vil3dref/datasets/scan_data/pcd_with_global_alignment/scene0000_00.pth'
pcd_data = torch.load(fname)
# numpy array, actually.
import pdb; pdb.set_trace()