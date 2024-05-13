import os
import tarfile

def tar_folders(folders_to_tar, tar_name):
    print(len(folders_to_tar))
    with tarfile.open(tar_name, 'w:gz') as tarf:  
        for folder_path in folders_to_tar:
            print(folder_path)
            if os.path.isdir(folder_path):
                tarf.add(folder_path, arcname=os.path.basename(folder_path))

src_dir = "/mnt/petrelfs/share_data/lvruiyuan/pcd_data"
output_dir = "."
for dataset in ["scannet", "matterport3d", "3rscan"]:
    tar_name = os.path.join(output_dir, f"{dataset}.tar.gz")
    real_dir = os.path.join(src_dir, dataset, 'scans')
    folders_to_tar = [os.path.join(real_dir, d) for d in os.listdir(real_dir) if os.path.isdir(os.path.join(real_dir, d))][:10]
    # import pdb; pdb.set_trace()
    tar_folders(folders_to_tar, tar_name)