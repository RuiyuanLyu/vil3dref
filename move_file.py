import os
import shutil
import logging
from tqdm import tqdm

SAFE_MODE_ON = False

def copy_tree(src, dst):
    if os.path.exists(src):
        if not os.path.exists(dst):
            os.makedirs(dst)
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if not SAFE_MODE_ON:
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)  # 使用 copy2 来保留元数据
        logging.info(f'Copied "{src}" to "{dst}"')
    else:
        logging.warning(f'Source directory "{src}" does not exist.')
        # print(f'Source directory "{src}" does not exist.')

if __name__ == "__main__":
    log_file_name = f"{__file__[:-3]}.log"
    print(f"log will be written to {log_file_name}")
    # clear the log file
    with open(log_file_name, 'w'): 
        pass
    logging.basicConfig(filename=log_file_name, level=logging.INFO, format='%(asctime)s - %(message)s')

    tgt_dir = "/mnt/hwfile/OpenRobotLab/lvruiyuan/3rscan"
    src_dir = "/mnt/petrelfs/share_data/lvruiyuan/3rscan"

    scene_ids = os.listdir(src_dir)
    tasks = []
    for scene_id in tqdm(scene_ids):
        src = os.path.join(src_dir, scene_id)
        dst = os.path.join(tgt_dir, scene_id)
        copy_tree(src, dst)
