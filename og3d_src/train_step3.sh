# step1: train the teacher model with groundtruth object labels
# configfile=configs/nr3d_gtlabel_model_es.yaml
# python train.py --config $configfile --output_dir ../datasets/exprs_es/gtlabels/es

# step2: train the pointnet encoder
# python train_pcd_backbone.py --config configs/pcd_classifier_es.yaml \
#     --output_dir ../datasets/exprs_es/pcd_clf_pre

# step3: train the student model with 3d point clouds

# Warning: 修改resume files
configfile=configs/nr3d_gtlabelpcd_mix_model_es.yaml
python -u train_mix.py --config $configfile \
    --output_dir ../datasets/exprs_mini/gtlabelpcd_mix_mmscan/mm_scan \
    --resume_files ../datasets/exprs_es/pcd_clf_pre_valfixed/ckpts/model_epoch_93.pt \
    ../datasets/exprs_es/gtlabels_valfixed/es/ckpts/model_epoch_22.pt
    # --test