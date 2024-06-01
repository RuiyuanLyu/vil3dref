export TOKENIZERS_PARALLELISM=true
# step1: train the teacher model with groundtruth object labels
configfile=configs/nr3d_gtlabel_model_es.yaml
python train.py --config $configfile --output_dir ../datasets/exprs_es/gtlabels_valfixed/es \
# --resume_files ../datasets/exprs_es/gtlabels/es/ckpts/model_epoch_6.pt 

# step2: train the pointnet encoder
# python train_pcd_backbone.py --config configs/pcd_classifier_es.yaml \
#     --output_dir ../datasets/exprs_es/pcd_clf_pre

# step3: train the student model with 3d point clouds
# configfile=configs/nr3d_gtlabelpcd_mix_model_es.yaml
# python train_mix.py --config $configfile \
#     --output_dir ../datasets/exprs_es/gtlabelpcd_mix/es_mini \
#     --resume_files ../datasets/exprs_neurips22/pcd_clf_pre/ckpts/model_epoch_100.pt \
#     ../datasets/exprs_neurips22/gtlabels/nr3d/ckpts/model_epoch_49.pt