# change the configfile for other datasets, e.g., scanrefer_gtlabel_model.yaml
configfile=configs/nr3d_gtlabel_model.yaml
python train_prev.py --config $configfile --output_dir ../datasets/exprs_neurips22/gtlabels/nr3d