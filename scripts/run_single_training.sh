DSET=data/N200-percentile_max-block-10/

python train_segmentation.py --batch_size=1 --lr_min=1e-7 --lr_max=1e-4 --dataset_folder ./${DSET} --epochs 500 --restarts 1 --outdir models-replicate/ --n_levels 4 --downsampling 2 --width 1.0
