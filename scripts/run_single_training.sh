# update DSET to targetfolder of prepared data:
DSET=data/N500-percentile_max-block-20/ 

# To train previous model with new data:
python train_segmentation_frompretrained.py --batch_size=1 --lr_min=1e-7 --lr_max=1e-4 --dataset_folder ./${DSET} --epochs 50 --restarts 1 --outdir models-replicate/ --n_levels 4 --downsampling 1 --width 1.0

# To train from scratch (adjust downsampling):
# python train_segmentation.py --batch_size=1 --lr_min=1e-7 --lr_max=1e-4 --dataset_folder ./${DSET} --epochs 500 --restarts 1 --outdir models-replicate/ --n_levels 4 --downsampling 2 --width 1.0

# To train from scratch using low patience:
# python train_segmentation_lowpatience.py --batch_size=1 --lr_min=1e-7 --lr_max=1e-4 --dataset_folder ./${DSET} --epochs 50 --restarts 1 --outdir models-replicate/ --n_levels 4 --downsampling 1 --width 1.0

