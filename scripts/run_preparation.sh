NSAMPLES=500
OUTDIR=data/
DELTA=20

# NOTE we call the prepare script for each folder individually to make
# sure that they are equally represented in the validation split.

for DATA in "/datafolder"; do
    python prepare_dataset.py --data_folder "$DATA" --label_folder "/targetfolder" --samples_per_stack $NSAMPLES --output_folder $OUTDIR --delta $DELTA --projector percentile_max;
done

