NSAMPLES=200
OUTDIR=data/
DELTA=10

# NOTE we call the prepare script for each folder individually to make
# sure that they are equally represented in the validation split.
for DATA in "/data_fmi/mdrive/groska/Fiona/Confocal/190225 johnson" "/data_fmi/mdrive/groska/Fiona/Confocal/190320 Merian" ; do
    python prepare_dataset.py --data_folder "$DATA" --label_folder "/tungstenfs/scratch/groska/Fiona/For Markus/" --samples_per_stack $NSAMPLES --output_folder $OUTDIR --delta $DELTA --projector percentile_max ;
done
