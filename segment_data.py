# Before running, adjust folder names (/datafolder), segmentation task (retchat.task.*) and model (models\*)

for DATA in "/datafolder" ; do
	rm "/home/fiona.muellner/faim-retina-chatbands/outputs/stacks_to_process.txt"
	luigi --local-scheduler --module retchat.tasks.run_segmentation_d20 retina.ParallelChatbandPredictionTask --output-folder 'outputs' --input-folder "$DATA" --model-dir models/N500-percentile_max-block-20_231027/segm/unet-L4-D1-None/run000 --model-weights-fname model_best.h5
done
