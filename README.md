# Retina-chatbands

A CNN-based detection of retina chatbands.

## Usage

### Train a new model
> TODO 

### Run a given model

```
luigi --module retchat.tasks.run_segmentation retina.ParallelChatbandPredictionTask --output-folder <SOME/OUTPUT/FOLDER> --input-folder <SOME/FOLDER/CONTAINING/INPUT/IMAGES> --split-fraction=10 --model-dir models/N200-percentile_max-block-10/segm/unet-L4-D2-None/run002/
```

The most important parameters are:

```--output-folder```: directory into which the segmentations are written.
```--model-dir```: directory containing the trained model
```--input-folder```: input folder which is scanned to identify stacks to process.
```--fname-patterns```: list of file patterns matching stacks that need to be processed. E.g. *conf488*stk, *stk
```--split-fraction```: split the work into split-fraction number of chunks. This can make it easier to distribute work over different machines and to restart processing in case of failures.


If you would like to see all parameters of the task, use:

```
luigi --module retchat.tasks.run_segmentation retina.ParallelChatbandPredictionTask --help
```


## Installation

> TODO 
