# Retina-chatbands

A CNN-based detection of retina chatbands.

## Usage

### Train a new model
> TODO 

### Run a given model

```
luigi --module retchat.tasks.run_segmentation retina.ParallelChatbandPredictionTask --output-folder <SOME/OUTPUT/FOLDER> --input-folder <SOME/FOLDER/CONTAINING/INPUT/IMAGES> --split-fraction=10 --model-dir models/N200-percentile_max-block-10/segm/unet-L4-D2-None/run002/
```

> TODO extend


## Installation

> TODO 
