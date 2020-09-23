# Retina-chatbands

A CNN-based detection of retina chatbands.

## Usage


### Run a given model

```
luigi --module retchat.tasks.run_segmentation retina.ParallelChatbandPredictionTask --output-folder <SOME/OUTPUT/FOLDER> --input-folder <SOME/FOLDER/CONTAINING/INPUT/IMAGES> --split-fraction=10 --model-dir models/N200-percentile_max-block-10/segm/unet-L4-D2-None/run002/
```

The most important parameters are:

```--output-folder```: directory into which the segmentations are written.

```--model-dir```: directory containing the trained model

```--input-folder```: input folder which is scanned to identify stacks to process.

```--fname-patterns```: list of file patterns matching stacks that need to be processed. E.g. ```*conf488*.stk, *.czi```.

```--split-fraction```: split the work into split-fraction number of chunks. This can make it easier to distribute work over different machines and to restart processing in case of failures.


If you would like to see all parameters of the task, use:

```
luigi --module retchat.tasks.run_segmentation retina.ParallelChatbandPredictionTask --help
```

### Train a new model

Training a model is done in two steps. First, the training data is
being prepared with ```prepare_dataset.py```. Then, the model is
trained on the dataset using ```train_segmentation.py```.

Each of them can be called using ```python <script.py> ...```. For details on the arguments, use

```
conda activate retina-env  # assuming you have setup a conda environment called retina-env.
cd faim-retina-chatbands/
python prepare_dataset.py --help
python train_segmentation.py --help
```

You find example calls of those two scripts in ```scripts/```. To run
these examples, simply call:

```
bash scripts/run_preparation.sh
bash scripts/run_single_training.sh
```

Note that these bash scripts contain the paths to the data on
tungsten. If you are running outside FMI, make sure to adjust the
paths accordingly.



## Installation

Frist, clone or download this repository.

Create a python environment (Optional, but recommended):

```
conda create -n retina-env python=3.6
conda activate retina-env
```

If you are on a machine with a CUDA-compatible GPU, you should first install the gpu version of tensorflow:

```
conda install tensorflow-gpu=1.12
```


Install the package into the current environment
```
cd faim-retina-chatbands/
pip install .
```
