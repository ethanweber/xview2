# xview2 challenge code

> Note that some of these instructions may be outdated due to the nature of the competition, but we are in the process of cleaning up code for reproducability. Training and testing will work for semantic segmentation, but the capibilities for instance segmentation with COCO data, as well as some handy visualizations, are in the works and not online yet.

This is the codebase used for our xview2 submission, which received 2nd place in Track 3: "Evaluation Only". The project is built on top of the [detectron2](https://github.com/facebookresearch/detectron2) repo by Facebook. The goal of this project is to do building damage assessment with before/after image pairs. We use a model to utilize this multi-temporal information. The prediction of our network is a 5-channel pixel-wise damage level prediction:

- 0: no building
- 1: undamaged building
- 2: building with minor damage
- 3: building with major damage
- 4: destroyed building

More specifics on the segmentation problem can be found at https://xview2.org/challenge.

# Clone and install dependencies

Start by cloning the repo: `git clone --recurse-submodules -j8 git@github.com:ethanweber/xview2.git`. Then, follow this in detail for installing dependencies: https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md. Install other packages as neeeded. A conda environment is convenient to manage everything.

# Download and format data

Download data from the xview2 site: https://xview2.org/download. Store the in the `data` folder, as decribed below. We do not use the "holdout set" in our work.

- data/
    - original_train/ (original w/o tier3)
    - test/ (test data)
    - train/ (combined w/ tier3)
    - train_gt/
    - train_images_quad/

Notice that the folders should be named in this format, where `data/train` contains both the "training set" and "additional tier3 training data". We use "train" for the experiments in our work.

# Model configuration

We store configs in the following format.

- configs/xview
    - *.yaml configs to use with detectron2

Our best config is located at `configs/xview/joint-11.yaml`. In this file, we see the following configuation:

```
_BASE_: "../Base-RCNN-FPN.yaml"
DATASETS:
  TRAIN: ("xview_semantic_damage_quad_train",)
  TEST: ("xview_semantic_damage_quad_val",)
INPUT:
  MIN_SIZE_TRAIN: (512,)
  MAX_SIZE_TRAIN: 512
  MIN_SIZE_TEST: 0
MODEL:
  META_ARCHITECTURE: "SemanticSegmentor"
  WEIGHTS: "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
  SEM_SEG_HEAD:
    NUM_CLASSES: 5
  RESNETS:
    DEPTH: 50
SOLVER:
  IMS_PER_BATCH: 8
  BASE_LR: 0.01
  STEPS: (210000, 250000)
  MAX_ITER: 270000
OUTPUT_DIR: "./outputs/output_joint_11"
TEST:
  EVAL_PERIOD: 5000
```

`xview_semantic_damage_quad_*` is the training and validation set used while training the model. This consists of pre/post images and their semantic segmentation ground truth labels. Notice that we use 512 as the image size, which is smaller than the original 1024 x 1024 images in originally downloaded xBD dataset. See `NB_make_quad_folder.ipynb` to create new this dataset, which is the origal dataset but split into quadrants for higher resolution.

Look at `detectron2_repo/detectron2/data/datasets/builtin.py`, where the datasets are registered by name. It's crucial the data exists where specified in the `data` folder. Note that this codebase originally reformed xBD to COCO, but we've moved away from this and switched to semenatic segmentation. The code is not maintained for COCO, but some notebook files demonstrate creating this data, such as `NB_create_xview_data.ipynb` and `NB_visualize_xview_coco_data.ipynb`.

# Train the network with a config

Go to the main directory and run a training process.

Example execution with one GPU. This is for the baseline localization model.
```
cd xview
export CUDA_VISIBLE_DEVICES=0
python detectron2_repo/tools/train_net.py --config-file configs/xview/joint-11.yaml
```

To run from a checkpoint: (make sure path to checkpoint is correct)
```
python detectron2_repo/tools/train_net.py --config-file configs/xview/joint-11.yaml MODEL.WEIGHTS "outputs/output_joint_11/model_0054999.pth"
```

# Looking at results

We compute the metrics used by xview2 and display them in Tensorboard during training. Original code for the metrics is located at `detectron2_repo/detectron2/evaluation/xview_evaluation.py`.

# Create submission

Use [NB_create_submission_from_model-quad.ipynb](NB_create_submission_from_model-quad.ipynb) to create the submission.

These two folders will be made from the script.
```
mkdir SUBMISSION_LOCALIZATION
mkdir SUBMISSION_DAMAGE
```

Then create a .zip folder containing all the images (in both folders) and submit to xview2. It will be marked by a timestamp. Upload this diretly to the xview website.

# View submission

Use [NB_visualize_submission_folder.ipynb](NB_visualize_submission_folder.ipynb) file to look at some of the predictions in your most recent submission folder.


# Handy notes

```
# avoid too many files open errors
ulimit -n 4096

# start a notebook
jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser

# simple fix to pycocotools
https://github.com/cocodataset/cocoapi/issues/49

# activate conda
eval "$(conda shell.bash hook)" && conda activate xview

# ssh / github issues
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa_personal

# transfer learning
https://github.com/facebookresearch/detectron2/issues/222
```
