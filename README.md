# xview challenge

Here we describe how to use this repo to train models, run experiments, and create submissions for the xview challenge.

# Repo Structure

- data/
    - original_train/ (original w/o tier3)
    - test/ (test data)
    - train/ (combined w/ tier3)
    - inria/
- datasets/
    - *.json files that have COCO datasets
- configs/
    - *.yaml configs to use with detectron2

# Cloning

Make sure to clone the detectron2 submodule.

`git clone --recurse-submodules -j8 git@github.com:ethanweber/xview.git`

# Install Dependencies

Follow this in detail: https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md. Install other packages as neeeded. A conda environment is easiest to manage everything.

# Create Data

Use [ethan_create_xview_data.ipynb](ethan_create_xview_data.ipynb) to create datasets. Note that this requires structuring the data folder as described in the `Repo Structure` section. This will create and put data in the `datasets/` directory.

# Make Sure Datasets are Registered with Detectron2

Go to `detectron2_repo/detectron2/data/datasets/builtin.py` and make sure that the dataset you just created is properly registered with detectron2. Most of the defaults are included with the repo.

# Confirm data is created correctly.

Use [visualize_xview_data.ipynb](visualize_xview_data.ipynb) to confirm that your dataset has been created properly in COCO format. Note that the datasets should have different number of classes for pre/localization and post/disaster datasets.

# Train the a Network with a Config

Go to the main directory and run a training process.

Example execution with one GPU. This is for the baseline localization model.
```
cd xview
export CUDA_VISIBLE_DEVICES=0
python detectron2_repo/tools/train_net.py --config-file configs/xview/mask_rcnn_R_50_FPN_1x-localization-00.yaml
```

To run from a checkpoint: (make sure path to checkpoint is correct)
```
python detectron2_repo/tools/train_net.py --config-file configs/xview/mask_rcnn_R_50_FPN_1x-localization-00.yaml MODEL.WEIGHTS "outputs/output_localization_00/model_0054999.pth"
```

# Testing Validation

The training script will report validation for the metrics that `xview` will test. Note that these are created with a file at `detectron2_repo/detectron2/evaluation/xview_evaluation.py`. The metrics are generated with `metrics/xview2_metrics`. All data will be stored in the OUTPUT_DIR defined by the .yaml config files.

Use [TestEvaluationMetrics.ipynb](TestEvaluationMetrics.ipynb) on a folder in `outputs/` (for the training session of interest) to see what some of the predictions vs. ground truths are. Be sure to specifiy if the challenge is localization/pre or disaster/post.

# Create Submission

Create two folders:
```
cd xview
mkdir SUBMISSION_LOCALIZATION
mkdir SUBMISSION_DAMAGE
```

Use [CreateSubmissionFromModel.ipynb](CreateSubmissionFromModel.ipynb) file.

Then create a .zip folder containing all the images (in both folders) and submit to xview2. It will be marked by a timestamp. Upload this diretly to the xview website.

# View Submission.

Use [TestSubmission.ipynb](TestSubmission.ipynb) file to look at some of the predictions in your most recent submission folder.


# Notes

```
# avoid too many files open errors
ulimit -n 4096

# start a notebook
jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser

# simple fix to pycocotools
https://github.com/cocodataset/cocoapi/issues/49

# activate conda
eval "$(conda shell.bash hook)" && conda activate xview

# run tensorboard
tensorboard --logdir outputs/output_localization_00/ --host=0.0.0.0 --port 8001
```