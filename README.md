# xview challenge

# structures

- data/
    - train.tar
    - train/
    - test.tar
    - test/
- annotation_formatter.py
- RunInstanceSegmentationInference.ipynb
- CreateSubmissionFromModel.ipynb
- ethan_create_xview_data.ipynb
- visualize_xview_data.ipynb
- TestEvaluationMetrics.ipynb

# Create Submission

Use [CreateSubmissionFromModel.ipynb](CreateSubmissionFromModel.ipynb) file.

Then create a .zip folder containing all the images (in both folders) and submit to xview2.

# Run the Model

Make sure that `detectron2_repo` submodule is up to date with the `master` branch.

Then update the following in `xview_evaluation.py`. This requires you to make the 3 folders in the xview directory. TODO: change this is they are there.
```
self._PRED_DIR = "/home/ethanweber/Documents/xview/metrics/PRED_DIR"
self._TARG_DIR = "/home/ethanweber/Documents/xview/metrics/TARG_DIR"
self._OUT_FP = "/home/ethanweber/Documents/xview/metrics/OUT_FP"
```

```
python detectron2_repo/tools/train_net.py --config-file detectron2_repo/configs/xview/mask_rcnn_R_50_FPN_3x.yaml --num-gpus 1 MODEL.WEIGHTS output/model_0054999.pth
```

# Notes

```
# avoid too many files open errors
ulimit -n 4096

# start a notebook
jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser

# simple fix to pycocotools
https://github.com/cocodataset/cocoapi/issues/49
```