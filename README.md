# xview challenge

# structures

- data/
    - train.tar
    - train/
    - test.tar
    - test/
- annotation_formatter.py
- ethan_create_xview_data.ipynb
- ethan_inference_xview.ipynb
- ethan_visualize_xview_data.ipynb

# notes

```
# avoid too many files open errors
ulimit -n 4096

# start a notebook
jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser

# simple fix to pycocotools
https://github.com/cocodataset/cocoapi/issues/49

# to run
python detectron2_repo/tools/train_net.py --config-file detectron2_repo/configs/xview/mask_rcnn_R_50_FPN_3x.yaml --num-gpus 1
```