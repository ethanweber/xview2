# xview challenge

# structures

- data/
    - train/
    - test/
- cocoapi/
- annotation_formatter.py

# coco tools

```
cd cocoapi/PythonAPI
python setup.py build
python setup.py install
```

# avoid too many files open error

```
ulimit -n 4096
```

# using maskrcnn-benchmark

```
update paths_catalog.py

https://github.com/facebookresearch/maskrcnn-benchmark/issues/18

https://github.com/facebookresearch/maskrcnn-benchmark/pull/1053

cd xview
export CUDA_VISIBLE_DEVICES=0
conda activate atenta #TODO: update this to explain what the conda setup procedure is
python maskrcnn-benchmark/tools/train_net.py --config-file e2e_mask_rcnn_R_50_FPN_1x.yaml

# don't show errors

python maskrcnn-benchmark/tools/train_net.py --config-file e2e_mask_rcnn_R_50_FPN_1x.yaml 2>&1 | grep -v " UserWarning: indexing with dtype torch.uint8"
```

# notes on training

log.txt is created for every run. remove this if you with to start from scratch

# start a notebook

```
jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser
```

# simple fix to pycocotools

https://github.com/cocodataset/cocoapi/issues/49

# to run

```
python detectron2_repo/tools/train_net.py --config-file detectron2_repo/configs/xview/mask_rcnn_R_50_FPN_3x.yaml --num-gpus 1
```