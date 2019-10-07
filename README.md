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

# using maskrcnn-benchmark

```
update paths_catalog.py

https://github.com/facebookresearch/maskrcnn-benchmark/issues/18

https://github.com/facebookresearch/maskrcnn-benchmark/pull/1053

cd xview
python maskrcnn-benchmark/tools/train_net.py --config-file e2e_mask_rcnn_R_50_FPN_1x.yaml
```