import sys
print("Running {}".format(sys.argv[0]))

test_pre = sys.argv[1]
test_post = sys.argv[2]
test_localization = sys.argv[3]
test_damage = sys.argv[4]

import torch, torchvision
from detectron2.evaluation.xview_evaluation import *

# Some basic setup
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import os
import numpy as np
import json
import shutil
import cv2
import random
import glob

import matplotlib.pyplot as plt

# Load damage model.
cfg_damage = get_cfg()
DAMAGE_MODEL_CONFIG = "./configs/xview/joint-11.yaml"
cfg_damage.merge_from_file(DAMAGE_MODEL_CONFIG)
# Load damage checkpoint.
cfg_damage.MODEL.WEIGHTS = os.path.join("model_weights.pth")
# cfg_damage.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 # set the testing threshold for this model
predictor_damage = DefaultPredictor(cfg_damage)

image_filenames = sorted(glob.glob("data/test_images_quad/*"))
count = 0
bl = None
br = None
tl = None
tr = None
temp = np.zeros((1024, 1024)).astype(int)
    
# Load the images.
image = cv2.imread(test_post)
pre_image = cv2.imread(test_pre)

# tl
temp_image = image[0:512,0:512]
temp_pre_image = pre_image[0:512,0:512]
outputs = predictor_damage(temp_image, temp_pre_image)
output = outputs["sem_seg"].argmax(dim=0).cpu()
tl = np.array(output, dtype=np.int)

# tr
temp_image = image[0:512,512:1024]
temp_pre_image = pre_image[0:512,512:1024]
outputs = predictor_damage(temp_image, temp_pre_image)
output = outputs["sem_seg"].argmax(dim=0).cpu()
tr = np.array(output, dtype=np.int)

# bl
temp_image = image[512:1024,0:512]
temp_pre_image = pre_image[512:1024,0:512]
outputs = predictor_damage(temp_image, temp_pre_image)
output = outputs["sem_seg"].argmax(dim=0).cpu()
bl = np.array(output, dtype=np.int)

# br
temp_image = image[512:1024,512:1024]
temp_pre_image = pre_image[512:1024,512:1024]
outputs = predictor_damage(temp_image, temp_pre_image)
output = outputs["sem_seg"].argmax(dim=0).cpu()
br = np.array(output, dtype=np.int)

temp[0:512,0:512] = tl
temp[0:512,512:1024] = tr
temp[512:1024,0:512] = bl
temp[512:1024,512:1024] = br

# Write to filepaths.
cv2.imwrite(test_localization, temp)
cv2.imwrite(test_damage, temp)
