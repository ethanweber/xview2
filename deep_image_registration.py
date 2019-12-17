from __future__ import print_function
import Registration
import matplotlib.pyplot as plt
from utils.utils import *
import cv2
import glob
import datetime
from os.path import join as pjoin

# http://code.activestate.com/recipes/303060-group-a-list-into-sequential-n-tuples/
def group(lst, n):
    """group([0,3,4,10,2,3], 2) => [(0,3), (4,10), (2,3)]
    
    Group a list into consecutive n-tuples. Incomplete tuples are
    discarded e.g.
    
    >>> group(range(10), 3)
    [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    """
    return zip(*[lst[i::n] for i in range(n)]) 


aligned_dir = 'aligned'
already_aligned_images = sorted(glob.glob("aligned/*"))
image_files = sorted(glob.glob("data_to_align/*"))
image_files_as_pairs = list(group(image_files, 2))

already_aligned_images = sorted(glob.glob("aligned/*"))
pre_images_name = [x.split('/')[1][8:] for x in already_aligned_images]
for post,pre in image_files_as_pairs[700:850]:
    print(datetime.datetime.now())
# designate image path here
    IX_path = pre
    IY_path = post
    pre_name = pre.split('/')[1]
    
    if pre_name not in pre_images_name:
        aligned_pre_name = 'aligned-'+pre_name
        aligned_file = pjoin(aligned_dir, aligned_pre_name)

        IX = cv2.imread(IX_path)
        IY = cv2.imread(IY_path)

        #initialize
        reg = Registration.CNN()
        #register
        X, Y, Z = reg.register(IX, IY)
        #generate regsitered image using TPS
        registered = tps_warp(Y, Z, IY, IX.shape)
        print(registered.shape)
        cv2.imwrite( aligned_file, registered );