"""
This file is used for formatting annotations for the xview training set.

Resources:
http://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
"""
from pycocotools import mask as maskUtils
import json
import glob
import os
import numpy as np
from PIL import Image # (pip install Pillow)
import numpy as np                                 # (pip install numpy)
from skimage import measure                        # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon # (pip install Shapely)


class AnnotationFormatter(object):
    """
    Annotation formatter.
    """

    info = {
        "description": "Inria Aerial Imagery Dataset",
        "url": "https://project.inria.fr/aerialimagelabeling/"
    }
    images = []
    annotations = []
    categories = [
        {
            "supercategory": "building",
            "id": 1,
            "name": "building"
        }
    ]

    def __init__(self):
        self.image_count = 0
        self.annotation_count = 0
        self.images = []
        self.annotations = []

    def create_sub_masks_grayscale(self,mask_image):
        width, height = mask_image.size
        # Initialize a dictionary of sub-masks indexed by RGB colors
        sub_masks = {}
        for x in range(width):
            for y in range(height):
                # Get the RGB values of the pixel
                pixel = mask_image.getpixel((x,y))

                # If the pixel is not black...
                if pixel != 0:
                    # Check to see if we've created a sub-mask...
                    pixel_str = str(pixel)
                    sub_mask = sub_masks.get(pixel_str)
                    if sub_mask is None:
                       # Create a sub-mask (one bit per pixel) and add to the dictionary
                        # Note: we add 1 pixel of padding in each direction
                        # because the contours module doesn't handle cases
                        # where pixels bleed to the edge of the image
                        sub_masks[pixel_str] = Image.new('1', (width+2, height+2))

                    # Set the pixel value to 1 (default is 0), accounting for padding
                    sub_masks[pixel_str].putpixel((x+1, y+1), 1)

        return sub_masks

    def create_sub_mask_annotation(self,sub_mask, image_id, category_id, annotation_id, is_crowd):
        # Find contours (boundary lines) around each sub-mask
        # Note: there could be multiple contours if the object
        # is partially occluded. (E.g. an elephant behind a tree)
        contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

        segmentations = []
        polygons = []
        for contour in contours:
            # Flip from (row, col) representation to (x, y)
            # and subtract the padding pixel
            for i in range(len(contour)):
                row, col = contour[i]
                contour[i] = (col - 1, row - 1)

            # Make a polygon and simplify it
            try:
                poly = Polygon(contour)
                poly = poly.simplify(1.0, preserve_topology=False)
                if poly.exterior != None:
                    polygons.append(poly)
                    segmentation = np.array(poly.exterior.coords).ravel().tolist()
                    segmentations.append(segmentation)
            except:
                pass
        # Combine the polygons to calculate the bounding box and area
        multi_poly = MultiPolygon(polygons)
        x, y, max_x, max_y = multi_poly.bounds
        width = max_x - x
        height = max_y - y
        bbox = (x, y, width, height)
        area = multi_poly.area

        annotation = {
            'segmentation': segmentations,
            'iscrowd': is_crowd,
            'image_id': image_id,
            'category_id': category_id,
            'id': annotation_id,
            'bbox': bbox,
            'area': area
        }

        return annotation
    is_crowd = 0


    def add_image_from_filename(self, file_name):
        self.image_count += 1
        mask_image = Image.open(file_name)

        width, height = mask_image.size
        image_id = self.image_count

        # read filename
        image_data = {
            "file_name": os.path.basename(file_name),
            "height": height,
            "width": width,
            "id": image_id
        }

        self.images.append(image_data)

        sub_masks = self.create_sub_masks_grayscale(mask_image)
        
        self.annotation_count+=1
        annotation_id = self.annotation_count
        
        annotation_data = []
        is_crowd = 0
        category_id = 1
        for color, sub_mask in sub_masks.items():
            annotation = self.create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd)
            self.annotations.append(annotation)

    def write_to_json(self, filename):
        data = {
            "info": self.info,
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories
        }
        with open(filename, 'w') as outfile:
            json.dump(data, outfile)
    


if __name__ == "__main__":
    formatter = AnnotationFormatter()
    annotation_files = glob.glob("data/inria/train/gt/*")
    from tqdm import tqdm
    for filename in tqdm(annotation_files):
        formatter.add_image_from_filename(filename)
    formatter.write_to_json("inria_buildings_annotations.json")