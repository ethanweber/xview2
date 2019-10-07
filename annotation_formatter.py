"""
This file is used for formatting annotations.

Resources:
http://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
"""
from pycocotools import mask as maskUtils
import json
import glob
import os
import numpy as np

class AnnotationFormatter(object):
    """
    Annotation formatter.
    """

    info = {
        "description": "xview2 dataset",
        "url": "https://xview2.org/"
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

    image_uid_to_image_id = {}
    annotation_uid_to_annotation_id = {}


    def __init__(self):
        self.image_count = 0
        self.annotation_count = 0

    def add_image_from_filename(self, filename):
        with open (filename, "r") as myfile:
            json_string = u'{}'.format(myfile.read())
        json_data = json.loads(json_string)
        width = int(json_data["metadata"]["width"])
        height = int(json_data["metadata"]["height"])
        image_uid = json_data["metadata"]["id"]
        image_id = self.image_count
        self.image_uid_to_image_id[image_uid] = image_id

        # read filename

        image_data = {
            "file_name": json_data["metadata"]["img_name"],
            "height": height,
            "width": width,
            "id": image_id
        }
        self.images.append(image_data)

        xy_features = json_data["features"]["xy"]
        for xy_feature in xy_features:
            # try:
            #     print(xy_feature["properties"]["subtype"])
            # except:
            #     pass
            polygon_text = xy_feature["wkt"]
            annotation_uid = xy_feature["properties"]["uid"]
            self.annotation_uid_to_annotation_id[annotation_uid] = self.annotation_count

            polygon_values = polygon_text[
                polygon_text.find("((") + 2:-2].replace(",", "").split(" ")
            
            polygon = []
            x_coords = []
            y_coords = []
            for i in range(0, len(polygon_values), 2):
                x, y = float(polygon_values[i]), float(polygon_values[i+1])
                x_coords.append(x)
                y_coords.append(y)
                polygon.append(x)
                polygon.append(y)
            
            bounding_box_width = max(x_coords) - min(x_coords)
            bounding_box_height = max(y_coords) - min(y_coords)
            bounding_box = [
                float(min(x_coords)),
                float(min(y_coords)),
                float(bounding_box_width),
                float(bounding_box_height)
            ]
            
            # height and width from original image
            rle = maskUtils.frPyObjects([polygon], height, width)
            area = maskUtils.area(rle)[0]

            annotation_data = {
                "segmentation": [polygon],
                "iscrowd": 0,
                "image_id": image_id,
                "category_id": 1,
                "id": self.annotation_count,
                "bbox": bounding_box,
                "area": float(area)
            }
            self.annotations.append(annotation_data)

            self.annotation_count += 1

        self.image_count += 1

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
    annotation_files = glob.glob("data/train/labels/*")
    for filename in annotation_files:
        formatter.add_image_from_filename(filename)
    formatter.write_to_json("coco_format.json")