#!/usr/bin/env python3
import json
import os
import re
import datetime

import cv2
import numpy as np
from itertools import groupby
from skimage import measure
from pycocotools import mask

convert = lambda text: int(text) if text.isdigit() else text.lower()
natrual_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))

    return rle


def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation

    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.

    """
    polygons = []
    contours = measure.find_contours(binary_mask, 0.5)
    # cvContours = cv2.findContours(binary_mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        polygons.append(segmentation)

    return polygons


def create_image_info(image_id, file_name, image_size,
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):
    image_info = {
        "id": image_id,
        "file_name": file_name,
        "width": image_size[1],
        "height": image_size[0],
        "date_captured": date_captured,
        "license": license_id,
        "coco_url": coco_url,
        "flickr_url": flickr_url
    }

    return image_info


def create_annotation_info(annotation_id, image_id, category_info, binary_mask,
                           tolerance=2, bounding_box=None):

    binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    area = mask.area(binary_mask_encoded)
    if area < 1:
        return None

    if bounding_box is None:
        bounding_box = mask.toBbox(binary_mask_encoded)

    if category_info["is_crowd"]:
        is_crowd = 1
        segmentation = binary_mask_to_rle(binary_mask)
    else:
        is_crowd = 0
        segmentation = binary_mask_to_polygon(binary_mask, tolerance)
        if not segmentation:
            return None
    annotation_infos = []
    for seg in segmentation:
        annotation_info = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_info["id"],
            "iscrowd": is_crowd,
            "area": area.tolist(),
            "bbox": bounding_box.tolist(),
            "segmentation": [seg],
            "width": binary_mask.shape[1],
            "height": binary_mask.shape[0],
        }
        annotation_id += 1
        annotation_infos.append(annotation_info)
    return annotation_infos, annotation_id


CATEGORIES = [
    {
        'id': 1,
        'name': 'maosi',
        'supercategory': 'flaws',
    },
    {
        'id': 2,
        'name': 'youwu',
        'supercategory': 'flaws',
    },
]


def main():
    coco_output = {
        "info": {},
        "licenses": [],
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1
    # filter for jpeg images
    image_root = "/Users/liyinggang/Downloads/hengshen_maosi/mask"
    image_names = os.listdir(image_root)
    image_files = [os.path.join(image_root, image_name) for image_name in image_names]
    # go through each image
    for image_filename in image_files[1:]:

        image = cv2.imread(image_filename)
        image_info = create_image_info(
            image_id, os.path.basename(image_filename), image.shape[:2])
        coco_output["images"].append(image_info)

        # filter for associated png annotations
        mask_img = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
        unique_labels = np.unique(mask_img)
        for class_id in unique_labels:
            if class_id == 0:  # 背景
                continue
            binary_mask = np.zeros(mask_img.shape)
            binary_mask[mask_img == class_id] = 1

            category_info = {'id': int(class_id), 'is_crowd': 0}
            annotation_infos, segmentation_id = create_annotation_info(
                segmentation_id, image_id, category_info, binary_mask, tolerance=0)

            if annotation_infos is not None:
                coco_output["annotations"].extend(annotation_infos,)
            # segmentation_id += 1

        image_id = image_id + 1

    with open('mask2coco.json', 'w') as output_json_file:
        json.dump(coco_output, output_json_file, indent=2)


if __name__ == "__main__":
    main()
