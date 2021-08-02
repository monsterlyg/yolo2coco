# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
"""
# This module provide
# Authors: jiaohanzhe(jiaohanzhe@baidu.com)
# Date: 2019/9/24 5:36 下午
"""
import os

import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

coco = COCO('mask2coco.json')
dataDir = '/Users/liyinggang/Downloads/hengshen_maosi/images'
writepath = '/Users/liyinggang/Downloads/hengshen_maosi/vis'

if not os.path.exists(writepath):
    os.mkdir(writepath)

catIds = coco.getCatIds()
# AREA_THRESHOLD = 100
mapper = {
    0: "background",
    1: 'maosi',
}
color_map = [(255, 255, 255), (255, 255, 0), (0, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)]

cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

# imgIds = coco.getImgIds(imgIds = [324158])
imgIds = coco.getImgIds()
print('IMAGE NUM: {}'.format(len(imgIds)))
imgId = np.random.randint(0, len(imgIds))
for i in tqdm(range(len(imgIds))):
    img = coco.loadImgs(imgIds[i])[0]
    img_path = os.path.join(dataDir, os.path.basename(img['file_name'])[:-4]+".jpeg")
    if not os.path.exists(img_path):
        continue
    orgimg = cv2.imread(img_path)
    cpimg = orgimg.copy()
    annIds = coco.getAnnIds(imgIds=[img['id']], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    for j in range(len(anns)):
        category = anns[j]["category_id"]
        # rle = coco.annToMask(anns[j])
        for seg in anns[j]['segmentation']:
            polypoints = []
            for k in range(len(seg) // 2):
                ipointx = seg[k * 2]
                ipointy = seg[k * 2 + 1]
                ipoint = [ipointx, ipointy]
                polypoints.append(ipoint)

            name = nms[anns[j]['category_id'] - 1]
            cv2.polylines(orgimg, [np.array(polypoints, dtype=np.int32)], True, color_map[category], 1)

    res = np.hstack((cpimg, orgimg))
    print(os.path.join(writepath, os.path.basename(img['file_name'])))
    cv2.imwrite(os.path.join(writepath, os.path.basename(img['file_name'])), res)
