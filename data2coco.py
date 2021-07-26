import numpy as np
import json
import os
from tqdm import tqdm
import shutil
import cv2
import pandas as pd
import argparse
import pathlib


class MyData2COCO:

    def __init__(self):
        self.images = []  # 存储images键对应的数据
        self.annotations = []  # 存储annotations键对应的数据
        self.categories = []  # 存储categories键对应的数据
        self.img_id = 0  # 统计image的id
        self.ann_id = 0  # 统计annotation的id

    def _categories(self, num_categories):  # num_categories 为总的类别数
        for i in range(0, num_categories):
            category = {}
            category['id'] = i
            category['name'] = str(i)  # 可根据实际需要修改
            category['supercategory'] = 'name'  # 可根据实际需要修改
            self.categories.append(category)

    def _image(self, path, h, w):  # 获取image信息
        image = {}
        image['height'] = h
        image['width'] = w
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path)
        return image

    def _annotation(self, label_id, bbox):
        bbox = list(bbox)
        area = bbox[2] * bbox[3]
        # points = [[bbox[0], bbox[1]], [bbox[0] + bbox[2], bbox[1]], [bbox[2], bbox[1] + bbox[3]],
        #           [bbox[0], bbox[1] + bbox[3]]]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = label_id
        annotation['segmentation'] = []
        annotation['bbox'] = bbox
        annotation['iscrowd'] = 0
        annotation['area'] = area
        return annotation

    @staticmethod
    def getAnno(pred, background_index=0):
        '''
        :param pred: segmentation pred out:numpy array
        :return:
        '''
        labels = list(np.unique(pred))
        if background_index in labels:
            labels.pop(labels.index(background_index))
        boxes = list()
        box_labels = list()
        for label in labels:
            pred_one = np.zeros_like(pred)
            pred_one[pred == label] = 1
            contours = cv2.findContours(pred_one, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnts = contours[0]
            for cnt in cnts:
                x, y, w, h = cv2.boundingRect(cnt)
                boxes.append([x, y, w, h])
                box_labels.append(int(label))
        return boxes, box_labels

    def to_coco(self, img_dir, num_categories, backgrpund_index=0,
                img_txt=None, abandon_thres=None):
        """
        """
        self._categories(num_categories)  # 初始化categories基本信息
        img_names = []
        if img_txt:
            with open(img_txt) as file:
                for line in file.readlines():
                    image = os.path.basename(line.strip())
                    img_names.append(image)
        else:
            img_names = os.listdir(img_dir)
        for img_name in tqdm(img_names):
            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            h, w = img.shape
            self.images.append(self._image(img_path, h, w))
            bboxs, labels = self.getAnno(img, backgrpund_index)
            for bbox, label in zip(bboxs, labels):
                if abandon_thres:
                    if bbox[2] * bbox[3] < abandon_thres:
                        continue
                annotation = self._annotation(label, bbox)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'mydata2coco'
        instance['license'] = ['none']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    def save_coco_json(self, instance, save_path):
        with open(save_path, 'w') as fp:
            fp.write(json.dumps(instance, indent=1, separators=(',', ': ')))


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_txt', type=str, default="test.txt", )
    parser.add_argument('--img_dir', type=str, default='mian_pred/', )
    parser.add_argument('--save_json', type=str, default='result.json')
    parser.add_argument('--abandon_thres', type=int, default=800)
    opt = parser.parse_args()
    return opt


color_map = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

def drwaBox():
    opt = arg_parse()
    root = "/Users/liyinggang/Downloads/hengyi-project/"
    img_ = "pred"
    ori_ = "mian_val_ori"
    save_dir = "/Users/liyinggang/Downloads/hengyi-project/pred-box"
    os.makedirs(save_dir, exist_ok=True)

    img_list = []

    path = pathlib.Path(os.path.join(root, img_))
    if path.is_file():
        img_list.append(img_)
    elif path.is_dir():
        for img_names in os.listdir(path):
            img_list.append(img_names.strip())
    testClass = MyData2COCO()
    for pred_name in tqdm(img_list):
        pred_path = os.path.join(root, img_, pred_name)
        ori_path = os.path.join(root, ori_, pred_name).replace('png', 'jpg')
        img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        bboxes, labels = testClass.getAnno(img, background_index=3)
        ori_img = cv2.imread(ori_path)
        for bbox, label in zip(bboxes, labels):
            ori_img = cv2.rectangle(ori_img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                    color_map[label], 6)
        cv2.imwrite(os.path.join(save_dir, pred_name), ori_img)


if __name__ == '__main__':
    drwaBox()
    # opt = arg_parse()
    # data2coco = MyData2COCO()
    # instance = data2coco.to_coco(opt.img_dir, img_txt=opt.img_txt, num_categories=3, backgrpund_index=3,
    #                              abandon_thres=opt.abandon_thres)
    # data2coco.save_coco_json(instance, opt.save_json)
    #
    # os.path.exists()