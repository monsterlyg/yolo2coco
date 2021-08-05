# put this file in yolov5 root dir
from numpy import random
import cv2
import numpy as np
import time
import argparse

import torch
import torchvision

from utils.datasets import letterbox
from utils.general import xywh2xyxy, scale_coords, non_max_suppression
from utils.plots import plot_one_box


def crop_img(img, crop_w, crop_h, overlap):
    import math
    h, w, _ = img.shape
    real_crop_h = crop_h - overlap
    pad_h = real_crop_h * math.ceil((h - crop_h) / real_crop_h) + crop_h - h
    real_crop_w = crop_w - overlap
    pad_w = real_crop_w * math.ceil((w - crop_w) / real_crop_w) + crop_w - w
    pad_img = np.zeros(shape=(h + pad_h, w + pad_w, 3))
    pad_img[:h, :w, :] = img

    cropped_imgs_offsets = list()
    h_cnt = (h + pad_h - crop_h) // real_crop_h + 1
    w_cnt = (w + pad_w - crop_w) // real_crop_w + 1
    for i in range(w_cnt):
        for j in range(h_cnt):
            offset = [real_crop_w * i, real_crop_h * j,
                      real_crop_w * i + crop_w, real_crop_h * j + crop_h]
            sub_img = pad_img[offset[1]:offset[3], offset[0]:offset[2], :]
            # cv2.imwrite("data/images/crop/{}_{}.jpg".format(i, j), sub_img)
            cropped_imgs_offsets.append([sub_img, offset])
    return cropped_imgs_offsets


def pred(img, ori_img, offset):
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = modelyolov5(img)[0]

    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)
    # Process detections
    rets = []
    for i, det in enumerate(pred):  # detections per image
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], ori_img.shape).round()
        for *xyxy, conf, cls in det:
            new_x1 = int(xyxy[0]) + offset[0]
            new_y1 = int(xyxy[1]) + offset[1]
            new_x2 = int(xyxy[2]) + offset[0]
            new_y2 = int(xyxy[3]) + offset[1]
            rets.append([new_x1, new_y1, new_x2, new_y2, float(conf), int(cls)])
        #     label = '%s %.2f' % (str(int(cls)), conf)
        #     plot_one_box(xyxy, ori_img, label=label, color=[random.randint(0, 255) for _ in range(3)],
        #                  line_thickness=3)
        # cv2.imwrite("bus.jpg", img0)
    return rets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/yolov5s.pt', help='sparse model weights')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--crop_size', type=int, default=600)
    parser.add_argument('--overlap', type=int, default=200)
    opt = parser.parse_args()
    print(opt)

    img_size = opt.img_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # another way of loading yolov5s
    modelyolov5 = torch.load(opt.weights, map_location=device)['model'].float().eval()
    modelyolov5.model[24].export = False  # onnx export
    modelyolov5.eval()

    path = 'data/images/img_1.png'
    img0 = cv2.imread(path)  # BGR
    # img = letterbox(img0, new_shape=416)[0]
    # pred(img)
    # Padded resize
    results = []
    sub_imgs_offsets = crop_img(img0, crop_w=opt.crop_size, crop_h=opt.crop_size, overlap=opt.overlap)
    for sub_img, offset in sub_imgs_offsets:
        img = letterbox(sub_img, new_shape=opt.img_size)[0]
        results.extend(pred(img, sub_img, offset))
    results = torch.Tensor(results)
    #results = post_nms(torch.Tensor(results))
    #这里的nms还可考虑 避免剔掉不同cls 做进一步优化
    kept_indice = torchvision.ops.nms(boxes=results[:, :4], scores=results[:, 4], iou_threshold=0.4)
    kept_rets = results[kept_indice, :]
    for *xyxy, conf, cls in kept_rets:
        label = '%s %.2f' % (str(int(cls)), conf)
        plot_one_box(xyxy, img0, label=label, color=[random.randint(0, 255) for _ in range(3)],
                     line_thickness=3)
    cv2.imwrite("data/images/datu_crop.jpg", img0)
