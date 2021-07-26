import json
import argparse

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class myParams:
    '''
    Params for coco evaluation api
    '''
    def __init__(self, iouType='segm', low_iou=0.1):
        self.low_iou_thres = low_iou
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None

    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(self.low_iou_thres, 0.95, int(np.round((0.95 - self.low_iou_thres) / .05)) + 1,
                                   endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(self.low_iou_thres, 0.95, int(np.round((0.95 - self.low_iou_thres) / .05)) + 1,
                                   endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1
        self.kpt_oks_sigmas = np.array(
            [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0


class MY_COCOeval(COCOeval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm', low_iou=0.1, high_iou=0.4):
        super().__init__(cocoGt, cocoDt, iouType)
        self.low_iou_thres = low_iou
        self.high_iou_thres = high_iou
        self.params = myParams(iouType="bbox", low_iou=self.low_iou_thres)

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''

        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=self.low_iou_thres, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=self.high_iou_thres, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=self.low_iou_thres)
            stats[2] = _summarize(1, maxDets=20, iouThr=self.high_iou_thres)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=self.low_iou_thres)
            stats[7] = _summarize(0, maxDets=20, iouThr=self.high_iou_thres)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()


def convert_coco_result(result_file):
    results = []
    with open(result_file, 'r') as fp:
        pred_annos = json.load(fp)
        for anno in pred_annos['annotations']:
            res = {}
            res["image_id"] = anno["image_id"]
            res["category_id"] = anno["category_id"]
            res["bbox"] = anno["bbox"]
            res["score"] = 1.0
            results.append(res)
    return results


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_json', type=str, )
    parser.add_argument('--dt_json', type=str, )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = arg_parse()
    opt.gt_json = "/Users/liyinggang/Downloads/val.json"
    opt.dt_json = "/Users/liyinggang/Downloads/s_val_outputs.json"
    cocoGt = COCO(opt.gt_json)  # 标注文件的路径及文件名，json文件形式

    cocoDt = cocoGt.loadRes(opt.dt_json)  # 自己的生成的结果的路径及文件名，json文件形式
    cocoEval = MY_COCOeval(cocoGt, cocoDt, "bbox", low_iou=0.1)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
