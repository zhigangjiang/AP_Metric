"""
@author: Zhigang Jiang
@date: 2022/2/12
@description:
"""

import numpy as np
from utils import draw_pr


def calc_dt_positive(dt_boxes: np.array, dt_scores: np.array,
                     gt_boxes: np.array, gt_difficulties: np.array,
                     calc_ious_fun, iou_thresh=0.5) -> (float, list):
    """
    判断检测为阳例的情况，正阳(TP)或负阳例(FP)
    :param dt_boxes: [n_dt x 4] 一张图的检测boxes
    :param dt_scores: [n_dt x 1] 一张图的检测scores
    :param gt_boxes: [n_gt x 4] 一张图的真实boxes
    :param gt_difficulties: [n_gt x 1] 一张图的真实difficulties标志，如果对应的gt_box是difficult的，则跳过
    :param calc_ious_fun: 计算iou的函数
    :param iou_thresh: iou阈值
    :return:
    n_gt: 一张图的去除difficult的gt_box数量
    results: [[score: float, tp: int], [score, tp], ...]
    """
    n_dt = len(dt_boxes)
    n_gt = len(gt_boxes) - sum(gt_difficulties)
    results = [[dt_score, 0] for dt_score in dt_scores]
    if n_gt == 0 or n_dt == 0:
        return n_gt, results

    ious, ious_idx = calc_ious_fun(gt_boxes, dt_boxes)
    detected = [False] * len(gt_boxes)  # gt检测标志

    for i, (iou, gt_i) in enumerate(zip(ious, ious_idx)):
        if iou > iou_thresh:  # 注意这里要先判断iou在判断difficult
            if gt_difficulties[gt_i]:  # 如果是difficult则增加跳过标志2
                results[i][1] = None
            elif not detected[gt_i]:  # 对应的gt没被匹配过
                results[i][1] = 1
                detected[gt_i] = True

    # 过滤difficult的结果
    results = [result for result in results if result[1] is not None]
    return n_gt, results


def calc_pr(gt_num: float, results: list) -> (np.array, np.array):
    """
    :param gt_num: 所有的的去除difficult的gt_box数量
    :param results: [[score: float, tp: int], [score, tp], ...]
    :return:
    """
    if gt_num == 0 or len(results) == 0:
        return [0], [0]

    results.sort(key=lambda result: -result[0])
    tp_list = [result[1] for result in results]  # 通过score排序
    recall = np.cumsum(tp_list) / gt_num
    n_dt = len(tp_list)
    precision = np.cumsum(tp_list) / (np.arange(n_dt) + 1)

    return recall, precision


def voc_ap(rec, prec, use_07_metric=False, show_curve=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """

    # correct AP calculation
    # first append sentinel values at the end
    prec = np.concatenate(([0.], prec, [0.]))
    rec = np.concatenate(([0.], rec, [1.]))
    org_precc = prec.copy()

    if use_07_metric:
        # 11 point metric
        ap = 0.
        max_prec = []
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            max_prec.append(p)
            ap = ap + p / 11.
        if show_curve:
            draw_pr(org_precc, rec, max_prec, np.arange(0., 1.1, 0.1), ap, '11point')
    else:

        # compute the precision envelope
        for i in range(prec.size - 1, 0, -1):
            prec[i - 1] = np.maximum(prec[i - 1], prec[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(rec[1:] != rec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((rec[i + 1] - rec[i]) * prec[i + 1])

        if show_curve:
            draw_pr(org_precc, rec, prec, rec, ap)
    return ap


def calc_eval(ids, dt_results, gt_results, calc_ious_fun, iou_thresh, use_07_metric, show_curve):
    all_n_gt = 0
    all_results = []

    for id in ids:
        gt_boxes = np.array(gt_results[id]['boxes'] if id in gt_results else [])
        gt_difficulties = np.array(gt_results[id]['difficulties'] if id in gt_results else [])
        pred_boxes = np.array(dt_results[id]['boxes'] if id in dt_results else [])
        pred_scores = np.array(dt_results[id]['scores'] if id in dt_results else [])

        n_gt, results = calc_dt_positive(pred_boxes, pred_scores, gt_boxes, gt_difficulties, calc_ious_fun, iou_thresh)
        all_n_gt += n_gt
        all_results += results

    rec, prec = calc_pr(all_n_gt, all_results)
    ap = voc_ap(rec, prec, use_07_metric, show_curve=show_curve)
    return rec, prec, ap
