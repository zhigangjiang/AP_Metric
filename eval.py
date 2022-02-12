"""
@author: Zhigang Jiang
@date: 2022/2/12
@description:
"""
from ap_metric import calc_eval
from utils import calc_ious, show


def get_dt_results(classname):
    return {
        'id_1': {
            'boxes': [[55, 50, 116, 117], [176, 124, 261, 219], [398, 116, 458, 184]],
            'scores': [0.3, 0.9, 0.5]
        },
        'id_2': {
            'boxes': [[30, 87, 190, 212], [7, 194, 132, 261], [214, 20, 286, 87], [337, 60, 397, 127],
                      [300, 166, 398, 261]],
            'scores': [0.85, 0.35, 0.1, 0.7, 0.8]
        },
        'id_3': {
            'boxes': [[151, 138, 313, 192]],
            'scores': [0.45]
        },
        'id_4': {
            'boxes': [],
            'scores': []
        },
        'id_5': {
            'boxes': [],
            'scores': []
        }
    }


def get_gt_results(classname):
    return {
        'id_1': {
            'boxes': [[185, 130, 286, 212], [59, 35, 108, 102], [362, 59, 423, 156]],
            'difficulties': [False, False, True]
        },
        'id_2': {
            'boxes': [[288, 180, 411, 247], [11, 97, 153, 207], [343, 58, 404, 121]],
            'difficulties': [False, False, False]
        },
        'id_3': {
            'boxes': [],
            'difficulties': []
        },
        'id_4': {
            'boxes': [[110, 82, 93, 121]],
            'difficulties': [True]
        },
        'id_5': {
            'boxes': [[94, 46, 153, 105]],
            'difficulties': [False]
        }
    }


def eval(classname, iou_thresh=0.5, use_07_metric=True, show_curve=False):
    dt_results = get_dt_results(classname)
    gt_results = get_gt_results(classname)
    # show(dt_results, gt_results)
    ids = dt_results.keys()
    rec, prec, ap = calc_eval(ids, dt_results, gt_results, calc_ious, iou_thresh, use_07_metric, show_curve)
    return rec, prec, ap


rec, prec, ap = eval(classname='', use_07_metric=True, show_curve=True)
print(ap)
