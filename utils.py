"""
@author: Zhigang Jiang
@date: 2022/2/12
@description:
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


def calc_ious(gt_boxes: np.array, dt_boxes: np.array):
    """
    普通图像的iou计算方法
    :param gt_boxes: [n_dt x 4]
    :param dt_boxes: [n_gt x 4]
    :return:
     max_ious: [n_dt x 4] 每个dt_box最匹配的gt_box的iou值，用于判断是否大于阈值
     max_ious_idx: [n_dt x 1] 每个dt_box最匹配的gt_box的索引，用于判断对应的gt_box是否为difficult
    """
    n_dt = len(dt_boxes)
    n_gt = len(gt_boxes)

    # ious记录了每个dt_box与gt_box的iou
    ious = np.zeros((n_dt, n_gt))
    for i in range(n_dt):
        dt_box = dt_boxes[i]
        i_x_min = np.maximum(gt_boxes[:, 0], dt_box[0])
        i_y_min = np.maximum(gt_boxes[:, 1], dt_box[1])
        i_x_max = np.minimum(gt_boxes[:, 2], dt_box[2])
        i_y_max = np.minimum(gt_boxes[:, 3], dt_box[3])
        i_w = np.maximum(i_x_max - i_x_min, 0.)
        i_h = np.maximum(i_y_max - i_y_min, 0.)
        inters_area = i_w * i_h
        dt_area = (dt_box[2] - dt_box[0]) * (dt_box[3] - dt_box[1])
        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        ious[i] = inters_area / (dt_area + gt_areas - inters_area)

    # 每个dt_box最匹配的gt_box的iou值
    max_ious = ious.max(axis=1)
    # 每个dt_box最匹配的gt_box的索引
    max_ious_idx = ious.argmax(axis=1)
    return max_ious, max_ious_idx


def draw_pr(org_prec, org_rec, max_prec, max_rec, ap, name='Approximated'):
    plt.plot(org_rec, org_prec,
             color='r', marker='o', mec='m', ms=3)
    plt.step(max_rec, max_prec,
             color='b', where='pre')
    plt.fill_between(max_rec, max_prec, step='pre', alpha=0.2,
                     color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1])
    plt.title(f'Precision-Recall curve({name}): AP={ap:0.5f}', fontsize=12)
    plt.xticks(np.arange(0, 1, 0.1))
    plt.yticks(np.arange(0, 1, 0.1))
    plt.grid(True)
    plt.legend(('PR-curve', f'{name}-PR-curve', f'{name}-AP'),
               loc='upper right', fontsize=6)
    plt.show()
    pass


def show(dt_results, gt_results):
    def draw_boxes(board, boxes, color, scores=None, difficulties=None):
        for i, box in enumerate(boxes):
            cv2.rectangle(board, (box[0], box[1]), (box[2], box[3]), color)
            if scores is not None:
                cv2.putText(board, str(scores[i]), ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
            if difficulties is not None and difficulties[i]:
                cv2.putText(board, str('X'), ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

    for k in dt_results:
        dt_result = dt_results[k]
        gt_result = gt_results[k]
        board = np.ones((284, 512, 3))
        draw_boxes(board, dt_result['boxes'], [0, 0, 1], scores=dt_result['scores'])
        draw_boxes(board, gt_result['boxes'], [0, 1, 0], difficulties=gt_result['difficulties'])
        plt.imshow(board)
        plt.title(k)
        plt.show()
