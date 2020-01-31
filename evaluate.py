import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
import cv2

class evaluation():
    def __init__(self):
        # class score weights
        # self.score_weight = {
        #     1: 0.15, 2: 0.09, 3: 0.09,
        #     4: 0.05, 5: 0.13, 6: 0.05,
        #     7: 0.12, 8: 0.13, 9: 0.07, 10:0.12}
        self.score_weight = np.array([0.15, 0.09, 0.09, 0.05, 0.13, 0.05, 0.12, 0.13, 0.07, 0.12])

    def IoU_thres(self, bbox):
        """
        Given the bbox, calculate the corresponding threshold
            Arg：
                bbox: a list of 4 digits, x, y, width, height
            Return:
                thres: threshold
        """
        sides = bbox[2:]
        m = np.min(sides)
        if m <= 0:
            raise ValueError('Long side is less or equal to zero. Check!')
        if m < 40:
            thres = 0.2
        elif (m < 120) & (m >= 40):
            thres = m / 200
        elif (m < 420) & (m >= 120):
            thres = m / 1500 + 0.52
        else:
            thres = 0.8
        return thres

    def IoU_calc(self, y_pred, y_true):
        """
        Compare two bboxes and calculate their IoU
        Args:
            y_pred: a list of 4 digits, x, y, width, height
            y_true: a list of 4 digits, x, y, width, height
        Returns:
            IoU: the IoU of the two
        """
        #  might introduce a very slight rounding error here
        y_pred = [int(np.round(x)) for x in y_pred]
        y_true = [int(np.round(x)) for x in y_true]
        # initiate two blank images
        width_pred = y_pred[0] + y_pred[2]
        height_pred = y_pred[1] + y_pred[-1]
        width_true = y_true[0] + y_true[2]
        height_true = y_true[1] + y_true[-1]
        width = np.max([width_pred, width_true])
        height = np.max([height_pred, height_true])
        pred_mat = np.zeros((height, width))
        true_mat = np.zeros((height, width))
        # mark areas
        pred_mat[y_pred[1] : y_pred[1] + y_pred[-1], y_pred[0] : y_pred[0] + y_pred[2]] = 1
        true_mat[y_true[1] : y_true[1] + y_true[-1], y_true[0] : y_true[0] + y_true[2]] = 1
        # to optimize calculation time
        if pred_mat.shape[0] > 1000:
            pred_mat = cv2.resize(pred_mat.astype(np.uint8), (800, 800))
            true_mat = cv2.resize(true_mat.astype(np.uint8), (800, 800))
        intersection = np.sum(pred_mat * true_mat)
        union = np.sum(np.clip(pred_mat + true_mat, 0, 1))
        IoU = intersection / union
        return IoU

    def mAP_calc(self, y_pred, y_true):
        """
        Calculate mAP for a list of predictions
            Args:
                y_pred: a pandas dataframe with columns ['id', 'category', 'bbox']
                y_true: a pandas dataframe with columns ['id', 'category', 'bbox']
                a note to categories and bbox: they should have the same sequence, i.e. category[0] corresponds
                to bbox[0]
            Return:
                AP: an np.array of AP score for all predictions, with the same sequence as the input dataframe.
                mAP: a single number, mAP
        """
        if (y_pred['id'] != y_true['id']).all():
            raise ValueError('the input dataframes image_id does not match!')
        prediction_len = len(y_pred)
        # initiate counters
        true_ct = np.zeros((prediction_len, 10))
        pred_ct = np.zeros((prediction_len, 10))
        # loop through all rows in the dataframe
        for i in tqdm(range(prediction_len)):
            pred_row = y_pred.loc[i, :]
            true_row = y_true.loc[i, :]
            cat, ct = np.unique(true_row.category, return_counts = True) # how many defects of each category in GT (groud truth)?

            for cat_, ct_ in zip(cat, ct):
                # keep track of the true labels
                true_ct[i, cat_ - 1] = ct_ # record in counter

            for n_j, j in enumerate(true_row.bbox):
                # get each true bbox and initiate its threshold
                true_category = true_row.category[n_j]
                thres = self.IoU_thres(j)
                IoU = 0
                ix = 0
                for n_k, k in enumerate(pred_row.bbox):
                    # compare true bbox to each pred bbox
                    IoU_ = self.IoU_calc(k, j)
                    if (IoU_ > IoU) & (IoU_ > thres):
                        # only look for largest IoU if multiple box satisfies one GT bbox
                        IoU = IoU_
                        ix = n_k
                if IoU == 0: # no overlay  found, skip
                    continue
                matched_cat = pred_row.category[ix]
                if matched_cat == true_category:
                    pred_ct[i, true_category - 1] += 1
        # post processing. to avoid div by 0 warning, use a mask
        zeros_row, zeros_col = np.where(true_ct == 0)
        match_row, match_col = np.where(true_ct == pred_ct)
        true_ct[zeros_row, zeros_col] = 1
        ratio = pred_ct / true_ct
        ratio[zeros_row, zeros_col] = 0
        # restoring match
        true_ct[zeros_row, zeros_col] = 0
        ratio[match_row, match_col] = 1
        AP = np.zeros((prediction_len,))
        for n, i in enumerate(ratio):
            nonzeros = len(np.nonzero(i)[0])
            if nonzeros == 0:
                continue
            else:
                AP[n] = np.sum(i) / nonzeros
        total_bbox_ct = np.zeros((10,))
        for i in range(10):
            total_bbox_ct[i] = len(np.nonzero(true_ct[:, i])[0])
        mAP = np.mean(ratio, axis = 0) * self.score_weight
        return AP, mAP
