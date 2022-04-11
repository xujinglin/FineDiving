import cv2.cv2
import torch
import numpy as np
import os
from os.path import join
from pydoc import locate
import sys
import torch.nn as nn
import torch.nn.functional as F

def import_class(name):
    return locate(name)

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def denormalize(label, label_range, upper = 100.0): 
    true_label = (label.float() / float(upper)) * (label_range[1] - label_range[0]) + label_range[0]
    return true_label

def normalize(label, label_range, upper = 100.0):
    norm_label = ((label - label_range[0]) / (label_range[1] - label_range[0]) ) * float(upper) 
    return norm_label

def segment_iou(target_segment, candidate_segments, args):
    tt1 = np.maximum(target_segment[0], candidate_segments[0])
    tt2 = np.minimum(target_segment[1], candidate_segments[1])

    segments_intersection = (tt2 - tt1).clip(0)
    segments_union = (candidate_segments[1] - candidate_segments[0]) \
      + (target_segment[1] - target_segment[0]) - segments_intersection
    tIoU = segments_intersection.astype(float) / (segments_union + sys.float_info.epsilon)
    return tIoU

def cal_tiou(tIoU_results, tiou_thresholds):
    tIoU_correct = np.zeros((len(tIoU_results), len(tiou_thresholds)))
    for tidx, tiou_thr in enumerate(tiou_thresholds):
        for idx in range(len(tIoU_results)):
            if tIoU_results[idx] >= tiou_thr:
                tIoU_correct[idx, tidx] = 1
            else:
                tIoU_correct[idx, tidx] = 0

    tIoU_correct_per_thr = tIoU_correct.sum(0)
    return tIoU_correct_per_thr

def seg_pool_1d(video_fea_1, video_1_st, video_1_ed, fix_size):
    video_fea_seg0 = F.interpolate(video_fea_1[:,:,:video_1_st], size=fix_size, mode='linear', align_corners=True)
    video_fea_seg1 = F.interpolate(video_fea_1[:,:,video_1_st:video_1_ed], size=fix_size, mode='linear', align_corners=True)
    video_fea_seg2 = F.interpolate(video_fea_1[:,:,video_1_ed:], size=fix_size, mode='linear', align_corners=True)
    video_1_segs = torch.cat([video_fea_seg0, video_fea_seg1, video_fea_seg2], 2)
    return video_1_segs

def seg_pool_3d(video_feamap_2, video_2_st, video_2_ed, fix_size):
    N, C, T, H, W = video_feamap_2.size()
    video_feamap_seg0 = F.interpolate(video_feamap_2[:, :, :video_2_st, :, :], size=[fix_size, H, W], mode='trilinear', align_corners=True)
    video_feamap_seg1 = F.interpolate(video_feamap_2[:, :, video_2_st:video_2_ed, :, :], size=[fix_size, H, W], mode='trilinear', align_corners=True)
    video_feamap_seg2 = F.interpolate(video_feamap_2[:, :, video_2_ed:, :, :], size=[fix_size, H, W], mode='trilinear', align_corners=True)
    video_2_segs_map = torch.cat([video_feamap_seg0, video_feamap_seg1, video_feamap_seg2], 2)
    return video_2_segs_map


