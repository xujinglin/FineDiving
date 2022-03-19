import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../"))

import torch
import torch.nn as nn
import time
import numpy as np
from utils.misc import segment_iou, cal_tiou, seg_pool_1d, seg_pool_3d


def network_forward_train(base_model, psnet_model, decoder, regressor_delta, pred_scores,
                          video_1, label_1_score, video_2, label_2_score, mse, optimizer, opti_flag,
                          epoch, batch_idx, batch_num, args, label_1_tas, label_2_tas, bce,
                          pred_tious_5, pred_tious_75):

    start = time.time()
    optimizer.zero_grad()

    ############# I3D featrue #############
    com_feature_12, com_feamap_12 = base_model(video_1, video_2)
    video_1_fea = com_feature_12[:,:,:com_feature_12.shape[2] // 2]
    video_2_fea = com_feature_12[:,:,com_feature_12.shape[2] // 2:]
    video_1_feamap = com_feamap_12[:,:,:com_feature_12.shape[2] // 2]
    video_2_feamap = com_feamap_12[:,:,com_feature_12.shape[2] // 2:]

    N,T,C,T_t,H_t,W_t = video_1_feamap.size()
    video_1_feamap = video_1_feamap.mean(-3)
    video_2_feamap = video_2_feamap.mean(-3)
    video_1_feamap_re = video_1_feamap.reshape(-1, T, C)
    video_2_feamap_re = video_2_feamap.reshape(-1, T, C)

    ############# Procedure Segmentation #############
    com_feature_12_u = torch.cat((video_1_fea, video_2_fea), 0)
    com_feamap_12_u = torch.cat((video_1_feamap_re, video_2_feamap_re), 0)

    u_fea_96, transits_pred = psnet_model(com_feature_12_u)
    u_feamap_96, transits_pred_map = psnet_model(com_feamap_12_u)
    u_feamap_96 = u_feamap_96.reshape(2*N, u_feamap_96.shape[1], u_feamap_96.shape[2], H_t, W_t)

    label_12_tas = torch.cat((label_1_tas, label_2_tas), 0)
    label_12_pad = torch.zeros(transits_pred.size())
    for bs in range(transits_pred.shape[0]):
        label_12_pad[bs, int(label_12_tas[bs, 0]), 0] = 1
        label_12_pad[bs, int(label_12_tas[bs, -1]), -1] = 1

    loss_tas = bce(transits_pred, label_12_pad.cuda())

    num = round(transits_pred.shape[1] / transits_pred.shape[-1])
    transits_st_ed = torch.zeros(label_12_tas.size())
    for bs in range(transits_pred.shape[0]):
        for i in range(transits_pred.shape[-1]):
            transits_st_ed[bs, i] = transits_pred[bs, i * num: (i + 1) * num, i].argmax(0).cpu().item() + i * num
    label_1_tas_pred = transits_st_ed[:transits_st_ed.shape[0] // 2]
    label_2_tas_pred = transits_st_ed[transits_st_ed.shape[0] // 2:]

    ############# Procedure-aware Cross-attention #############
    u_fea_96_1 = u_fea_96[:u_fea_96.shape[0] // 2].transpose(1, 2)
    u_fea_96_2 = u_fea_96[u_fea_96.shape[0] // 2:].transpose(1, 2)

    u_feamap_96_1 = u_feamap_96[:u_feamap_96.shape[0] // 2].transpose(1, 2)
    u_feamap_96_2 = u_feamap_96[u_feamap_96.shape[0] // 2:].transpose(1, 2)

    if epoch / args.max_epoch <= args.prob_tas_threshold:
        video_1_segs = []
        for bs_1 in range(u_fea_96_1.shape[0]):
            video_1_st = int(label_1_tas[bs_1][0].item())
            video_1_ed = int(label_1_tas[bs_1][1].item())
            video_1_segs.append(seg_pool_1d(u_fea_96_1[bs_1].unsqueeze(0), video_1_st, video_1_ed, args.fix_size))
        video_1_segs = torch.cat(video_1_segs, 0).transpose(1, 2)

        video_2_segs = []
        for bs_2 in range(u_fea_96_2.shape[0]):                 
            video_2_st = int(label_2_tas[bs_2][0].item())
            video_2_ed = int(label_2_tas[bs_2][1].item())
            video_2_segs.append(seg_pool_1d(u_fea_96_2[bs_2].unsqueeze(0), video_2_st, video_2_ed, args.fix_size))
        video_2_segs = torch.cat(video_2_segs, 0).transpose(1, 2)   

        video_1_segs_map = []
        for bs_1 in range(u_feamap_96_1.shape[0]):
            video_1_st = int(label_1_tas[bs_1][0].item())
            video_1_ed = int(label_1_tas[bs_1][1].item())
            video_1_segs_map.append(seg_pool_3d(u_feamap_96_1[bs_1].unsqueeze(0), video_1_st, video_1_ed, args.fix_size))
        video_1_segs_map = torch.cat(video_1_segs_map, 0)
        video_1_segs_map = video_1_segs_map.reshape(video_1_segs_map.shape[0], video_1_segs_map.shape[1], video_1_segs_map.shape[2], -1).transpose(2, 3)
        video_1_segs_map = torch.cat([video_1_segs_map[:,:,:,i] for i in range(video_1_segs_map.shape[-1])], 2).transpose(1, 2)

        video_2_segs_map = []
        for bs_2 in range(u_fea_96_2.shape[0]):
            video_2_st = int(label_2_tas[bs_2][0].item())
            video_2_ed = int(label_2_tas[bs_2][1].item())
            video_2_segs_map.append(seg_pool_3d(u_feamap_96_2[bs_2].unsqueeze(0), video_2_st, video_2_ed, args.fix_size))
        video_2_segs_map = torch.cat(video_2_segs_map, 0)
        video_2_segs_map = video_2_segs_map.reshape(video_2_segs_map.shape[0], video_2_segs_map.shape[1], video_2_segs_map.shape[2], -1).transpose(2, 3)
        video_2_segs_map = torch.cat([video_2_segs_map[:, :, :, i] for i in range(video_2_segs_map.shape[-1])], 2).transpose(1, 2)
    else:
        video_1_segs = []
        for bs_1 in range(u_fea_96_1.shape[0]):
            video_1_st = int(label_1_tas_pred[bs_1][0].item())
            video_1_ed = int(label_1_tas_pred[bs_1][1].item())
            if video_1_st == 0:
                video_1_st = 1
            if video_1_ed == 0:
                video_1_ed = 1
            video_1_segs.append(seg_pool_1d(u_fea_96_1[bs_1].unsqueeze(0), video_1_st, video_1_ed, args.fix_size))
        video_1_segs = torch.cat(video_1_segs, 0).transpose(1, 2)   

        video_2_segs = []
        for bs_2 in range(u_fea_96_2.shape[0]):                 
            video_2_st = int(label_2_tas_pred[bs_2][0].item())
            video_2_ed = int(label_2_tas_pred[bs_2][1].item())
            if video_2_st == 0:
                video_2_st = 1
            if video_2_ed == 0:
                video_2_ed = 1
            video_2_segs.append(seg_pool_1d(u_fea_96_2[bs_2].unsqueeze(0), video_2_st, video_2_ed, args.fix_size))
        video_2_segs = torch.cat(video_2_segs, 0).transpose(1, 2)   

        video_1_segs_map = []
        for bs_1 in range(u_feamap_96_1.shape[0]):
            video_1_st = int(label_1_tas_pred[bs_1][0].item())
            video_1_ed = int(label_1_tas_pred[bs_1][1].item())
            if video_1_st == 0:
                video_1_st = 1
            if video_1_ed == 0:
                video_1_ed = 1
            video_1_segs_map.append(seg_pool_3d(u_feamap_96_1[bs_1].unsqueeze(0), video_1_st, video_1_ed, args.fix_size))
        video_1_segs_map = torch.cat(video_1_segs_map, 0)
        video_1_segs_map = video_1_segs_map.reshape(video_1_segs_map.shape[0], video_1_segs_map.shape[1], video_1_segs_map.shape[2], -1).transpose(2, 3)
        video_1_segs_map = torch.cat([video_1_segs_map[:, :, :, i] for i in range(video_1_segs_map.shape[-1])], 2).transpose(1, 2)

        video_2_segs_map = []
        for bs_2 in range(u_fea_96_2.shape[0]):
            video_2_st = int(label_2_tas_pred[bs_2][0].item())
            video_2_ed = int(label_2_tas_pred[bs_2][1].item())
            if video_2_st == 0:
                video_2_st = 1
            if video_2_ed == 0:
                video_2_ed = 1
            video_2_segs_map.append(seg_pool_3d(u_feamap_96_2[bs_2].unsqueeze(0), video_2_st, video_2_ed, args.fix_size))
        video_2_segs_map = torch.cat(video_2_segs_map, 0)
        video_2_segs_map = video_2_segs_map.reshape(video_2_segs_map.shape[0], video_2_segs_map.shape[1], video_2_segs_map.shape[2], -1).transpose(2, 3)
        video_2_segs_map = torch.cat([video_2_segs_map[:, :, :, i] for i in range(video_2_segs_map.shape[-1])], 2).transpose(1, 2)

    decoder_video_12_map_list = []
    decoder_video_21_map_list = []
    for i in range(args.step_num):
        decoder_video_12_map = decoder(video_1_segs[:, i*args.fix_size:(i+1)*args.fix_size,:],
                                                      video_2_segs_map[:, i*args.fix_size*H_t*W_t:(i+1)*args.fix_size*H_t*W_t,:])     # N,15,256/64
        decoder_video_21_map = decoder(video_2_segs[:, i*args.fix_size:(i+1)*args.fix_size,:],
                                          video_1_segs_map[:, i*args.fix_size*H_t*W_t:(i+1)*args.fix_size*H_t*W_t,:])    # N,15,256/64
        decoder_video_12_map_list.append(decoder_video_12_map)
        decoder_video_21_map_list.append(decoder_video_21_map)

    decoder_video_12_map = torch.cat(decoder_video_12_map_list, 1)
    decoder_video_21_map = torch.cat(decoder_video_21_map_list, 1)

    ############# Fine-grained Contrastive Regression #############
    decoder_12_21 = torch.cat((decoder_video_12_map, decoder_video_21_map), 0)
    delta = regressor_delta(decoder_12_21)
    delta = delta.mean(1)
    loss_aqa = mse(delta[:delta.shape[0]//2], (label_1_score - label_2_score)) \
               + mse(delta[delta.shape[0]//2:], (label_2_score - label_1_score))

    loss = loss_aqa + loss_tas
    loss.backward()
    optimizer.step()

    end = time.time()
    batch_time = end - start

    score = (delta[:delta.shape[0]//2].detach() + label_2_score)
    pred_scores.extend([i.item() for i in score])

    tIoU_results = []
    for bs in range(transits_pred.shape[0] // 2):
        tIoU_results.append(segment_iou(np.array(label_12_tas.squeeze(-1).cpu())[bs],
                                        np.array(transits_st_ed.squeeze(-1).cpu())[bs],
                                        args))

    tiou_thresholds = np.array([0.5, 0.75])
    tIoU_correct_per_thr = cal_tiou(tIoU_results, tiou_thresholds)
    Batch_tIoU_5 = tIoU_correct_per_thr[0]
    Batch_tIoU_75 = tIoU_correct_per_thr[1]
    pred_tious_5.extend([Batch_tIoU_5])
    pred_tious_75.extend([Batch_tIoU_75])

    if batch_idx % args.print_freq == 0:
        print('[Training][%d/%d][%d/%d] \t Batch_time: %.2f \t Batch_loss: %.4f \t '
              'lr1 : %0.5f \t lr2 : %0.5f'
              % (epoch, args.max_epoch, batch_idx, batch_num, batch_time, loss.item(), 
                 optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))


def network_forward_test(base_model, psnet_model, decoder, regressor_delta, pred_scores,
                         video_1, video_2_list, label_2_score_list,
                         args, label_1_tas, label_2_tas_list,
                         pred_tious_test_5, pred_tious_test_75):
    score = 0
    tIoU_results = []
    for video_2, label_2_score, label_2_tas in zip(video_2_list, label_2_score_list, label_2_tas_list):

        ############# I3D featrue #############
        com_feature_12, com_feamap_12 = base_model(video_1, video_2) 
        video_1_fea = com_feature_12[:, :, :com_feature_12.shape[2] // 2]
        video_2_fea = com_feature_12[:, :, com_feature_12.shape[2] // 2:]
        video_1_feamap = com_feamap_12[:, :, :com_feature_12.shape[2] // 2]
        video_2_feamap = com_feamap_12[:, :, com_feature_12.shape[2] // 2:]

        N, T, C, T_t, H_t, W_t = video_1_feamap.size()
        video_1_feamap = video_1_feamap.mean(-3)
        video_2_feamap = video_2_feamap.mean(-3)
        video_1_feamap_re = video_1_feamap.reshape(-1, T, C)
        video_2_feamap_re = video_2_feamap.reshape(-1, T, C)

        ############# Procedure Segmentation #############
        com_feature_12_u = torch.cat((video_1_fea, video_2_fea), 0)
        com_feamap_12_u = torch.cat((video_1_feamap_re, video_2_feamap_re), 0)

        u_fea_96, transits_pred = psnet_model(com_feature_12_u)
        u_feamap_96, transits_pred_map = psnet_model(com_feamap_12_u)
        u_feamap_96 = u_feamap_96.reshape(2 * N, u_feamap_96.shape[1], u_feamap_96.shape[2], H_t, W_t)

        label_12_tas = torch.cat((label_1_tas, label_2_tas), 0)
        num = round(transits_pred.shape[1] / transits_pred.shape[-1])
        transits_st_ed = torch.zeros(label_12_tas.size())
        for bs in range(transits_pred.shape[0]):
            for i in range(transits_pred.shape[-1]):
                transits_st_ed[bs, i] = transits_pred[bs, i * num: (i + 1) * num, i].argmax(0).cpu().item() + i * num
        label_1_tas_pred = transits_st_ed[:transits_st_ed.shape[0] // 2]
        label_2_tas_pred = transits_st_ed[transits_st_ed.shape[0] // 2:]

        ############# Procedure-aware Cross-attention #############
        u_fea_96_1 = u_fea_96[:u_fea_96.shape[0] // 2].transpose(1, 2)
        u_fea_96_2 = u_fea_96[u_fea_96.shape[0] // 2:].transpose(1, 2)
        u_feamap_96_1 = u_feamap_96[:u_feamap_96.shape[0] // 2].transpose(1, 2)
        u_feamap_96_2 = u_feamap_96[u_feamap_96.shape[0] // 2:].transpose(1, 2)

        video_1_segs = []
        for bs_1 in range(u_fea_96_1.shape[0]):
            video_1_st = int(label_1_tas_pred[bs_1][0].item())
            video_1_ed = int(label_1_tas_pred[bs_1][1].item())
            if video_1_st == 0:
                video_1_st = 1
            if video_1_ed == 0:
                video_1_ed = 1
            video_1_segs.append(seg_pool_1d(u_fea_96_1[bs_1].unsqueeze(0), video_1_st, video_1_ed, args.fix_size))
        video_1_segs = torch.cat(video_1_segs, 0).transpose(1, 2)   

        video_2_segs = []
        for bs_2 in range(u_fea_96_2.shape[0]):                 
            video_2_st = int(label_2_tas_pred[bs_2][0].item())
            video_2_ed = int(label_2_tas_pred[bs_2][1].item())
            if video_2_st == 0:
                video_2_st = 1
            if video_2_ed == 0:
                video_2_ed = 1
            video_2_segs.append(seg_pool_1d(u_fea_96_2[bs_2].unsqueeze(0), video_2_st, video_2_ed, args.fix_size))
        video_2_segs = torch.cat(video_2_segs, 0).transpose(1, 2)   

        video_1_segs_map = []
        for bs_1 in range(u_feamap_96_1.shape[0]):
            video_1_st = int(label_1_tas_pred[bs_1][0].item())
            video_1_ed = int(label_1_tas_pred[bs_1][1].item())
            if video_1_st == 0:
                video_1_st = 1
            if video_1_ed == 0:
                video_1_ed = 1
            video_1_segs_map.append(seg_pool_3d(u_feamap_96_1[bs_1].unsqueeze(0), video_1_st, video_1_ed, args.fix_size))
        video_1_segs_map = torch.cat(video_1_segs_map, 0)
        video_1_segs_map = video_1_segs_map.reshape(video_1_segs_map.shape[0], video_1_segs_map.shape[1], video_1_segs_map.shape[2], -1).transpose(2, 3)
        video_1_segs_map = torch.cat([video_1_segs_map[:, :, :, i] for i in range(video_1_segs_map.shape[-1])], 2).transpose(1, 2)

        video_2_segs_map = []
        for bs_2 in range(u_fea_96_2.shape[0]):
            video_2_st = int(label_2_tas_pred[bs_2][0].item())
            video_2_ed = int(label_2_tas_pred[bs_2][1].item())
            if video_2_st == 0:
                video_2_st = 1
            if video_2_ed == 0:
                video_2_ed = 1
            video_2_segs_map.append(seg_pool_3d(u_feamap_96_2[bs_2].unsqueeze(0), video_2_st, video_2_ed, args.fix_size))
        video_2_segs_map = torch.cat(video_2_segs_map, 0)
        video_2_segs_map = video_2_segs_map.reshape(video_2_segs_map.shape[0], video_2_segs_map.shape[1], video_2_segs_map.shape[2], -1).transpose(2, 3)
        video_2_segs_map = torch.cat([video_2_segs_map[:, :, :, i] for i in range(video_2_segs_map.shape[-1])], 2).transpose(1, 2)

        decoder_video_12_map_list = []
        decoder_video_21_map_list = []
        for i in range(args.step_num):
            decoder_video_12_map = decoder(video_1_segs[:, i * args.fix_size:(i + 1) * args.fix_size, :],
                                                     video_2_segs_map[:,
                                                     i * args.fix_size * H_t * W_t:(i + 1) * args.fix_size * H_t * W_t,
                                                     :])
            decoder_video_21_map = decoder(video_2_segs[:, i * args.fix_size:(i + 1) * args.fix_size, :],
                                              video_1_segs_map[:, i * args.fix_size * H_t * W_t:(i + 1) * args.fix_size * H_t * W_t,
                                              :])
            decoder_video_12_map_list.append(decoder_video_12_map)
            decoder_video_21_map_list.append(decoder_video_21_map)

        decoder_video_12_map = torch.cat(decoder_video_12_map_list, 1)
        decoder_video_21_map = torch.cat(decoder_video_21_map_list, 1)

        ############# Fine-grained Contrastive Regression #############
        decoder_12_21 = torch.cat((decoder_video_12_map, decoder_video_21_map), 0)
        delta = regressor_delta(decoder_12_21)
        delta = delta.mean(1)
        score += (delta[:delta.shape[0]//2].detach() + label_2_score)

        for bs in range(transits_pred.shape[0] // 2):
            tIoU_results.append(segment_iou(np.array(label_12_tas.squeeze(-1).cpu())[bs],
                                            np.array(transits_st_ed.squeeze(-1).cpu())[bs], args))

    pred_scores.extend([i.item() / len(video_2_list) for i in score])

    tIoU_results_mean = [sum(tIoU_results) / len(tIoU_results)]
    tiou_thresholds = np.array([0.5, 0.75])
    tIoU_correct_per_thr = cal_tiou(tIoU_results_mean, tiou_thresholds)
    pred_tious_test_5.extend([tIoU_correct_per_thr[0]])
    pred_tious_test_75.extend([tIoU_correct_per_thr[1]])


def save_checkpoint(base_model, psnet_model, decoder, regressor_delta, optimizer, epoch,
                    epoch_best_aqa, rho_best, L2_min, RL2_min, prefix, args):
    torch.save({
        'base_model': base_model.state_dict(),
        'psnet_model': psnet_model.state_dict(),
        'decoder': decoder.state_dict(),
        'regressor_delta': regressor_delta.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'epoch_best_aqa': epoch_best_aqa,
        'rho_best': rho_best,
        'L2_min': L2_min,
        'RL2_min': RL2_min,
    }, os.path.join(args.experiment_path, prefix + '.pth'))

def save_outputs(pred_scores, true_scores, args):
    save_path_pred = os.path.join(args.experiment_path, 'pred.npy')
    save_path_true = os.path.join(args.experiment_path, 'true.npy')
    np.save(save_path_pred, pred_scores)
    np.save(save_path_true, true_scores)
