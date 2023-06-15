import numpy as np
import torch
import torch.nn as nn

from scipy import stats
from tools import builder, helper
from utils import misc
import time
import pickle

def train_net(args):
    print('Trainer start ... ')
    # build dataset
    train_dataset, test_dataset = builder.dataset_builder(args)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs_train,
                                            shuffle=True, num_workers=int(args.workers),
                                            pin_memory=True, worker_init_fn=misc.worker_init_fn)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs_test,
                                                  shuffle=False, num_workers=int(args.workers),
                                                  pin_memory=True)

    # build model
    base_model, psnet_model, decoder, regressor_delta = builder.model_builder(args)

    # CUDA
    global use_gpu
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        base_model = base_model.cuda()
        psnet_model = psnet_model.cuda()
        decoder = decoder.cuda()
        regressor_delta = regressor_delta.cuda()
        torch.backends.cudnn.benchmark = True

    optimizer, scheduler = builder.build_opti_sche(base_model, psnet_model, decoder, regressor_delta, args)

    start_epoch = 0
    global epoch_best_tas, pred_tious_best_5, pred_tious_best_75, epoch_best_aqa, rho_best, L2_min, RL2_min
    epoch_best_tas = 0
    pred_tious_best_5 = 0
    pred_tious_best_75 = 0
    epoch_best_aqa = 0
    rho_best = 0
    L2_min = 1000
    RL2_min = 1000

    # resume ckpts
    if args.resume:
        start_epoch, epoch_best_aqa, rho_best, L2_min, RL2_min = builder.resume_train(base_model, psnet_model, decoder,
                                                                                      regressor_delta, optimizer, args)
        print('resume ckpts @ %d epoch(rho = %.4f, L2 = %.4f , RL2 = %.4f)'
              % (start_epoch - 1, rho_best, L2_min, RL2_min))

    # DP
    base_model = nn.DataParallel(base_model)
    psnet_model = nn.DataParallel(psnet_model)
    decoder = nn.DataParallel(decoder)
    regressor_delta = nn.DataParallel(regressor_delta)

    # loss
    mse = nn.MSELoss().cuda()
    bce = nn.BCELoss().cuda()

    # training phase
    for epoch in range(start_epoch, args.max_epoch):
        pred_tious_5 = []
        pred_tious_75 = []
        true_scores = []
        pred_scores = []

        base_model.train()  
        psnet_model.train()
        decoder.train()
        regressor_delta.train()

        if args.fix_bn:
            base_model.apply(misc.fix_bn)
        for idx, (data, target) in enumerate(train_dataloader):
            # num_iter += 1
            opti_flag = True

            # video_1 is query and video_2 is exemplar
            video_1 = data['video'].float().cuda()
            video_2 = target['video'].float().cuda()
            label_1_tas = data['transits'].float().cuda() + 1
            label_2_tas = target['transits'].float().cuda() + 1
            label_1_score = data['final_score'].float().reshape(-1, 1).cuda()
            label_2_score = target['final_score'].float().reshape(-1, 1).cuda()

            # forward

            helper.network_forward_train(base_model, psnet_model, decoder, regressor_delta, pred_scores,
                                         video_1, label_1_score, video_2, label_2_score, mse, optimizer,
                                         opti_flag, epoch, idx+1, len(train_dataloader),
                                         args, label_1_tas, label_2_tas, bce,
                                         pred_tious_5, pred_tious_75)
            true_scores.extend(data['final_score'].numpy())

        # evaluation results
        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)
        rho, p = stats.spearmanr(pred_scores, true_scores)
        L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
        RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / true_scores.shape[0]
        pred_tious_mean_5 = sum(pred_tious_5) / len(train_dataset)
        pred_tious_mean_75 = sum(pred_tious_75) / len(train_dataset)

        print('[Training] EPOCH: %d, tIoU_5: %.4f, tIoU_75: %.4f'
              % (epoch, pred_tious_mean_5, pred_tious_mean_75))
        print('[Training] EPOCH: %d, correlation: %.4f, L2: %.4f, RL2: %.4f, lr1: %.4f, lr2: %.4f'%(epoch, rho, L2, RL2, 
            optimizer.param_groups[0]['lr'],  optimizer.param_groups[1]['lr']))

        validate(base_model, psnet_model, decoder, regressor_delta, test_dataloader, epoch, optimizer, args)
        helper.save_checkpoint(base_model, psnet_model, decoder, regressor_delta, optimizer, epoch,
                               epoch_best_aqa, rho_best, L2_min, RL2_min, 'last', args)
        print('[TEST] EPOCH: %d, best correlation: %.6f, best L2: %.6f, best RL2: %.6f' % (epoch_best_aqa,
                                                                                           rho_best, L2_min, RL2_min))
        print('[TEST] EPOCH: %d, best tIoU_5: %.6f, best tIoU_75: %.6f' % (epoch_best_tas,
                                                                           pred_tious_best_5, pred_tious_best_75))

        # scheduler lr
        if scheduler is not None:
            scheduler.step()


def validate(base_model, psnet_model, decoder, regressor_delta, test_dataloader, epoch, optimizer, args):

    print("Start validating epoch {}".format(epoch))
    global use_gpu
    global epoch_best_aqa, rho_best, L2_min, RL2_min, epoch_best_tas, pred_tious_best_5, pred_tious_best_75

    true_scores = []
    pred_scores = []
    pred_tious_test_5 = []
    pred_tious_test_75 = []

    base_model.eval()  
    psnet_model.eval()
    decoder.eval()
    regressor_delta.eval()

    batch_num = len(test_dataloader)
    with torch.no_grad():
        datatime_start = time.time()

        for batch_idx, (data, target) in enumerate(test_dataloader, 0):
            datatime = time.time() - datatime_start
            start = time.time()

            video_1 = data['video'].float().cuda()
            video_2_list = [item['video'].float().cuda() for item in target]
            label_1_tas = data['transits'].float().cuda() + 1
            label_2_tas_list = [item['transits'].float().cuda() + 1 for item in target]
            label_2_score_list = [item['final_score'].float().reshape(-1, 1).cuda() for item in target]

            helper.network_forward_test(base_model, psnet_model, decoder, regressor_delta, pred_scores,
                                        video_1, video_2_list, label_2_score_list,
                                        args, label_1_tas, label_2_tas_list,
                                        pred_tious_test_5, pred_tious_test_75)

            batch_time = time.time() - start
            if batch_idx % args.print_freq == 0:
                print('[TEST][%d/%d][%d/%d] \t Batch_time %.2f \t Data_time %.2f'
                    % (epoch, args.max_epoch, batch_idx, batch_num, batch_time, datatime))
            datatime_start = time.time()
            true_scores.extend(data['final_score'].numpy())
        
        # evaluation results
        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)
        rho, p = stats.spearmanr(pred_scores, true_scores)
        L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
        RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / true_scores.shape[0]
        pred_tious_test_mean_5 = sum(pred_tious_test_5) / (len(test_dataloader) * args.bs_test)
        pred_tious_test_mean_75 = sum(pred_tious_test_75) / (len(test_dataloader) * args.bs_test)

        if pred_tious_test_mean_5 > pred_tious_best_5:
            pred_tious_best_5 = pred_tious_test_mean_5
        if pred_tious_test_mean_75 > pred_tious_best_75:
            pred_tious_best_75 = pred_tious_test_mean_75
            epoch_best_tas = epoch
        print('[TEST] EPOCH: %d, tIoU_5: %.6f, tIoU_75: %.6f' % (epoch, pred_tious_best_5, pred_tious_best_75))

        if L2_min > L2:
            L2_min = L2
        if RL2_min > RL2:
            RL2_min = RL2
        if rho > rho_best:
            rho_best = rho
            epoch_best_aqa = epoch
            print('-----New best found!-----')
            helper.save_outputs(pred_scores, true_scores, args)
            helper.save_checkpoint(base_model, psnet_model, decoder, regressor_delta, optimizer, epoch, epoch_best_aqa,
                                   rho_best, L2_min, RL2_min, 'last', args)
        print('[TEST] EPOCH: %d, correlation: %.6f, L2: %.6f, RL2: %.6f' % (epoch, rho, L2, RL2))

def test_net(args):
    print('Tester start ... ')

    train_dataset, test_dataset = builder.dataset_builder(args)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs_test,
                                                  shuffle=False, num_workers=int(args.workers),
                                                  pin_memory=True)

    # build model
    base_model, psnet_model, decoder, regressor_delta = builder.model_builder(args)

    # load checkpoints
    builder.load_model(base_model, psnet_model, decoder, regressor_delta, args)

    # CUDA
    global use_gpu
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        base_model = base_model.cuda()
        psnet_model = psnet_model.cuda()
        decoder = decoder.cuda()
        regressor_delta = regressor_delta.cuda()
        torch.backends.cudnn.benchmark = True

    # DP
    base_model = nn.DataParallel(base_model)
    psnet_model = nn.DataParallel(psnet_model)
    decoder = nn.DataParallel(decoder)
    regressor_delta = nn.DataParallel(regressor_delta)

    test(base_model, psnet_model, decoder, regressor_delta, test_dataloader, args)

def test(base_model, psnet_model, decoder, regressor_delta, test_dataloader, args):
    global use_gpu
    global epoch_best_aqa, rho_best, L2_min, RL2_min
    global epoch_best_tas, pred_tious_best_5, pred_tious_best_75

    true_scores = []
    pred_scores = []
    pred_tious_test_5 = []
    pred_tious_test_75 = []

    base_model.eval()
    psnet_model.eval()
    decoder.eval()
    regressor_delta.eval()

    batch_num = len(test_dataloader)
    with torch.no_grad():
        datatime_start = time.time()

        for batch_idx, (data, target) in enumerate(test_dataloader, 0):
            datatime = time.time() - datatime_start
            start = time.time()

            video_1 = data['video'].float().cuda()
            video_2_list = [item['video'].float().cuda() for item in target]
            label_1_tas = data['transits'].float().cuda() + 1
            label_2_tas_list = [item['transits'].float().cuda() + 1 for item in target]
            label_2_score_list = [item['final_score'].float().reshape(-1, 1).cuda() for item in target]

            helper.network_forward_test(base_model, psnet_model, decoder, regressor_delta, pred_scores,
                                        video_1, video_2_list, label_2_score_list,
                                        args, label_1_tas, label_2_tas_list,
                                        pred_tious_test_5, pred_tious_test_75)

            batch_time = time.time() - start
            if batch_idx % args.print_freq == 0:
                print('[TEST][%d/%d] \t Batch_time %.2f \t Data_time %.2f'
                      % (batch_idx, batch_num, batch_time, datatime))
            datatime_start = time.time()
            true_scores.extend(data['final_score'].numpy())

        # evaluation results
        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)
        rho, p = stats.spearmanr(pred_scores, true_scores)
        L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
        RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / \
              true_scores.shape[0]
        pred_tious_test_mean_5 = sum(pred_tious_test_5) / (len(test_dataloader) * args.bs_test)
        pred_tious_test_mean_75 = sum(pred_tious_test_75) / (len(test_dataloader) * args.bs_test)

        print('[TEST] tIoU_5: %.6f, tIoU_75: %.6f' % (pred_tious_test_mean_5, pred_tious_test_mean_75))
        print('[TEST] correlation: %.6f, L2: %.6f, RL2: %.6f' % (rho, L2, RL2))
