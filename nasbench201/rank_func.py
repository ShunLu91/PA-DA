import torch
import logging
import numpy as np
from copy import deepcopy
from scipy.stats import kendalltau
from utils import obtain_accuracy, AverageMeter


def valid_specific_path(valid_loader, model, sampled_arch, criterion, device):
    val_loss, val_top1, val_top5 = AverageMeter(), AverageMeter(), AverageMeter()
    model.eval()
    with torch.no_grad():
        for step, (val_inputs, val_targets) in enumerate(valid_loader):
            val_inputs = val_inputs.to(device)
            val_targets = val_targets.to(device)

            # prediction
            logits = model(val_inputs, sampled_arch)
            loss = criterion(logits, val_targets)
            # record
            prec1, prec5 = obtain_accuracy(
                logits.data, val_targets.data, topk=(1, 5)
            )
            val_loss.update(loss.item(), val_inputs.size(0))
            val_top1.update(prec1.item(), val_inputs.size(0))
            val_top5.update(prec5.item(), val_inputs.size(0))
    return val_loss.avg, val_top1.avg, val_top5.avg


def rank_supernet(args, valid_loader, model, criterion):
    logging.info('---------- Start to rank on NAS-Bench-201 ----------')
    nasbench201 = np.load('./dataset/nasbench201/nasbench201_dict.npy', allow_pickle=True).item()
    if args.debug:
        new_dict = {}
        for i in range(5):
            new_dict[str(i)] = deepcopy(nasbench201[str(i)])
        nasbench201 = deepcopy(new_dict)
    nasbench201_len = len(nasbench201)
    tmp_pred = []
    tmp_target = []
    prediction = {}
    for step, item in enumerate(nasbench201):
        model_id = int(item)
        operation = nasbench201[item]['operation']
        if args.dataset in ['cifar10', 'cifar10_rebuild_loader']:
            target = nasbench201[item]['cifar10_test']
        elif args.dataset in ['cifar100', 'cifar100_rebuild_loader']:
            target = nasbench201[item]['cifar100_test']
        elif args.dataset in ['imagenet16', 'imagenet16_rebuild_loader']:
            target = nasbench201[item]['imagenet16_test']
        else:
            raise NotImplementedError

        val_loss, val_top1, val_top5 = valid_specific_path(valid_loader, model, operation, criterion, args.device)
        tmp_pred.append(val_top1)
        tmp_target.append(target)
        prediction[model_id] = {'id': model_id, 'model_gene': operation, 'pred': val_top1, 'target': target}
        if step % args.rank_print_freq == 0 or (step + 1) == nasbench201_len:
            logging.info("model_id: %d  gene: %s  loss: %.3f  top1: %.3f  target: %.3f"
                         % (model_id, str(operation), val_loss, val_top1, target))
            logging.info("Evaluated: %05d\tWaiting: %05d\tCurrent Kendall's Tau: %.3f" %
                         (len(tmp_pred), nasbench201_len-len(tmp_pred), kendalltau(tmp_pred, tmp_target)[0]))

    # Save predictions
    print('\n')
    np.save(args.pred_path, prediction)
    logging.info('Finish ranking and save predictions to : %s' % args.pred_path)
    final_ranking = kendalltau(tmp_pred, tmp_target)[0]
    logging.info("Final_pred: %05d\tFinal_target: %05d\tFinal_Kendall's Tau: %.3f" %
                 (len(tmp_pred), len(tmp_target), final_ranking))

    return final_ranking
