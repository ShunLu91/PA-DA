import argparse
import logging
import random
import sys
import time
from copy import deepcopy

import numpy as np
import torch
from scipy.stats import kendalltau

from datasets.dataloader import get_dataloader
from models.cell_operations import NAS_BENCH_201
from models.supernet import Supernet201
from utils import obtain_accuracy, AverageMeter, set_seed, run_func, time_record

parser = argparse.ArgumentParser("Train 201 Supernet")
# dataset
parser.add_argument("--data_root", type=str, default='./dataset/', help="The path to dataset")
parser.add_argument("--dataset", type=str, default='cifar10', help="Dataset.")
parser.add_argument("--search_space_name", type=str, default='nas-bench-201', help="The search space name.")
parser.add_argument("--num_classes", type=int, default=10, help="Dataset Classes")
# supernet
parser.add_argument("--max_nodes", type=int, default=4, help="The maximum number of nodes.")
parser.add_argument("--channel", type=int, default=16, help="The number of channels.")
parser.add_argument("--num_cells", type=int, default=5, help="The number of cells in one stage.")
# training settings
parser.add_argument("--exp_name", type=str, default='debug_baseline', help='exp_name for saving results')
parser.add_argument("--method", type=str, default='spos', choices=['spos', 'fairnas', 'sumnas'])
parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
parser.add_argument("--inner_lr", type=float, default=0.05, help="Learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
parser.add_argument("--wd", type=float, default=2.5e-4, help="Weight decay")
parser.add_argument("--epochs", type=int, default=250, help="Training epochs")
parser.add_argument("--gpu_id", type=int, default=0, help="Training GPU")
parser.add_argument("--train_batch_size", type=int, default=256, help="Train batch size")
parser.add_argument("--valid_batch_size", type=int, default=512, help="Valid batch size")
parser.add_argument("--print_freq", type=int, default=50, help="print frequency when training")
parser.add_argument("--rank_print_freq", type=int, default=100, help="print frequency when ranking")
parser.add_argument("--seed", type=int, default=0, help="manual seed")
parser.add_argument("--debug", default=False, action='store_true', help="for debug")

args = parser.parse_args()
args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
args.ckpt_path = 'checkpoints/%s.pt' % args.exp_name
args.pred_path = 'results/%s.npy' % args.exp_name
if args.debug:
    args.epochs = 5

# logging config
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logging.info(args)
set_seed(args.seed)


def mix_grad(grad_list, weight_list):
    """
    calc weighted average of gradient
    """
    mixed_grad = []
    for g_list in zip(*grad_list):
        g_list = torch.stack([weight_list[i] * g_list[i] for i in range(len(weight_list))])
        mixed_grad.append(torch.sum(g_list, dim=0))
    return mixed_grad


def apply_grad(model, grad):
    """
    assign gradient to model(nn.Module) instance. return the norm of gradient
    """
    for p, g in zip(model.parameters(), grad):
        if p.grad is None:
            p.grad = g
        else:
            p.grad += g


def train(epoch, train_loader, model, criterion, optimizer, inner_optimizer=None):
    train_loss = AverageMeter()
    train_top1 = AverageMeter()
    train_top5 = AverageMeter()

    model.train()
    path_list = []
    num_candidate_ops = 5
    candidate_ops = list(range(5))
    candidate_edges = 6

    for step, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)
        if args.method == 'spos':
            # randomly sample an arch
            sampled_arch = [
                random.choice(candidate_ops) for _ in range(candidate_edges)
            ]
            optimizer.zero_grad()
            logits = model(inputs, sampled_arch)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

        elif args.method == 'fairnas':
            # shuffle the ops to get sub-models with strict fairness
            for _ in range(candidate_edges):
                random.shuffle(candidate_ops)
                path_list.append(deepcopy(candidate_ops))
            # inner loop
            optimizer.zero_grad()
            for _path_id in range(num_candidate_ops):
                sampled_arch = [_operations[_path_id] for _operations in path_list]
                logits = model(inputs, sampled_arch)
                loss = criterion(logits, targets)
                loss.backward()
                # record training metrics
                prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
                train_loss.update(loss.item(), inputs.size(0))
                train_top1.update(prec1.item(), inputs.size(0))
                train_top5.update(prec5.item(), inputs.size(0))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

        elif args.method == 'sumnas':
            # record the supernet weights
            weights_before = deepcopy(model.state_dict())
            grad_list = []

            # shuffle the ops to get sub-models fairly
            for _ in range(candidate_edges):
                random.shuffle(candidate_ops)
                path_list.append(deepcopy(candidate_ops))
            # inner loop
            for _path_id in range(num_candidate_ops):
                sampled_arch = [_operations[_path_id] for _operations in path_list]
                # inner optimization
                for _step in range(args.adaption_steps):
                    inner_optimizer.zero_grad()
                    logits = model(inputs, sampled_arch)
                    loss = criterion(logits, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    inner_optimizer.step()
                    # record training metrics
                    prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
                    train_loss.update(loss.item(), inputs.size(0))
                    train_top1.update(prec1.item(), inputs.size(0))
                    train_top5.update(prec5.item(), inputs.size(0))
                # record reptile gradient
                outer_grad = []
                weights_after = deepcopy(model.state_dict())
                for p_0, p_T in zip(weights_before.items(), weights_after.items()):
                    outer_grad.append(-(p_T[1] - p_0[1]).detach())
                grad_list.append(outer_grad)
                model.load_state_dict(weights_before)
            # outer loop
            optimizer.zero_grad()
            weight = torch.ones(len(grad_list)) / len(grad_list)
            grad = mix_grad(grad_list, weight)
            apply_grad(model, grad)
            optimizer.step()

        else:
            raise ValueError('Wrong training method for the supernet: %s' % args.method)
        # record
        prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
        train_loss.update(loss.item(), inputs.size(0))
        train_top1.update(prec1.item(), inputs.size(0))
        train_top5.update(prec5.item(), inputs.size(0))

        if step % args.print_freq == 0 or step + 1 == len(train_loader):
            logging.info('[Training] Epoch %03d/%03d, step %03d/%03d, loss: %.3f, top1: %.3f, top5: %.3f'
                         % (epoch, args.epochs, step, len(train_loader), train_loss.avg, train_top1.avg, train_top5.avg))
    return train_loss.avg, train_top1.avg, train_top5.avg


def valid(valid_loader, model, criterion):
    val_loss, val_top1, val_top5 = AverageMeter(), AverageMeter(), AverageMeter()
    model.eval()
    with torch.no_grad():
        for step, (val_inputs, val_targets) in enumerate(valid_loader):
            val_inputs = val_inputs.to(args.device)
            val_targets = val_targets.to(args.device)

            # randomly sample an arch
            candidate_ops = range(5)
            candidate_edges = 6
            sampled_arch = [
                random.choice(candidate_ops) for _ in range(candidate_edges)
            ]

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


def rank_supernet(valid_loader, model, criterion):
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
        target = nasbench201[item]['cifar10_test']
        val_loss, val_top1, val_top5 = valid_specific_path(valid_loader, model, operation, criterion, args.device)
        tmp_pred.append(val_top1)
        tmp_target.append(target)
        prediction[model_id] = {'id': model_id, 'model_gene': operation, 'pred': val_top1, 'target': target}
        if step % args.rank_print_freq == 0 or (step + 1) == nasbench201_len:
            logging.info("model_id: %d  gene: %s  loss: %.3f  top1: %.3f  target: %.3f"
                         % (model_id, str(operation), val_loss, val_top1, target))
            logging.info("Evaluated: %05d\tWaiting: %05d\tCurrent Kendall's Tau: %.5f" %
                         (len(tmp_pred), nasbench201_len-len(tmp_pred), kendalltau(tmp_pred, tmp_target)[0]))

    # save predictions
    print('\n')
    np.save(args.pred_path, prediction)
    logging.info('Finish ranking and save predictions to : %s' % args.pred_path)
    final_ranking = kendalltau(tmp_pred, tmp_target)[0]
    logging.info("Final_pred: %05d\tFinal_target: %05d\tFinal_Kendall's Tau: %.5f" %
                 (len(tmp_pred), len(tmp_target), final_ranking))

    return final_ranking


def main():
    # time record
    train_start = time.time()

    # dataloader
    train_loader, valid_loader = get_dataloader(args, model=None, dataset=args.dataset)

    # supernet
    model = Supernet201(
        C=args.channel, N=args.num_cells, max_nodes=args.max_nodes,
        num_classes=args.num_classes, search_space=NAS_BENCH_201
    ).to(args.device)

    # training settings
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd)
    if args.method == 'sumnas':
        args.adaption_steps = 2
        optimizer = torch.optim.SGD(model.parameters(), 1.0, weight_decay=4e-5)
        inner_optimizer = torch.optim.SGD(model.parameters(), 0.05, momentum=0.9)
    else:
        inner_optimizer = None
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = torch.nn.CrossEntropyLoss()
    best_val_top1 = 0.0

    logging.info('---------- Start to train supernet ----------')
    for epoch in range(args.epochs):
        # train supernet
        train_loss, train_top1, cnn_top5 = train(epoch, train_loader, model, criterion, optimizer, inner_optimizer)
        logging.info(
            "[Epoch: %s/%s] train_loss=%.3f, train_top1=%.3f, train_top5=%.3f" %
            (epoch, args.epochs, train_loss, train_top1, cnn_top5)
        )

        # valid supernet
        val_loss, val_top1, val_top5 = valid(valid_loader, model, criterion)
        logging.info(
            "[Validation], val_loss=%.3f, val_top1=%.3f, val_top5=%.3f, best_top1=%.3f" %
            (val_loss, val_top1, val_top5, best_val_top1)
        )
        if best_val_top1 < val_top1:
            best_val_top1 = val_top1

        # save latest checkpoint
        torch.save(model.state_dict(), args.ckpt_path)
        logging.info('Save latest checkpoint to %s' % args.ckpt_path)

        # scheduler step
        scheduler.step()
        print('\n')

    # time record
    supernet_training_elapse = time_record(train_start, prefix='Supernet training')
    print('\n')

    # load best supernet weights
    latest_pretrained_weights = torch.load(args.ckpt_path)
    model.load_state_dict(latest_pretrained_weights)
    model.eval()

    # ranking supernet
    final_ranking = rank_supernet(valid_loader, model, criterion)

    # write results
    with open('./results/ranking.txt', 'a') as f:
        f.write("EXP: %s \t Seed: %s \t Kendall' Tau: %.6f \t Training_Elapse: %s \n"
                % (args.exp_name, args.seed, final_ranking, supernet_training_elapse))


if __name__ == "__main__":
    run_func(args, main)
