import argparse
import logging
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from datasets.dataloader import get_dataloader
from models.cell_operations import NAS_BENCH_201
from models.cell_operations import ResNetBasicblock
from models.supernet import Supernet201
from rank_func import rank_supernet
from utils import obtain_accuracy, AverageMeter, set_seed, run_func, time_record

parser = argparse.ArgumentParser("Train 201 Supernet")
# dataset
parser.add_argument("--data_root", type=str, default='./dataset/', help="The path to dataset")
parser.add_argument("--dataset", type=str, default='cifar10_rebuild_loader', help="Dataset.")
parser.add_argument("--search_space_name", type=str, default='nas-bench-201', help="The search space name.")
parser.add_argument("--num_classes", type=int, default=10, help="Dataset Classes")
# supernet
parser.add_argument("--max_nodes", type=int, default=4, help="The maximum number of nodes.")
parser.add_argument("--channel", type=int, default=16, help="The number of channels.")
parser.add_argument("--num_cells", type=int, default=5, help="The number of cells in one stage.")
# training settings
parser.add_argument("--exp_name", type=str, default='debug_panda', help='exp_name for saving results')
parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
parser.add_argument("--tau", type=float, default=0.5, help="Momentum")
parser.add_argument("--percent_train", type=float, default=0.5, help="training samples percent to go backward")
parser.add_argument("--wd", type=float, default=2.5e-4, help="Weight decay")
parser.add_argument("--epochs", type=int, default=250, help="Training epochs")
parser.add_argument("--gpu_id", type=int, default=0, help="Training GPU")
parser.add_argument("--train_batch_size", type=int, default=256, help="Train batch size")
parser.add_argument("--valid_batch_size", type=int, default=512, help="Valid batch size")
parser.add_argument("--print_freq", type=int, default=50, help="print frequency when training")
parser.add_argument("--update_freq", type=int, default=98, help="Frequency for updating class prob")
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

args.up_prob = np.array([1 / 5 for _ in range(5)])
args.ip_prob = args.up_prob
logging.info(args)
set_seed(args.seed)


def path_importance_sampling(ip_prob, candidate_ops, candidate_edges):
    sampled_arch = np.random.choice(candidate_ops, size=candidate_edges, p=ip_prob).tolist()
    return sampled_arch


def operatio_gradient_norm_prob(model, sampled_arch, path_grads_norm):
    """
    Compute op probability according to the average op gradients norm.

    Args:
        model: model after loss.backward().
        sampled_arch: sampled forward path.
        up_prob: uniform sampling probability.

    Returns:

    """
    weights_norm = [[0.0] for _ in range(5)]
    for tmp_cell in model.cells:
        if isinstance(tmp_cell, ResNetBasicblock):
            continue
        for edge_idx, tmp_edge in enumerate(tmp_cell.edges):
            op_idx = sampled_arch[edge_idx]
            if op_idx in [2, 3]:
                op_forward = tmp_cell.edges[tmp_edge][op_idx].op[1]
                weights_norm[op_idx].append(op_forward.weight.grad.norm().item())
                weights_norm[op_idx].append(op_forward.bias.grad.norm().item())

    for op_idx, tmp_norm in enumerate(weights_norm):
        tmp_norm = np.sum(tmp_norm)
        if tmp_norm != 0.0:
            tmp_norm /= sampled_arch.count(op_idx)
            path_grads_norm[op_idx] = np.mean([path_grads_norm[op_idx], tmp_norm])

    return path_grads_norm


def sample_prob_with_grads_norm(path_grads_norm):
    last_ip_prob = args.ip_prob
    weights_norm_prob = np.array(path_grads_norm) / np.sum(path_grads_norm)
    args.ip_prob = weights_norm_prob * args.path_tau + args.up_prob * (1 - args.path_tau)
    logging.info('Update importance_path_probability: '
                 'old=[%.3f, %.3f, %.3f, %.3f, %.3f], now=[%.3f, %.3f, %.3f, %.3f, %.3f]' %
                 (*last_ip_prob, *args.ip_prob))


def train(epoch, train_data, train_loader, model, criterion_per_sample, optimizer):
    train_loss = AverageMeter()
    train_top1 = AverageMeter()
    train_top5 = AverageMeter()

    model.train()
    candidate_edges = 6
    candidate_ops = list(range(5))
    train_grad_norm = []
    train_sample_indices = []
    path_grads_norm = [0.0 for _ in range(5)]
    logging.info('Epoch: %d, importance_path_sampling_prob: [%.3f, %.3f, %.3f, %.3f, %.3f]' %
                 (epoch, *args.ip_prob))

    # sample a minibatch of instances to train
    for step, (inputs, targets, indices) in enumerate(train_loader):
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)

        # randomly sample an arch to train
        sampled_arch = path_importance_sampling(args.ip_prob, candidate_ops, candidate_edges)
        optimizer.zero_grad()
        logits = model(inputs, sampled_arch)
        loss_per_sample = criterion_per_sample(logits, targets)

        # compute the grad norm per sample
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            one_hot_targets = F.one_hot(targets, num_classes=args.num_classes)
            grad_norm = (torch.norm(probs - one_hot_targets, dim=-1).detach().cpu().numpy())
        loss = loss_per_sample.mean()

        # optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        # update grads norm of each trainable parameter
        path_grads_norm = operatio_gradient_norm_prob(model, sampled_arch, path_grads_norm)

        # record grad norm of each sample
        train_grad_norm.extend(grad_norm)
        train_sample_indices.extend(indices.cpu().numpy())

        # record training metrics
        prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
        train_loss.update(loss.item(), inputs.size(0))
        train_top1.update(prec1.item(), inputs.size(0))
        train_top5.update(prec5.item(), inputs.size(0))

        if step % args.print_freq == 0 or step + 1 == len(train_loader):
            logging.info('[Training] Epoch %03d/%03d, step %03d/%03d, loss: %.3f, top1: %.3f, top5: %.3f, tau: %.3f, '
                         'path_tau: %.3f, prob: [%.3f, %.3f, %.3f, %.3f, %.3f], tmp_path: %s'
                         % (epoch, args.epochs, step, len(train_loader),
                            train_loss.avg, train_top1.avg, train_top5.avg, args.tau,
                            args.path_tau, *args.ip_prob, sampled_arch))

    # update path sampling probability
    sample_prob_with_grads_norm(path_grads_norm)

    # update per instance sample prob and training indices of the next epoch
    sample_grad_norm = np.zeros(len(train_data))
    for _idx, _grad_norm in zip(train_sample_indices, train_grad_norm):
        if sample_grad_norm[_idx] != 0:
            sample_grad_norm[_idx] = np.mean([sample_grad_norm[_idx], _grad_norm])
        else:
            sample_grad_norm[_idx] = _grad_norm
    sum_grad_norm = np.sum(sample_grad_norm)
    cur_prob = sample_grad_norm / sum_grad_norm
    new_prob = cur_prob * args.tau + (1 - args.tau) * train_data.initial_prob
    indices_next_epoch = np.random.choice(train_data.initial_indices, size=len(train_data), replace=True, p=new_prob)
    train_data.update_running_indices(indices_next_epoch)

    return train_loss.avg, train_top1.avg, train_top5.avg


def valid(valid_loader, model, criterion):
    val_loss, val_top1, val_top5 = AverageMeter(), AverageMeter(), AverageMeter()
    model.eval()
    candidate_ops = range(5)
    candidate_edges = 6

    with torch.no_grad():
        for step, (val_inputs, val_targets) in enumerate(valid_loader):
            val_inputs = val_inputs.to(args.device)
            val_targets = val_targets.to(args.device)

            # randomly sample an arch
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


def main():
    # time record
    train_start = time.time()

    # supernet
    model = Supernet201(
        C=args.channel, N=args.num_cells, max_nodes=args.max_nodes,
        num_classes=args.num_classes, search_space=NAS_BENCH_201
    ).to(args.device)

    # dataloader
    train_data, valid_data, train_loader, valid_loader = get_dataloader(args, model, dataset=args.dataset)

    # training settings
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = torch.nn.CrossEntropyLoss()
    criterion_per_sample = torch.nn.CrossEntropyLoss(reduction='none')
    best_val_top1 = 0.0
    logging.info('---------- Start to train supernet ----------')

    for epoch in range(args.epochs):
        args.tau = epoch / args.epochs
        args.path_tau = epoch / args.epochs

        # train supernet
        train_loss, train_top1, cnn_top5 = train(
            epoch, train_data, train_loader, model, criterion_per_sample, optimizer,
        )
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

    # load latest supernet weights
    latest_pretrained_weights = torch.load(args.ckpt_path)
    model.load_state_dict(latest_pretrained_weights)
    model.eval()

    # ranking supernet
    final_ranking = rank_supernet(args, valid_loader, model, criterion)

    # write results
    with open('./results/ranking.txt', 'a') as f:
        f.write("EXP: %s \t Seed: %s \t Kendall' Tau: %.6f \t Training_Elapse: %s \n"
                % (args.exp_name, args.seed, final_ranking, supernet_training_elapse))


if __name__ == "__main__":
    run_func(args, main)
