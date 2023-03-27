import logging
import os
import random
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return "{name}(val={val}, avg={avg}, count={count})".format(
            name=self.__class__.__name__, **self.__dict__
        )


def obtain_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def set_seed(seed):
    """
        Fix all seeds
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    cudnn.enabled = True
    cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def time_record(start, prefix=None):
    import logging
    import time
    end = time.time()
    duration = end - start
    hour = duration // 3600
    minute = (duration - hour * 3600) // 60
    second = duration - hour * 3600 - minute * 60
    if prefix is not None:
        logging.info('%s Elapsed Time: %dh %dmin %ds' % (prefix, hour, minute, second))
    else:
        logging.info('Total Elapsed Time: %dh %dmin %ds' % (hour, minute, second))

    return '%dh %dmin %ds' % (hour, minute, second)


def gpu_monitor(gpu, sec, used=100):
    import time
    import pynvml
    import logging

    wait_min = sec // 60
    divisor = 1024 * 1024
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    if meminfo.used / divisor < used:
        logging.info('GPU-{} is free, start runing!'.format(gpu))
        return False
    else:
        logging.info('GPU-{}, Memory: total={}MB used={}MB free={}MB, waiting {}min...'.format(
            gpu,
            meminfo.total / divisor,
            meminfo.used / divisor,
            meminfo.free / divisor,
            wait_min)
        )
        time.sleep(sec)
        return True


def run_func(args, main):
    import time
    if torch.cuda.is_available():
        while gpu_monitor(args.gpu_id, sec=300, used=7000):
            pass
    start_time = time.time()
    result = main()
    time_record(start_time)
    # email_sender(result=result, config=args)


class SamplingParams(object):
    """Training parameters for the training of class sampling"""

    def __init__(self, steps_per_epoch, classes):
        self.classes = classes
        self.total_steps = 0
        self.steps_per_epoch = steps_per_epoch
        self.reset()

    def reset(self):
        self.ssg = [0.0 for _ in range(self.classes)]
        self.count = [0.0 for _ in range(self.classes)]

    def update(self, class_train, grad_norm, batch_size):
        self.total_steps += 1
        self.ssg[class_train] += grad_norm
        self.count[class_train] += batch_size


def get_grad_norm(net):
    arr_sum = 0.
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if layer.weight.grad is not None:
                arr_sum += layer.weight.grad.norm()

    return arr_sum.item()
