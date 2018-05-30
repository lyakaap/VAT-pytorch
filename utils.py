from collections import OrderedDict
import logging
import logzero
from pathlib import Path
from tensorboardX import SummaryWriter
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, top_k=(1,)):
    """Computes the precision@k for the specified values of k"""
    max_k = max(top_k)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    if len(res) == 1:
        res = res[0]

    return res


def save_checkpoint(model, epoch, filename, optimizer=None):
    if optimizer is None:
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
        }, filename)
    else:
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename)


def load_checkpoint(model, path, optimizer=None):
    resume = torch.load(path)

    if ('module' in list(resume['state_dict'].keys())[0]) \
            and not (isinstance(model, torch.nn.DataParallel)):
        new_state_dict = OrderedDict()
        for k, v in resume['state_dict'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(resume['state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(resume['optimizer'])
        return model, optimizer
    else:
        return model


def set_logger(path, loglevel=logging.INFO, tf_board_path=None):
    path_dir = '/'.join(path.split('/')[:-1])
    if not Path(path_dir).exists():
        Path(path_dir).mkdir(parents=True)
    logzero.loglevel(loglevel)
    logzero.formatter(logging.Formatter('[%(asctime)s %(levelname)s] %(message)s'))
    logzero.logfile(path)

    if tf_board_path is not None:
        tb_path_dir = '/'.join(tf_board_path.split('/')[:-1])
        if not Path(tb_path_dir).exists():
            Path(tb_path_dir).mkdir(parents=True)
        writer = SummaryWriter(tf_board_path)

        return writer
