import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _l2_normalize(d):
    if isinstance(d, Variable):
        d = d.data.numpy()
    elif isinstance(d, torch.Tensor):
        d = d.numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2, 3))).reshape((-1, 1, 1, 1)) + 1e-16)
    return torch.from_numpy(d)


def vat_loss(model, X, xi=0.1, eps=1.0, Ip=1, use_gpu=True):
    """VAT loss function
    :param model: networks to train
    :param X: Variable, input
    :param xi: hyperparameter of VAT (default: 1.0)
    :param eps: hyperparameter of VAT (default: 1.0)
    :param Ip: iteration times of computing adv noise (default: 1)
    :param use_gpu: use gpu or not (default: True)
    :return: LDS, model prediction (for classification-loss calculation)
    """
    kl_div = nn.KLDivLoss()
    if use_gpu:
        kl_div.cuda()

    pred = model(X)

    # prepare random unit tensor
    d = torch.rand(X.shape)
    d = Variable(_l2_normalize(d))
    if use_gpu:
        d = d.cuda()
        
    # calc adversarial direction
    for ip in range(Ip):
        d.requires_grad = True
        pred_hat = model(X + d / xi)
        adv_distance = kl_div(F.log_softmax(pred_hat, dim=1), pred.detach())
        adv_distance.backward()
        d = Variable(_l2_normalize(d.grad.data))
        model.zero_grad()

    # calc LDS
    r_adv = d * eps
    pred_hat = model(X + r_adv)
    pred = model(X)
    LDS = kl_div(F.log_softmax(pred_hat, dim=1), pred.detach())
    return LDS, pred
