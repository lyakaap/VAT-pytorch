import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
            
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


def _kl_div(log_probs, probs):
    # pytorch KLDLoss is averaged over all dim if size_average=True
    kld = F.kl_div(log_probs, probs, size_average=False)
    return kld / log_probs.shape[0]


class VATLoss(nn.Module):

    def __init__(self, model, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param model: networks to train
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.model = model
        self.xi = xi
        self.eps = eps
        self.ip = ip
        self.ent_min = ent_min

    def forward(self, x, d, a):
        with torch.no_grad():
            pred = F.softmax(self.model(x), dim=1)

        # prepare random unit tensor
        d = torch.rand(x.shape).to(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        d = _l2_normalize(d)

        # calc adversarial direction
        for _ in range(self.ip):
            d.requires_grad_()
            with _disable_tracking_bn_stats(self.model):
                pred_hat = self.model(x + self.xi * d)
            adv_distance = _kl_div(F.log_softmax(pred_hat, dim=1), pred)
            adv_distance.backward()
            d = _l2_normalize(d.grad)
            self.model.zero_grad()

        # calc LDS
        r_adv = d * self.eps
        with _disable_tracking_bn_stats(self.model):
            pred_hat = self.model(x + r_adv)
        lds = _kl_div(F.log_softmax(pred_hat, dim=1), pred)

        return lds
