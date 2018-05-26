import torch
import torch.nn as nn
import torch.nn.functional as F


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, 1, 1)
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATLoss(nn.Module):

    def __init__(self, model, xi=0.1, eps=1.0, ip=1):
        """VAT loss
        :param model: networks to train
        :param xi: hyperparameter of VAT (default: 1.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.model = model
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, x):

        kl_div = nn.KLDivLoss()

        with torch.no_grad():
            pred = self.model(x)

        # prepare random unit tensor
        d = torch.rand(x.shape).to(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        d = _l2_normalize(d)

        # calc adversarial direction
        for _ in range(self.ip):
            d.requires_grad = True
            pred_hat = self.model(x + d / self.xi)
            adv_distance = kl_div(F.log_softmax(pred_hat, dim=1), pred)
            adv_distance.backward()
            d = _l2_normalize(d.grad)
            self.model.zero_grad()

        # calc LDS
        r_adv = d * self.eps
        pred_hat = self.model(x + r_adv)
        lds = kl_div(F.log_softmax(pred_hat, dim=1), pred)

        return lds
