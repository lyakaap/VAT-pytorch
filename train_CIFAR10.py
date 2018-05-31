import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from vat import VATLoss
import data_utils
import utils


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3)
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(-1, 128)
        x = self.fc1(x)
        return x


def train(args, model, device, data_iterators, optimizer):
    model.train()
    for i in tqdm(range(args.iters)):
        
        # reset
        if i % args.log_interval == 0:
            ce_losses = utils.AverageMeter()
            vat_losses = utils.AverageMeter()
            prec1 = utils.AverageMeter()
        
        x_l, y_l = next(data_iterators['labeled'])
        x_ul, _ = next(data_iterators['unlabeled'])

        x_l, y_l = x_l.to(device), y_l.to(device)
        x_ul = x_ul.to(device)

        optimizer.zero_grad()

        vat_loss = VATLoss(xi=args.xi, eps=args.eps, ip=args.ip)
        cross_entropy = nn.CrossEntropyLoss()

        lds = vat_loss(model, x_ul)
        output = model(x_l)
        classification_loss = cross_entropy(output, y_l)
        loss = classification_loss + args.alpha * lds
        loss.backward()
        optimizer.step()

        acc = utils.accuracy(output, y_l)
        ce_losses.update(classification_loss.item(), x_l.shape[0])
        vat_losses.update(lds.item(), x_ul.shape[0])
        prec1.update(acc.item(), x_l.shape[0])

        if i % args.log_interval == 0:
            print(f'\nIteration: {i}\t'
                  f'CrossEntropyLoss {ce_losses.val:.4f} ({ce_losses.avg:.4f})\t'
                  f'VATLoss {vat_losses.val:.4f} ({vat_losses.avg:.4f})\t'
                  f'Prec@1 {prec1.val:.3f} ({prec1.avg:.3f})')


def test(model, device, data_iterators):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in tqdm(data_iterators['test']):
            with torch.no_grad():
                x, y = x.to(device), y.to(device)
                outputs = model(x)
            correct += torch.eq(outputs.max(dim=1)[1], y).detach().cpu().float().sum()

        test_acc = correct / len(data_iterators['test'].dataset) * 100.

    print(f'\nTest Accuracy: {test_acc:.4f}%\n')


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--iters', type=int, default=10000, metavar='N',
                        help='number of iterations to train (default: 10000)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA',
                        help='regularization coefficient (default: 0.01)')
    parser.add_argument('--xi', type=float, default=10.0, metavar='XI',
                        help='hyperparameter of VAT (default: 0.1)')
    parser.add_argument('--eps', type=float, default=1.0, metavar='EPS',
                        help='hyperparameter of VAT (default: 1.0)')
    parser.add_argument('--ip', type=int, default=1, metavar='IP',
                        help='hyperparameter of VAT (default: 1)')
    parser.add_argument('--workers', type=int, default=8, metavar='W',
                        help='number of CPU')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_iterators = data_utils.get_iters(
        root_path='.',
        l_batch_size=args.batch_size,
        ul_batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        workers=args.workers
    )

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    train(args, model, device, data_iterators, optimizer)
    test(model, device, data_iterators)


if __name__ == '__main__':
    main()
