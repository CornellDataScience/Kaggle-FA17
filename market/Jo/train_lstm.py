import os
import shutil
import argparse
import time

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from yellowfin import YFOptimizer

learning_rate = 0.0015
decay = 0.001
epochs = 80

parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')

parser.add_argument('--name', default='DenseNet_BC_100_12', type=str,
                    help='name of experiment')
parser.add_argument('--lr', '--learning-rate', default=0.0015, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.001, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--adam', '-a',
                    help='whether or not to use adam', action='store_true')
parser.add_argument('--yf',
                    help='whether or not to use yellowfin', action='store_true')
parser.add_argument('--layers', default=100, type=int,
                    help='total number of layers (default: 100)')

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

from dataloader import train_loader, val_loader

best_loss = 1000

class MarketLSTM(nn.Module):
    def __init__(self, hidden_dim):
        super(MarketLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(1, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).cuda().double(),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).cuda().double())

    def forward(self, input):
        lstm_out, self.hidden = self.lstm(
            input.view(-1, 1, 1), self.hidden)
        outputs = self.fc(lstm_out)
        return outputs


def main():
    global args, best_loss
    args = parser.parse_args()
    
    configure("runs/%s" % (args.name))

    model = MarketLSTM(args.layers)

    # get the number of model parameters

    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    kwargs = {'num_workers': 1, 'pin_memory': True}

    model = model.cuda().double()

    cudnn.benchmark = True

    # define loss function (criterion) and pptimizer
    criterion = nn.MSELoss().cuda()
    if args.adam:
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)
    elif args.yf:
        optimizer = YFOptimizer(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    for epoch in range(epochs):
        train(train_loader(), model, criterion, optimizer, epoch)
        loss = validate(val_loader(), model, criterion, epoch)


        # remember best prec@1 and save checkpoint
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec': best_loss,
        }, is_best)
    print('Best accuracy: ', best_loss)


def train(train_loader, model, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        model.zero_grad()
        model.hidden = model.init_hidden()

        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input)

        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))

        # compute gradient and do SGD step
        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses))
    # log to TensorBoard
    log_value('train_loss', losses.avg, epoch)


def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'
                    .format(
                        i, len(val_loader), batch_time=batch_time, loss=losses))

    # log to TensorBoard
    log_value('val_loss', losses.avg, epoch)
    return losses.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/" % (args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' %
                        (args.name) + 'model_best.pth.tar')


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

if __name__ == '__main__':
    main()
