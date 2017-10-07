import argparse
import os, sys
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import densenet as dn

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

from data import output_dataset, le

parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--model', default='', type=str,
                    help='path to best model')

best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()

    # create model
    if os.path.isfile(args.model):
        model = torch.load(args.resume)
    else:
        print("no such model")
        sys.exit()

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    
    # for training on multiple GPUs. 
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    cudnn.benchmark = True

    model.eval()

    out = np.array([])

    for input, idx in output_dataset:
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)

        # compute output
        output = model(input_var)
        finalPred = le.inverse_transform(torch.max(output, 1))

        out = np.append([[idx,finalPred]], axis = 0)

    print(out)

    #for image, idx in output_dataset:

if __name__ == '__main__':
    main()