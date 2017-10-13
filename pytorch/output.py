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
import numpy as np
import pandas as pd

import densenet as dn
import resnet as rn

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

from data import output_dataset, le

parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--model', default='', type=str,
                    help='path to best model')

parser.add_argument('--layers', default=100, type=int,
                    help='total number of layers (default: 100)')
parser.add_argument('--growth', default=12, type=int,
                    help='number of new channels per layer (default: 12)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--reduce', default=0.5, type=float,
                    help='compression rate in transition stage (default: 0.5)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='To not use bottleneck block')
parser.add_argument('--type', default='dn3', type=str,
                    help='type of network')

best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()

    # create model
    
    if args.type == "dn3":
        model = dn.DenseNet3(args.layers, 120, args.growth, reduction=args.reduce,
                             bottleneck=args.bottleneck)
    elif args.type == "resnet":
        model = rn.ResNetTransfer(120)

    else: raise Exception('No such model exists - choose dn3 or resnet')

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    
    # for training on multiple GPUs. 
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    cudnn.benchmark = True

    if os.path.isfile(args.model):
        checkpoint = torch.load(args.model)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        sys.exit()
        
    model.eval()

    out = pd.read_csv('sample_submission.csv')

    out = out.set_index("id")

    for input, img in output_dataset:
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True).unsqueeze(0)

        # compute output
        output = torch.nn.functional.softmax(model(input_var))

        #print(output[0].data.numpy())
        print(img)

        print(output[0].data.cpu().numpy())

        out.loc[img] = output[0].data.cpu().numpy()
    out.to_csv("out.csv")

if __name__ == '__main__':
    main()