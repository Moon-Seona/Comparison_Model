import numpy as np
import pandas as pd
import torch

from CBMF import *
from util import *

def main(model_name, aux_name, lr, lambda1, lambda2, epoch, batch, device):

    device = torch.device(device)

    if model_name != 'MF' :
        print('Check model name!')
        exit(1)

    clothing, clothing_train, arts, patio, home, phone, sports, user, item = dataload()

    usernum = user[-1] + 1

    if aux_name == 'arts' :
        domain = clothing.append(arts)

    itemnum = item[-1] + 1
    #print(usernum, itemnum)

    dfdf

if __name__ == '__main__' :
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='MF')
    parser.add_argument('--aux_name', default='arts')
    parser.add_argument('--lr', type=float, default='1e-3')
    parser.add_argument('--lambda1', type=int, default=50)
    parser.add_argument('--lambda2', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    print('Called with args: ')
    print(args)
    main(args.model_name, args.aux_name, args.lr, args.lambda1, args.lambda2, args.epoch, args.batch, args.device)