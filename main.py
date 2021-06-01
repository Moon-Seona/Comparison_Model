#import numpy as np
#import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

#from CBMF import MF
from MF import *
from util import *

SEED = 2020
torch.manual_seed(SEED)

def main(model_name, aux_name, lr, lambda1, lambda2, k, epoch, batch, device):

    device = torch.device(device)

    clothing, clothing_train, arts, patio, home, phone, sports, user, item = dataload()

    trainset = clothing[clothing_train.rating != 0]
    testset = clothing[clothing_train.rating == 0]

    if aux_name == 'arts' :
        trainset = trainset.append(arts)
    elif aux_name == 'patio' :
        trainset  = trainset.append(patio)
    elif aux_name == 'home' :
        trainset = trainset.append(home)
    elif aux_name == 'phone' :
        trainset = trainset.append(phone)
    elif aux_name == 'sports' :
        trainset = trainset.append(sports)
    else :
        print('Check aux domain name!')
        exit(1)

    itemnum = max(trainset.item) + 1
    usernum = max(trainset.user) + 1
    #print(usernum, itemnum)

    if model_name == 'MF':
        model = biasedMF(usernum, itemnum, k).to(device)
    else:
        print('Check model name!')
        exit(1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # weight_decay 추가 ?

    average = torch.tensor(trainset.rating.values).mean()
    trainset = TensorDataset(torch.tensor(trainset.user.values), torch.tensor(trainset.item.values), torch.tensor(trainset.rating.values))
    train_data_loader = DataLoader(trainset, batch_size=batch, shuffle=True)

    best_epoch = -1
    best_loss = float('inf')
    best_mae = float('inf')
    best_rmse = float('inf')

    for n in tqdm(range(epoch)):
        total_loss = 0
        model.train()
        t = tqdm(train_data_loader, smoothing=0, mininterval=1.0)
        for iter, (u, i, rating) in enumerate(t):
            u, i, rating = u.to(device), i.to(device), rating.to(device)
            pred, x, theta = model(u, i, average.to(device))
            # regularization
            reg_x = (lambda1 / 2) * torch.pow(x, 2).sum()
            reg_theta = (lambda1 / 2) * torch.pow(theta, 2).sum()
            reg_bias_user = (lambda2 / 2) * torch.pow(model.user_bias[u], 2).sum()
            reg_bias_item = (lambda2 / 2) * torch.pow(model.item_bias[i], 2).sum()

            cost = torch.pow(rating-pred-model.user_bias[u]-model.item_bias[i], 2).sum() + reg_x + reg_theta + reg_bias_user + reg_bias_item
            #print(cost)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            total_loss += cost.item()
            t.set_description('(Loss: %g)' % cost)

        print('eopch: ', n, 'train loss: ', round(total_loss/len(t), 4))

        model.eval()
        with torch.no_grad() :
            u  = torch.tensor(testset.user.values).to(device)
            i  = torch.tensor(testset.item.values).to(device)
            rating  = torch.tensor(testset.rating.values).to(device)
            pred, _, _ = model(u, i, average.to(device))

            diff = pred - rating
            mae = torch.abs(diff).mean()
            rmse = torch.sqrt(torch.pow(diff, 2).mean())

            print('MAE: ', mae, 'RMSE: ', rmse)

            if mae <= best_mae or rmse <= best_rmse :
                best_epoch = n
                torch.save(model, f'save_dir/{model_name}_{aux_name}_{best_epoch}.pt')
                best_mae = mae
                best_rmse = rmse

    print('best epoch: ', best_epoch, 'best mae: ', best_mae, 'best rmse: ', best_rmse)

if __name__ == '__main__' :
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='MF')
    parser.add_argument('--aux_name', default='home')
    parser.add_argument('--lr', type=float, default='1e-3')
    parser.add_argument('--lambda1', type=int, default=50)
    parser.add_argument('--lambda2', type=int, default=10)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    print('Called with args: ')
    print(args)
    main(args.model_name, args.aux_name, args.lr, args.lambda1, args.lambda2, args.k, args.epoch, args.batch, args.device)