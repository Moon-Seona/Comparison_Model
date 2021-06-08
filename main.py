import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


from MF import *
from util import *

SEED = 2020
torch.manual_seed(SEED)

def main(model_name, aux_name, lr, lambda1, lambda2, k, epoch, batch, device):

    device = torch.device(device)

    clothing, clothing_train, arts, patio, home, phone, sports, user, item = dataload()

    trainset = clothing[clothing_train.rating != 0]
    testset = clothing[clothing_train.rating == 0]

    if model_name == 'biasedMF' or model_name == 'MF' :
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
        elif aux_name == 'multi' :
            trainset = trainset.append(arts)
            trainset = trainset.append(patio)
            trainset = trainset.append(home)
            trainset = trainset.append(phone)
            trainset = trainset.append(sports)
            #print(trainset.shape)
        else :
            print('Check aux domain name!')
            exit(1)

    elif model_name == 'CBMF' :
        if aux_name == 'arts' :
            aux = arts
        elif aux_name == 'patio' :
            aux = patio
        elif aux_name == 'sports' :
            aux = sports
        elif aux_name == 'phone' :
            aux = phone
        elif aux_name == 'home' :
            aux = home
        else :
            print('check aux domain name!')
            exit(1)
        aux_itemnum = max(aux.item) + 1
        aux_usernum = max(aux.user) + 1

    itemnum = max(trainset.item) + 1
    usernum = max(trainset.user) + 1

    #print(usernum, itemnum, aux_usernum, aux_itemnum)

    if model_name == 'biasedMF':
        model = biasedMF(usernum, itemnum, k).to(device)
    elif model_name == 'CBMF':
        # get predict rating
        model = CBMF(usernum, itemnum, k).to(device)
        aux_model = CBMF(aux_usernum, aux_itemnum, k).to(device)
        # UserWarning: torch.range is deprecated in favor of torch.arange and will be removed in 0.5. Note that arange generates values in [start; end), not [start; end].
        main_users = torch.range(0, usernum-1, dtype=torch.long) # min(trainset.user)
        main_items = torch.range(0, itemnum-1, dtype=torch.long) # min(trainset.item),
        aux_users = torch.range(0, aux_usernum-1, dtype=torch.long) # min(aux.user),
        aux_items = torch.range(0, aux_itemnum-1, dtype=torch.long) # min(aux.user),
        # why except min? aux domain changes range in each domains..
        #print(main_users, main_items, aux_users, aux_items)
        aux_optimizer = torch.optim.Adam(aux_model.parameters(), lr=lr)
        #trainset.append(aux)
    else:
        print('Check model name!')
        exit(1)

    # add weight decay?
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    average = torch.tensor(trainset.rating.values).mean()
    trainset = TensorDataset(torch.tensor(trainset.user.values), torch.tensor(trainset.item.values), torch.tensor(trainset.rating.values))
    train_data_loader = DataLoader(trainset, batch_size=batch, shuffle=True)
    if model_name == 'CBMF' :
        aux_average = torch.tensor(aux.rating.values).mean()
        aux = TensorDataset(torch.tensor(aux.user.values), torch.tensor(aux.item.values), torch.tensor(aux.rating.values))
        aux_data_loader = DataLoader(aux, batch_size=batch, shuffle=True)

    best_epoch = -1
    best_loss = float('inf')
    best_mae = float('inf')
    best_rmse = float('inf')

    for n in tqdm(range(epoch)):
        total_loss = 0
        model.train()
        # new epoch start, already have to know user's cluster and item's cluster
        if model_name == 'CBMF' :
            # calculate cluster
            model.eval()
            aux_model.eval()

            main_users, main_items, aux_users, aux_items = main_users.to(device), main_items.to(device), aux_users.to(device), aux_items.to(device)

            _, main_users_mf, main_items_mf = model(main_users, main_items, average.to(device))
            _, aux_users_mf, aux_items_mf = aux_model(aux_users, aux_items, aux_average.to(device))

            new_rating, new_rating2 = cluster_rating(main_users, main_items, aux_users, aux_items, main_users_mf, main_items_mf, aux_users_mf, aux_items_mf, trainset, aux)

            # cluster trainset rating example
            trainset_rating2 = 0
            trainset2 = TensorDataset(torch.tensor(trainset.user.values), torch.tensor(trainset.item.values), torch.tensor(trainset_rating2))
            train_data_loader2 = DataLoader(trainset2, batch_size=batch, shuffle=True)

            model.train()
            aux_model.train()

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

        print('eopch: ', n, 'train loss: ', round(total_loss/len(t)/batch, 4))

        if model_name == 'CBMF' :
            t2 = tqdm(aux_data_loader, smoothing=0, mininterval=1.0)
            for iter, (u, i, rating) in enumerate(t2):
                u, i, rating = u.to(device), i.to(device), rating.to(device)
                pred, x, theta = model(u, i, aux_average.to(device))
                # regularization
                reg_x = (lambda1 / 2) * torch.pow(x, 2).sum()
                reg_theta = (lambda1 / 2) * torch.pow(theta, 2).sum()
                reg_bias_user = (lambda2 / 2) * torch.pow(model.user_bias[u], 2).sum()
                reg_bias_item = (lambda2 / 2) * torch.pow(model.item_bias[i], 2).sum()

                cost = torch.pow(rating - pred - model.user_bias[u] - model.item_bias[i], 2).sum() + reg_x + reg_theta + reg_bias_user + reg_bias_item
                # print(cost)
                aux_optimizer.zero_grad()
                cost.backward()
                aux_optimizer.step()

                total_loss += cost.item()
                t.set_description('(Loss: %g)' % cost)

            print('eopch: ', n, 'train loss: ', round(total_loss / len(t) / batch, 4))


        model.eval()
        with torch.no_grad() :
            u  = torch.tensor(testset.user.values).to(device)
            i  = torch.tensor(testset.item.values).to(device)
            rating  = torch.tensor(testset.rating.values).to(device)
            pred, x, theta = model(u, i, average.to(device))
            # regularization
            reg_x = (lambda1 / 2) * torch.pow(x, 2).sum()
            reg_theta = (lambda1 / 2) * torch.pow(theta, 2).sum()
            reg_bias_user = (lambda2 / 2) * torch.pow(model.user_bias[u], 2).sum()
            reg_bias_item = (lambda2 / 2) * torch.pow(model.item_bias[i], 2).sum()
            loss = torch.pow(rating - pred - model.user_bias[u] - model.item_bias[i], 2).sum() + reg_x + reg_theta + reg_bias_user + reg_bias_item
            #print(reg_x, reg_theta, reg_bias_item, reg_bias_user)

            diff = pred - rating
            mae = torch.abs(diff).mean()
            rmse = torch.sqrt(torch.pow(diff, 2).mean())
            loss = loss/len(u)

            print('test loss: ', loss, 'MAE: ', mae, 'RMSE: ', rmse)

            if mae <= best_mae: # or rmse <= best_rmse :
                best_epoch = n
                torch.save(model, f'save_dir/{model_name}_{aux_name}_{best_epoch}.pt')
                best_mae = mae
                best_rmse = rmse

    print('best epoch: ', best_epoch, 'best mae: ', best_mae, 'best rmse: ', best_rmse)

if __name__ == '__main__' :
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='CBMF')
    parser.add_argument('--aux_name', default='arts')
    parser.add_argument('--lr', type=float, default=0.001) # 0.001
    parser.add_argument('--lambda1', type=int, default=5) # 50
    parser.add_argument('--lambda2', type=float, default=0.5) # 10
    parser.add_argument('--k', type=int, default=20) # 5
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    print('Called with args: ')
    print(args)
    main(args.model_name, args.aux_name, args.lr, args.lambda1, args.lambda2, args.k, args.epoch, args.batch, args.device)