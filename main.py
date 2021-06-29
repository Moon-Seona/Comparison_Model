import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
#from fast_pytorch_kmeans import KMeans
from MF import *
from util import *
from datetime import datetime
import os.path
from model import *

SEED = 2020
torch.manual_seed(SEED)

def main(model_name, aux_name, lr, lr2, k, epoch, batch, n_cluster, alpha, sample, sample_ratio, top, mode, weight_decay, weight_decay2, device):

    device = torch.device(device)

    clothing, clothing_train, arts, patio, home, phone, sports, user, item = dataload()
    #print(len(user))

    trainset = clothing[clothing_train.rating != 0]
    testset = clothing[clothing_train.rating == 0]

    clod_start_user = np.unique(testset.user)
    test = {}
    for i in clod_start_user:
        if len(testset[(testset.user == i) & (testset.rating == 5)]['item']) == 0:
            # test[i]['item'] = -1
            # test[i]['count'] = 0
            continue

        test[i] = {}
        itemlist = list(testset[(testset.user == i) & (testset.rating == 5)]['item'])
        test[i]['pos_item'] = itemlist  # np.random.choice(itemlist, 1).item()
        # test[i]['neg_item'] = testset[testset.rating<5]['item']
        # print(itemlist, test[i]['item'])
        test[i]['count'] = 1  # len(testset[(testset.user == i) & (testset.rating==5)]['item'])
        # print(test[i])

    if sample : # sample all domain for campare with biasedCBMF
        trainset = trainset.sample(frac=sample_ratio) # , random_state=SEED
        #print(trainset.item.unique().shape)
        # replace=True 제거
        arts = arts.sample(frac=sample_ratio)
        patio = patio.sample(frac=sample_ratio)
        sports = sports.sample(frac=sample_ratio)
        phone = phone.sample(frac=sample_ratio)
        home = home.sample(frac=sample_ratio)

        #clothing_train = trainset.append(testset)
        #print(clothing_train.shape)

    if mode == 'multi' :
        if model_name == 'CDCF' :
            pred = torch.load('/home/moon/Experiments/recommandation_system/multi_domain/multi_domain/dataset/rating_matrix/new_cdcf.pt')
        elif model_name == 'AF' :
            pred = torch.load('/home/moon/Experiments/recommandation_system/multi_domain/multi_domain/dataset/rating_matrix/af.pt')
            pred = pred[:,:max(user)+1,:] # all user -> main domain user
        else :
            pred = multi_data(model_name, sample, sample_ratio, k, SEED)
        print('WEIGHT MULTI DOMAIN')
        domain_list = [arts, patio, sports, phone, home]
        domain_count = torch.zeros((len(domain_list), len(user)))
        #print(domain_count.shape)
        for i, domain in enumerate(domain_list):
            domain_count[i] = get_count(clothing_train, domain, len(user))
        for n in range(15) :
            print('Weight: ', n)
            weight_n = torch.zeros((len(domain_list), len(user)))
            for j, domain in enumerate(domain_count):
                weight_n[j] = sigmoid(domain_count[j], n)
            weight_d = weight_n.sum(axis=0)
            #weight_trainset = pred[0,:,:]
            weight_trainset = torch.tensor(np.expand_dims((weight_n / weight_d), axis=-1)) * pred
            weight_trainset = weight_trainset.sum(axis=0)
            weight_trainset = rating_range(weight_trainset)

            weight_rmse = weight_trainset[torch.tensor(testset.user.values), torch.tensor(testset.item.values)]

            mae = torch.abs(weight_rmse-torch.tensor(testset.rating.values)).mean()
            rmse = torch.sqrt(torch.pow((weight_rmse-torch.tensor(testset.rating.values)), 2).mean())
            test_recall = recall(top, weight_trainset, test)
            print('MAE: ', mae, 'RMSE: ', rmse, 'recall: ', round(test_recall, 4))
        exit(1)
        #print(trainset.shape, arts.shape, patio.shape, sports.shape, phone.shape, home.shape, testset.shape)
    now = datetime.now()

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
        #print(trainset.shape, trainset.user.unique().shape, trainset.item.unique().shape)
        #print(trainset.shape[0] / (trainset.user.unique().shape[0] *trainset.item.unique().shape[0]) * 100)
    else :
        print('Check aux domain name!')
        exit(1)

    trainset = trainset.reset_index()
    del trainset['index']
    #print(trainset.index, trainset.columns)

    # if model_name == 'CDCF' :
    #     # CDCF correlation
    #     file = f'dataset/{aux_name}_pearson_correlation_{sample}_{sample_ratio}.pt'
    #     if ~os.path.isfile(file):
    #         #trainset = trainset.append(clothing_train[clothing_train.rating==0])
    #         correlation(trainset, user, aux_name, sample, sample_ratio)

    itemnum = max(trainset.item) + 1
    usernum = max(trainset.user) + 1

    #print(usernum, itemnum, aux_usernum, aux_itemnum)

    if model_name == 'MF' :
        model = MF(usernum, itemnum, k).to(device)
    elif model_name == 'biasedMF':
        model = biasedMF(usernum, itemnum, k).to(device)
    elif model_name == 'CBMF':
        # get predict rating
        model = MF(usernum, itemnum, k).to(device)
        cluster_model = CBMF(usernum, itemnum, k).to(device)
        optimizer2 = torch.optim.Adam(cluster_model.parameters(), lr=lr2, weight_decay=weight_decay2)
    elif model_name == 'biasedCBMF' :
        model = biasedMF(usernum, itemnum, k).to(device)
        cluster_model = biasedCBMF(usernum, itemnum, k).to(device)
        optimizer2 = torch.optim.Adam(cluster_model.parameters(), lr=lr2, weight_decay=weight_decay2)
    elif model_name == 'CDCF' :
        model = CDCF(trainset, usernum, itemnum, k)
    #    #pred = CDCF(aux_name)
    #     pred = torch.load('/home/moon/Experiments/recommandation_system/multi_domain/multi_domain/dataset/rating_matrix/cdcf.pt')
    #     print(pred.shape)
    else:
        print('Check model name!')
        exit(1)

    # add weight decay?
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    average = torch.tensor(trainset.rating.values).mean()
    trainset1 = TensorDataset(torch.tensor(trainset.index.values), torch.tensor(trainset.user.values), torch.tensor(trainset.item.values), torch.tensor(trainset.rating.values))
    train_data_loader = DataLoader(trainset1, batch_size=batch, shuffle=True)

    best_epoch = -1
    best_loss = float('inf')
    best_mae = float('inf')
    best_rmse = float('inf')
    best_hy = torch.zeros(testset.shape)

    #pred, _, _ = model(torch.tensor(user).to(device), torch.tensor(item).to(device), average.to(device))
    #print(pred.shape)

    for n in tqdm(range(epoch)):

        ### Training

        total_loss = 0
        total_loss2 = 0
        model.train()
        t = tqdm(train_data_loader, smoothing=0, mininterval=1.0)
        for iter, (idx, u, i, rating) in enumerate(t):
            pred, x, theta = model(u.to(device), i.to(device), average.to(device))

            cost = torch.pow(rating.to(device) - torch.diag(pred, 0), 2).sum()

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            total_loss += cost.item()
            t.set_description('(Loss: %g)' % cost)

        if 'CBMF' in model_name:
            cluster_model.train()
            # why model? (why not cluster_model?)
            # in paper, fig2. clustering latent matrices P and Q to archieve clusters of users and items and producing the coarse matrix.

            user_cluster, item_cluster, new_rating = construct_cluster(model, trainset.to_numpy(), n_cluster)
            t = tqdm(train_data_loader, smoothing=0, mininterval=1.0)
            for iter, (idx, u, i, rating) in enumerate(t):
                #print(idx)
                u, i = u.to(device), i.to(device)

                pred2, x, theta = cluster_model(user_cluster[u], item_cluster[i], average.to(device))

                cluster_rating =  new_rating[idx, 2] #torch.zeros((u.shape[0]), dtype=torch.float).to(device)

                # 격하게 for문 제거하고싶다....ㅠㅠㅠㅠㅠ
                # for index in range(u.shape[0]):
                #     # check = new_rating[(new_rating[:, 0] == u[index]) & (new_rating[:, 1] == i[index]), 2]
                #     # print(u[index], i[index])
                #     # print(new_rating[:, 0] == u[index])
                #     # print(new_rating[(new_rating[:, 0] == u[index]) & (new_rating[:, 1] == i[index])])
                #     # print(check)
                #     # print(check[0])
                #     cluster_rating[index] = new_rating[(new_rating[:, 0] == u[index]) & (new_rating[:, 1] == i[index]), 2][0]

                # modify optimizer to optimizer2
                cost = torch.pow(cluster_rating - torch.diag(pred2, 0), 2).sum()
                optimizer2.zero_grad()
                cost.backward()
                optimizer2.step()

                #total_loss += cost.item()
                total_loss2 += cost.item()
                t.set_description('(Cluster Loss: %g)' % cost)

        if 'CBMF' in model_name :
            print('eopch: ', n, 'MF loss: ', round(total_loss / len(t) / batch, 4), 'Cluster loss: ', round(total_loss2/len(t)/batch, 4))
        else :
            print('eopch: ', n, 'train loss: ', round(total_loss / len(t) / batch, 4))

        ### Validation

        model.eval()
        with torch.no_grad():
            u = torch.tensor(testset.user.values).to(device)
            i = torch.tensor(testset.item.values).to(device)
            rating = torch.tensor(testset.rating.values).to(device)

            pred, x, theta = model(u, i, average.to(device))
            pred = torch.diag(pred, 0)
            loss = torch.pow(rating - pred, 2).sum()

            if 'CBMF' in model_name:
                cluster_model.eval()
                pred2, x, theta = cluster_model(user_cluster[u], item_cluster[i], average)

                pred2 = torch.diag(pred2, 0)
                pred = pred * (1 - alpha) + pred2 * (alpha)

                loss = torch.pow(rating - pred, 2).sum()  # + reg_x2 + reg_theta2 + reg_bias_user2 + reg_bias_item2
                # loss = loss*(1-alpha) + loss2*alpha
            diff = pred - rating
            mae = torch.abs(diff).mean()
            rmse = torch.sqrt(torch.pow(diff, 2).mean())
            print('MAE: ', round(mae.item(), 4), 'RMSE: ', round(rmse.item(), 4), 'loss: ',
                  round(loss.item() / u.shape[0], 4))
            if mae <= best_mae:  # best_loss > loss : # rmse <= best_rmse: #
                best_epoch = n
                torch.save(model, f'save_dir/{model_name}_{aux_name}_{best_epoch}_{n_cluster}_{lr2}.pt')
                best_mae = mae
                best_rmse = rmse
                best_hy = pred
                best_loss = loss

    model = torch.load(f'save_dir/{model_name}_{aux_name}_{best_epoch}_{n_cluster}_{lr2}.pt')

    pred, _, _ = model(torch.tensor(user).to(device), torch.tensor(item).to(device), average.to(device))

    test_recall = recall(top, pred.detach().cpu(), test)
    print('best epoch: ', best_epoch, 'best loss: ', round(best_loss.item() / u.shape[0], 4), 'best mae: ',
          round(best_mae.item(), 4), 'best rmse: ', round(best_rmse.item(), 4), 'best recall: ', round(test_recall, 4))

    torch.save(pred, f'save_dir/{model_name}_{aux_name}_{sample}_{sample_ratio}_{k}_{SEED}_{lr2}_hy.pt')

if __name__ == '__main__' :
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='biasedCBMF')
    parser.add_argument('--aux_name', default='arts')
    parser.add_argument('--lr', type=float, default=0.0003) # 0.001
    parser.add_argument('--lr2', type=float, default=0.0001)
    parser.add_argument('--k', type=int, default=10) # 5
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--cluster', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--sample', type=bool, default=False)
    parser.add_argument('--sample_ratio', type=float, default=0.1)
    parser.add_argument('--top', type=int, default=100) # 20
    parser.add_argument('--mode', default='cross')
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--weight_decay2', type=float, default=0.0001) # lr2 정해진 다음에 실험 예정
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    print('Called with args: ')
    print(args)
    main(args.model_name, args.aux_name, args.lr, args.lr2, args.k, args.epoch, args.batch, args.cluster,
         args.alpha, args.sample, args.sample_ratio, args.top, args.mode, args.weight_decay, args.weight_decay2, args.device)