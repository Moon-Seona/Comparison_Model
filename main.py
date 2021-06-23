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

def main(model_name, aux_name, lr, lambda1, lambda2, k, epoch, batch, cluster, alpha, sample, sample_ratio, top, mode, weight_decay, device):

    device = torch.device(device)

    clothing, clothing_train, arts, patio, home, phone, sports, user, item = dataload()
    #print(len(user))

    trainset = clothing[clothing_train.rating != 0]
    testset = clothing[clothing_train.rating == 0]

    clod_start_user = np.unique(testset.user)
    test = {}
    for i in clod_start_user:
        if len(testset[(testset.user == i) & (testset.rating==5)]['item']) == 0:
            #test[i]['item'] = -1
            #test[i]['count'] = 0
            continue
        test[i] = {}
        itemlist = list(testset[(testset.user == i) & (testset.rating==5)]['item'])
        test[i]['item'] = np.random.choice(itemlist, 1).item()
        #print(itemlist, test[i]['item'])
        test[i]['count'] = 1 #len(testset[(testset.user == i) & (testset.rating==5)]['item'])
        #print(test[i])

    if sample : # sample all domain for campare with biasedCBMF
        trainset = trainset.sample(frac=sample_ratio, replace=True, random_state=SEED)
        #print(trainset.item.unique().shape)
        arts = arts.sample(frac=sample_ratio, random_state=SEED)
        patio = patio.sample(frac=sample_ratio, random_state=SEED)
        sports = sports.sample(frac=sample_ratio, random_state=SEED)
        phone = phone.sample(frac=sample_ratio, random_state=SEED)
        home = home.sample(frac=sample_ratio, random_state=SEED)

        #clothing_train = trainset.append(testset)
        #print(clothing_train.shape)

    if mode == 'multi' :
        if model_name == 'CDCF' :
            pred = torch.load('/home/moon/Experiments/recommandation_system/multi_domain/multi_domain/dataset/rating_matrix/cdcf.pt')
        elif model_name == 'AF' :
            pred = torch.load('/home/moon/Experiments/recommandation_system/multi_domain/multi_domain/dataset/rating_matrix/af.pt')
            pred = pred[:,:max(user)+1,:] # all user -> main domain user
        else :
            pred = multi_data(model_name, sample, sample_ratio, k)
        print('WEIGHT MULTI DOMAIN')
        domain_list = [arts, patio, sports, phone, home]
        domain_count = torch.zeros((len(domain_list), len(user)))
        #print(domain_count.shape)
        for i, domain in enumerate(domain_list):
            domain_count[i] = get_count(clothing_train, domain, len(user))
        for n in range(20) :
            print('Weight: ', n)
            weight_n = torch.zeros((len(domain_list), len(user)))
            for j, domain in enumerate(domain_count):
                weight_n[j] = sigmoid(domain_count[j], n)
            weight_d = weight_n.sum(axis=0)
            #weight_trainset = pred[4,:,:]
            weight_trainset = torch.tensor(np.expand_dims((weight_n / weight_d), axis=-1)) * pred
            weight_trainset = weight_trainset.sum(axis=0)
            weight_trainset = rating_range(weight_trainset)

            weight_rmse = weight_trainset[torch.tensor(testset.user.values), torch.tensor(testset.item.values)]

            rmse = torch.sqrt(torch.pow((weight_rmse-torch.tensor(testset.rating.values)), 2).mean())
            test_recall = recall(top, weight_trainset, test)
            print('RMSE: ', rmse, 'recall: ', round(test_recall, 4))

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
    '''
    # CDCF correlation
    file = f'dataset/{aux_name}_pearson_correlation_{sample}_{sample_ratio}.pt'
    if ~os.path.isfile(file):
        correlation(trainset, user, aux_name, sample, sample_ratio)
    '''
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
        optimizer2 = torch.optim.Adam(cluster_model.parameters(), lr=lr, weight_decay=weight_decay)
    elif model_name == 'biasedCBMF' :
        model = biasedMF(usernum, itemnum, k).to(device)
        cluster_model = biasedCBMF(usernum, itemnum, k).to(device)
        optimizer2 = torch.optim.Adam(cluster_model.parameters(), lr=lr, weight_decay=weight_decay)
    # elif model_name == 'CDCF' :
    #    #pred = CDCF(aux_name)
    #     pred = torch.load('/home/moon/Experiments/recommandation_system/multi_domain/multi_domain/dataset/rating_matrix/cdcf.pt')
    #     print(pred.shape)
    else:
        print('Check model name!')
        exit(1)

    # add weight decay?
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    average = torch.tensor(trainset.rating.values).mean()
    trainset1 = TensorDataset(torch.tensor(trainset.user.values), torch.tensor(trainset.item.values), torch.tensor(trainset.rating.values))
    train_data_loader = DataLoader(trainset1, batch_size=batch, shuffle=True)

    best_epoch = -1
    best_loss = float('inf')
    best_mae = float('inf')
    best_rmse = float('inf')
    best_hy = torch.zeros(testset.shape)

    #pred, _, _ = model(torch.tensor(user).to(device), torch.tensor(item).to(device), average.to(device))
    #print(pred.shape)

    for n in tqdm(range(epoch)):
        if model_name == 'CBMF' or model_name == 'biasedCBMF':
            rating2 = []
            model.eval()
            with torch.no_grad():
                #t = tqdm(train_data_loader, smoothing=0, mininterval=1.0)
                for iter, (u, i, rating) in enumerate(train_data_loader):
                    u, i, rating = u.to(device), i.to(device), rating.to(device)
                    _, x, theta = cluster_model(u, i, average.to(device))
                    new_rating = cluster_rating(x, theta, rating, cluster)
                    rating2.extend(new_rating)
                rating2 = torch.tensor(rating2)

        total_loss = 0
        model.train()
        t = tqdm(train_data_loader, smoothing=0, mininterval=1.0)
        for iter, (u, i, rating) in enumerate(t):
            u, i, rating = u.to(device), i.to(device), rating.to(device)
            pred, x, theta = model(u, i, average.to(device))
            pred = torch.diag(pred, 0)
            # regularization
            #reg_x = (lambda1 / 2) * torch.pow(x, 2).sum()
            #reg_theta = (lambda1 / 2) * torch.pow(theta, 2).sum()
            cost = torch.pow(rating - pred, 2).sum() #+ reg_x + reg_theta
            if model_name == 'biasedMF' or model_name == 'biasedCBMF' :
                reg_bias_user = (lambda2 / 2) * torch.pow(model.user_bias[u], 2).sum()
                reg_bias_item = (lambda2 / 2) * torch.pow(model.item_bias[i], 2).sum()
                cost = torch.pow(rating-pred, 2).sum() #+ reg_x + reg_theta + reg_bias_user + reg_bias_item
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            # cluster
            if model_name == 'CBMF' or model_name == 'biasedCBMF':
                pred2, x2, theta2 = cluster_model(u, i, average.to(device))
                pred2 = torch.diag(pred2, 0)
                rating2 = cluster_rating(x, theta, rating)
                #reg_x2 = (lambda1 / 2) * torch.pow(x2, 2).sum()
                #reg_theta2 = (lambda1 / 2) * torch.pow(theta2, 2).sum()
                rating2 = rating2.to(device)
                #reg_bias_user2 = (lambda2 / 2) * torch.pow(cluster_model.user_bias[u], 2).sum()
                #reg_bias_item2 = (lambda2 / 2) * torch.pow(cluster_model.item_bias[i], 2).sum()
                cost2 = torch.pow(rating2[iter*batch:(iter+1)*batch]-pred2, 2).sum() #+ reg_x2 + reg_theta2 + reg_bias_user2 + reg_bias_item2
                optimizer2.zero_grad()
                cost2.backward()
                optimizer2.step()

            total_loss += cost.item()
            t.set_description('(Loss: %g)' % cost)

        print('eopch: ', n, 'train loss: ', round(total_loss/len(t)/batch, 4))

        model.eval()
        with torch.no_grad() :
            u  = torch.tensor(testset.user.values).to(device)
            i  = torch.tensor(testset.item.values).to(device)
            rating  = torch.tensor(testset.rating.values).to(device)
            pred, x, theta = model(u, i, average.to(device))
            pred = torch.diag(pred, 0)
            #reg_x = (lambda1 / 2) * torch.pow(x, 2).sum()
            #reg_theta = (lambda1 / 2) * torch.pow(theta, 2).sum()
            loss = torch.pow(rating - pred, 2).sum() #+ reg_x + reg_theta

            if model_name == 'biasedMF' or model_name == 'biasedCBMF' :
                #reg_bias_user = (lambda2 / 2) * torch.pow(model.user_bias[u], 2).sum()
                #reg_bias_item = (lambda2 / 2) * torch.pow(model.item_bias[i], 2).sum()
                loss = torch.pow(rating-pred-model.user_bias[u]-model.item_bias[i], 2).sum() #+ reg_x + reg_theta + reg_bias_user + reg_bias_item

            if model_name == 'CBMF' or model_name == 'biasedCBMF':
                pred2, x, theta = cluster_model(u, i, average.to(device))
                pred2 = torch.diag(pred2, 0)
                pred = pred * (1-alpha) + pred2 * (alpha) # pred 비율을 늘려야 할듯 !
                #reg_x2 = (lambda1 / 2) * torch.pow(x2, 2).sum()
                #reg_theta2 = (lambda1 / 2) * torch.pow(theta2, 2).sum()
                rating2 = rating2.to(device)
                #reg_bias_user2 = (lambda2 / 2) * torch.pow(cluster_model.user_bias[u], 2).sum()
                #reg_bias_item2 = (lambda2 / 2) * torch.pow(cluster_model.item_bias[i], 2).sum()
                loss2 = torch.pow(rating2[iter * batch:(iter + 1) * batch] - pred2, 2).sum() #+ reg_x2 + reg_theta2 + reg_bias_user2 + reg_bias_item2
                loss = loss*(1-alpha) + loss2*alpha

            diff = pred - rating
            mae = torch.abs(diff).mean()
            rmse = torch.sqrt(torch.pow(diff, 2).mean())
            print('RMSE: ', round(rmse.item(), 4), 'loss: ', round(loss.item()/u.shape[0], 4))
            #test_loss = loss*(1-alpha) + loss2*alpha
            if best_loss > loss : # rmse <= best_rmse: #
                best_epoch = n
                torch.save(model, f'save_dir/{model_name}_{aux_name}_{best_epoch}_{cluster}_{now}.pt')
                best_mae = mae
                best_rmse = rmse
                best_hy = pred
                best_loss = loss

    model = torch.load(f'save_dir/{model_name}_{aux_name}_{best_epoch}_{cluster}_{now}.pt')

    pred, _, _ = model(torch.tensor(user).to(device), torch.tensor(item).to(device), average.to(device))

    test_recall = recall(top, pred.detach().cpu(), test)
    print('best epoch: ', best_epoch, 'best loss: ', round(best_loss.item()/u.shape[0], 4), 'best rmse: ', round(best_rmse.item(), 4), 'best recall: ', round(test_recall, 4))

    torch.save(pred, f'save_dir/{model_name}_{aux_name}_{sample}_{sample_ratio}_{k}_hy.pt')

if __name__ == '__main__' :
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='biasedMF')
    parser.add_argument('--aux_name', default='arts')
    parser.add_argument('--lr', type=float, default=0.0003) # 0.001
    parser.add_argument('--lambda1', type=int, default=5) # 50
    parser.add_argument('--lambda2', type=float, default=0.5) # 10
    parser.add_argument('--k', type=int, default=10) # 5
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--cluster', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--sample', type=bool, default=False)
    parser.add_argument('--sample_ratio', type=float, default=0.1)
    parser.add_argument('--top', type=int, default=20)
    parser.add_argument('--mode', default='cross')
    parser.add_argument('--weight_decay', default=0.00001)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    print('Called with args: ')
    print(args)
    main(args.model_name, args.aux_name, args.lr, args.lambda1, args.lambda2, args.k, args.epoch, args.batch, args.cluster,
         args.alpha, args.sample, args.sample_ratio, args.top, args.mode, args.weight_decay, args.device)