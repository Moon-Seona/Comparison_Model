import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import random
from kmeans_pytorch import kmeans
from fast_pytorch_kmeans import KMeans

SEED = 2020
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def replaceIndex(dataframe, item_index, user_index):
    dataframe['item'] = dataframe['item'].apply(lambda x: item_index[x])
    dataframe['user'] = dataframe['user'].apply(lambda x: user_index[x])
    return dataframe

def set_index(clothing, clothing_train, patio, home, arts, phone, sports):
    total_users = np.hstack(
        [clothing['user'].values, patio['user'].values, home['user'].values, arts['user'].values, phone['user'].values,
         sports['user'].values])
    total_items = np.hstack(
        [clothing['item'].values, patio['item'].values, home['item'].values, arts['item'].values, phone['item'].values,
         sports['item'].values])

    users, user_indices = np.unique(total_users, return_index=True)
    items, item_indices = np.unique(total_items, return_index=True)

    users = np.array([total_users[index] for index in sorted(user_indices)])
    items = np.array([total_items[index] for index in sorted(item_indices)])

    item_index = {}
    user_index = {}

    for i, item in tqdm(enumerate(items)):
        item_index[item] = i
    for i, user in tqdm(enumerate(users)):
        user_index[user] = i

    clothing = replaceIndex(clothing, item_index, user_index)
    clothing_train = replaceIndex(clothing_train, item_index, user_index)
    patio = replaceIndex(patio, item_index, user_index)
    phone = replaceIndex(phone, item_index, user_index)
    home = replaceIndex(home, item_index, user_index)
    sports = replaceIndex(sports, item_index, user_index)
    arts = replaceIndex(arts, item_index, user_index)

    clothing = del_column(clothing)
    clothing_train = del_column(clothing_train)
    patio = del_column(patio)
    phone = del_column(phone)
    home = del_column(home)
    sports = del_column(sports)
    arts = del_column(arts)


    return clothing, clothing_train, patio, home, arts, phone, sports

def rating_range(hypothesis):
    hypothesis[hypothesis > 5] = 5
    hypothesis[hypothesis < 1] = 1
    return hypothesis

def del_column(domain):
    del domain['Unnamed: 0']
    del domain['Unnamed: 0.1']
    return domain

def dataload():
    # pycharm
    # '~/Experiments/recommandation_system/multi_domain/multi_domain'
    # jupyter
    # '../..'
    path = '~/Experiments/recommandation_system/multi_domain/multi_domain'
    # target
    clothing = pd.read_csv(f'{path}/dataset/new_clothing.csv')
    clothing_train = pd.read_csv(f'{path}/dataset/new_clothingset.csv')
    # auxiliary
    arts = pd.read_csv(f'{path}/dataset/new_arts.csv')
    home = pd.read_csv(f'{path}/dataset/new_home.csv')
    patio = pd.read_csv(f'{path}/dataset/new_patio.csv')
    phone = pd.read_csv(f'{path}/dataset/new_phone.csv')
    sports = pd.read_csv(f'{path}/dataset/new_sports.csv')
    # string to num
    clothing, clothing_train, patio, home, arts, phone, sports = set_index(clothing, clothing_train, patio, home, arts,
                                                                           phone, sports)
    item = clothing['item'].unique()
    user = clothing['user'].unique()

    # only testset
    #clothing = clothing[clothing_train.rating == 0]

    return clothing, clothing_train, arts, patio, home, phone, sports, user, item

def cluster_rating(users_mf, items_mf , rating, cluster) :
    device = 'cuda:0'

    num_clusters = cluster # 수 변경?
    #
    # # cluster idx, cluster center
    # user_cluster_idx, _ = kmeans(users_mf, num_clusters=num_clusters, device=device)
    # item_cluster_idx, _ = kmeans(items_mf, num_clusters=num_clusters, device=device)
    #aux_user_cluster_idx, _ = kmeans(aux_users_mf, num_clusters=num_clusters, device=device)
    #aux_item_cluster_idx, _ = kmeans(aux_items_mf, num_clusters=num_clusters, device=device)
    # cluster rating mean..
    # threshold = 5 ?
    kmeans = KMeans(n_clusters=num_clusters, mode='euclidean', verbose=1) # 여기를 변경해야댐!!!!!!!!!!!!!!!!!!!
    user_cluster_idx = kmeans.fit_predict(users_mf)
    item_cluster_idx = kmeans.fit_predict(items_mf)

    # for i in range(10) :
    #     print(i, user_cluster_idx[user_cluster_idx==i].shape)
    #     print(i, item_cluster_idx[item_cluster_idx==i].shape)

    new_rating = torch.zeros(rating.shape, dtype=torch.double).to(device)
    user_cluster_idx, item_cluster_idx = user_cluster_idx.to(device), item_cluster_idx.to(device)

    for i in range(num_clusters) : # user
        for j in range(num_clusters) : # item
            # user, item list
            idx = (user_cluster_idx == i) & (item_cluster_idx == j)
            t = rating[idx]
            if len(t) <= 5 : # threshold
                continue
            new_rating[idx] = torch.mean(t)
            #print(i, j, t.shape)
    #print(new_rating.shape, new_rating.mean(), new_rating)
    return new_rating


def recall(k, score, dic):
    '''
    k : top-k
    score : predict rating
    '''
    total = 0
    count = 0
    total_numerator = 0  # 분자
    total_denominator = 0  # 분모

    all_item = np.arange(score.shape[1])  # torch.range(0, score.shape[1]-1, dtype=int)

    for user_i, user_id in enumerate(tqdm(list(dic.keys()))):  # test user
        # top-k에서 맞춘 개수 / user 4점 이상인 test item 수
        users_item = dic[user_id]['item']  # 5점 받은 item
        no_item = np.setdiff1d(all_item, users_item)
        sample_item = np.random.choice(no_item, 1000)
        check = np.concatenate((users_item, sample_item))
        check2 = np.setdiff1d(all_item, check)
        score[user_id][check2] = 0

        values, index = torch.topk(score[user_id], k)

        numerator = np.intersect1d(users_item, index).shape[0]
        denominator = dic[user_id]['count']
        # denominator2 = k

        if denominator == 0:
            continue
        elif denominator < k and numerator >= k:
            numerator = denominator
        # recall
        total_denominator += denominator
        # total_denominator2 += denominator2
        total_numerator += numerator
        count += 1

    return total_numerator / total_denominator

def multi_data(model_name, sample) :
    pred1 = torch.load(f'save_dir/{model_name}_arts_{sample}_hy.pt').detach().cpu()
    pred2 = torch.load(f'save_dir/{model_name}_patio_{sample}_hy.pt').detach().cpu()
    pred3 = torch.load(f'save_dir/{model_name}_sports_{sample}_hy.pt').detach().cpu()
    pred4 = torch.load(f'save_dir/{model_name}_phone_{sample}_hy.pt').detach().cpu()
    pred5 = torch.load(f'save_dir/{model_name}_home_{sample}_hy.pt').detach().cpu()
    return torch.stack([pred1, pred2, pred3, pred4, pred5])

def get_count(main, domain) :
    # count 미리 저장
    domain_append = main.append(domain)
    domain_append = domain_append[domain_append.user.isin(main.user) & domain_append.item.isin(domain.item)]
    # 주 도메인의 사용자가 보조 도메인 arts에 평점 매긴 개수
    domain_count = domain_append.groupby('user').count().rating
    # 모든 사용자에 대해 일반화
    total_count = np.zeros(len(main.user.unique()))
    #print(total_count[list(domain_count.index)[0]])
    total_count[domain_count.index] = domain_count.values
    total_count = torch.tensor(total_count)
    return total_count

def sigmoid(x, n):
    x = x - n
    return 1 / (1 +torch.exp(-x))