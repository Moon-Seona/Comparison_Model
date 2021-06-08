import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import random
from kmeans_pytorch import kmeans

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

def cluster_rating(main_users, main_items, aux_users, aux_items, main_users_mf, main_items_mf, aux_users_mf, aux_items_mf, main, aux) :
    device = 'cuda:0'

    num_clusters = 10

    # cluster idx, cluster center
    main_user_cluster_idx, _ = kmeans(main_users_mf, num_clusters=num_clusters, device=device)
    main_item_cluster_idx, _ = kmeans(main_items_mf, num_clusters=num_clusters, device=device)
    aux_user_cluster_idx, _ = kmeans(aux_users_mf, num_clusters=num_clusters, device=device)
    aux_item_cluster_idx, _ = kmeans(aux_items_mf, num_clusters=num_clusters, device=device)
    # cluster rating mean..
    # threshold = 5 ?
    main_rating = torch.tensor([li[2] for li in main])
    aux_rating = torch.tensor([li[2] for li in aux])
    # main
    cluster_i = [] # user
    cluster_j = [] # item
    new_rating = torch.zeros(main_rating.shape, dtype=torch.double)
    for (u,i,rating) in tqdm(main) : # 2 seconds
        #print(u,i,rating)
        cluster_i.append(main_user_cluster_idx[u].item())
        cluster_j.append(main_item_cluster_idx[i].item())

    cluster_i = torch.tensor(cluster_i)
    cluster_j = torch.tensor(cluster_j)

    for i in range(num_clusters) : # user
        for j in range(num_clusters) : # item
            # user, item list
            idx = (cluster_i == i) & (cluster_j == j)
            t = main_rating[idx]
            if len(t) <= 5 : # threshold
                continue
            new_rating[idx] = torch.mean(t)
            #print(i, j, t.shape)
    #print(new_rating.shape, new_rating.mean(), new_rating)
    # aux
    cluster_i2 = []  # user
    cluster_j2 = []  # item
    new_rating2 = torch.zeros(aux_rating.shape, dtype=torch.double)
    for (u, i, rating) in tqdm(aux):  # 2 seconds
        # print(u,i,rating)
        cluster_i2.append(aux_user_cluster_idx[u].item())
        cluster_j2.append(aux_item_cluster_idx[i].item())

    cluster_i2 = torch.tensor(cluster_i2)
    cluster_j2 = torch.tensor(cluster_j2)

    for i in range(num_clusters):  # user
        for j in range(num_clusters):  # item
            # user, item list
            idx = (cluster_i2 == i) & (cluster_j2 == j)
            t = aux_rating[idx]
            if len(t) <= 5:  # threshold
                continue
            new_rating2[idx] = torch.mean(t)
    #print(new_rating2[new_rating2==0].shape)
    return new_rating, new_rating2