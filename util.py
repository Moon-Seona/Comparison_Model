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
    # add under 3 lines
    del dataframe['Unnamed: 0']
    del dataframe['Unnamed: 0.1']
    return dataframe.drop_duplicates()

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

    return clothing, clothing_train, patio, home, arts, phone, sports

def rating_range(hypothesis):
    hypothesis[hypothesis > 5] = 5
    hypothesis[hypothesis < 1] = 1
    return hypothesis

def dataload():
    # pycharm
    # '~/Experiments/recommandation_system/multi_domain/multi_domain'
    # jupyter
    # '../..'
    #path = '~/Experiments/recommandation_system/multi_domain/multi_domain'
    path  = '.'
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

    kmeans = KMeans(n_clusters=num_clusters, mode='euclidean', verbose=1)
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

def construct_cluster(model, rating_matrix, n_clusters):
    device = 'cuda:0'

    rating_matrix = torch.LongTensor(rating_matrix).to(device)

    kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=1)

    users_feature = model.embed_user.weight
    items_feature = model.embed_item.weight

    user_cluster_idx = kmeans.fit_predict(users_feature).to(device)
    item_cluster_idx = kmeans.fit_predict(items_feature).to(device)

    # for i in range(10) :
    #     print(i, user_cluster_idx[user_cluster_idx==i].shape)
    #     print(i, item_cluster_idx[item_cluster_idx==i].shape)

    new_rating = rating_matrix.clone().float()

    for i in range(n_clusters):  # user
        for j in range(n_clusters):  # item

            # print(user_cluster_idx[rating_matrix[:,0]] == i, item_cluster_idx[rating_matrix[:,1]] == j)

            t = rating_matrix[(user_cluster_idx[rating_matrix[:, 0]] == i) & (item_cluster_idx[rating_matrix[:, 1]] == j), 2]

            if len(t) <= 5:  # threshold
                continue

            new_rating[(user_cluster_idx[rating_matrix[:, 0]] == i) & (
                        item_cluster_idx[rating_matrix[:, 1]] == j), 2] = torch.mean(t.float())
            # print(i, j, t.shape)
    # print(new_rating.shape, new_rating.mean(), new_rating)
    return user_cluster_idx, item_cluster_idx, new_rating

def recall(k, score, dic):
    '''
    k : top-k
    score : predict rating
    '''
    total_numerator = 0  # 분자
    total_denominator = 0  # 분모
    count = 0
    total = 0

    all_item = np.arange(score.shape[1])  # torch.range(0, score.shape[1]-1, dtype=int)

    for user_i, user_id in enumerate(tqdm(list(dic.keys()))):  # test user
        # top-k에서 맞춘 개수 / user 4점 이상인 test item 수

        for i in range(100):
            users_item = [dic[user_id]['pos_item'][0]] #np.random.choice(dic[user_id]['pos_item'],1)  # 5점 받은 item
            no_item = np.setdiff1d(all_item, dic[user_id]['pos_item'])
            #print(np.unique(score[user_id]), np.unique(score[user_id]).shape)
            #if user_i == 10 :
            #    break

            #np.random.seed(1000)
            sample_item = np.random.choice(no_item, 1000, replace=False) # 중복 제거

            check = np.concatenate((users_item, sample_item)) # sample_item
            #check2 = np.setdiff1d(all_item, check)
            #score[user_id][check2] = -1


            values, index = torch.topk(score[user_id][check], k)
            #if values[-1] <=  score[user_id][users_item][0]:
                #print(values[-1], score[user_id][users_item][0])
            #    numerator = 1
            #else:
            #    numerator = 0
            #print(values, index)
            numerator = np.intersect1d(users_item, check[index]).shape[0]
            denominator = dic[user_id]['count']
            if denominator == 0:
                #total_numerator += 1
                #total_denominator += 1
                #total += 1 #(numerator / denominator)
                #count += 1
                continue
            #if denominator < k and numerator >= k:
            #    numerator = denominator
            # recall
            total_denominator += denominator
            total_numerator += numerator
            total += (numerator/denominator)
            count+=1
            # if user_i == 1 :
            #     print(numerator, denominator)
            #     if i == 10 :
            #         break
    #print(count)
    #print(total_numerator/total_denominator == total/count)
    return total_numerator / total_denominator

def multi_data(model_name, sample, sample_ratio, k, seed) :
    # [arts, patio, sports, phone, home]
    pred1 = torch.load(f'save_dir/{model_name}_arts_{sample}_{sample_ratio}_{k}_{seed}_hy.pt').detach().cpu()
    pred2 = torch.load(f'save_dir/{model_name}_patio_{sample}_{sample_ratio}_{k}_{seed}_hy.pt').detach().cpu()
    pred3 = torch.load(f'save_dir/{model_name}_sports_{sample}_{sample_ratio}_{k}_{seed}_hy.pt').detach().cpu()
    pred4 = torch.load(f'save_dir/{model_name}_phone_{sample}_{sample_ratio}_{k}_{seed}_hy.pt').detach().cpu()
    pred5 = torch.load(f'save_dir/{model_name}_home_{sample}_{sample_ratio}_{k}_{seed}_hy.pt').detach().cpu()
    return torch.stack([pred1, pred2, pred3, pred4, pred5])

def get_count(main, domain, usernum) :
    # count 미리 저장
    domain_append = main.append(domain)
    domain_append = domain_append[domain_append.user.isin(main.user) & domain_append.item.isin(domain.item)]
    # 주 도메인의 사용자가 보조 도메인 arts에 평점 매긴 개수
    domain_count = domain_append.groupby('user').count().rating
    #print(domain_append.shape, domain_count.shape)
    # 모든 사용자에 대해 일반화
    # print(domain_count.index, domain_count.values)
    total_count = np.zeros(usernum)
    #print(total_count[list(domain_count.index)[0]])
    total_count[domain_count.index] = domain_count.values
    total_count = torch.tensor(total_count)
    return total_count

def sigmoid(x, n):
    x = x - n
    return 1 / (1 +torch.exp(-x))