import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import random

SEED = 2020
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def replaceIndex(dataframe, item_index, user_index):
    dataframe['item'] = dataframe['item'].apply(lambda x: item_index[x])
    dataframe['user'] = dataframe['user'].apply(lambda x: user_index[x])    
    return dataframe

def set_index(clothing, clothing_train, patio, home, arts, phone, sports) :
    total_users = np.hstack([clothing['user'].values, patio['user'].values, home['user'].values, arts['user'].values, phone['user'].values, sports['user'].values])
    total_items = np.hstack([clothing['item'].values, patio['item'].values, home['item'].values, arts['item'].values, phone['item'].values, sports['item'].values])

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

def rating_range(hypothesis) :
    hypothesis[hypothesis>5] = 5
    hypothesis[hypothesis<1] = 1
    return hypothesis
    
def dataload() :
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
    clothing, clothing_train, patio, home, arts, phone, sports = set_index(clothing, clothing_train, patio, home, arts, phone, sports)
    item = clothing['item'].unique()
    user = clothing['user'].unique()

    clothing = clothing[clothing_train.rating == 0] # only testset

    return clothing, clothing_train, arts, patio, home, phone, sports, user, item

def get_trainset(main, aux, user, item, test_bool) :
    # triplet을 만들면, 굳이 아래 numpy를 만들어야 하나?
    # main
    main_user = main.user.to_numpy()
    main_item = main.item.to_numpy()
    main_rating = main.rating.to_numpy()
    # aux
    item2 = aux.item.to_numpy() # use for random choice
    if test_bool :
        aux = aux.loc[aux.item.isin(main_item)]
    else :
        aux = aux.loc[aux.user.isin(main_user)]
    aux_user = aux.user.to_numpy()
    aux_item = aux.item.to_numpy()
    aux_rating = aux.rating.to_numpy()
    linked_user = np.intersect1d(main_user, aux_user)
    trainset1 = [] # user, item1, item2
    trainset2 = [] # rating1, rating2

    for i in user :
        main_idx = np.where(main_user == i)[0] # length: 0도 가능
        aux_idx = np.where(aux_user == i)[0]
        # 두 도메인 모두 rating이 있는 경우: linked user
        if len(main_idx) == 0 :
            item_d2 = aux_item[aux_idx]
            rating_d2 = aux_rating[aux_idx]
            for j in range(len(aux_idx)) :
                triplet = []
                label = []
                triplet.append(i)
                triplet.append(random.choice(item))
                triplet.append(item_d2[j])
                label.append(0)
                label.append(rating_d2[j])
                trainset1.append(triplet)
                trainset2.append(label)
        elif len(aux_idx) == 0 : # only main item
            item_d1 = main_item[main_idx]
            rating_d1 = main_rating [main_idx]
            for j in range(len(main_idx)) :
                triplet = []
                label = []
                triplet.append(i)
                triplet.append(item_d1[j])
                triplet.append(random.choice(item2))
                label.append(rating_d1[j])
                label.append(0)
                trainset1.append(triplet)
                trainset2.append(label)
        else: # 두 도메인 모두 item  #if i in linked_user
            # 1. 길이 비교
            # 더 긴쪽이 우선...
            item_d1 = main_item[main_idx]
            item_d2 = aux_item[aux_idx]
            rating_d1 = main_rating[main_idx]
            rating_d2 = aux_rating[aux_idx]
            if len(main_idx) > len(aux_idx) :
                for j in range(len(aux_idx)) : # user, item d1, item d2
                    triplet = []
                    label = []
                    triplet.append(i)
                    triplet.append(item_d1[j])
                    triplet.append(item_d2[j])
                    label.append(rating_d1[j])
                    label.append(rating_d2[j])
                    trainset1.append(triplet)
                    trainset2.append(label)
                for k in range(len(aux_idx), len(main_idx)) :
                    triplet = []
                    label = []
                    triplet.append(i)
                    triplet.append(item_d1[k])
                    triplet.append(random.choice(item2))
                    label.append(rating_d1[k])
                    label.append(0)
                    trainset1.append(triplet)
                    trainset2.append(label)
            # 만약 error 가 난다면, 이유는 아이템 수가 같은 경우를 고려하지 않았기 떄문..
            else : # aux item 수가 더 많은 경우
                for j in range(len(main_idx)) : # user, item d1, item d2
                    triplet = []
                    label = []
                    triplet.append(i)
                    triplet.append(item_d1[j])
                    triplet.append(item_d2[j])
                    label.append(rating_d1[j])
                    label.append(rating_d2[j])
                    trainset1.append(triplet)
                    trainset2.append(label)
                for k in range(len(main_idx), len(aux_idx)) :
                    triplet = []
                    label = []
                    triplet.append(i)
                    triplet.append(random.choice(item))
                    triplet.append(item_d2[k])
                    label.append(0)
                    label.append(rating_d2[k])
                    trainset1.append(triplet)
                    trainset2.append(label)
    trainset = []
    trainset.append(trainset1)
    trainset.append(trainset2)

    return trainset

class Dataset:

    def __init__(self, name):
        self.domain = name
        if name != 'arts' and name != 'patio' and name != 'sports' and name != 'phone' and name != 'home' :
            print('mode name error!!')
            exit(1)
        
        self.clothing, self.clothing_train, self.arts, self.patio, self.home, self.phone, self.sports, self.user, self.item = dataload()
        
        if self.domain == 'arts' :
            self.trainset = get_trainset(self.clothing_train, self.arts, self.user, self.item, False)
            self.testset = get_trainset(self.clothing, self.arts, self.user, self.item, True)
        elif self.domain == 'patio' :
            self.trainset = get_trainset(self.clothing_train, self.patio, self.user, self.item, False)
            self.testset = get_trainset(self.clothing, self.patio, self.user, self.item, True)
        elif self.domain == 'home' :
            self.trainset = get_trainset(self.clothing_train, self.home, self.user, self.item, False)
            self.testset = get_trainset(self.clothing, self.home, self.user, self.item, True)
        elif self.domain == 'phone':
            self.trainset = get_trainset(self.clothing_train, self.phone, self.user, self.item, False)
            self.testset = get_trainset(self.clothing, self.phone, self.user, self.item, True)
        else : # sports
            self.trainset = get_trainset(self.clothing_train, self.sports, self.user, self.item, False)
            self.testset = get_trainset(self.clothing, self.sports, self.user, self.item, True)

    def get_number(self):
        # user
        num_user = self.user.max() + 1
        # item
        num_item_d1 = self.item.max() + 1
        if self.domain == 'arts' :
            num_item_d2 = max(self.arts.item.unique()) + 1
        elif self.domain == 'patio' :
            num_item_d2 = max(self.patio.item.unique()) + 1
        elif self.domain == 'home' :
            num_item_d2 = max(self.home.item.unique()) + 1
        elif self.domain == 'phone' :
            num_item_d2 = max(self.phone.item.unique()) + 1
        else : # sports
            num_item_d2 = max(self.sports.item.unique()) + 1
        return num_user, num_item_d1, num_item_d2

    def neg_sampling(self, num): # 필요 없을 것 같은데??
        item_dict = self.train_neg_dict
        user_list = []
        item_list = []
        #print(item_dict)
        for user in list(self.user_set):
            items = random.sample(item_dict[user], 20)
            item_list += items
            user_list += [user] * len(items)
        #print(user_list)
        #print(item_list)
        result = np.transpose(np.array([user_list, item_list]))
        return random.sample(result.tolist(), num)
    
    def get_data(self):
        return self.trainset, self.testset