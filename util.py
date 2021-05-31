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

    clothing = clothing[clothing_train.rating == 0]  # only testset

    return clothing, clothing_train, arts, patio, home, phone, sports, user, item