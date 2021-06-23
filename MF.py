import torch
import torch.nn as nn
from kmeans_pytorch import kmeans
from util import *

# class Clustering(nn.Module):
#     def __init__(self, clustering_num):
#         super(Clustering, self).__init__()
#
#         self.clustering_num = clustering_num
#
#     def forward(self, ):
#         cluster = self.clustering_num

class MF(nn.Module):
    def __init__(self, usernum, itemnum, factor_num):
        super(MF, self).__init__()

        self.embed_user = nn.Embedding(usernum, factor_num)
        self.embed_item = nn.Embedding(itemnum, factor_num)

    def forward(self, user, item, average):

        user = self.embed_user(user)
        item = self.embed_item(item)

        #predict = (user * item).sum(axis=1) + average
        predict = torch.matmul(user, item.t()) + average

        return predict, user, item

class biasedMF(nn.Module):
    def __init__(self, usernum, itemnum, factor_num):
        super(biasedMF, self).__init__()

        self.embed_user = nn.Embedding(usernum, factor_num)
        self.embed_item = nn.Embedding(itemnum, factor_num)
        self.user_bias = nn.Parameter(torch.zeros((usernum,)))
        self.item_bias = nn.Parameter(torch.zeros((itemnum,)))

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

    def forward(self, user, item, average):
        user_bias = self.user_bias[user]
        item_bias = self.item_bias[item]

        user = self.embed_user(user)
        item = self.embed_item(item)
        #print(user.shape, item.shape)
        predict = torch.matmul(user, item.t()) + average + torch.unsqueeze(user_bias, 1) + torch.unsqueeze(item_bias, 0)

        return predict, user, item

class CBMF(nn.Module):
    def __init__(self, usernum, itemnum, factor_num):
        super(CBMF, self).__init__()

        self.embed_user = nn.Embedding(usernum, factor_num)
        self.embed_item = nn.Embedding(itemnum, factor_num)
        self.user_bias = nn.Parameter(torch.zeros((usernum,)))
        self.item_bias = nn.Parameter(torch.zeros((itemnum,)))

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

    def forward(self, user, item, average):
        #user_bias = self.user_bias[user]
        #item_bias = self.item_bias[item]
        user = self.embed_user(user)
        item = self.embed_item(item)
        # print(user.shape, item.shape)
        # clustering
        #rating2 = cluster_rating(user, item, rating)
        predict = torch.matmul(user, item.t()) + average

        return predict, user, item

class biasedCBMF(nn.Module):
    def __init__(self, usernum, itemnum, factor_num):
        super(biasedCBMF, self).__init__()

        self.embed_user = nn.Embedding(usernum, factor_num)
        self.embed_item = nn.Embedding(itemnum, factor_num)
        self.user_bias = nn.Parameter(torch.zeros((usernum,)))
        self.item_bias = nn.Parameter(torch.zeros((itemnum,)))

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

    def forward(self, user, item, average):
        user_bias = self.user_bias[user]
        item_bias = self.item_bias[item]
        user = self.embed_user(user)
        item = self.embed_item(item)
        # print(user.shape, item.shape)
        # clustering
        #rating2 = cluster_rating(user, item, rating)
        predict = torch.matmul(user, item.t()) + average + torch.unsqueeze(user_bias, 1) + torch.unsqueeze(item_bias, 0)

        return predict, user, item