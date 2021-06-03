import torch
import torch.nn as nn
from kmeans_pytorch import kmeans

class Clustering(nn.Module):
    def __init__(self, clustering_num):
        super(Clustering, self).__init__()

        self.clustering_num = clustering_num

    def forward(self):
        cluster = self.clustering_num

class MF(nn.Module):
    def __init__(self, usernum, itemnum, factor_num):
        super(MF, self).__init__()

        self.embed_user = nn.Embedding(usernum, factor_num)
        self.embed_item = nn.Embedding(itemnum, factor_num)

    def forward(self, user, item, average):

        user = self.embed_user(user)
        item = self.embed_item(item)

        predict = (user * item).sum(axis=1) + average

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

        predict = (user * item).sum(axis=1) + average + user_bias + item_bias

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
        user_bias = self.user_bias[user]
        item_bias = self.item_bias[item]

        user = self.embed_user(user)
        item = self.embed_item(item)

        cluster_idx, _ = kmeans(user, num_clusters=10, device=torch.device('cuda:0')) # 여기서 해도 되나요...?
        cluster_pred = 0
        # 0.3 = alpha, 0.7 = 1-alpha
        predict = (user * item).sum(axis=1)*0.7 + cluster_pred*0.3  + average + user_bias + item_bias

        return predict, user, item