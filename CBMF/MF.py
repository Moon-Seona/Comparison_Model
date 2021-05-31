import torch
import torch.nn as nn

class MF(nn.Module):
    def __init__(self, usernum, itemnum, factor_num):
        super(MF, self).__init__()

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

        predict = user * item + average + user_bias + item_bias

        return predict