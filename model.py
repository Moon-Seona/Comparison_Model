import os
import torch
import torch.nn as nn
from reader import Dataset
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class CoNet(nn.Module):

    def __init__(self, config, print_summary=False):
        super(CoNet, self).__init__()
        self.print_summary = print_summary
        self.dataset = Dataset(config['aux_domain'])
        self.trainset, self.testset = self.dataset.get_data()

        self.edim = config["edim"]
        self.lr = config["lr"]
        self.reg = config["reg"]
        self.batch_size = config["batch_size"]
        self.cross_layer = config["cross_layer"]
        self.std = config["std"]
        self.epoch = config["epoch"]
        self.initialise_nn()

        self.num_user, self.num_item_d1, self.num_item_d2 = self.dataset.get_number()
 
        self.U = nn.Embedding(self.num_user, self.edim)
        self.V_d1 = nn.Embedding(self.num_item_d1, self.edim)
        self.V_d2 = nn.Embedding(self.num_item_d2, self.edim)
        self.sigmoid = nn.Sigmoid()
    
    def create_nn(self, layers):
        weights = {}
        biases = {}
        for l in range(len(self.layers) - 1):
            weights[l] = torch.normal(mean=0, std=self.std, size=(layers[l], layers[l+1]), requires_grad=True, device=device)
            biases[l] = torch.normal(mean=0, std=self.std, size=(layers[l+1],), requires_grad=True, device=device)
        return weights, biases
    
    def initialise_nn(self):
        edim = 2*self.edim
        i=0
        self.layers = [edim]
        while edim>8:
            i+=1
            edim/=2
            self.layers.append(int(edim))
        #print(self.layers)
        assert (self.cross_layer <= i)

        #weights and biases: apps
        weights_d1, biases_d1 = self.create_nn(self.layers)
        self.weights_d1 = nn.ParameterList([nn.Parameter(weights_d1[i]) for i in range(len(self.layers) - 1)])
        self.biases_d1 = nn.ParameterList([nn.Parameter(biases_d1[i]) for i in range(len(self.layers) - 1)])
        self.W_d1 = nn.Parameter(torch.normal(mean=0, std=self.std, size=(self.layers[-1], 1), requires_grad=True, device=device))
        self.b_d1 = nn.Parameter(torch.normal(mean=0, std=self.std, size=(1,), requires_grad=True, device=device))

        #weights and biases: news
        weights_d2, biases_d2 = self.create_nn(self.layers)
        self.weights_d2 = nn.ParameterList([nn.Parameter(weights_d2[i]) for i in range(len(self.layers) - 1)])
        self.biases_d2 = nn.ParameterList([nn.Parameter(biases_d2[i]) for i in range(len(self.layers) - 1)])
        self.W_d2 = nn.Parameter(torch.normal(mean=0, std=self.std, size=(self.layers[-1], 1), requires_grad=True, device=device))
        self.b_d2 = nn.Parameter(torch.normal(mean=0, std=self.std, size=(1,), requires_grad=True, device=device))

        #weights: shared layers
        weights_shared = {}
        for l in range(self.cross_layer):
            weights_shared[l] = torch.normal(mean=0, std=self.std, size=(self.layers[l], self.layers[l+1]), requires_grad=True, device=device)
        self.weights_shared = [weights_shared[i] for i in range(self.cross_layer)]
        
    def forward(self, user, item_d1, item_d2):
        user_emb = self.U(torch.LongTensor(user)).to(device=device) # P
        item_emb_d1 = self.V_d1(torch.LongTensor(item_d1)).to(device=device) # Q
        item_emb_d2 = self.V_d2(torch.LongTensor(item_d2)).to(device=device) # Q

        cur_d1 = torch.cat((user_emb, item_emb_d1), 1)
        cur_d2 = torch.cat((user_emb, item_emb_d2), 1)
        pre_d1 = cur_d1
        pre_d2 = cur_d2
        for l in range(len(self.layers) - 1):
            #print(cur_d2.shape)
            #print(self.weights_d2[l].shape)
            cur_d1 = torch.add(torch.matmul(cur_d1, self.weights_d1[l]), self.biases_d1[l])
            cur_d2 = torch.add(torch.matmul(cur_d2, self.weights_d2[l]), self.biases_d2[l])

            if (l < self.cross_layer):
                #print("cur_d1.shape", cur_d1.shape)
                #print("cur_d2.shape", cur_d2.shape)
                #print("w_.shape", self.weights_shared[l].shape)
                cur_d1 = torch.matmul(pre_d2, self.weights_shared[l]) # L1 norm 추가해서 SCoNet 완성
                cur_d2 = torch.matmul(pre_d1, self.weights_shared[l])
            cur_d1 = nn.functional.relu(cur_d1)
            cur_d2 = nn.functional.relu(cur_d2)
            pre_d1 = cur_d1
            pre_d2 = cur_d2

        z_d1 = torch.matmul(cur_d1, self.W_d1) + self.b_d1
        z_d2 = torch.matmul(cur_d2, self.W_d2) + self.b_d2
        #print("z_apps", z_apps.shape)
        #print("z_news", z_news.shape)

        # sigmoid 제거!!!
        return z_d1, z_d2
    
    def fit(self):
        
        params = [{"params": self.parameters(), "lr":self.lr},
                  {"params": self.weights_shared, "lr": self.lr, "weight_decay":self.reg}]
        optimizer = torch.optim.Adam(params)
        criterion = nn.MSELoss()
        # 여기서 self.tainset, self.testset 불러와서 바로 훈련 가능하도록 만들기
        [train_data, labels] = self.trainset
        train_data = torch.tensor(train_data)
        labels = torch.tensor(labels, device=device)
        # print(train_data.shape, labels.shape)
        
        [test_data, test_labels] = self.testset
        test_data = torch.tensor(test_data)
        test_labels = torch.tensor(test_labels, device=device)
        
        labels_d1, labels_d2 = labels[:,0], labels[:,1]
        user, item_d1, item_d2 = train_data[:,0], train_data[:,1], train_data[:,2]
        
        test_labels_d1, test_labels_d2 = test_labels[:,0], test_labels[:,1]
        test_user, test_item_d1, test_item_d2 = test_data[:,0], test_data[:,1], test_data[:,2]
        
        for epoch in tqdm(range(self.epoch)):
            total_loss = 0
            permutation = torch.randperm(user.shape[0]) #  Returns a random permutation of integers from 0 to n - 1. => dataloader shuffle 과 같은 역할
            max_idx = int((len(permutation) // (self.batch_size/2) -1) * (self.batch_size/2))
            #range(0, max_idx, self.batch_size)
            for batch in range(0, max_idx, self.batch_size):
                #print(batch)
                optimizer.zero_grad()
                idx = permutation[batch : batch + self.batch_size]  
                pred_d1, pred_d2 = self.forward(user[idx], item_d1[idx], item_d2[idx])
                loss_d1 = criterion(labels_d1[idx].float(), torch.squeeze(pred_d1))
                loss_d2 = criterion(labels_d2[idx].float(), torch.squeeze(pred_d2))
                loss = loss_d1 + loss_d2
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
#                 if batch % (self.batch_size*10) == 0 :
#                     print(f'epoch: {epoch}, batch: {batch}, loss: {round(loss.item(), 4)}')
            
            print("epoch: {}, \t loss: {:.4f}".format(epoch, round(total_loss/self.batch_size, 4)))
            # testset 확인
            with torch.no_grad() :
                pred, _ = self.forward(test_user, test_item_d1, test_item_d2)
                test_loss = criterion(test_labels_d1.float(), torch.squeeze(pred))
                diff = pred - test_labels_d1
                diff = diff.cpu().detach()
                test_mae = np.absolute(diff).mean()
                test_rmse = np.sqrt((diff ** 2).mean())

                print("Test MAE: {:4f}, Test RMSE: {:4f}, Test loss: {:4f}".format(test_mae.item(), test_rmse.item(), test_loss.item()))
                #print('Test_diff :', diff)
                #print('test loss: {:4f}'.format(test_loss.item()))
                
                