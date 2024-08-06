import torch
import torch.optim as optim
from torch.autograd import Variable
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import  global_mean_pool
from random import shuffle
import random
import argparse
import numpy as np

from model import GNN
from utils import gen_ran_output,load_data4pretrain, mkdir, graph_views

class GraphCL(torch.nn.Module):

    def __init__(self, gnn, hid_dim=16, pool=None):
        super(GraphCL, self).__init__()
        self.gnn = gnn
        if pool is None:
            self.pool = global_mean_pool
        else:
            self.pool = pool

        self.projection_head = torch.nn.Sequential(torch.nn.Linear(hid_dim, hid_dim),
                                                   torch.nn.ReLU(inplace=True),
                                                   torch.nn.Linear(hid_dim, hid_dim))

    def forward_cl(self, x, edge_index, batch):
        x = self.pool(self.gnn(x, edge_index), batch.long())
        x = self.projection_head(x)
        return x

    def loss_cl(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / ((sim_matrix.sum(dim=1) - pos_sim) + 1e-4)
        loss = - torch.log(loss).mean() + 10
        return loss


class PreTrain(torch.nn.Module):
    def __init__(self, pretext="GraphCL", gnn_type='GCN',
                 input_dim=None, hid_dim=None, gln=2):
        super(PreTrain, self).__init__()
        self.pretext = pretext
        self.gnn_type=gnn_type
        self.gln = gln
        self.gnn = GNN(input_dim=input_dim, hid_dim=hid_dim, out_dim=hid_dim, gcn_layer_num=gln, gnn_type=gnn_type)

        if pretext in ['GraphCL', 'SimGRACE']:
            self.model = GraphCL(self.gnn, hid_dim=hid_dim, pool=None)
        else:
            raise ValueError("pretext should be GraphCL, SimGRACE")

    def get_loader(self, graph_list, batch_size,
                   aug1=None, aug2=None, aug_ratio=None, pretext="GraphCL"):

        if len(graph_list) % batch_size == 1:
            raise KeyError(
                "batch_size {} makes the last batch only contain 1 graph, \n which will trigger a zero bug in GraphCL!")

        if pretext == 'GraphCL':
            shuffle(graph_list)
            if aug1 is None:
                aug1 = random.sample(['dropN', 'permE', 'maskN'], k=1)
            if aug2 is None:
                aug2 = random.sample(['dropN', 'permE', 'maskN'], k=1)
            if aug_ratio is None:
                aug_ratio = random.randint(1, 3) * 1.0 / 10  # 0.1,0.2,0.3

            print("===graph views: {} and {} with aug_ratio: {}".format(aug1, aug2, aug_ratio))

            view_list_1 = []
            view_list_2 = []
            for g in graph_list:
                view_g = graph_views(data=g, aug=aug1, aug_ratio=aug_ratio)
                view_g = Data(x=view_g.x, edge_index=view_g.edge_index)
                view_list_1.append(view_g)
                view_g = graph_views(data=g, aug=aug2, aug_ratio=aug_ratio)
                view_g = Data(x=view_g.x, edge_index=view_g.edge_index)
                view_list_2.append(view_g)

            loader1 = DataLoader(view_list_1, batch_size=batch_size, shuffle=False,
                                 num_workers=1)  # you must set shuffle=False !
            loader2 = DataLoader(view_list_2, batch_size=batch_size, shuffle=False,
                                 num_workers=1)  # you must set shuffle=False !

            return loader1, loader2
        elif pretext == 'SimGRACE':
            loader = DataLoader(graph_list, batch_size=batch_size, shuffle=False, num_workers=1)
            return loader, None  # if pretext==SimGRACE, loader2 is None
        else:
            raise ValueError("pretext should be GraphCL, SimGRACE")

    def train_simgrace(self, model, loader, optimizer):
        model.train()
        train_loss_accum = 0
        total_step = 0
        for step, data in enumerate(loader):
            optimizer.zero_grad()
            data = data.to(device)
            x2 = gen_ran_output(data, model) 
            x1 = model.forward_cl(data.x, data.edge_index, data.batch)
            x2 = Variable(x2.detach().data.to(device), requires_grad=False)
            loss = model.loss_cl(x1, x2)
            loss.backward()
            optimizer.step()
            train_loss_accum += float(loss.detach().cpu().item())
            total_step = total_step + 1

        return train_loss_accum / total_step

    def train_graphcl(self, model, loader1, loader2, optimizer):
        model.train()
        train_loss_accum = 0
        total_step = 0
        for step, batch in enumerate(zip(loader1, loader2)):
            batch1, batch2 = batch
            optimizer.zero_grad()
            x1 = model.forward_cl(batch1.x.to(device), batch1.edge_index.to(device), batch1.batch.to(device))
            x2 = model.forward_cl(batch2.x.to(device), batch2.edge_index.to(device), batch2.batch.to(device))
            loss = model.loss_cl(x1, x2)

            loss.backward()
            optimizer.step()

            train_loss_accum += float(loss.detach().cpu().item())
            total_step = total_step + 1

        return train_loss_accum / float(total_step)

    def train(self, dataname, graph_list, batch_size=10, aug1='dropN', aug2="permE", aug_ratio=None, lr=0.01,
              decay=0.0001, epochs=100):

        loader1, loader2 = self.get_loader(graph_list, batch_size, aug1=aug1, aug2=aug2,
                                           pretext=self.pretext)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=decay)

        train_loss_min = 1000000
        for epoch in range(1, epochs + 1):  # 1..100
            if self.pretext == 'GraphCL':
                train_loss = self.train_graphcl(self.model, loader1, loader2, optimizer)
                # print('loss',train_loss)
            elif self.pretext == 'SimGRACE':
                train_loss = self.train_simgrace(self.model, loader1, optimizer)
            else:
                raise ValueError("pretext should be GraphCL, SimGRACE")

            print("***epoch: {}/{} | train_loss: {:.8}".format(epoch, epochs, train_loss))

            if train_loss_min > train_loss:
                train_loss_min = train_loss
                torch.save(self.model.gnn.state_dict(),
                           "./pre_trained_gnn/{}.{}.{}.{}.pth".format(dataname, self.pretext, self.gnn_type, self.gln))
                # do not use '../pre_trained_gnn/' because hope there should be two folders: (1) '../pre_trained_gnn/'  and (2) './pre_trained_gnn/'
                # only selected pre-trained models will be moved into (1) so that we can keep reproduction
                print("+++model saved ! {}.{}.{}.{}.pth".format(dataname, self.pretext, self.gnn_type, self.gln))



if __name__ == '__main__':
            # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0.0001,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=3,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--gnn_type', type=str, default="GCN")
    parser.add_argument('--dataset', type=str, default = 'ogbg_molhiv', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--num_part', type=int, default = 500, help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--eval_train', type=int, default = 1, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    parser.add_argument('--pretext', type=str, default = 'GraphCL', help='learning methods for model pretrain')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 256)')
    args = parser.parse_args()
    
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)
    
    print(args)

    mkdir('./pre_trained_gnn/')

    pretext = args.pretext
    dataname, num_parts = args.dataset, args.num_part

    print('Use {} for {} based model pretrain'.format(dataname, args.gnn_type))
    graph_list, input_dim, hid_dim = load_data4pretrain(dataname, num_parts, 'graph_level')
    print(graph_list[0])
    print(input_dim, hid_dim)
    pt = PreTrain(pretext, args.gnn_type, input_dim, hid_dim, gln=args.num_layer)
    pt.model.to(device) 
    pt.train(dataname, graph_list, batch_size=args.batch_size, aug1='dropN', aug2="permE", aug_ratio=None,lr=args.lr, decay=args.decay,epochs=args.epochs)
