import sys
import os
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch.nn import functional as F
import pickle as pk
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, to_undirected
from torch_geometric.nn.dense.linear import Linear
from utils import act

import copy
import random
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from torch_geometric.utils import negative_sampling
from tqdm import tqdm
from model_laddersider_edge import LadderSide_GNN, LadderSide_GNN_merge
from utils import load_state_dict_from_path


class GINConv(MessagePassing):

    def __init__(self, in_dim, out_dim, aggr = "add"):
        super(GINConv, self).__init__(aggr=aggr)
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, 2*in_dim), torch.nn.ReLU(), torch.nn.Linear(2*in_dim, out_dim))
        self.aggr = aggr

    def forward(self, x, edge_index):
        #add self loops in the edge space
        edge_index,_ = add_self_loops(edge_index, num_nodes = x.size(0))
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return self.mlp(aggr_out)
    

class GNN_LP(torch.nn.Module):
    def __init__(self, input_dim, hid_dim=None, out_dim=None, gcn_layer_num=2, gnn_type='GAT'):
        super().__init__()
        
        if gnn_type == 'GCN':
            GraphConv = GCNConv
        elif gnn_type == 'GAT':
            GraphConv = GATConv
        elif gnn_type == 'GIN':
            GraphConv = GINConv
        else:
            raise KeyError('gnn_type can be only GAT, GCN and GIN')
        self.gcn_layer_num = gcn_layer_num
        self.gnn_type = gnn_type
        if hid_dim is None:
            hid_dim = int(0.618 * input_dim)  # "golden cut"
        if out_dim is None:
            out_dim = hid_dim

        if gcn_layer_num < 2:
            raise ValueError('GNN layer_num should >=2 but you set {}'.format(gcn_layer_num))
        elif gcn_layer_num == 2:
            self.conv_layers = torch.nn.ModuleList([GraphConv(input_dim, hid_dim), GraphConv(hid_dim, out_dim)])
        else:
            layers = [GraphConv(input_dim, hid_dim)]
            for i in range(gcn_layer_num - 2):
                layers.append(GraphConv(hid_dim, hid_dim))
            layers.append(GraphConv(hid_dim, out_dim))
            self.conv_layers = torch.nn.ModuleList(layers)
        
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(gcn_layer_num-1):
            self.batch_norms.append(torch.nn.BatchNorm1d(hid_dim))
        self.batch_norms.append(torch.nn.BatchNorm1d(out_dim))

    def encode(self, x, edge_index):
        h = x
        for layer in range(self.gcn_layer_num):
            h = self.conv_layers[layer](h, edge_index)
            h = self.batch_norms[layer](h)
            if layer == self.gcn_layer_num - 1:
                #remove relu for the last layer
                h = F.dropout(h, training = self.training)
            else:
                h = F.dropout(act(h), training = self.training)
        return h
    
    def decode(self, z,edge_label_index):
        # 所有节点的表示向量
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        r = (src * dst).sum(dim=-1)
        return r

    def forward(self, x, edge_index, edge_label_index):
        h = self.encode(x, edge_index)
        return self.decode(h, edge_label_index)
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_metrics(out, edge_label):
    edge_label = edge_label.cpu().numpy()
    out = out.cpu().numpy()
    pred = (out > 0.5).astype(int)
    auc = roc_auc_score(edge_label, out)
    ap = average_precision_score(edge_label, out)
    f1 = f1_score(edge_label, pred)
    return auc, f1, ap
    

def negative_sample():
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes, 
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')
    edge_label_index = torch.cat([train_data.edge_label_index, neg_edge_index], dim=-1)
    edge_label = torch.cat([train_data.edge_label,train_data.edge_label.new_zeros(neg_edge_index.size(1))], dim=-1)
    return edge_label, edge_label_index


def test(model, data):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
        model.train()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())


def train(model, train_data, val_data, test_data):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001) #0.005
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    min_epochs = 10
    best_model = None
    best_val_auc = 0
    final_test_auc = 0

    model.train()
    for epoch in range(1000):
        optimizer.zero_grad()
        edge_label, edge_label_index = negative_sample()
        out = model(train_data.x, train_data.edge_index, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()
        val_auc = test(model, val_data)
        test_auc = test(model, test_data)
        if epoch+1 > min_epochs and val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item():.4f}, Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}')
    return final_test_auc


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=2,
                        help='number of GNN message passing layers (default: 2).')
    parser.add_argument('--hid_dim', type=int, default=100,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--gnn_type', type=str, default="GIN")
    parser.add_argument('--dataset', type=str, default = 'Flickr', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = './pre_trained_gnn/ogbg_molhiv.GraphCL.GIN.2.pth', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--eval_train', type=int, default = 1, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    parser.add_argument('--tuning_methods', type=str, default = 'finetune', help='tuning methods for transfer learning')
    parser.add_argument('--tuning', type=int, default = 1, help='tuning or not')
    args = parser.parse_args()

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)
    print(device)

    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.2, num_test=0.2, is_undirected=True, add_negative_train_samples=False, disjoint_train_ratio=0)
    ])

    
    dataname = args.dataset
    dataset = pk.load(open('./dataset/{}/feature_reduced.data'.format(dataname), 'br'))
    print(len(dataset))
    dataset = transform(dataset[0])
    train_data, val_data, test_data = dataset[0], dataset[1], dataset[2]

    print(train_data)
    print(val_data)
    print(test_data)
    
    input_dim = out_dim = 100

    # model = LadderSide_GNN(input_dim, hid_dim=args.hid_dim, out_dim=out_dim, gcn_layer_num=args.num_layer, gnn_type=args.gnn_type,down_dim=16).to(device)
    model = LadderSide_GNN_merge(input_dim, hid_dim=args.hid_dim, out_dim=out_dim, gcn_layer_num=args.num_layer, gnn_type=args.gnn_type,down_dim=16).to(device)

    model,_ = load_state_dict_from_path(model, args.input_model_file, device)
    for name, param in model.named_parameters():
        print(name, param.size())
    print('\n\n')

    # model = GNN_LP(input_dim, hid_dim=args.hid_dim, out_dim=out_dim, gcn_layer_num=args.num_layer, gnn_type=args.gnn_type).to(device)
    # model.load_state_dict(torch.load(args.input_model_file,map_location=device))
    # model_stat = torch.load(args.input_model_file,map_location=device)
    # for key in model_stat.keys():
    #     print(key,model_stat[key].size())

    print(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    
    test_auc = train(model, train_data, val_data, test_data)
    print('Final Test AUC:', test_auc)
