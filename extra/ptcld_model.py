from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn import EdgeConv, global_max_pool, radius_graph
import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
import os

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, MLP, DynamicEdgeConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set, knn
from torch_geometric.utils import add_self_loops
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor
from torch_geometric.nn.inits import reset

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


class ptcld_GNN(torch.nn.Module):
    def __init__(self, input_dim, hid_dim=None, out_dim=None, gcn_layer_num=2, gnn_type='GAT', k=20):
        super().__init__()
        self.k=k
        self.transfer = MLP([input_dim, hid_dim])
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
            self.conv_layers = torch.nn.ModuleList([GraphConv(hid_dim, hid_dim, aggr='max'), GraphConv(hid_dim, out_dim, aggr='max')])
        else:
            layers = [GraphConv(hid_dim, hid_dim, aggr='max')]
            for i in range(gcn_layer_num - 2):
                layers.append(GraphConv(hid_dim, hid_dim, aggr='max'))
            layers.append(GraphConv(hid_dim, out_dim, aggr='max'))
            self.conv_layers = torch.nn.ModuleList(layers)
        
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(gcn_layer_num-1):
            self.batch_norms.append(torch.nn.BatchNorm1d(hid_dim))
        self.batch_norms.append(torch.nn.BatchNorm1d(out_dim))

        self.reset_parameters()
    
    def reset_parameters(self):
        print('reset the transfer para')
        reset(self.transfer)

    def forward(self, x ,batch):
        if isinstance(x, torch.Tensor):
            x: PairTensor = (x, x)
        b: PairOptTensor = (None, None)
        if isinstance(batch, torch.Tensor):
            b = (batch, batch)

        edge_index = knn(x[0], x[1], self.k, b[0], b[1]).flip([0])
        h = self.transfer(x[0])
        # print(h.shape)
        for layer in range(self.gcn_layer_num):
            h = self.conv_layers[layer](h, edge_index)
            # print(layer, h.shape)
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.gcn_layer_num - 1:
                #remove relu for the last layer
                h = F.dropout(h, training = self.training)
            else:
                h = F.dropout(F.relu(h), training = self.training)
        
        node_emb = h
        return node_emb

class GNN_ptcldpred(torch.nn.Module):
    def __init__(self, input_dim, num_class, hid_dim=None, out_dim=None, gcn_layer_num=2, gnn_type='GAT', pool=None, k=20):
        super().__init__()

        self.gnn = ptcld_GNN(input_dim, hid_dim, out_dim, gcn_layer_num, gnn_type, k)
        if pool is None:
            self.pool = global_max_pool
        else:
            self.pool = pool
        # self.graph_pred_linear = torch.nn.Sequential(torch.nn.Linear(hid_dim, num_class), torch.nn.Softmax(dim=1))
        self.graph_pred_linear = torch.nn.Linear(out_dim, num_class)

    def forward(self, x, batch):
        graph_emb = self.pool(self.gnn(x, batch), batch.long())
        graph_pred = self.graph_pred_linear(graph_emb)
        return graph_pred

    def from_pretrained(self, model_file, device):
        self.gnn.load_state_dict(torch.load(model_file, map_location=device), strict=False)


class MLP_ptcldpred(torch.nn.Module):
    def __init__(self, input_dim, num_class, hid_dim=None, out_dim=None, gcn_layer_num=2, pool=None, k=20):
        super().__init__()
        self.k=k
        self.transfer = MLP([input_dim, hid_dim])

        self.gcn_layer_num = gcn_layer_num

        if hid_dim is None:
            hid_dim = int(0.618 * input_dim)  # "golden cut"
        if out_dim is None:
            out_dim = hid_dim

        if gcn_layer_num < 2:
            raise ValueError('GNN layer_num should >=2 but you set {}'.format(gcn_layer_num))
        elif gcn_layer_num == 2:
            self.conv_layers = torch.nn.ModuleList([Lin(hid_dim, hid_dim), Lin(hid_dim, out_dim)])
        else:
            layers = [Lin(hid_dim, hid_dim)]
            for i in range(gcn_layer_num - 2):
                layers.append(Lin(hid_dim, hid_dim))
            layers.append(Lin(hid_dim, out_dim))
            self.conv_layers = torch.nn.ModuleList(layers)
        
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(gcn_layer_num-1):
            self.batch_norms.append(torch.nn.BatchNorm1d(hid_dim))
        self.batch_norms.append(torch.nn.BatchNorm1d(out_dim))

        if pool is None:
            self.pool = global_max_pool
        else:
            self.pool = pool
        self.graph_pred_linear = torch.nn.Linear(out_dim, num_class)

        self.reset_parameters()
    
    def reset_parameters(self):
        print('reset the transfer para')
        reset(self.transfer)

    def forward(self, x, batch):
        h = self.transfer(x)
        # print(h.shape)
        for layer in range(self.gcn_layer_num):
            h = self.conv_layers[layer](h)
            # print(layer, h.shape)
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.gcn_layer_num - 1:
                #remove relu for the last layer
                h = F.dropout(h, training = self.training)
            else:
                h = F.dropout(F.relu(h), training = self.training)
        
        node_emb = h
        graph_emb = self.pool(node_emb, batch.long())
        graph_pred = self.graph_pred_linear(graph_emb)

        return graph_pred