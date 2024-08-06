import torch
from torch import Tensor
import torch.nn.functional as F
import pickle as pk
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, to_undirected
from torch_geometric.nn.dense.linear import Linear
from utils import act

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from random import shuffle
import random

from utils import gen_ran_output,load_data4pretrain, mkdir, graph_views

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


class GNN(torch.nn.Module):
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

    def forward(self, x, edge_index):
        h = x
        for layer in range(self.gcn_layer_num):
            h = self.conv_layers[layer](h, edge_index)
            h = self.batch_norms[layer](h)
            if layer == self.gcn_layer_num - 1:
                #remove relu for the last layer
                h = F.dropout(h, training = self.training)
            else:
                h = F.dropout(act(h), training = self.training)
        
        node_emb = h
        return node_emb

    
class GNN_nodepred(torch.nn.Module):
    def __init__(self, input_dim, num_class, hid_dim=None, out_dim=None, gcn_layer_num=2, gnn_type='GAT'):
        super().__init__()

        self.gnn = GNN(input_dim, hid_dim, out_dim, gcn_layer_num, gnn_type)
        # self.graph_pred_linear = torch.nn.Sequential(torch.nn.Linear(hid_dim, num_class), torch.nn.Softmax(dim=1))
        self.graph_pred_linear = torch.nn.Linear(hid_dim, num_class)

    def forward(self, x, edge_index):
        node_emb = self.gnn(x, edge_index)
        node_pred = self.graph_pred_linear(node_emb)
        return node_pred

    def from_pretrained(self, model_file, device):
        self.gnn.load_state_dict(torch.load(model_file, map_location=device))


class MLP_nodepred(torch.nn.Module):
    def __init__(self, input_dim, num_class, hid_dim=16, out_dim=None, gcn_layer_num=2):
        super().__init__()
        self.num_layer = gcn_layer_num

        if self.num_layer < 2:
            raise ValueError("Number of side net layers must be greater than 1.")

        ###List of MLPs
        self.mlp = torch.nn.ModuleList()
        if self.num_layer==2:
            self.mlp.append(Linear(input_dim, hid_dim))
            self.mlp.append(Linear(hid_dim, out_dim))
        elif self.num_layer>2:
            self.mlp.append(Linear(input_dim, hid_dim))
            for layer in range(1, self.num_layer-1):
                self.mlp.append(Linear(hid_dim, hid_dim))
            self.mlp.append(Linear(hid_dim, out_dim))
        self.graph_pred_linear = Linear(out_dim, num_class)

        ###List of batchnorms
        # self.batch_norm = torch.nn.BatchNorm1d(out_dim)
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.num_layer-1):
            self.batch_norms.append(torch.nn.BatchNorm1d(hid_dim))
        self.batch_norms.append(torch.nn.BatchNorm1d(out_dim))

    def forward(self, x, edge_index):
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.mlp[layer](h_list[layer])
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                h = F.dropout(h, training = self.training)
            else:
                h = F.dropout(F.relu(h), training = self.training)
            h_list.append(h)

        node_representation = h_list[-1]
        out = self.graph_pred_linear(node_representation)

        return node_representation


if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    device_idx = 1
    device = torch.device("cuda:" + str(device_idx)) if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    output_model_file = './pre_trained_gnn/CiteSeer.GraphCL.GCN.pth'
    dataname, num_parts = 'Cora', 200
    dataset = pk.load(open('./dataset/{}/feature_reduced.data'.format(dataname), 'br'))
    print('xxxx',dataset)
    data = dataset.data
    num_class = dataset.num_classes
    
    print("oooo", data)
    x = data.x.detach()
    edge_index = data.edge_index
    edge_index = to_undirected(edge_index)
    
    # model = SideMLP(input_dim=input_dim, hid_dim=16, out_dim=out_dim, gcn_layer_num=3)
    gnn_type = 'GCN'
    input_dim = out_dim = x.shape[1]
    model = GNN_nodepred(input_dim, num_class, 100, out_dim, gcn_layer_num=5, gnn_type=gnn_type)
    print(model)
    # for i in model.named_parameters():
    #     print(i)
    model.from_pretrained(output_model_file, device)
    # for i in model.named_parameters():
    #     print(i)
    res = model(x, edge_index)
    print(res)
    print(res.shape)