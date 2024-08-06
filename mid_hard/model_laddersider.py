import torch
from torch import Tensor
import torch.nn.functional as F
import pickle as pk
from torch_geometric.utils import to_undirected
from torch_geometric.nn.dense.linear import Linear
from utils import act
import torch.nn as nn
import torch
from model import *
from utils import load_state_dict_from_path
import copy

    
class LadderSide_GNN(GNN):
    def __init__(self, input_dim, hid_dim=None, out_dim=None, gcn_layer_num=2, gnn_type='GAT', down_dim=16):
        super().__init__(input_dim, hid_dim, out_dim, gcn_layer_num, gnn_type)

        self.side_MLP = nn.ModuleList()
        self.side_norms = nn.ModuleList()
        self.alphas = nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for i in range(self.gcn_layer_num)])
        
        self.first_downsample = nn.Linear(input_dim, down_dim)
        self.side_downsamples = nn.ModuleList(
            [nn.Linear(hid_dim, down_dim) 
            for i in range(self.gcn_layer_num-1)]
        )
        self.side_downsamples.append(nn.Linear(out_dim, down_dim))

        for i in range(self.gcn_layer_num - 1):
            self.side_MLP.append(nn.Linear(down_dim, down_dim))
            self.side_norms.append(nn.BatchNorm1d(down_dim))
        self.side_MLP.append(nn.Linear(down_dim, out_dim))
        self.side_norms.append(nn.BatchNorm1d(out_dim))

        self.freeze_original_params(self.gcn_layer_num)

    def forward(self, x, edge_index):
        side_x = self.first_downsample(F.dropout(x, training = self.training))
        h_list = [x]
        side_list = [side_x] 
        for layer in range(self.gcn_layer_num):
            h = self.conv_layers[layer](h_list[layer], edge_index)
            h = self.batch_norms[layer](h)
            if layer == self.gcn_layer_num - 1:
                #remove relu for the last layer
                h = F.dropout(h, training = self.training)
            else:
                h = F.dropout(act(h), training = self.training)
            
            ## side_forward
            gate = torch.sigmoid(self.alphas[layer])
            # gate = self.alphas[layer]
            side_h = gate * side_list[layer] + (1 - gate) * self.side_downsamples[layer](h)
            side_h = self.side_MLP[layer](side_h)
            side_h = self.side_norms[layer](side_h)

            if layer == self.gcn_layer_num - 1:
                side_h = F.dropout(side_h, training = self.training)
                # print("side last",side_h)
            else:
                side_h = F.dropout(act(side_h), training = self.training)

            # print('layer',layer, h, side_h)
            h_list.append(h)
            side_list.append(side_h)
        node_emb = side_list[-1]
        return node_emb

    def freeze_original_params(self, num_layer):
        for param in self.parameters():
            param.requires_grad = False
        # print('layer',self.gnns[0])
        for param in self.first_downsample.parameters():
            param.requires_grad = True
        for i in range(num_layer):
            self.alphas[i].requires_grad = True
            for param in self.side_MLP[i].parameters():
                param.requires_grad = True
            for param in self.side_norms[i].parameters():
                param.requires_grad = True
            for param in self.side_downsamples[i].parameters():
                param.requires_grad = True

    def unfreeze_original_params(self):
        for param in self.parameters():
            param.requires_grad = True


class GNN_nodepred_ladderside(GNN_nodepred):
    def __init__(self, input_dim, num_class, hid_dim=None, out_dim=None, gcn_layer_num=2, gnn_type='GAT', down_dim=16):
        super().__init__(input_dim, num_class, hid_dim, out_dim, gcn_layer_num, gnn_type)

        self.gnn = LadderSide_GNN(input_dim, hid_dim, out_dim, gcn_layer_num, gnn_type, down_dim)

    def from_pretrained(self, model_file, device):
        model, _ = load_state_dict_from_path(self.gnn, model_file, device)
        return model


class LadderSide_GNN_merge(GNN):
    def __init__(self, input_dim, hid_dim=None, out_dim=None, gcn_layer_num=2, gnn_type='GAT', down_dim=16):
        super().__init__(input_dim, hid_dim, out_dim, gcn_layer_num, gnn_type)
        
        self.conv_layers2 = copy.deepcopy(self.conv_layers)
        self.batch_norms2 = copy.deepcopy(self.batch_norms)
        self.alpha_merge = nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for i in range(gcn_layer_num)])
        self.alpha_merge_bth = nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for i in range(gcn_layer_num)])

        self.side_MLP = nn.ModuleList()
        self.side_norms = nn.ModuleList()
        self.alphas = nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for i in range(self.gcn_layer_num)])
        
        self.first_downsample = nn.Linear(input_dim, down_dim)
        self.side_downsamples = nn.ModuleList(
            [nn.Linear(hid_dim, down_dim) 
            for i in range(self.gcn_layer_num-1)]
        )
        self.side_downsamples.append(nn.Linear(out_dim, down_dim))

        for i in range(self.gcn_layer_num - 1):
            self.side_MLP.append(nn.Linear(down_dim, down_dim))
            self.side_norms.append(nn.BatchNorm1d(down_dim))
        self.side_MLP.append(nn.Linear(down_dim, out_dim))
        self.side_norms.append(nn.BatchNorm1d(out_dim))

        self.freeze_original_params(self.gcn_layer_num)

    def forward(self, x, edge_index):
        side_x = self.first_downsample(F.dropout(x, training = self.training))
        h_list = [x]
        side_list = [side_x] 
        for layer in range(self.gcn_layer_num):
            h = self.conv_layers[layer](h_list[layer], edge_index)
            h2 = self.conv_layers2[layer](h_list[layer], edge_index)
            # gate_init = torch.sigmoid(self.alpha_merge[layer])
            gate_init = self.alpha_merge[layer]
            h = gate_init*h + (1-gate_init)*h2
                                    
            h = self.batch_norms[layer](h)
            h2 = self.batch_norms2[layer](h)
            # gate_bth = torch.sigmoid(self.alpha_merge_bth[layer])
            gate_bth = self.alpha_merge_bth[layer]
            h = gate_bth*h + (1-gate_bth)*h2
            
            if layer == self.gcn_layer_num - 1:
                #remove relu for the last layer
                h = F.dropout(h, training = self.training)
            else:
                h = F.dropout(act(h), training = self.training)
            
            ## side_forward
            # gate = torch.sigmoid(self.alphas[layer])
            gate = self.alphas[layer]
            side_h = gate * side_list[layer] + (1 - gate) * self.side_downsamples[layer](h)
            side_h = self.side_MLP[layer](side_h)
            side_h = self.side_norms[layer](side_h)

            if layer == self.gcn_layer_num - 1:
                side_h = F.dropout(side_h, training = self.training)
                # print("side last",side_h)
            else:
                side_h = F.dropout(act(side_h), training = self.training)

            # print('layer',layer, h, side_h)
            h_list.append(h)
            side_list.append(side_h)
        node_emb = side_list[-1]
        return node_emb

    def freeze_original_params(self, num_layer):
        for param in self.parameters():
            param.requires_grad = False
        
        for param in self.alpha_merge:
            param.requires_grad = True  
        for param in self.alpha_merge_bth:
            param.requires_grad = True   

        for param in self.first_downsample.parameters():
            param.requires_grad = True
        for i in range(num_layer):
            self.alphas[i].requires_grad = True
            for param in self.side_MLP[i].parameters():
                param.requires_grad = True
            for param in self.side_norms[i].parameters():
                param.requires_grad = True
            for param in self.side_downsamples[i].parameters():
                param.requires_grad = True

    def unfreeze_original_params(self):
        for param in self.parameters():
            param.requires_grad = True        

class GNN_nodepred_merg_ladderside(GNN_nodepred):
    def __init__(self, input_dim, num_class, hid_dim=None, out_dim=None, gcn_layer_num=2, gnn_type='GAT', down_dim=16):
        super().__init__(input_dim, num_class, hid_dim, out_dim, gcn_layer_num, gnn_type)

        self.gnn = LadderSide_GNN_merge(input_dim, hid_dim, out_dim, gcn_layer_num, gnn_type, down_dim)

    def from_pretrained(self, model_file, device):
        model, _ = load_state_dict_from_path(self.gnn, model_file, device)
        return model

if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    device_idx = 1
    device = torch.device("cuda:" + str(device_idx)) if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    output_model_file = './pre_trained_gnn/ogbn_arxiv.GraphCL.GCN.3.pth'
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
    gnn_type = 'GIN'
    input_dim = out_dim = x.shape[1]
    model = GNN_nodepred_merg_ladderside(input_dim, num_class, hid_dim=100, out_dim=out_dim, gcn_layer_num=3, gnn_type='GCN', down_dim=16)
    print(model)
    for i in model.named_parameters():
        print(i)
    model.from_pretrained(output_model_file, device)
    for i in model.named_parameters():
        print(i)
    res = model(x, edge_index)
    print(res)
    print(res.shape)