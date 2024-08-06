import torch
from torch_geometric.utils import to_undirected
import torch.nn.functional as F
from model import *  
import copy
from utils import load_state_dict_from_path
import torch.nn as nn
import pickle as pk



class SideLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.input_size = in_dim
        self.latent_size = out_dim
        # down projection
        self.lin = nn.Linear(self.input_size, self.latent_size)
        torch.nn.init.xavier_uniform_(self.lin.weight.data)

    def forward(self, x):
        output = self.lin(x)
        return output


class SideMLP(torch.nn.Module):
    def __init__(self, input_dim, hid_dim=16, out_dim=None, gcn_layer_num=2):
        super().__init__()
        self.num_layer = gcn_layer_num

        if self.num_layer < 2:
            raise ValueError("Number of side net layers must be greater than 1.")

        ###List of MLPs
        self.mlp = torch.nn.ModuleList()
        if self.num_layer==2:
            self.mlp.append(SideLayer(input_dim, hid_dim))
            self.mlp.append(SideLayer(hid_dim, out_dim))
        elif self.num_layer>2:
            self.mlp.append(SideLayer(input_dim, hid_dim))
            for layer in range(1, self.num_layer-1):
                self.mlp.append(SideLayer(hid_dim, hid_dim))
            self.mlp.append(SideLayer(hid_dim, out_dim))


        #List of batchnorms
        # self.batch_norm = torch.nn.BatchNorm1d(out_dim)
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.num_layer-1):
            self.batch_norms.append(torch.nn.BatchNorm1d(hid_dim))
        self.batch_norms.append(torch.nn.BatchNorm1d(out_dim))

    def forward(self, x):
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.mlp[layer](h_list[layer])
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, training = self.training)
            else:
                h = F.dropout(F.relu(h), training = self.training)
            h_list.append(h)

        node_representation = h_list[-1]

        return node_representation


class GNN_nodepred_side(torch.nn.Module):
    def __init__(self, feat_dim, num_class, out_dim, num_layer=5, hid_dim=100, gnn_type='GIN', num_side_layer=3, side_hid_dim=16):
        super().__init__()

        self.gnn = GNN(feat_dim, hid_dim, out_dim, gcn_layer_num=num_layer, gnn_type=gnn_type)
        self.siders = SideMLP(feat_dim, side_hid_dim, out_dim, gcn_layer_num=num_side_layer)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        # self.graph_pred_linear = torch.nn.Sequential(torch.nn.Linear(hid_dim, num_class), torch.nn.Softmax(dim=1))
        self.graph_pred_linear = torch.nn.Linear(hid_dim, num_class)
        self.freeze_original_params()
    
    def forward(self, x, edge_index):
        node_representation = self.gnn(x, edge_index)
        side_representation = self.siders(x)
        # gate = torch.sigmoid(self.alpha)
        gate = self.alpha
        node_representation = (1-gate)*node_representation+gate*side_representation

        return self.graph_pred_linear(node_representation)
    
    def freeze_original_params(self):
        for param in self.parameters():
            param.requires_grad = False
        
        self.alpha.requires_grad = True
        for param in self.graph_pred_linear.parameters():
            param.requires_grad = True
        for param in self.siders.parameters():
            param.requires_grad = True

    def unfreeze_original_params(self):
        for param in self.parameters():
            param.requires_grad = True

    def from_pretrained(self, model_file, device):
        model, _ = load_state_dict_from_path(self.gnn, model_file, device)     
        return model


class GNN_merge(GNN):
    def __init__(self, input_dim, hid_dim=None, out_dim=None, gcn_layer_num=2, gnn_type='GAT'):
        super().__init__(input_dim, hid_dim, out_dim, gcn_layer_num, gnn_type)
        self.conv_layers2 = copy.deepcopy(self.conv_layers)
        self.batch_norms2 = copy.deepcopy(self.batch_norms)
        self.alpha_merge = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for i in range(gcn_layer_num)])
        self.alpha_merge_bth = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for i in range(gcn_layer_num)])

    def forward(self, x, edge_index):
        h_list = [x]
        for layer in range(self.gcn_layer_num):
            h = self.conv_layers[layer](h_list[layer], edge_index)
            h2 = self.conv_layers2[layer](h_list[layer], edge_index)
            
            # gate = torch.sigmoid(self.alpha_merge[layer])
            gate = self.alpha_merge[layer]
            h = gate*h + (1-gate)*h2

            h = self.batch_norms[layer](h)
            h2 = self.batch_norms2[layer](h)
            # gate_bth = torch.sigmoid(self.alpha_merge_bth[layer])
            gate_bth = self.alpha_merge_bth[layer]
            h = gate_bth*h + (1-gate_bth)*h2
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.gcn_layer_num - 1:
                #remove relu for the last layer
                h = F.dropout(h, training = self.training)
            else:
                h = F.dropout(act(h), training = self.training)
            
            h_list.append(h)
        
        node_emb = h_list[-1]
        return node_emb


class GNN_nodepred_merg_side(torch.nn.Module):
    def __init__(self, feat_dim, num_class, out_dim, num_layer=3, hid_dim=100, gnn_type='GIN', num_side_layer=3, side_hid_dim=32):
        super().__init__()

        self.gnn = GNN_merge(feat_dim, hid_dim, out_dim, gcn_layer_num=num_layer, gnn_type=gnn_type)
        self.siders = SideMLP(feat_dim, side_hid_dim, out_dim, gcn_layer_num=num_side_layer)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        # self.graph_pred_linear = torch.nn.Sequential(torch.nn.Linear(hid_dim, num_class), torch.nn.Softmax(dim=1))
        self.graph_pred_linear = torch.nn.Linear(hid_dim, num_class)
        self.freeze_original_params()
    
    def wght_layer_cal(self, model):
        cnt = 0
        for i in model.conv_layers.state_dict().keys():
            cnt+=1
        return cnt
    
    def forward(self, x, edge_index):
        node_representation = self.gnn(x, edge_index)
        side_representation = self.siders(x)
        # gate = torch.sigmoid(self.alpha)
        gate = self.alpha
        node_representation = (1-gate)*node_representation+gate*side_representation

        return self.graph_pred_linear(node_representation)
    
    def freeze_original_params(self):
        for param in self.parameters():
            param.requires_grad = False
        
        for param in self.gnn.alpha_merge:
            param.requires_grad = True   
        for param in self.gnn.alpha_merge_bth:
            param.requires_grad = True  
        self.alpha.requires_grad = True

        for param in self.graph_pred_linear.parameters():
            param.requires_grad = True
        for param in self.siders.parameters():
            param.requires_grad = True

    def unfreeze_original_params(self):
        for param in self.parameters():
            param.requires_grad = True

    def from_pretrained(self, model_file, device):
        model, _ = load_state_dict_from_path(self.gnn, model_file, device)     
        return model


if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    device_idx = 1
    device = torch.device("cuda:" + str(device_idx)) if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    output_model_file = './pre_trained_gnn/ogbn_arxiv.GraphCL.GCN.2.pth'
    dataname, num_parts = 'Cora', 200
    dataset = pk.load(open('./dataset/{}/feature_reduced.data'.format(dataname), 'br'))
    print('xxxx',dataset)
    data = dataset.data
    num_class = dataset.num_classes
    
    print("oooo", data)
    x = data.x.detach()
    edge_index = data.edge_index
    edge_index = to_undirected(edge_index)
    
    gnn_type = 'GCN'
    input_dim = out_dim = x.shape[1]
    model = GNN_nodepred_merg_side(feat_dim=input_dim, num_class=num_class, out_dim=out_dim, gnn_type=gnn_type)
    print(model)
    for i in model.named_parameters():
        print(i)
    
    model.from_pretrained(output_model_file, device)
    res = model(x, edge_index)
    # for i in model.named_parameters():
    #     print(i)
    print(res)
    print(res.shape)
    # model.unfreeze_original_params()
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    for i in model.named_parameters():
        print(i)

