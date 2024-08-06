import torch
from torch_geometric.nn import global_max_pool
import torch.nn.functional as F
from ptcld_model import *  
import argparse
import copy

from torch_geometric.data import DataLoader
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
                # h = h = self.batch_norm(h)
                h = F.dropout(h, training = self.training)
            else:
                h = F.dropout(F.relu(h), training = self.training)
            h_list.append(h)

        node_representation = h_list[-1]

        return node_representation


class GNN_ptcldpred_side(torch.nn.Module):
    def __init__(self, feat_dim, num_class, out_dim, num_layer=5, hid_dim=100, gnn_type='GIN', num_side_layer=3, side_hid_dim=16, pool=None, k=20):
        super().__init__()

        self.gnn = ptcld_GNN(feat_dim, hid_dim, out_dim, num_layer, gnn_type, k)
        if pool is None:
            self.pool = global_max_pool
        else:
            self.pool = pool
        self.siders = SideMLP(feat_dim, side_hid_dim, out_dim, gcn_layer_num=num_side_layer)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        # self.graph_pred_linear = torch.nn.Sequential(torch.nn.Linear(hid_dim, num_class), torch.nn.Softmax(dim=1))
        self.graph_pred_linear = torch.nn.Linear(hid_dim, num_class)
        self.freeze_original_params()
    
    def forward(self, x, batch):
        node_representation = self.gnn(x, batch)
        side_representation = self.siders(x)
        # gate = torch.sigmoid(self.alpha)
        gate = self.alpha
        node_representation = (1-gate)*node_representation+gate*side_representation
        graph_representation = self.pool(node_representation, batch.long())

        return self.graph_pred_linear(graph_representation)
    
    def freeze_original_params(self):
        for param in self.parameters():
            param.requires_grad = False
        
        for param in self.gnn.transfer.parameters():
            param.requires_grad = True
        for param in self.graph_pred_linear.parameters():
            param.requires_grad = True
        
        self.alpha.requires_grad = True
        for param in self.siders.parameters():
            param.requires_grad = True

    def unfreeze_original_params(self):
        for param in self.parameters():
            param.requires_grad = True

    def from_pretrained(self, model_file, device):
        model, _ = load_state_dict_from_path(self.gnn, model_file, device)     
        return model


class ptcld_GNN_merge(ptcld_GNN):
    def __init__(self, input_dim, hid_dim=None, out_dim=None, gcn_layer_num=2, gnn_type='GAT', k=20):
        super().__init__(input_dim, hid_dim, out_dim, gcn_layer_num, gnn_type, k)
        self.conv_layers2 = copy.deepcopy(self.conv_layers)
        self.batch_norms2 = copy.deepcopy(self.batch_norms)
        self.alpha_merge = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for i in range(gcn_layer_num)])
        self.alpha_merge_bth = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for i in range(gcn_layer_num)])

    def forward(self, x, batch):
        if isinstance(x, torch.Tensor):
            x: PairTensor = (x, x)
        b: PairOptTensor = (None, None)
        if isinstance(batch, torch.Tensor):
            b = (batch, batch)

        edge_index = knn(x[0], x[1], self.k, b[0], b[1]).flip([0])
        h = self.transfer(x[0])
        
        h_list = [h]
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
                h = F.dropout(F.relu(h), training = self.training)
            
            h_list.append(h)
        
        node_emb = h_list[-1]
        return node_emb


class GNN_ptcldpred_merg_side(torch.nn.Module):
    def __init__(self, feat_dim, num_class, out_dim, num_layer=5, hid_dim=100, gnn_type='GIN', num_side_layer=3, side_hid_dim=16, pool=None, k=20):
        super().__init__()

        self.gnn = ptcld_GNN_merge(feat_dim, hid_dim, out_dim, num_layer, gnn_type, k)
        if pool is None:
            self.pool = global_max_pool
        else:
            self.pool = pool
        self.siders = SideMLP(feat_dim, side_hid_dim, out_dim, gcn_layer_num=num_side_layer)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        # self.graph_pred_linear = torch.nn.Sequential(torch.nn.Linear(hid_dim, num_class), torch.nn.Softmax(dim=1))
        self.graph_pred_linear = torch.nn.Linear(hid_dim, num_class)
        self.freeze_original_params()
    
    def forward(self, x, batch):
        node_representation = self.gnn(x, batch)
        side_representation = self.siders(x)
        # gate = torch.sigmoid(self.alpha)
        gate = self.alpha
        node_representation = (1-gate)*node_representation+gate*side_representation
        graph_representation = self.pool(node_representation, batch.long())

        return self.graph_pred_linear(graph_representation)
    
    def freeze_original_params(self):
        for param in self.parameters():
            param.requires_grad = False
        
        for param in self.gnn.alpha_merge:
            param.requires_grad = True   
        for param in self.gnn.alpha_merge_bth:
            param.requires_grad = True
        self.alpha.requires_grad = True
        for param in self.gnn.transfer.parameters():
            param.requires_grad = True

        
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
    device_idx = 0
    device = torch.device("cuda:" + str(device_idx)) if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    output_model_file = './pre_trained_gnn/ogbg_molhiv.GraphCL.GCN.2.pth'
    
    pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)

    train_dataset = ModelNet('./modelnet', '10', True, transform, pre_transform)
    num_class = train_dataset.num_classes
    print(num_class,train_dataset[0])
    test_dataset = ModelNet('./modelnet', '10', False, transform, pre_transform)
    print(test_dataset[0])


    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                            num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                            num_workers=1)
    
    # model = SideMLP(input_dim=input_dim, hid_dim=16, out_dim=out_dim, gcn_layer_num=3)
    gnn_type = 'GCN'
    # model = GNN_merge(input_dim, hid_dim=100, out_dim=out_dim, gcn_layer_num=3, gnn_type=gnn_type)
    model = GNN_ptcldpred_merg_side(feat_dim=3, num_class=num_class, out_dim=100, num_layer=2, hid_dim=100, 
                                  gnn_type=gnn_type, num_side_layer=3, side_hid_dim=32)
    print(model)
    for i in model.named_parameters():
        print(i)
    
    model.from_pretrained(output_model_file, device)
    batch = next(iter(train_loader))
    res = model(batch.pos, batch.batch)

    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    for i in model.named_parameters():
        print(i)
    
    print(res)
    print(res.shape)

