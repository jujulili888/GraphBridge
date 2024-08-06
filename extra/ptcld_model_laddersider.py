import torch
from torch import Tensor
import torch.nn.functional as F
import pickle as pk
from utils import act
import torch.nn as nn
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from random import shuffle
import random
from ptcld_model import *
from utils import load_state_dict_from_path
import copy

    
class LadderSide_ptcld_GNN(ptcld_GNN):
    def __init__(self, input_dim, hid_dim=None, out_dim=None, gcn_layer_num=2, gnn_type='GAT', down_dim=16, k=20):
        super().__init__(input_dim, hid_dim, out_dim, gcn_layer_num, gnn_type, k)

        self.side_MLP = nn.ModuleList()
        self.side_norms = nn.ModuleList()
        self.alphas = nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for i in range(self.gcn_layer_num)])
        
        self.first_downsample = nn.Linear(hid_dim, down_dim)
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

    def forward(self, x, batch):
        if isinstance(x, torch.Tensor):
            x: PairTensor = (x, x)
        b: PairOptTensor = (None, None)
        if isinstance(batch, torch.Tensor):
            b = (batch, batch)
        
        edge_index = knn(x[0], x[1], self.k, b[0], b[1]).flip([0])
        h = self.transfer(x[0])

        side_x = self.first_downsample(F.dropout(h, training = self.training))
        h_list = [h]
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

        for param in self.transfer.parameters():
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


class GNN_ptcldpred_ladderside(GNN_ptcldpred):
    def __init__(self, input_dim, num_class, hid_dim=None, out_dim=None, gcn_layer_num=2, gnn_type='GAT', down_dim=16, pool=None, k=20):
        super().__init__(input_dim, num_class, hid_dim, out_dim, gcn_layer_num, gnn_type, pool, k)

        self.gnn = LadderSide_ptcld_GNN(input_dim, hid_dim, out_dim, gcn_layer_num, gnn_type, down_dim, k)

    def from_pretrained(self, model_file, device):
        model, _ = load_state_dict_from_path(self.gnn, model_file, device)
        return model


class LadderSide_ptcld_GNN_merge(ptcld_GNN):
    def __init__(self, input_dim, hid_dim=None, out_dim=None, gcn_layer_num=2, gnn_type='GAT', down_dim=16, k=20):
        super().__init__(input_dim, hid_dim, out_dim, gcn_layer_num, gnn_type, k)
        
        self.conv_layers2 = copy.deepcopy(self.conv_layers)
        self.batch_norms2 = copy.deepcopy(self.batch_norms)
        self.alpha_merge = nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for i in range(gcn_layer_num)])
        self.alpha_merge_bth = nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for i in range(gcn_layer_num)])

        self.side_MLP = nn.ModuleList()
        self.side_norms = nn.ModuleList()
        self.alphas = nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for i in range(self.gcn_layer_num)])
        
        self.first_downsample = nn.Linear(hid_dim, down_dim)
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

    def forward(self, x, batch):
        if isinstance(x, torch.Tensor):
            x: PairTensor = (x, x)
        b: PairOptTensor = (None, None)
        if isinstance(batch, torch.Tensor):
            b = (batch, batch)
        
        edge_index = knn(x[0], x[1], self.k, b[0], b[1]).flip([0])
        h = self.transfer(x[0])
        side_x = self.first_downsample(F.dropout(h, training = self.training))
        h_list = [h]
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
        
        for param in self.transfer.parameters():
            param.requires_grad = True  
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

class GNN_ptcldpred_merg_ladderside(GNN_ptcldpred):
    def __init__(self, input_dim, num_class, hid_dim=None, out_dim=None, gcn_layer_num=2, gnn_type='GAT', down_dim=16, pool=None, k=20):
        super().__init__(input_dim, num_class, hid_dim, out_dim, gcn_layer_num, gnn_type, pool, k)

        self.gnn = LadderSide_ptcld_GNN_merge(input_dim, hid_dim, out_dim, gcn_layer_num, gnn_type, down_dim, k)

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
    
    gnn_type = 'GCN'
    model = GNN_ptcldpred_merg_ladderside(input_dim=3, num_class=num_class, hid_dim=100, out_dim=100, gcn_layer_num=2, gnn_type=gnn_type, down_dim=16)

    print(model)
    for i in model.named_parameters():
        print(i)
    
    model.from_pretrained(output_model_file, device)
    batch = next(iter(train_loader))
    res = model(batch.pos, batch.batch)
    # for i in model.named_parameters():
    #     print(i)
    # model.unfreeze_original_params()
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    for i in model.named_parameters():
        print(i)
    
    print(res)
    print(res.shape)