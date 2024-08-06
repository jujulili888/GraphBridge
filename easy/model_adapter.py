import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from model import *  
import argparse

from loader import MoleculeDataset_aug
from torch_geometric.data import DataLoader
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import global_mean_pool
from util import load_state_dict_from_path

num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 

class AdapterLayer(nn.Module):
    def __init__(self, emb_dim, down_dim):
        super().__init__()

        self.adapter_input_size = emb_dim
        self.adapter_latent_size = down_dim
        self.non_linearity = torch.nn.ReLU()
        # down projection
        self.down_proj = nn.Linear(self.adapter_input_size, self.adapter_latent_size)
        # up projection
        self.up_proj = nn.Linear(self.adapter_latent_size, self.adapter_input_size)

        self.init_weights()

    def init_weights(self):
        self.down_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.down_proj.bias.data.zero_()
        self.up_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.up_proj.bias.data.zero_()

    def forward(self, x):
        output = self.up_proj(self.non_linearity(self.down_proj(x)))
        return output


class AdapterGINConv(GINConv):
    def __init__(self, emb_dim, aggr='add', down_dim=16):
        super().__init__(emb_dim, aggr)
        #multi-layer perceptron
        self.adapter = AdapterLayer(emb_dim, down_dim)
        self.adapter_mp = AdapterLayer(emb_dim, down_dim)
        self.alpha1 = nn.Parameter(torch.tensor(0.0))
        self.alpha2 = nn.Parameter(torch.tensor(0.0))


    def forward(self, x, edge_index, edge_attr):
        out_adpt1 = self.adapter(x)
        #add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))
        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)
        
        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])
        out_init, out_adpt2 = self.propagate(edge_index, x=x, edge_attr=edge_embeddings)
        out = out_init + self.alpha1*out_adpt1+self.alpha2*out_adpt2
        return out

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return (self.mlp(aggr_out), self.adapter_mp(aggr_out))


class AdapterGNN(GNN):
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0):
        super().__init__(num_layer, emb_dim, JK, drop_ratio)
        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(AdapterGINConv(emb_dim, aggr = "add"))
        self.freeze_original_params(self.num_layer)


        # ###List of batchnorms
        # self.batch_norms = torch.nn.ModuleList()
        # for layer in range(num_layer):
        #     self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def freeze_original_params(self, num_layer):
        for param in self.parameters():
            param.requires_grad = False
        # print('layer',self.gnns[0])
        for i in range(num_layer):
            self.gnns[i].alpha1.requires_grad = True
            self.gnns[i].alpha2.requires_grad = True
            for param in self.batch_norms[i].parameters():
                param.requires_grad = True
            for param in self.gnns[i].adapter.parameters():
                param.requires_grad = True
            for param in self.gnns[i].adapter_mp.parameters():
                param.requires_grad = True

    def unfreeze_original_params(self):
        for param in self.parameters():
            param.requires_grad = True


class GNN_graphpred_PET(GNN_graphpred):
    def __init__(self, num_layer, emb_dim, num_tasks, JK = "last", drop_ratio = 0, graph_pooling = "mean", is_adapt_tune=True):
        super().__init__(num_layer, emb_dim, num_tasks, JK, drop_ratio, graph_pooling)
        self.is_adapt_tune = is_adapt_tune
        if self.is_adapt_tune:
            self.gnn = AdapterGNN(num_layer, emb_dim, JK, drop_ratio)
        else:
            self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio)


    def from_pretrained(self, model_file, device):
        model, _ = load_state_dict_from_path(self.gnn, model_file, device)
        return model




if __name__ == "__main__":
            # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=1,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--output_model_file', type = str, default = 'models_graphcl/graphcl_80.pth', help='filename to output the pre-trained model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--aug1', type=str, default = 'none')
    parser.add_argument('--aug_ratio1', type=float, default = 0.2)
    parser.add_argument('--aug2', type=str, default = 'none')
    parser.add_argument('--aug_ratio2', type=float, default = 0.2)
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    args = parser.parse_args()


    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


    #set up dataset
    dataset = MoleculeDataset_aug("dataset/" + args.dataset, dataset=args.dataset)
    print("data",dataset)
    dataset.aug, dataset.aug_ratio = args.aug1, args.aug_ratio1
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=False)
    batch = next(iter(dataloader))
    print(batch)
    # print('x_feat',batch.x)
    # print('e_idx',batch.edge_index)
    # print('e_attr',batch.edge_attr)
    # print('batch',batch.batch)
    # model = AdapterGINConv(emb_dim=300, aggr='add')
    
    # x_embedding1 = torch.nn.Embedding(num_atom_type, 300)
    # x_embedding2 = torch.nn.Embedding(num_chirality_tag, 300)
    # # print(batch.x)
    # x_new = x = x_embedding1(batch.x[:,0]) + x_embedding2(batch.x[:,1])
    # # print(x_new.shape)
    # print(res)
    model = GNN_graphpred_PET(args.num_layer, args.emb_dim, 2, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling)
    print(model)
    for i in model.named_parameters():
        print(i)
    res = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    # model.from_pretrained(args.output_model_file)
    print(res)
    print(res.shape)

    # model.unfreeze_original_params()
    # print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    # for i in model.named_parameters():
    #     print(i)

