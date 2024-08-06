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


num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 

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
    def __init__(self, num_layer, emb_dim, down_dim=16, drop_ratio = 0):
        super().__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio

        if self.num_layer < 2:
            raise ValueError("Number of side net layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.mlp = torch.nn.ModuleList()
        if self.num_layer==2:
            self.mlp.append(SideLayer(emb_dim, down_dim))
            self.mlp.append(SideLayer(down_dim, emb_dim))
        elif self.num_layer>2:
            self.mlp.append(SideLayer(emb_dim, down_dim))
            for layer in range(1, num_layer-1):
                self.mlp.append(SideLayer(down_dim, down_dim))
            self.mlp.append(SideLayer(down_dim, emb_dim))


        ###List of batchnorms
        self.batch_norm = torch.nn.BatchNorm1d(emb_dim)
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.mlp[layer](h_list[layer])
            # h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = h = self.batch_norm(h)
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        node_representation = h_list[-1]

        return node_representation


class GNN_graphpred_side(GNN_graphpred):
    def __init__(self, num_layer, emb_dim, num_tasks, num_side_layer=3, down_dim=16, JK = "last", drop_ratio = 0, graph_pooling = "mean"):
        super().__init__(num_layer, emb_dim, num_tasks, JK, drop_ratio, graph_pooling)

        self.siders = SideMLP(num_side_layer, emb_dim, down_dim, drop_ratio)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.freeze_original_params()
    
    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)
        side_representation = self.siders(x, edge_index, edge_attr)
        node_representation = (1-self.alpha)*node_representation+self.alpha*side_representation

        return self.graph_pred_linear(self.pool(node_representation, batch))
    
    def freeze_original_params(self):
        for param in self.parameters():
            param.requires_grad = False
        
        self.alpha.requires_grad = True
        for param in self.graph_pred_linear.parameters():
            param.requires_grad = True
        for param in self.siders.parameters():
            param.requires_grad = True
        self.siders.x_embedding1.weight.requires_grad = False
        self.siders.x_embedding2.weight.requires_grad = False

    def unfreeze_original_params(self):
        for param in self.parameters():
            param.requires_grad = True

    def from_pretrained(self, model_file, device):
        model, _ = load_state_dict_from_path(self.gnn, model_file, device)     
        self.siders.state_dict()['x_embedding1.weight'].copy_(model.state_dict()['x_embedding1.weight'])
        self.siders.state_dict()['x_embedding2.weight'].copy_(model.state_dict()['x_embedding2.weight'])
        return model


if __name__ == "__main__":
            # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of transfer of graph neural networks')
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
    print

    #set up dataset
    dataset = MoleculeDataset_aug("dataset/" + args.dataset, dataset=args.dataset)
    print("data",dataset)
    dataset.aug, dataset.aug_ratio = args.aug1, args.aug_ratio1
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=False)
    batch = next(iter(dataloader))
    print(batch)

    model = GNN_graphpred_side(args.num_layer, args.emb_dim, 2, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling)
    # model = SideMLP(args.num_layer, args.emb_dim)
    print(model)
    # for i in model.named_parameters():
    #     print(i)
    res = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    model.from_pretrained(args.output_model_file)


