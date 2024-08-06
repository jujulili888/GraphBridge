import torch
import torch.nn.functional as F
from model import *  
import argparse
from util import load_state_dict_from_path


num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 


class LadderSide_GNN(GNN):
    def __init__(self, num_layer, emb_dim, down_dim=16, JK = "last", drop_ratio = 0):
        super().__init__(num_layer, emb_dim, JK, drop_ratio)

        self.side_MLP = nn.ModuleList()
        self.side_norms = nn.ModuleList()
        self.alphas = nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for i in range(self.num_layer)])
 
        
        self.first_downsample = nn.Linear(emb_dim, down_dim)
        self.side_downsamples = nn.ModuleList(
            [nn.Linear(emb_dim, down_dim) 
            for i in range(self.num_layer)]
        )

        for i in range(num_layer - 1):
            self.side_MLP.append(nn.Linear(down_dim, down_dim))
            self.side_norms.append(nn.BatchNorm1d(down_dim))
        self.side_MLP.append(nn.Linear(down_dim, emb_dim))
        self.side_norms.append(nn.BatchNorm1d(emb_dim))

        self.freeze_original_params(self.num_layer)
        # print(self.alphas)
        # print(self.side_downsamples)

    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])
        side_x = self.first_downsample(F.dropout(x, self.drop_ratio, training = self.training))
        h_list = [x]
        side_list = [side_x] 
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            
            ## side_forward
            # print(side_list[layer].shape, self.side_downsamples[layer](h).shape)
            side_h = self.alphas[layer] * side_list[layer] + (1 - self.alphas[layer]) * self.side_downsamples[layer](h)
            side_h = self.side_MLP[layer](side_h)
            side_h = self.side_norms[layer](side_h)
            if layer == self.num_layer - 1:
                side_h = F.dropout(side_h, self.drop_ratio, training = self.training)
            else:
                side_h = F.dropout(F.relu(side_h), self.drop_ratio, training = self.training)

            h_list.append(h)
            side_list.append(side_h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(side_list, dim = 1)
        elif self.JK == "last":
            node_representation = side_list[-1]
        elif self.JK == "max":
            side_list = [h.unsqueeze_(0) for h in side_list]
            node_representation = torch.max(torch.cat(side_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            side_list = [h.unsqueeze_(0) for h in side_list]
            node_representation = torch.sum(torch.cat(side_list, dim = 0), dim = 0)[0]

        return node_representation

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


class GNN_graphpred_ladderside(GNN_graphpred):
    def __init__(self, num_layer, emb_dim, num_tasks, down_dim=16, JK = "last", drop_ratio = 0, graph_pooling = "mean"):
        super().__init__(num_layer, emb_dim, num_tasks, JK, drop_ratio, graph_pooling)

        self.gnn = LadderSide_GNN(num_layer, emb_dim, down_dim, JK, drop_ratio)

    def from_pretrained(self, model_file, device):
        model, _ = load_state_dict_from_path(self.gnn, model_file, device)
        return model


if __name__ == "__main__":
            # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
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


    
    model = GNN_graphpred_ladderside(args.num_layer, args.emb_dim, 2, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling)
    print(model)
    for i in model.named_parameters():
        print(i)
    model.from_pretrained(args.output_model_file, device)
    # model.to(device)
    for i in model.named_parameters():
        print(i)


