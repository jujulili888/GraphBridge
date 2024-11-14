from collections import defaultdict
import pickle as pk
from torch_geometric.utils import subgraph, k_hop_subgraph, to_undirected
import torch
import numpy as np
from torch_geometric.transforms import SVDFeatureReduction
from torch_geometric.datasets import Planetoid, Amazon, Flickr, Reddit, Reddit2

from torch_geometric.data import Data, Batch
import random
import warnings
from utils import mkdir
from random import shuffle
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import DataLoader
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import copy



def SVD_FeatureReduction(out_channels, x):
    if x.size(-1) > out_channels:
        U, S, _ = torch.linalg.svd(x)
        x = torch.mm(U[:, :out_channels],
                            torch.diag(S[:out_channels]))
    return x


if __name__ == '__main__':
    device_idx = 1
    device = torch.device("cuda:" + str(device_idx)) if torch.cuda.is_available() else torch.device("cpu")
    dataname = 'Flickr' 
    #
    if dataname in ['CiteSeer', 'PubMed', 'Cora']:
        dataset = Planetoid(root='./dataset/', name=dataname)
        data = dataset.data
        data = data.to(device)
        print(data)
        feature_reduce = SVDFeatureReduction(out_channels=100)
        data = feature_reduce(data).cpu()
        print(data.x.device)
        dataset.data = data
        print(dataset.data, dataset.data.x.device)
    elif dataname=='Computers':
        dataset = Amazon(root='./dataset/', name=dataname)
        label_mask = np.load('./dataset/{}/all_list_computers.npy'.format(dataname))
        data = dataset.data
        print(len(label_mask))
        train_idx, valid_idx, test_idx = label_mask[:int(0.7*len(label_mask))], label_mask[int(0.7*len(label_mask)):int(0.8*len(label_mask))], label_mask[int(0.8*len(label_mask)):]
        mask_all = torch.zeros(data.x.shape[0], dtype=torch.bool)
        train_mask = mask_all.clone()
        train_mask[train_idx] = 1
        train_mask = torch.BoolTensor(train_mask)    
        val_mask = mask_all.clone()
        val_mask[valid_idx] = 1
        val_mask = torch.BoolTensor(val_mask)
        test_mask = mask_all.clone()
        test_mask[test_idx] = 1
        test_mask = torch.BoolTensor(test_mask)
        print(train_mask, val_mask, test_mask)
        feature_reduce = SVDFeatureReduction(out_channels=100)
        data = feature_reduce(data)
        data = Data(x=data.x, edge_index=data.edge_index, y=data.y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        dataset.data = data
        print(dataset.data)
    elif dataname in ['ogbn-products', 'ogbn-arxiv', 'ogbn-proteins', 'ogbn-mag']:
        dataset = PygNodePropPredDataset(root='./dataset/', name=dataname) 
        data = dataset.data
        print(data)
        # mask_unification
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        mask_all = torch.zeros(data.num_nodes, dtype=torch.bool)
        train_mask = mask_all.clone()
        train_mask[train_idx] = 1
        train_mask = torch.BoolTensor(train_mask)    
        val_mask = mask_all.clone()
        val_mask[valid_idx] = 1
        val_mask = torch.BoolTensor(val_mask)
        test_mask = mask_all.clone()
        test_mask[test_idx] = 1
        test_mask = torch.BoolTensor(test_mask)
        label = data.y.squeeze(1)
        #  feature matrix split for feature dimension reduction
        x = copy.deepcopy(data.x)
        max_len = x.shape[0]//40000
        print(max_len)
        x_ls = []
        for i in range(max_len):
            x_blk = x[i*40000:(i+1)*40000,:]
            x_reduce = SVD_FeatureReduction(out_channels=100, x=x_blk.to(device)).cpu()
            print(x_reduce.shape)
            x_ls.append(x_reduce)
        x_ls.append(SVD_FeatureReduction(out_channels=100, x=x[max_len*40000:,:].to(device)).cpu())
        x_2 = torch.vstack(x_ls)
        print(x_2.shape, x_2.device)
        data = Data(x=x_2, edge_index=data.edge_index, y=label, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        print(data)
        # save new graph data
        dataset.data = data
        print(dataset,dataset.data)
    elif dataname in ['ogbg-molhiv','ogbg-molpcba','ogbg-moltox21','ogbg-molbace', 'ogbg-molbbbp', 
                      'ogbg-molclintox', 'ogbg-molmuv','ogbg-molsider', 'ogbg-moltoxcast', 'ogbg-ppa']:
        dataset = PygGraphPropPredDataset(root='./dataset/', name=dataname) 
        print(dataset.data)
        split_idx = dataset.get_idx_split()
        atom_encoder = AtomEncoder(emb_dim = 100)
        bond_encoder = BondEncoder(emb_dim = 100)
        atom_emb = atom_encoder(dataset.data.x)
        dataset.data.x = atom_emb
        print(dataset.data)
        print(dataset[split_idx['train']][0], len(dataset), dataset.num_tasks)
        print(dataset[split_idx['valid']][0],dataset[split_idx['test']][0])
    elif dataname == 'Flickr':
        dataset = Flickr(root='./dataset/Flickr/')
        print(dataset.data.y)
        data = dataset.data
        x = copy.deepcopy(data.x)
        max_len = x.shape[0]//40000
        print(max_len)
        x_ls = []
        for i in range(max_len):
            x_blk = x[i*40000:(i+1)*40000,:]
            x_reduce = SVD_FeatureReduction(out_channels=100, x=x_blk.to(device)).cpu()
            print(x_reduce.shape)
            x_ls.append(x_reduce)
        x_ls.append(SVD_FeatureReduction(out_channels=100, x=x[max_len*40000:,:].to(device)).cpu())
        x_2 = torch.vstack(x_ls)
        print(x_2.shape, x_2.device)
        data.x = x_2
        dataset.data = data
        print(dataset, dataset.data)
    elif dataname == 'Reddit2':
        dataset = Reddit2(root='./dataset/Reddit2/')
        print(dataset.data)
        data = dataset.data
        x = copy.deepcopy(data.x)
        max_len = x.shape[0]//40000
        print(max_len)
        x_ls = []
        for i in range(max_len):
            x_blk = x[i*40000:(i+1)*40000,:]
            x_reduce = SVD_FeatureReduction(out_channels=100, x=x_blk.to(device)).cpu()
            print(x_reduce.shape)
            x_ls.append(x_reduce)
        x_ls.append(SVD_FeatureReduction(out_channels=100, x=x[max_len*40000:,:].to(device)).cpu())
        x_2 = torch.vstack(x_ls)
        print(x_2.shape, x_2.device)
        data.x = x_2
        dataset.data = data
        print(dataset, dataset.data)

    pk.dump(dataset, open('./dataset/{}/feature_reduced.data'.format("Flickr"), 'bw'))
    print('./dataset/{}/feature_reduced.data finish processing!!'.format("Flickr"))

    