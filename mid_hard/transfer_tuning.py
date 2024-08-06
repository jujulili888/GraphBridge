import torch
import torch.optim as optim
import pickle as pk
import torch.nn as nn
import time
import copy
import numpy as np
from model import * 
from model_sider import *
from model_laddersider import *
import argparse
from torch_geometric.utils import to_undirected
from torchinfo import summary



def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(features, g)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
    

def para_cnt(model):
    para_cnt = 0
    for name, params in model.named_parameters():
        if params.requires_grad==True:
            print(name, params.shape)
            if len(params.shape)==2:
                w,h = params.shape
                para_cnt += w*h
            elif len(params.shape)==3:
                w,h,c = params.shape
                para_cnt += w*h*c
            elif len(params.shape)==1:
                l = params.shape[0]
                para_cnt += l
            else:
                para_cnt +=1
            print(para_cnt)
    return para_cnt


def train_model(args, g, features, labels, masks, model, optimizer):
    train_mask = masks[0]
    val_mask = masks[1]
    test_mask = masks[2]
    loss_fcn = nn.CrossEntropyLoss()

    dur = []
    best_model=None
    best=0
    best_epoch = 0
    test_acc_ls = []
    patience = 50
    count = 0

    t_start = time.time()
    for epoch in range(args.epochs+1):
        model.train()
        t0 = time.time()
        # forward
        logits = model(features, g)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        # loss.requires_grad_(True)
        loss.backward()
        optimizer.step()

        dur.append(time.time() - t0)

        train_acc = evaluate(g, features, labels, train_mask, model)
        val_acc = evaluate(g, features, labels, val_mask, model)
        test_acc = evaluate(g, features, labels, test_mask, model)

        if best<val_acc:
            count = 0
            best=val_acc
            best_model=copy.deepcopy(model)
            best_epoch = epoch
        else:
            count+=1
            if count>patience:
                break
        if epoch % 100 == 0:
            print("Time consumption: {:.4f}".format(np.sum(dur)))
            print("Epoch {:05d} | Loss {:.4f} | train Accuracy  {:.4f} ".format(epoch, loss.item(), train_acc))
            print("Epoch {:05d} | Val Accuracy  {:.4f} | Test Accuracy  {:.4f} ".format(epoch, val_acc, test_acc))
        
        test_acc_ls.append(test_acc)
        loss = 0.0

    print("Total time consumption: {:.4f}s".format(time.time() - t_start))
    
    return best_model, best_epoch, test_acc_ls
if __name__ == "__main__":
        # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=2,
                        help='number of GNN message passing layers (default: 2).')
    parser.add_argument('--hid_dim', type=int, default=100,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--gnn_type', type=str, default="GCN")
    parser.add_argument('--dataset', type=str, default = 'Cora', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = './mid_hard/pre_trained_gnn/ogbn_arxiv.GraphCL.GCN.2.pth', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--eval_train', type=int, default = 1, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    parser.add_argument('--tuning_methods', type=str, default = 'finetune', help='tuning methods for transfer learning')
    parser.add_argument('--tuning', type=int, default = 1, help='tuning or not')
    args = parser.parse_args()


    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)
    print(device)
    
    dataname = args.dataset
    dataset = pk.load(open('./dataset/{}/feature_reduced.data'.format(dataname), 'br'))
    dataset = dataset
    data = dataset.data.to(device)
    num_class = dataset.num_classes
    print('xxxx',dataset, data, num_class)
    
    # data preprocess
    x = data.x.detach()
    edge_index = data.edge_index
    edge_index = to_undirected(edge_index)

    
    print()
    input_dim = out_dim = 100
    num_class = 7
    if args.tuning_methods == 'finetune':
        model = GNN_nodepred(input_dim, num_class, args.hid_dim, out_dim, gcn_layer_num=args.num_layer, gnn_type=args.gnn_type)
    elif args.tuning_methods == 'sidetune':
        model = GNN_nodepred_side(feat_dim=input_dim, num_class=num_class, out_dim=out_dim, num_layer=args.num_layer, hid_dim=args.hid_dim, 
                                  gnn_type=args.gnn_type, num_side_layer=2, side_hid_dim=16)
    elif args.tuning_methods == 'laddersidetune':
        model = GNN_nodepred_ladderside(input_dim, num_class, args.hid_dim, out_dim, gcn_layer_num=args.num_layer, gnn_type=args.gnn_type, down_dim=16)
    elif args.tuning_methods == 'merge_sidetune':
        model = GNN_nodepred_merg_side(feat_dim=input_dim, num_class=num_class, out_dim=out_dim, num_layer=args.num_layer, hid_dim=args.hid_dim, 
                                  gnn_type=args.gnn_type, num_side_layer=2, side_hid_dim=16)
    elif args.tuning_methods == 'merge_laddersidetune':
        model = GNN_nodepred_merg_ladderside(input_dim, num_class, args.hid_dim, out_dim, gcn_layer_num=args.num_layer, gnn_type=args.gnn_type, down_dim=16)

    model.to(device)
    if  args.input_model_file != "" and args.tuning:
        model.from_pretrained(args.input_model_file, device)
        print("Load pretrain model: {}".format(args.input_model_file))
        print()
    # print(model)
    print(summary(model))
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    labels = data.y
    masks = [data.train_mask, data.val_mask, data.test_mask]
    print(sum(data.train_mask==True))
    print(sum(data.val_mask==True))
    print(sum(data.test_mask==True))
    before_acc= evaluate(edge_index, x, labels, data.test_mask, model)
    print("ACCURACY BEFORE TUNING: {:.4f} ".format(before_acc))
    best_model, best_epoch, test_acc_ls = train_model(args, edge_index, x, labels, masks, model, optimizer)
    after_acc= evaluate(edge_index, x, labels, data.test_mask, best_model)
    print("ACCURACY AFTER TUNING: {:.4f}, Epoch:{:04d}".format(after_acc, best_epoch))

    with open('outputs/result.log', 'a+') as f:
        f.write(args.input_model_file + ' ' + args.dataset+ ' ' + args.gnn_type + ' ' + args.tuning_methods + ' ' +str(args.tuning) + ' ' + str(after_acc) +  ' ' + str(best_epoch))
        f.write('\n')