from torch.nn import Sequential as Seq, Linear as Lin, ReLU
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from ptcld_model import *
from ptcld_model_sider import *
from ptcld_model_laddersider import *

from tqdm import tqdm
import matplotlib.pyplot as plt


def train(model, device, loader, optimizer, criterion):
    model.train()
    train_size = 0
    total_loss = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch.pos, batch.batch)
        optimizer.zero_grad()
        loss = criterion(pred, batch.y)
        loss.backward()
        total_loss += loss.detach().cpu().item() * len(batch.y.detach().cpu())
        optimizer.step()
        train_size += len(batch.y)
        
        del batch, pred
        torch.cuda.empty_cache()
        
    return total_loss / train_size


def eval(model, device, loader):
    model.eval()

    correct = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):

        with torch.no_grad():
            batch = batch.to(device)
            pred = model(batch.pos, batch.batch)
            pred = pred.max(dim=1)[1]
            correct += pred.eq(batch.y).sum().cpu().item()
        del batch, pred
        torch.cuda.empty_cache()
    
    return correct / len(loader.dataset)


def set_seed(seed_value=42):
    """Set the seed for generating random numbers for PyTorch and other libraries to ensure reproducibility.

    Args:
        seed_value (int, optional): The seed value. Defaults to 42.
    """
    print("Setting seed..." + str(seed_value) + " for reproducibility")
    # Set the seed for generating random numbers in Python's random library.
    random.seed(seed_value)
    
    # Set the seed for generating random numbers in NumPy, which can also affect randomness in cases where PyTorch relies on NumPy.
    np.random.seed(seed_value)
    
    # Set the seed for generating random numbers in PyTorch. This affects the randomness of various PyTorch functions and classes.
    torch.manual_seed(seed_value)
    
    # If you are using CUDA, and want to generate random numbers on the GPU, you need to set the seed for CUDA as well.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU, if you are using more than one GPU.
        # Additionally, for even more deterministic behavior, you might need to set the following environment, though it may slow down the performance.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=101,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0.0001,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=2,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--hid_dim', type=int, default=100,
                        help='embedding dimensions (default: 100)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--gnn_type', type=str, default="GIN")
    parser.add_argument('--input_model_file', type=str, default = './pre_trained_gnn/ogbn_arxiv.SimGRACE.GAT.2.pth', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--eval_train', type=int, default = 1, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 1, help='number of workers for dataset loading')
    parser.add_argument('--tuning_methods', type=str, default = 'sidetune', help='tuning methods for transfer learning')
    parser.add_argument('--tuning', type=int, default = 1, help='tuning or training from scratch')
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    print("PyTorch version:", torch.__version__)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    
    pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)

    train_dataset = ModelNet('./modelnet', '10', True, transform, pre_transform)
    num_class = train_dataset.num_classes
    print(num_class,train_dataset[0], train_dataset)
    test_dataset = ModelNet('./modelnet', '10', False, transform, pre_transform)
    print(test_dataset[0], test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    if args.tuning_methods == 'finetune':
        model = GNN_ptcldpred(input_dim=3, num_class=num_class, hid_dim=args.hid_dim, out_dim=args.hid_dim, gcn_layer_num=args.num_layer, gnn_type=args.gnn_type)
    elif args.tuning_methods == 'sidetune':
        model = GNN_ptcldpred_side(feat_dim=3, num_class=num_class, out_dim=args.hid_dim, num_layer=args.num_layer, hid_dim=args.hid_dim, 
                                  gnn_type=args.gnn_type, num_side_layer=2, side_hid_dim=16)
    elif args.tuning_methods == 'merge_sidetune':
        model = GNN_ptcldpred_merg_side(feat_dim=3, num_class=num_class, out_dim=args.hid_dim, num_layer=args.num_layer, hid_dim=args.hid_dim, 
                                  gnn_type=args.gnn_type, num_side_layer=2, side_hid_dim=16)
    elif args.tuning_methods == 'laddersidetune':
        model = GNN_ptcldpred_ladderside(input_dim=3, num_class=num_class, hid_dim=args.hid_dim, out_dim=args.hid_dim, gcn_layer_num=args.num_layer, 
                                         gnn_type=args.gnn_type, down_dim=16)
    elif args.tuning_methods == 'merge_laddersidetune':
        model = GNN_ptcldpred_merg_ladderside(input_dim=3, num_class=num_class, hid_dim=args.hid_dim, out_dim=args.hid_dim, gcn_layer_num=args.num_layer, 
                                         gnn_type=args.gnn_type, down_dim=16)
    else:
        model = MLP_ptcldpred(input_dim=3, num_class=num_class, hid_dim=16, out_dim=args.hid_dim, gcn_layer_num=args.num_layer)
    model.to(device)
    print(model)
    for i in model.named_parameters():
        print(i)
    print(args)
    if args.input_model_file != "" and args.tuning:
        model.from_pretrained(args.input_model_file, device)
        print("Load pretrain model: {}".format(args.input_model_file))
        print()
    for i in model.named_parameters():
        print(i)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.decay)
    criterion = nn.CrossEntropyLoss()
    print(optimizer)

    train_acc_list = []
    test_acc_list = []
    loss_ls = []
    bf_train_acc = eval(model, device, train_loader)
    bf_test_acc = eval(model, device, test_loader)
    print("ACCURACY BEFORE TUNING: {:.4f}, {:.4f}".format(bf_train_acc, bf_test_acc))
    for epoch in range(1, args.epochs):
        
        print("====epoch " + str(epoch))
        loss = train(model, device, train_loader, optimizer, criterion)
        loss_ls.append(loss)
        print("loss: %f" %(loss))
        
        print("====Evaluation")
        train_acc = eval(model, device, train_loader)
        test_acc = eval(model, device, test_loader)
        print("train: %f test: %f" %(train_acc, test_acc))

        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)
        del loss, train_acc, test_acc
        torch.cuda.empty_cache()
        print("")
    print('Best Results: {:.4f}'.format(max(test_acc_list)))
    
    plt.switch_backend('Agg')

    plt.figure()                   
    plt.plot(loss_ls,'b',label = 'loss')       
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()      
    plt.savefig("outputs/{}_{}.jpg".format(args.gnn_type, args.tuning_methods))
    print('successfully save the image!')

    with open('outputs/result.log', 'a+') as f:
        f.write(args.input_model_file+ ' ' + args.gnn_type + ' ' + args.tuning_methods + ' ' +str(args.tuning) + ' ' + str(max(test_acc_list)))
        f.write('\n')
    
