from ogb.linkproppred import PygLinkPropPredDataset
import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling
from tqdm import trange

dataset_name = 'ogbl-ddi'
dataset = PygLinkPropPredDataset(name=dataset_name)
print(f'The {dataset_name} dataset has {len(dataset)} graph(s).')
ddi_graph = dataset[0]
print(f'DDI 图: {ddi_graph}')
print(f'节点数量 |V|: {ddi_graph.num_nodes}')
print(f'边的数量 |E|: {ddi_graph.num_edges}')
print(f'无向图？: {ddi_graph.is_undirected()}')
print(f'节点平均度: {ddi_graph.num_edges / ddi_graph.num_nodes:.2f}')
print(f'节点特征: {ddi_graph.num_node_features}')
print(f'有孤立点？: {ddi_graph.has_isolated_nodes()}')
print(f'有自循环？: {ddi_graph.has_self_loops()}')

split_edges = dataset.get_edge_split()
train_edges, valid_edges, test_edges = split_edges['train'], split_edges['valid'], split_edges['test']
print(train_edges)
print(f'训练集正边的数量: {train_edges["edge"].shape[0]}')
print(f'验证集正边的数量: {valid_edges["edge"].shape[0]}')
print(f'验证集负边的数量: {valid_edges["edge_neg"].shape[0]}')
print(f'测试集正边的数量: {test_edges["edge"].shape[0]}')
print(f'测试集负边的数量: {valid_edges["edge_neg"].shape[0]}')


class GraphSAGE(torch.nn.Module):
    """使用 GraphSAGE 架构构建的图神经网络。"""

    def __init__(self, conv, in_channels, hidden_channels, out_channels, num_layers, dropout):
        '''
        in_channels：初始节点嵌入的维度。由于药物没有节点特征，我们将随机初始化这些向量。
        hidden_channels：中间节点嵌入的维度。隐藏层的维度。
        out_channels：输出节点嵌入的维度。
        num_layers：我们的 GNN 中的层数K。这是应用 GraphSAGE 运算符的次数。
        dropout：Dropout 应用于权重矩阵 W1 和 W2。
        '''
        super(GraphSAGE, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        assert (num_layers >= 2), 'Have at least 2 layers'##至少两层卷积
        # 在每一个layer中增加conv，上一层与下一层的维度必须一致
        # 我们还应用了归一化，之后输出节点嵌入。每个卷积层都是 L2 归一化的。
        self.convs.append(conv(in_channels, hidden_channels, normalize=True))
        for l in range(num_layers - 2):
            self.convs.append(conv(hidden_channels, hidden_channels, normalize=True))
        self.convs.append(conv(hidden_channels, out_channels, normalize=True))

        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is not None: ## 如果有edge_attr
            return self.forward_with_edge_attr(x, edge_index, edge_attr)

        # x 是初始节点嵌入的矩阵，形状 [N, in_channels]
        for i in range(self.num_layers - 1):
            # 第 i 层进行消息传递和聚合
            x = self.convs[i](x, edge_index)
            # x 的形状为 [N, hidden_channels]
            # 通过非线性激活函数relu
            x = F.relu(x)

            x = F.dropout(x, p=self.dropout, training=self.training)

        # 生成最终嵌入， x 的形状为 [N, out_channels]
        x = self.convs[self.num_layers - 1](x, edge_index)
        return x
  
    def forward_with_edge_attr(self, x, edge_index, edge_attr):
        # x 是初始节点嵌入的矩阵，形状 [N, in_channels]
        for i in range(self.num_layers - 1):
            # 第 i 层进行消息传递和聚合
            x = self.convs[i](x, edge_index, edge_attr)
            # x 的形状为 [N, hidden_channels]
            # 通过非线性激活函数relu
            x = F.relu(x)

            x = F.dropout(x, p=self.dropout,training=self.training)

        # 生成最终嵌入， x 的形状为 [N, out_channels]
        x = self.convs[self.num_layers - 1](x, edge_index, edge_attr)
        return x


class LinkPredictor(torch.nn.Module):
    """将两个输入转换为单个输出的通用网络。"""

    def __init__(self, in_channels, hidden_channels, dropout, out_channels=1,
                concat=lambda x, y: x * y):
        super(LinkPredictor, self).__init__()

        self.model = nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(), 
                                  nn.Dropout(p=dropout), nn.Linear(hidden_channels, out_channels), nn.Sigmoid())
        
        self.concat = concat
    
    def forward(self, u, v):
        x = self.concat(u, v)
        return self.model(x)


##训练我们的完整模型（GraphSAGE + LinkPredictor）
def train(graphsage_model, link_predictor, initial_node_embeddings, edge_index, 
          pos_train_edges, optimizer, batch_size, edge_attr=None):
    
    total_loss, total_examples = 0, 0

    # 设置我们的模型进行训练
    graphsage_model.train()
    link_predictor.train()

    # 迭代成批的训练边（“正边”）
    # （最后一次迭代的边数可能比 batch_size 少）
    for pos_samples in DataLoader(pos_train_edges, batch_size, shuffle=True):
        optimizer.zero_grad()

        # 运行 GraphSAGE 前向传递
        node_embeddings = graphsage_model(initial_node_embeddings, edge_index, edge_attr)
        
        #对由 attr:'edge_index'给出的图的随机对负边进行采样。
        # neg_samples 是一个尺寸为 [2, batch_size] 的张量
        neg_samples = negative_sampling(edge_index, 
                                        num_nodes=initial_node_embeddings.size(0),
                                        num_neg_samples=len(pos_samples),
                                        method='dense')
        
        # 在正边嵌入上运行链接预测器前向传递
        pos_preds = link_predictor(node_embeddings[pos_samples[:, 0]], 
                                    node_embeddings[pos_samples[:, 1]])
        
        # 在负边嵌入上运行链接预测器前向传递
        neg_preds = link_predictor(node_embeddings[neg_samples[0]], 
                                    node_embeddings[neg_samples[1]])

        preds = torch.concat((pos_preds, neg_preds))
        labels = torch.concat((torch.ones_like(pos_preds), 
                                torch.zeros_like(neg_preds)))

        loss = F.binary_cross_entropy(preds, labels)

        loss.backward()
        optimizer.step()

        num_examples = len(pos_preds)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
    
    return total_loss / total_examples


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#设置参数
graphsage_in_channels = 256 
graphsage_hidden_channels = 256 
graphsage_out_channels = 256 
graphsage_num_layers = 2 
dropout = 0.5 

###注意，因为数据库ddi本身没有附带节点特征矩阵，所以我们要创立初始嵌入。torch.nn.Embedding
initial_node_embeddings = torch.nn.Embedding(ddi_graph.num_nodes, graphsage_in_channels).to(device)
graphsage_model = GraphSAGE(SAGEConv, graphsage_in_channels, 
                                 graphsage_hidden_channels,
                                 graphsage_out_channels,
                                 graphsage_num_layers, 
                                 dropout).to(device)


link_predictor_in_channels = graphsage_out_channels
link_predictor_hidden_channels = link_predictor_in_channels

link_predictor = LinkPredictor(in_channels=link_predictor_in_channels, 
                               hidden_channels=link_predictor_hidden_channels, 
                               dropout=dropout).to(device)


##参数
lr = 0.005 
batch_size = 65536 
epochs = 2  
eval_steps = 5 
optimizer = torch.optim.Adam(list(graphsage_model.parameters()) + list(link_predictor.parameters()),lr=lr)

pos_valid_edges = valid_edges['edge'].to(device)
neg_valid_edges = valid_edges['edge_neg'].to(device)
pos_test_edges = test_edges['edge'].to(device)
neg_test_edges = test_edges['edge_neg'].to(device)

from ogb.linkproppred import Evaluator
evaluator = Evaluator(name = dataset_name)

@torch.no_grad()
def test(graphsage_model, link_predictor, initial_node_embeddings, edge_index, pos_valid_edges, neg_valid_edges, pos_test_edges, neg_test_edges, batch_size, evaluator, edge_attr=None):
    graphsage_model.eval()
    link_predictor.eval()

    final_node_embeddings = graphsage_model(initial_node_embeddings, edge_index, edge_attr)

    pos_valid_preds = []
    for pos_samples in DataLoader(pos_valid_edges, batch_size):
        pos_preds = link_predictor(final_node_embeddings[pos_samples[:, 0]], 
                                    final_node_embeddings[pos_samples[:, 1]])
        pos_valid_preds.append(pos_preds.squeeze())
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)
    

    neg_valid_preds = []
    for neg_samples in DataLoader(neg_valid_edges, batch_size):
        neg_preds = link_predictor(final_node_embeddings[neg_samples[:, 0]], 
                                    final_node_embeddings[neg_samples[:, 1]])
        neg_valid_preds.append(neg_preds.squeeze())
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for pos_samples in DataLoader(pos_test_edges, batch_size):
        pos_preds = link_predictor(final_node_embeddings[pos_samples[:, 0]], 
                                    final_node_embeddings[pos_samples[:, 1]])
        pos_test_preds.append(pos_preds.squeeze())
    pos_test_pred = torch.cat(pos_test_preds, dim=0)
    
    neg_test_preds = []
    for neg_samples in DataLoader(neg_test_edges, batch_size):
        neg_preds = link_predictor(final_node_embeddings[neg_samples[:, 0]], 
                                    final_node_embeddings[neg_samples[:, 1]])
        neg_test_preds.append(neg_preds.squeeze())
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    # Calculate Hits@20
    evaluator.K = 20
    valid_hits = evaluator.eval({'y_pred_pos': pos_valid_pred, 'y_pred_neg': neg_valid_pred})
    test_hits = evaluator.eval({'y_pred_pos': pos_test_pred, 'y_pred_neg': neg_test_pred})

    return valid_hits, test_hits


epochs_bar = trange(1, epochs + 1, desc='Loss n/a')

edge_index = ddi_graph.edge_index.to(device)
pos_train_edges = train_edges['edge'].to(device)

losses = []
valid_hits_list = []
test_hits_list = []
for epoch in epochs_bar:
    loss = train(graphsage_model, link_predictor, initial_node_embeddings.weight, edge_index, pos_train_edges, optimizer, batch_size)
    losses.append(loss)

    epochs_bar.set_description(f'Loss {loss:0.4f}')

    if epoch % eval_steps == 0:
        valid_hits, test_hits = test(graphsage_model, link_predictor, initial_node_embeddings.weight, edge_index, pos_valid_edges, neg_valid_edges, pos_test_edges, neg_test_edges, batch_size, evaluator)
        print()
        print(f'Epoch: {epoch}, Validation Hits@20: {valid_hits["hits@20"]:0.4f}, Test Hits@20: {test_hits["hits@20"]:0.4f}')
        valid_hits_list.append(valid_hits['hits@20'])
        test_hits_list.append(test_hits['hits@20'])
    else:
        valid_hits_list.append(valid_hits_list[-1] if valid_hits_list else 0)
        test_hits_list.append(test_hits_list[-1] if test_hits_list else 0)
