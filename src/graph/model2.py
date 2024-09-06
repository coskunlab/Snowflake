import torch
import torch_geometric
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphConv, SAGEConv, GINConv, LayerNorm
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
import lightning.pytorch as pl
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn as nn
from torch_geometric.utils import softmax
from torch_geometric.nn.aggr import Aggregation
from torchmetrics import AUROC, F1Score, Accuracy
import math

class MLPModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5, task=None, num_classes=None):
        super().__init__()
        self.dropout = dropout
        self.n_feat = in_channels
        self.n_classes = out_channels
        self.n_hidden = hidden_channels
        self.n_layers = num_layers

        self.layers = [Linear(self.n_feat, self.n_hidden)]
        if self.n_layers > 2:
            for _ in range(self.n_layers-2):
                self.layers.append(Linear(self.n_hidden, self.n_hidden))
                self.layers.append(LayerNorm(self.n_hidden))
        self.layers.append(Linear(self.n_hidden, self.n_classes))
        self.layers = torch.nn.Sequential(*self.layers)
    
    def forward(self, x, *args, **kwargs):
        for layer in range(self.n_layers):
            x = self.layers[layer](x)
            if layer == self.n_layers - 1:
                #remove relu for the last layer
                x = F.dropout(x, self.dropout, training=self.training)
            else:
                x = F.dropout(F.relu(x), self.dropout, training=self.training)
        return x


# GNN module
gnn_layer_by_name = {"GCN": GCNConv, "GAT": GATConv, "GraphConv": GraphConv, "SAGEConv": SAGEConv, "GINConv": GINConv, }

class GNNModel(torch.nn.Module):
    def __init__(self, layer_name, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.n_feat = in_channels
        self.n_classes = out_channels
        self.n_hidden = hidden_channels
        self.n_layers = num_layers

        gnn_layer = gnn_layer_by_name[layer_name]

        layers = []
        in_channels, out_channels = self.n_feat, self.n_hidden
        
        for _ in range(self.n_layers - 1):
            layers += [
                gnn_layer(in_channels=in_channels, out_channels=out_channels, **kwargs),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(self.dropout),
                LayerNorm(out_channels)
            ]
            in_channels = self.n_hidden
        layers += [gnn_layer(in_channels=in_channels, out_channels=self.n_classes , **kwargs)]
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x, edge_index):
        """
        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        for layer in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(layer, torch_geometric.nn.MessagePassing):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x


class NodeLevelGNN(pl.LightningModule):
    def __init__(self, model_name, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()

        if model_name == "MLP":
            self.model = MLPModel(**model_kwargs)
        elif model_name == 'GINConv':
            self.model = torch_geometric.nn.models.GIN(dropout=0.5, **model_kwargs)
        elif model_name == 'GAT':
            self.model = torch_geometric.nn.models.GAT(dropout=0.5, v2=True, heads=8, **model_kwargs)
        else:
            self.model = GNNModel(layer_name=model_name, **model_kwargs)
        self.loss_module = torch.nn.CrossEntropyLoss()

    def forward(self, data, mode="train"):
        x, edge_index = data.x, data.edge_index
        out = self.model(x, edge_index)

        loss = self.loss_module(out, data.node_types)
        pred = out.argmax(dim=1)
        acc = (pred == data.node_types).sum().float() / len(pred)
        return loss, acc

    def configure_optimizers(self):
        # We use SGD here, but Adam works as well
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=5e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="val")
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log("test_acc", acc)

"""Attention pooling module"""
class Attention_module(Aggregation):
    def __init__(self, D1 = 20, D2 = 10):
        super(Attention_module, self).__init__()
        self.attention_Tanh = [
            nn.Linear(D1, D2),
            nn.Tanh()]
        
        self.attention_Sigmoid = [
            nn.Linear(D1, D2),
            nn.Sigmoid()]

        self.attention_Tanh = nn.Sequential(*self.attention_Tanh)
        self.attention_Sigmoid = nn.Sequential(*self.attention_Sigmoid)
        self.attention_Concatenate = nn.Linear(D2, 1)

    def forward(self, x, index=None, ptr=None, dim_size = None, dim= -2): # 20->10->2
        tanh_res = self.attention_Tanh(x)
        sigmoid_res = self.attention_Sigmoid(x)
        Attention_score = tanh_res.mul(sigmoid_res)
        Attention_score = self.attention_Concatenate(Attention_score)  # N x n_classes
        # return Attention_score, x
        gate = softmax(Attention_score, index, ptr, dim_size, dim) # N*K
        return self.reduce(gate*x, index, ptr, dim_size, dim)

"""Initial weights"""
def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()

class GraphLevelGNN(pl.LightningModule):
    def __init__(self, model_name, c_out, graph_pooling="mean", **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        c_hidden = model_kwargs.get('out_channels', 16)
        self.task = model_kwargs.get('task', 'binary')
        task = self.task
        num_classes = model_kwargs.get('num_classes', 2)
        self.pooling = graph_pooling 
        
        if model_name == "MLP":
            self.model = MLPModel(**model_kwargs)
        elif model_name == 'GINConv':
            self.model = torch_geometric.nn.models.GIN(dropout=0.5, norm='BatchNorm', **model_kwargs)
        elif model_name == 'GAT':
            self.model = torch_geometric.nn.models.GAT(dropout=0.5, norm='BatchNorm', v2=True, heads=8, **model_kwargs)
        elif model_name == 'PNA':
            self.model = torch_geometric.nn.models.PNA(dropout=0.5, norm='BatchNorm', **model_kwargs)
        else:
            self.model = GNNModel(layer_name=model_name, **model_kwargs)
        
        self.fnn_layer = nn.Linear(c_hidden, c_hidden)
        self.selu = nn.SELU()
        self.head = torch.nn.Linear(c_hidden, c_out)

        self.loss_module = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task=task, num_classes=num_classes)
        self.train_auroc = AUROC(task=task, num_classes=num_classes)
        self.train_f1 = F1Score(task=task, num_classes=num_classes)
        self.valid_acc = Accuracy(task=task, num_classes=num_classes)
        self.valid_auroc = AUROC(task=task, num_classes=num_classes)
        self.valid_f1 = F1Score(task=task,num_classes=num_classes)
        self.test_acc = Accuracy(task=task,num_classes=num_classes)
        self.test_auroc = AUROC(task=task,num_classes=num_classes)
        self.test_f1 = F1Score(task=task,num_classes=num_classes)
        
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Linear(c_hidden, 1))
        elif graph_pooling == "attention2":
            self.pool = Attention_module(D1 = c_hidden, D2=c_hidden//2)
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            self.pool = Set2Set(c_hidden, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        initialize_weights(self)

    def forward(self, data, mode="train"):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if 'batch' in data else torch.zeros((len(data.x),)).long().to(data.x.device)

        x = self.model(x, edge_index)
        x = self.fnn_layer(x)
        x = self.selu(x)
        x = self.pool(x, batch) 
        out = self.head(x)
        out= F.softmax(out, dim = 1)

        return out

    def configure_optimizers(self):
        # We use SGD here, but Adam works as well
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=5e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        y = batch['y']
        out = self.forward(batch)
        loss = self.loss_module(out, batch.y)

        y_hat = out.argmax(dim=1)

        if self.task == 'binary':
            self.train_acc(y_hat, y)
            self.train_auroc(out[:,1], y)
            self.train_f1(y_hat, y)
        else:
            self.train_acc(out, y)
            self.train_auroc(out, y)
            self.train_f1(out, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        self.log("train_auc", self.train_auroc, on_step=False, on_epoch=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch['y']
        out = self.forward(batch)
        loss = self.loss_module(out, batch.y)

        y_hat = out.argmax(dim=1)

        if self.task == 'binary':
            self.valid_acc(y_hat, y)
            self.valid_auroc(out[:,1], y)
            self.valid_f1(y_hat, y)
        else:
            self.valid_acc(out, y)
            self.valid_auroc(out, y)
            self.valid_f1(out, y)

        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_auc", self.valid_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.valid_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        y = batch['y']
        out = self.forward(batch)

        y_hat = out.argmax(dim=1)
        if self.task == 'binary':
            self.test_acc(y_hat, y)
            self.test_auroc(out[:,1], y)
            self.test_f1(y_hat, y)
        else:
            self.test_acc(out, y)
            self.test_auroc(out, y)
            self.test_f1(out, y)

        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_auc", self.test_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
    
    

class GNNModel_Pos(torch.nn.Module):
    def __init__(self, layer_name, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.n_feat = in_channels
        self.n_classes = out_channels
        self.n_hidden = hidden_channels
        self.n_layers = num_layers

        gnn_layer = gnn_layer_by_name[layer_name]

        layers = []
        in_channels, out_channels = self.n_feat, self.n_hidden
        
        for _ in range(self.n_layers - 1):
            layers += [
                gnn_layer(in_channels=in_channels, out_channels=out_channels, **kwargs),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(self.dropout),
                LayerNorm(out_channels)
            ]
            in_channels = self.n_hidden
        layers += [gnn_layer(in_channels=in_channels, out_channels=self.n_classes , **kwargs)]
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x, edge_index, edge_weight):
        """
        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        for layer in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(layer, torch_geometric.nn.MessagePassing):
                x = layer(x, edge_index, edge_weight)
            else:
                x = layer(x)
        return x

class GraphLevelGNN_Pos(pl.LightningModule):
    def __init__(self, model_name, c_out, graph_pooling="mean", **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        c_hidden = model_kwargs.get('out_channels', 16)
        self.task = model_kwargs.get('task', 'binary')
        task = self.task
        num_classes = model_kwargs.get('num_classes', 2)
        self.pooling = graph_pooling 
        
        if model_name == 'GINConv':
            self.model = torch_geometric.nn.models.GIN(dropout=0.5, norm='BatchNorm', **model_kwargs)
        elif model_name == 'GAT':
            self.model = torch_geometric.nn.models.GAT(dropout=0.5, norm='BatchNorm', edge_dim=2, v2=True, heads=8, **model_kwargs)
        else:
            self.model = GNNModel(layer_name=model_name, **model_kwargs)
        
        self.fnn_layer = nn.Linear(c_hidden, c_hidden)
        self.selu = nn.SELU()
        self.head = torch.nn.Linear(c_hidden, c_out)

        self.loss_module = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task=task, num_classes=num_classes)
        self.train_auroc = AUROC(task=task, num_classes=num_classes)
        self.train_f1 = F1Score(task=task, num_classes=num_classes)
        self.valid_acc = Accuracy(task=task, num_classes=num_classes)
        self.valid_auroc = AUROC(task=task, num_classes=num_classes)
        self.valid_f1 = F1Score(task=task,num_classes=num_classes)
        self.test_acc = Accuracy(task=task,num_classes=num_classes)
        self.test_auroc = AUROC(task=task,num_classes=num_classes)
        self.test_f1 = F1Score(task=task,num_classes=num_classes)
        
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Linear(c_hidden, 1))
        elif graph_pooling == "attention2":
            self.pool = Attention_module(D1 = c_hidden, D2=c_hidden//2)
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            self.pool = Set2Set(c_hidden, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        initialize_weights(self)

    '''def forward(self, data, mode="train"):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if 'batch' in data else torch.zeros((len(data.x),)).long().to(data.x.device)
        x = self.model(x, edge_index, edge_attr=edge_attr)
        x = self.fnn_layer(x)
        x = self.selu(x)
        x = self.pool(x, batch) 
        out = self.head(x)
        out= F.softmax(out, dim = 1)

        return out'''

    def forward(self, x, edge_index, edge_attr=None, data=None):
        batch = data.batch if 'batch' in data else torch.zeros((len(data.x),)).long().to(data.x.device)
        x = self.model(x, edge_index, edge_attr=edge_attr)
        x = self.fnn_layer(x)
        x = self.selu(x)
        x = self.pool(x, batch) 
        out = self.head(x)
        out= F.softmax(out, dim = 1)
        
        return out
    
    def latent(self, x, edge_index, edge_attr=None, data=None):
        batch = data.batch if 'batch' in data else torch.zeros((len(data.x),)).long().to(data.x.device)
        x = self.model(x, edge_index, edge_attr=edge_attr)
        x = self.fnn_layer(x)
        x = self.selu(x)
        x = self.pool(x, batch) 
        return x

    def captum(self, data, x):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        batch = data.batch if 'batch' in data else torch.zeros((len(data.x),)).long().to(data.x.device)
        x = self.model(x, edge_index, edge_attr=edge_attr)
        x = self.fnn_layer(x)
        x = self.selu(x)
        x = self.pool(x, batch) 
        out = self.head(x)
        out= F.softmax(out, dim = 1)

        return out


    def configure_optimizers(self):
        # We use SGD here, but Adam works as well
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=5e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        y = batch['y']
        out = self.forward(batch)
        loss = self.loss_module(out, batch.y)

        y_hat = out.argmax(dim=1)
        if self.task == 'binary':
            self.train_acc(y_hat, y)
            self.train_auroc(out[:,1], y)
            self.train_f1(y_hat, y)
        else:
            self.train_acc(out, y)
            self.train_auroc(out, y)
            self.train_f1(out, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        self.log("train_auc", self.train_auroc, on_step=False, on_epoch=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch['y']
        out = self.forward(batch)
        loss = self.loss_module(out, batch.y)

        y_hat = out.argmax(dim=1)

        if self.task == 'binary':
            self.valid_acc(y_hat, y)
            self.valid_auroc(out[:,1], y)
            self.valid_f1(y_hat, y)
        else:
            self.valid_acc(out, y)
            self.valid_auroc(out, y)
            self.valid_f1(out, y)

        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_auc", self.valid_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.valid_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        y = batch['y']
        out = self.forward(batch)

        y_hat = out.argmax(dim=1)

        if self.task == 'binary':
            self.test_acc(y_hat, y)
            self.test_auroc(out[:,1], y)
            self.test_f1(y_hat, y)
        else:
            self.test_acc(out, y)
            self.test_auroc(out, y)
            self.test_f1(out, y)

        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_auc", self.test_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)

class BilinearFusion(nn.Module):
    def __init__(self, skip=1, use_bilinear=1, gate1=1, gate2=1, dim1=128, dim2=128, scale_dim1=1, scale_dim2=1, mmhid=16, dropout_rate=0.25):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1//scale_dim1, dim2//scale_dim2
        skip_dim = dim1_og+dim2_og if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1), 32), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(32+skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        init_max_weights(self)

    def forward(self, vec1, vec2):
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            h1 = self.linear_h1(vec1)
            o1 = self.linear_o1(h1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            h2 = self.linear_h2(vec2)
            o2 = self.linear_o2(h2)

        ### Fusion
        o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1) # BATCH_SIZE X 1024
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip: out = torch.cat((out, vec1, vec2), 1)
        out = self.encoder2(out)
        return out

class GraphLevelGNN_Moprh(pl.LightningModule):
    def __init__(self, model_name, num_feat, c_out, graph_pooling="mean", fusion='concat',**model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        c_hidden = model_kwargs.get('out_channels', 16)
        self.fusion = fusion

        # self.feat_embedding = torch.nn.Linear(num_feat, c_hidden)
        # torch.nn.init.xavier_uniform_(self.feat_embedding.weight.data)
        self.feat_embedding = nn.Sequential(*[nn.Linear(num_feat, c_hidden), nn.ELU(), nn.AlphaDropout(p=0.5, inplace=False), 
                                              nn.Linear(c_hidden, c_hidden), nn.ELU(), nn.AlphaDropout(p=0.5, inplace=False)])

        if model_name == "MLP":
            self.model = MLPModel(**model_kwargs)
        elif model_name == 'GINConv':
            self.model = torch_geometric.nn.models.GIN(dropout=0.5, norm='BatchNorm', **model_kwargs)
        elif model_name == 'GAT':
            self.model = torch_geometric.nn.models.GAT(dropout=0.5, norm='BatchNorm', v2=True, heads=8, **model_kwargs)
        elif model_name == 'PNA':
            self.model = torch_geometric.nn.models.PNA(dropout=0.5, norm='BatchNorm', **model_kwargs)
        else:
            self.model = GNNModel(layer_name=model_name, **model_kwargs)
        
        if self.fusion=='concat':
            self.head = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Linear(c_hidden*2, c_hidden), 
                                            torch.nn.LeakyReLU(),
                                            torch.nn.Dropout(0.5), torch.nn.Linear(c_hidden, c_out))
        elif self.fusion == 'bilinear':
            self.head = BilinearFusion(dim1=c_hidden, dim2=c_hidden, 
                                       scale_dim1=1, gate1=1, 
                                       scale_dim2=1, gate2=1, 
                                       skip=True, mmhid=c_hidden)
            self.pred = torch.nn.Linear(c_hidden, c_out)
        
        self.fnn_layer = nn.Linear(c_hidden, c_hidden)
        self.selu = nn.SELU()

        self.loss_module = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="binary")
        self.train_auroc = AUROC(task="binary")
        self.train_f1 = F1Score(task="binary")
        self.valid_acc = Accuracy(task="binary")
        self.valid_auroc = AUROC(task="binary")
        self.valid_f1 = F1Score(task="binary")
        self.test_acc = Accuracy(task="binary")
        self.test_auroc = AUROC(task="binary")
        self.test_f1 = F1Score(task="binary")

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Linear(c_hidden, 1))
        elif graph_pooling == "attention2":
            self.pool = Attention_module(D1 = c_hidden, D2=c_hidden//2)
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            self.pool = Set2Set(c_hidden, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")


    def forward(self, data, mode="train"):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if 'batch' in data else torch.zeros((len(data.x),)).long().to(data.x.device)

        x = self.model(x, edge_index)
        x = self.fnn_layer(x)
        x = self.selu(x)
        x = self.pool(x, batch) 
        if self.fusion == 'concat':
            x = torch.concat((x, self.feat_embedding(data.features.float())), dim=1)
            out = self.head(x)
            out= F.softmax(out, dim = 1)
        elif self.fusion == 'bilinear':
            x = self.head(x, self.feat_embedding(data.features.float()))
            out = self.pred(x)
            out= F.softmax(out, dim = 1)
        # loss = self.loss_module(out, data.y)
        # pred = out.argmax(dim=1)
        # acc = (pred == data.y).sum().float() / len(pred)
        return out
    
    def captum(self, X, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if 'batch' in data else torch.zeros((len(data.x),)).long().to(data.x.device)

        x = self.model(x, edge_index)
        x = self.fnn_layer(x)
        x = self.selu(x)
        x = self.pool(x, batch) 

        x = self.head(x, self.feat_embedding(X))
        out = self.pred(x)
        out= F.softmax(out, dim = 1)  
        return out  

    def configure_optimizers(self):
        # We use SGD here, but Adam works as well
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=5e-4)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        y = batch['y']
        out = self.forward(batch)
        loss = self.loss_module(out, batch.y)

        y_hat = out.argmax(dim=1)
        self.train_acc(y_hat, y)
        self.train_auroc(out[:,1], y)
        self.train_f1(y_hat, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        self.log("train_auc", self.train_auroc, on_step=False, on_epoch=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch['y']
        out = self.forward(batch)
        loss = self.loss_module(out, batch.y)

        y_hat = out.argmax(dim=1)
        self.valid_acc(y_hat, y)
        self.valid_auroc(out[:,1], y)
        self.valid_f1(y_hat, y)

        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_auc", self.valid_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.valid_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        y = batch['y']
        out = self.forward(batch)

        y_hat = out.argmax(dim=1)
        self.test_step_acc(y_hat, y)
        self.test_step_auroc(out[:,1], y)
        self.test_step_f1(y_hat, y)

        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_auc", self.test_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)

