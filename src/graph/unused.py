class GraphLevelGNN(pl.LightningModule):
    def __init__(self, model_name, c_out, graph_pooling="mean", **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        c_hidden = model_kwargs.get('out_channels', 16)
        self.pool = graph_pooling 

        if model_name == "MLP":
            self.model = MLPModel(**model_kwargs)
        elif model_name == 'GINConv':
            self.model = torch_geometric.nn.models.GIN(dropout=0.5, norm='BatchNorm', **model_kwargs)
        elif model_name == 'GAT':
            self.model = torch_geometric.nn.models.GAT(dropout=0.5, norm='BatchNorm', v2=True, heads=8, **model_kwargs)
        else:
            self.model = GNNModel(layer_name=model_name, **model_kwargs)
        self.head = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Linear(c_hidden, c_hidden), 
                                torch.nn.LeakyReLU(),
                                torch.nn.Dropout(0.5), torch.nn.Linear(c_hidden, c_out))
        self.loss_module = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="binary")
        self.train_auroc = AUROC(task="binary")
        self.train_f1 = F1Score(task="binary")
        self.valid_acc = Accuracy(task="binary")
        self.valid_auroc = AUROC(task="binary")
        self.valid_f1 = F1Score(task="binary")
        
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
        x = self.pool(x, batch) 
        out = self.head(x)

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
        self.train_auroc(y_hat, y)
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
        self.valid_auroc(y_hat, y)
        self.valid_f1(y_hat, y)

        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_auc", self.valid_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.valid_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss