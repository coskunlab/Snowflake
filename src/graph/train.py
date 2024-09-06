import lightning.pytorch as pl
from torch_geometric.loader import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import os
from .model import NodeLevelGNN, GraphLevelGNN, GraphLevelGNN_Moprh, GraphLevelGNN_Pos
from lightning.pytorch.loggers import CSVLogger, WandbLogger
import wandb

def train_node_classifier(model_name, train_set, val_set, test_set, dataset,
                            CHECKPOINT_PATH, AVAIL_GPUS, epochs=100, **model_kwargs):
    pl.seed_everything(42)
    train_loader = DataLoader(train_set, batch_size=32)
    val_loader = DataLoader(val_set, batch_size=32)
    test_loader = DataLoader(test_set, batch_size=32)

    wandb_logger = WandbLogger()

    # Create a PyTorch Lightning trainer
    root_dir = os.path.join(CHECKPOINT_PATH, "NodeLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[ModelCheckpoint(dirpath=root_dir, save_weights_only=True, mode="max", monitor="val_acc", filename=f'NodeLevel{model_name}'), LearningRateMonitor(logging_interval="step")],
        accelerator="cuda",
        devices=AVAIL_GPUS,
        max_epochs=epochs,
        enable_progress_bar=False,
        log_every_n_steps=10,
        logger=[CSVLogger(save_dir=root_dir), wandb_logger],
    )  # 0 because epoch size is 1
    # trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(root_dir, "NodeLevel%s.ckpt" % model_name)
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = NodeLevelGNN(
            model_name=model_name, in_channels=dataset.num_node_features, out_channels=dataset.num_classes, **model_kwargs
        )
        model = NodeLevelGNN.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything()
        model = NodeLevelGNN(
            model_name=model_name, in_channels=dataset.num_node_features, out_channels=dataset.num_classes, **model_kwargs
        )
        trainer.fit(model, train_loader, val_loader)
        model = NodeLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on the test set
    test_result = trainer.test(model, dataloaders=test_loader , verbose=False)
    batch = next(iter(test_loader))
    batch = batch.to(model.device)
    _, train_acc = model.forward(batch, mode="train")
    _, val_acc = model.forward(batch, mode="val")
    result = {"train": train_acc, "val": val_acc, "test": test_result[0]["test_acc"]}
    return model, result, trainer


def train_graph_classifier(model_name, train_set, val_set, test_set, dataset,
                            CHECKPOINT_PATH, AVAIL_GPUS, epochs=100, **model_kwargs):
    pl.seed_everything(42)
    train_loader = DataLoader(train_set, batch_size=8)
    val_loader = DataLoader(val_set, batch_size=8)
    test_loader = DataLoader(test_set, batch_size=8)
    c_out = dataset.num_classes
    c_hidden = model_kwargs.get('hidden_channels', 16)

    wandb_logger = WandbLogger()

    # Create a PyTorch Lightning trainer
    root_dir = os.path.join(CHECKPOINT_PATH, "GraphLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[ModelCheckpoint(dirpath=root_dir, save_weights_only=True, mode="max", monitor="val_acc", filename=f'GraphLevel{model_name}'), LearningRateMonitor(logging_interval="step")],
        accelerator="cuda",
        devices=AVAIL_GPUS,
        max_epochs=epochs,
        enable_progress_bar=False,
        log_every_n_steps=10,
        logger=[CSVLogger(save_dir=root_dir), wandb_logger],
    )  # 0 because epoch size is 1
    # trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(root_dir, "GraphLevel%s.ckpt" % model_name)
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = GraphLevelGNN(
            model_name=model_name, in_channels=dataset.num_node_features, out_channels=c_hidden, c_out=c_out, **model_kwargs
        )
        model = GraphLevelGNN.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything()
        model = GraphLevelGNN(
            model_name=model_name, in_channels=dataset.num_node_features, out_channels=c_hidden, c_out=c_out, **model_kwargs
        )
        trainer.fit(model, train_loader, val_loader)
        model = GraphLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on the test set
    test_result = trainer.test(model, dataloaders=test_loader , verbose=False)
    batch = next(iter(test_loader))
    batch = batch.to(model.device)
    _, train_acc = model.forward(batch, mode="train")
    _, val_acc = model.forward(batch, mode="val")
    result = {"train": train_acc, "val": val_acc, "test": test_result[0]["test_acc"]}
    return model, result, trainer

def train_graph_classifier_morph(model_name, train_set, val_set, test_set, dataset,
                            CHECKPOINT_PATH, AVAIL_GPUS, epochs=100, **model_kwargs):
    pl.seed_everything(42)
    train_loader = DataLoader(train_set, batch_size=8)
    val_loader = DataLoader(val_set, batch_size=8)
    test_loader = DataLoader(test_set, batch_size=8)
    c_out = dataset.num_classes
    c_hidden = model_kwargs.get('hidden_channels', 16)
    num_feat = len(dataset[0].features_names)

    wandb_logger = WandbLogger()

    # Create a PyTorch Lightning trainer
    root_dir = os.path.join(CHECKPOINT_PATH, "GraphLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[ModelCheckpoint(dirpath=root_dir, save_weights_only=True, mode="max", monitor="val_acc", filename=f'GraphLevel{model_name}'), LearningRateMonitor(logging_interval="step")],
        accelerator="cuda",
        devices=AVAIL_GPUS,
        max_epochs=epochs,
        enable_progress_bar=False,
        log_every_n_steps=10,
        logger=[CSVLogger(save_dir=root_dir), wandb_logger],
    )  # 0 because epoch size is 1
    # trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(root_dir, "GraphLevel%s.ckpt" % model_name)
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = GraphLevelGNN_Moprh(
            model_name=model_name, in_channels=dataset.num_node_features, num_feat=num_feat, out_channels=c_hidden, c_out=c_out, **model_kwargs
        )
        model = GraphLevelGNN_Moprh.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything()
        model = GraphLevelGNN_Moprh(
            model_name=model_name, in_channels=dataset.num_node_features, num_feat=num_feat, out_channels=c_hidden, c_out=c_out, **model_kwargs
        )
        trainer.fit(model, train_loader, val_loader)
        model = GraphLevelGNN_Moprh.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on the test set
    test_result = trainer.test(model, dataloaders=test_loader , verbose=False)
    batch = next(iter(test_loader))
    batch = batch.to(model.device)
    _, train_acc = model.forward(batch, mode="train")
    _, val_acc = model.forward(batch, mode="val")
    result = {"train": train_acc, "val": val_acc, "test": test_result[0]["test_acc"]}
    return model, result, trainer


def train_graph_classifier_kfold(model_name, train_set, val_set, test_set, dataset,
                            CHECKPOINT_PATH, AVAIL_GPUS, epochs=100, batch_size=64, morph=False,pos=False, **model_kwargs):
    
    pl.seed_everything(42)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    c_out = dataset.nc
    c_hidden = model_kwargs.get('hidden_channels', 16)

    wandb_logger = WandbLogger()

    # Create a PyTorch Lightning trainer
    root_dir = os.path.join(CHECKPOINT_PATH, "GraphLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[
            ModelCheckpoint(dirpath=root_dir, save_weights_only=True, mode="max", monitor="val_auc", filename=f'GraphLevel{model_name}'), 
            LearningRateMonitor(logging_interval="epoch")],
        accelerator="cuda",
        devices=AVAIL_GPUS,
        max_epochs=epochs,
        enable_progress_bar=False,
        log_every_n_steps=10,
        logger=[CSVLogger(save_dir=root_dir), wandb_logger],
    )  # 0 because epoch size is 1
    # trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    pretrained_filename = os.path.join(root_dir, "GraphLevel%s.ckpt" % model_name)
    if morph:
        num_feat = len(dataset[0].features[0])
        print('Training with Morphological features')
        if os.path.isfile(pretrained_filename):
            print("Found pretrained model, loading...")
            model = GraphLevelGNN_Moprh(
                model_name=model_name, in_channels=dataset.num_node_features, num_feat=num_feat, out_channels=c_hidden, c_out=c_out, **model_kwargs
            )
            # trainer.validate(model=model, dataloaders=val_loader)
        else:
            pl.seed_everything()
            model = GraphLevelGNN_Moprh(
                model_name=model_name, in_channels=dataset.num_node_features, num_feat=num_feat, out_channels=c_hidden, c_out=c_out, **model_kwargs
            )
            trainer.fit(model, train_loader, val_loader)
        
    else:
        if pos==False:
            if os.path.isfile(pretrained_filename):
                print("Found pretrained model, loading...")
                model = GraphLevelGNN(
                    model_name=model_name, in_channels=dataset.num_node_features, out_channels=c_hidden, c_out=c_out, **model_kwargs
                )
                # trainer.validate(model=model, dataloaders=val_loader)

            else:
                pl.seed_everything()
                model = GraphLevelGNN(
                    model_name=model_name, in_channels=dataset.num_node_features, out_channels=c_hidden, c_out=c_out, **model_kwargs
                )
                trainer.fit(model, train_loader, val_loader)
        else:
            pl.seed_everything()
            model = GraphLevelGNN_Pos(
                model_name=model_name, in_channels=dataset.num_node_features, out_channels=c_hidden, c_out=c_out, **model_kwargs
            )
            trainer.fit(model, train_loader, val_loader)

    test_result = trainer.test(model, dataloaders=test_loader , verbose=False)
    print(trainer.logged_metrics)


# Small function for printing the test scores
def print_results(result_dict):
    if "train" in result_dict:
        print("Train accuracy: %4.2f%%" % (100.0 * result_dict["train"]))
    if "val" in result_dict:
        print("Val accuracy:   %4.2f%%" % (100.0 * result_dict["val"]))
    print("Test accuracy:  %4.2f%%" % (100.0 * result_dict["test"]))