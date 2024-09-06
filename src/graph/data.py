# Dataset class
import os.path as osp
import pandas as pd
import torch
from torch_geometric.data import Dataset
import torch.utils.data as data
import morph
import torch_geometric.data
from sklearn.preprocessing import StandardScaler

class GraphDataset(Dataset):
    def __init__(self, root, filePath, n_c, y_name='label',transform=None, pre_transform=None):
        self.filePath = osp.join(root, filePath)
        try:
            self.nc = n_c
        except:pass
        self.y_name = y_name
        super().__init__(root, transform, pre_transform)

    @property
    def processed_file_names(self):
        self.fileDF = pd.read_csv(self.filePath)
        return self.fileDF['Path'].tolist()

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        filePath = osp.join(self.processed_dir, self.processed_file_names[idx])
        data = torch.load(filePath)
        data.y = data[self.y_name]
        data.name = self.processed_file_names[idx]
        return data

class GraphDatasetPos(GraphDataset):
    def __init__(self, root, filePath, n_c, y_name='label',transform=None, pre_transform=None):
        super().__init__(root, filePath, n_c, y_name,transform, pre_transform)

    def get(self, idx):
        filePath = osp.join(self.processed_dir, self.processed_file_names[idx])
        data = torch.load(filePath)
        # pos_transformed = data.pos - torch.mean(data.pos, axis=0)
        # # normalize
        # means = pos_transformed .mean(dim=0, keepdim=True)
        # stds = pos_transformed .std(dim=0, keepdim=True)
        # normalized_data = (pos_transformed  - means) / stds
        pos = data.node_types.unsqueeze(0).t()
        # print(pos)
        data.x = torch.concat((data.x, pos), axis=1)
        data.y = data[self.y_name]
        data.name = self.processed_file_names[idx]
        return data

class GraphDatasetMorph(GraphDataset):
    def __init__(self, root, morph_path, filePath, n_c, y_name='label',transform=None, pre_transform=None):
        scaler = StandardScaler()
        self.outlinePCA =  morph.OutlinePCA.load(morph_path)
        self.outlinePCA.df_weights['R'] = scaler.fit_transform(self.outlinePCA.df_weights[['R']])
        super().__init__(root, filePath, n_c, y_name,transform, pre_transform)
    
    def get(self, idx):
        filePath = osp.join(self.processed_dir, self.processed_file_names[idx])
        data = torch.load(filePath)
        data.y = data[self.y_name]
        data.name = self.processed_file_names[idx]
        name = self.processed_file_names[idx].split('\\')[-1].split('_')
        query = '_'.join(name[:-1]) + f'_id{name[-1][:-3]}'
        morph_features = self.outlinePCA.df_weights[self.outlinePCA.df_weights.id == query]
        data.features = torch.tensor(morph_features.values[:, :-1].astype(float))
        data.features_names = morph_features.columns[:-1].tolist()
        return data

class GraphDatasetMLP(GraphDataset):
    def __init__(self, root, raw_folder_name, n_c, 
                y_name='label', transform=None, pre_transform=None):
        super().__init__(root, raw_folder_name, n_c, 
                y_name=y_name, transform=transform, pre_transform=pre_transform)

    def get(self, idx):
        filePath = osp.join(self.processed_dir, self.processed_file_names[idx])
        data = torch.load(filePath)
        data_new = torch_geometric.data.Data()
        data_new.x = data['x'].mean(axis=0, keepdim=True)
        data_new.y = data[self.y_name]
        data_new.name = self.processed_file_names[idx]
        return data_new

class GraphDatasetMLP_Moprh(GraphDatasetMorph):
    def __init__(self, root, morph_path, raw_folder_name, n_c, 
                y_name='label', transform=None, pre_transform=None):
        super().__init__(root, morph_path, raw_folder_name, n_c, 
                y_name=y_name, transform=transform, pre_transform=pre_transform)

    def get(self, idx):
        filePath = osp.join(self.processed_dir, self.processed_file_names[idx])
        data = torch.load(filePath)
        data_new = torch_geometric.data.Data()
        data_new.x = data['x'].mean(axis=0, keepdim=True)
        data_new.y = data[self.y_name]
        data_new.name = self.processed_file_names[idx]
        
        name = self.processed_file_names[idx].split('\\')[-1].split('_')
        query = '_'.join(name[:-1]) + f'_id{name[-1][:-3]}'
        morph_features = self.outlinePCA.df_weights[self.outlinePCA.df_weights.id == query]
        data_new.features = torch.tensor(morph_features.values[:, :-2].astype(float))
        data_new.features_names = morph_features.columns[:-2].tolist()
        return data_new


def train_test_val_split(dataset, test_ratio=0.4, val_ratio=0.2):
    seed = torch.Generator().manual_seed(42)

    dataset = dataset.shuffle()
    test_size = int(len(dataset)*test_ratio)
    train_size = len(dataset) - int(len(dataset)*test_ratio)
    train_set, test_set = data.random_split(dataset, [train_size, test_size], generator=seed)

    val_size = int(train_size*val_ratio)
    train_size = train_size - val_size
    train_set, val_set = data.random_split(train_set, [train_size, val_size], generator=seed)

    return train_set, val_set, test_set