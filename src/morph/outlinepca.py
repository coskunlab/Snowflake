from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import os 
import pickle 

class OutlinePCA(object):
    def __init__(self, morph_data, n_components=16, random_state=111):
        self.gc_one_hot = morph_data.gc_one_hot
        self.bd_reg_foll = morph_data.bd_reg_foll
        self.bd_reg_gc = morph_data.bd_reg_gc
        self.df_foll_info = morph_data.df_new
        self.df = morph_data.df_bd
        self.morph_data = morph_data
        print('Fitting Follicle and GC boundaries PCA')
        self.foll_pca = self.fit_pca(self.bd_reg_foll, n_components=n_components, random_state=random_state)
        self.gc_pca = self.fit_pca(self.bd_reg_gc, n_components=n_components, random_state=random_state)
        print(f'Constructing latent space variation')
        self.latent_rep()
        self.get_weights_df(self.df_foll_info)

    def fit_pca(self, x, n_components=16, random_state=111):
        pca = PCA(n_components=n_components, random_state=random_state)
        pca.fit(x)
        return pca

    def plot_variance(self):
        # Get variance
        foll_variance = np.cumsum(self.foll_pca.explained_variance_ratio_)
        component_number_foll = np.arange(len(foll_variance)) + 1
        gc_variance = np.cumsum(self.foll_pca.explained_variance_ratio_)
        component_number_gc = np.arange(len(gc_variance)) + 1

        # Plot variance
        fig, axes = plt.subplots(1, 2, figsize=(8,3), sharey=True)
        axes[0].plot(component_number_foll, foll_variance)
        axes[0].set(ylim=(0.5,1), 
                    xlabel="n-components", 
                    ylabel="Explained variance",
                    title="Fraction of foll segmentation\nvariance captured")
        axes[1].plot(component_number_gc, gc_variance)
        axes[1].set(xlabel="n-components", 
                    title="Fraction of gc segmentation\nvariance captured")
        plt.tight_layout()

    def latent_rep(self,map_points=[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]):
        reconstruct = lambda mean, weight, component: (mean+weight*component).reshape(*original_shape)
        original_shape = self.bd_reg_foll[0].shape
        map_points = np.array(map_points)

        # Transform data with PCA
        self.foll_weights = self.foll_pca.transform(self.bd_reg_foll)
        self.gc_weights = self.gc_pca.transform(self.bd_reg_gc)
  
        # Get mapping at std deviation
        foll_std = np.std(self.foll_weights, 0).T
        gc_std = np.std(self.gc_weights, 0).T

        folls_std_map = foll_std[:, np.newaxis]*map_points
        gc_std_map = gc_std[:, np.newaxis]*map_points

        ## Reconstruct the cells along the principle components
        folls_along_princ_comps = []
        for foll_std, comp in zip(folls_std_map, self.foll_pca.components_):
            folls = [reconstruct(self.foll_pca.mean_, weight, comp) for weight in foll_std]
            folls_along_princ_comps.append(folls)
            
        gcs_along_princ_comps = []
        for gc_std, comp in zip(gc_std_map, self.gc_pca.components_):
            gcs = [reconstruct(self.gc_pca.mean_, weight, comp) for weight in gc_std]
            gcs_along_princ_comps.append(gcs)

        self.latent_folls = folls_along_princ_comps
        self.latent_gcs = gcs_along_princ_comps
        self.inverse_pca_foll = self.foll_pca.inverse_transform(self.foll_weights)
        self.inverse_pca_gc = self.gc_pca.inverse_transform(self.gc_weights)

    def get_weights_df(self, df_foll_info):
        # Create weight dataframe
        df_foll_weights = pd.DataFrame(self.foll_weights, columns=[f'Foll_SM_{i+1}' for i in range(len(self.foll_weights[0]))])
        df_gc_weights = pd.DataFrame(self.gc_weights, columns=[f'GC_SM_{i+1}' for i in range(len(self.gc_weights[0]))])
        self.df_weights = pd.concat([df_foll_weights, df_gc_weights], axis=1)
        self.df_weights['R'] = df_foll_info['R']
        self.df_weights['GC'] = self.gc_one_hot
        self.df_weights['id'] = df_foll_info['id']

    def plot_latent_rep(self, N, map_points=[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]):
        ## Cell reconstruction
        fig, axes = plt.subplots(6, len(map_points)*2, figsize=(15,7), sharey=True)
        for i, row in enumerate(axes):
            for j in range(len(map_points)):
                self.plot(self.latent_folls[i][j], row[j], N=N, color='m')
            for j in range(len(map_points)):
                self.plot(self.latent_gcs[i][j], row[j+len(map_points)], N=N, color='c')

        ## Complicated labeling, simple versions don't work with 3d projections
        title_props = {'size':25, 'horizontalalignment':'center'}
        label_props = {'size':18, 'horizontalalignment':'center', 'verticalalignment':'center'}
        fig.text(.25,.92, "Foll reconstructions", **title_props)
        fig.text(.75,.92, "GC reconstructions", **title_props)
        for i, x in zip(map_points, np.linspace(.042,.468,len(map_points))):
            fig.text(x,.9,f'{i}'+r'$\sigma$', **label_props)
        for i, x in zip(map_points, np.linspace(.532,.96,len(map_points))):
            fig.text(x,.9,f'{i}'+r'$\sigma$', **label_props)
        for i, y in enumerate(np.linspace(.82,.12,6)):
            fig.text(0,y, "PCA %i"%(i+1), rotation='vertical', **label_props)

        plt.tight_layout(rect=[0, 0.03, 1, 0.90])
        plt.subplots_adjust(wspace=0, hspace=0)

    def plot_latent_reconstruction(self, N, N_rows=10, N_cols=10, figsize=(10,10)):
        ## Reconstruction
        fig, axes = plt.subplots(N_rows, N_cols, figsize=figsize, sharey=True, sharex=True)
        for i, row in enumerate(axes):
            for j in range(len(row)):
                self.plot(self.bd_reg_foll[i*N_rows+j], row[j], N=N, color='r', alpha=0.5)
                self.plot(self.inverse_pca_foll[i*N_rows+j], row[j], N=N, color='m', ls='--')
                if self.gc_one_hot[i*N_rows+j] == 1:
                    self.plot(self.bd_reg_gc[i*N_rows+j], row[j], N=N, color='b', alpha=0.5)
                    self.plot(self.inverse_pca_gc[i*N_rows+j], row[j], N=N, color='c', ls='--')
        plt.tight_layout(rect=[0, 0.03, 1, 0.90])
        plt.subplots_adjust(wspace=0, hspace=0)
    
    def plot_random_data(self, N, N_rows=12, N_cols=7, random_state=4):
        figsize=(N_cols,N_rows)
        n_sample = N_rows*N_cols
        df_sampled = self.df.sample(n=n_sample, random_state=random_state).sort_index()
        index_sampled = df_sampled.index.tolist()
        df_sampled.reset_index(inplace=True, drop=True)

        ## Orignal data 
        fig, axes = plt.subplots(N_rows, N_cols, figsize=figsize, )
        for i, row in enumerate(axes):
            for j in range(len(row)):
                # For each boundary resample to N points and register
                foll_data = df_sampled.loc[i*N_cols+j, 'foll'].T
                gc_data = df_sampled.loc[i*N_cols+j, 'gc'].T
                self.plot2(foll_data[1], foll_data[0], row[j], N=N, color='r', alpha=1)
                if self.gc_one_hot[index_sampled][i*N_cols+j] == 1:
                    self.plot2(gc_data[1], gc_data[0], row[j], N=N, color='b', alpha=1)
        plt.tight_layout(rect=[0, 0.03, 1, 0.90])
        plt.subplots_adjust(wspace=0, hspace=0)    

        ## Reconstruction using resample function
        fig, axes = plt.subplots(N_rows, N_cols, figsize=figsize)
        for i, row in enumerate(axes):
            for j in range(len(row)):
                # For each boundary resample to N points and register
                bd_foll_resampled = self.morph_data.bd_resample((df_sampled.loc[i*N_cols+j, 'foll']), N)
                bd_gc_resampled = self.morph_data.bd_resample((df_sampled.loc[i*N_cols+j, 'gc']), N)
                self.plot2(bd_foll_resampled[1], bd_foll_resampled[0], row[j], N=N, color='r', alpha=1)
                if self.gc_one_hot[index_sampled][i*N_cols+j] == 1:
                    self.plot2(bd_gc_resampled[1], bd_gc_resampled[0], row[j], N=N, color='b', alpha=1)
        plt.tight_layout(rect=[0, 0.03, 1, 0.90])
        plt.subplots_adjust(wspace=0, hspace=0)    

        ## Reconstruction using inverse PCA
        fig, axes = plt.subplots(N_rows, N_cols, figsize=figsize)
        for i, row in enumerate(axes):
            for j in range(len(row)):
                # For each boundary resample to N points and register
                bd_foll_resampled = self.inverse_pca_foll[index_sampled][i*N_cols+j]
                bd_gc_resampled = self.inverse_pca_gc[index_sampled][i*N_cols+j]
                self.plot(bd_foll_resampled, row[j], N=N, color='r', alpha=1)
                if self.gc_one_hot[index_sampled][i*N_cols+j] == 1:
                    self.plot(bd_gc_resampled, row[j], N=N, color='b', alpha=1)
        plt.tight_layout(rect=[0, 0.03, 1, 0.90])
        plt.subplots_adjust(wspace=0, hspace=0)    

    def plot_all_data(self, N, N_rows=15, N_cols=15, legends=True):
        figsize = (N_rows, N_cols)

        for k in range(len(self.gc_one_hot)//(N_rows*N_cols)+1):
        ## Reconstruction
            fig, axes = plt.subplots(N_rows, N_cols, figsize=figsize, sharey=True, sharex=True)
            for i, row in enumerate(axes):
                for j in range(len(row)):
                    try:
                        bd_foll_resampled = self.inverse_pca_foll[k*N_rows*N_cols:][i*N_cols+j]
                        bd_gc_resampled = self.inverse_pca_gc[k*N_rows*N_cols:][i*N_cols+j]
                        self.plot(bd_foll_resampled, row[j], N=N, color='r', alpha=1)
                        if self.gc_one_hot[N_rows*N_cols*k + i*N_rows+j] == 1:
                            self.plot(bd_gc_resampled, row[j], N=N, color='b', alpha=1)
                        if legends:
                            label_props = {'size':12, 'horizontalalignment':'center', 'verticalalignment':'center'}
                            row[j].set_title(self.df.loc[k*N_rows*N_cols+i*N_cols+j, 'id'], **label_props)    
                    except:
                        row[j].axis('off')
            plt.tight_layout(rect=[0, 0.03, 1, 0.90])
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.show()

    @staticmethod
    def plot(data, ax, N=100, **kwargs):
        x = np.concatenate([data[:N], [data[0]]])
        y = np.concatenate([data[N:], [data[N]]])
        ax.plot(x, y, linewidth=2, **kwargs)
        ax.set_aspect('equal')
        # ax.set_xlim(-1.5, 1.5)
        # ax.set_ylim(-1.1, 1.1)
        ax.axis('off')

    @staticmethod
    def plot2(x,y, ax, N=100, **kwargs):
        x_ = np.append(x, x[0])
        y_ = np.append(y, y[0])
        try:
            ax.plot(x_, y_, linewidth=2, **kwargs)
        except:
            pass
        ax.set_aspect('equal')
        # ax.set_xlim(-1.5, 1.5)
        # ax.set_ylim(-1.1, 1.1)
        ax.axis('off')    

    def save_pickle(self, path):
        try:
            os.remove(path)
            print("File exist. Deleted")
        except FileNotFoundError:
            pass

        # Open a file and use dump()
        with open(path, 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        with open(path, 'rb') as file:
            return pickle.load(file)

        