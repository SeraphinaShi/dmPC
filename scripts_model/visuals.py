import os
import pandas as pd
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import torch


def plot_c_PCA_latent_help(c_data, c_latent_list, c_meta_hist, n_rounds, k=1, k_sub=None, plot_save_path=''):
    
    
    color_labels = np.array(list(map(str, c_meta_hist['code'].unique())))
    for b in range(n_rounds):
        color_labels = np.union1d(color_labels, c_meta_hist[f'code_b{b}'].unique())   
    color_values = sns.color_palette("Set2", 8)
    color_map = dict(zip(color_labels, color_values))

    if  k_sub != None:
        color_labels_sub = c_meta_hist[f'code_b{n_rounds - 1}'].astype(str).values
        for b in range(n_rounds):
            color_labels_sub = np.union1d(color_labels_sub, c_meta_hist[f'code_sub_b{b}'].unique())   
        color_values_sub = sns.color_palette("Set2", 8)
        color_map_sub = dict(zip(color_labels_sub, color_values_sub))

    pca = PCA(n_components=5)
    pca.fit(c_data)
    components = pca.transform(c_data)

    n_pics = n_rounds+1

    if k_sub == None:
        fig, ax = plt.subplots(nrows=1, ncols=n_pics, figsize=(n_pics*5, 5))

        ax[0].scatter(components[:,0],components[:,1], color=c_meta_hist['code'].astype(str).map(color_map))
        handlelist = [plt.plot([], marker="o", ls="", color=color)[0] for color in color_values]
        ax[0].legend(handlelist, color_labels, title="Prior cluster")
        ax[0].set_xlabel('pc1')
        ax[0].set_ylabel('pc2')
        ax[0].set_title(f'PCA on input data (k={k})')

        for b in range(n_rounds):
            n_p = b + 1

            data_b = c_latent_list[k][b]
            pca = PCA(n_components=5)
            pca.fit(data_b)
            components = pca.transform(data_b)

            ax[n_p].scatter(components[:,0],components[:,1], color = c_meta_hist[f'code_b{b}'].astype(str).map(color_map))
            handlelist = [plt.plot([], marker="o", ls="", color=color)[0] for color in color_values]
            ax[n_p].legend(handlelist, color_labels, title="Updated cluster")
            ax[n_p].set_xlabel('pc1')
            ax[n_p].set_xlabel('pc2')
            ax[n_p].set_title(f'PCA on latent space in round {b}, (k={k})')
    else:
        fig, ax = plt.subplots(nrows=2, ncols=n_pics, figsize=(n_pics*5, 2*5))

        ax[0,0].scatter(components[:,0],components[:,1], color=c_meta_hist['code'].astype(str).map(color_map))
        handlelist = [plt.plot([], marker="o", ls="", color=color)[0] for color in color_values]
        ax[0,0].legend(handlelist, color_labels, title="Prior cluster")
        ax[0,0].set_xlabel('pc1')
        ax[0,0].set_ylabel('pc2')
        ax[0,0].set_title(f'PCA on input data (k={k})')

        ax[1,0].scatter(components[:,0],components[:,1], color=c_meta_hist[f'code_b{n_rounds-1}'].astype(str).map(color_map_sub))
        handlelist = [plt.plot([], marker="o", ls="", color=color)[0] for color in color_values_sub]
        ax[1,0].legend(handlelist, color_labels_sub, title="Prior cluster")
        ax[1,0].set_xlabel('pc1')
        ax[1,0].set_ylabel('pc2')
        ax[1,0].set_title(f'PCA on input data (k={k_sub})')

        for b in range(n_rounds):
            n_p = b + 1

            data_b = c_latent_list[k][b]

            pca = PCA(n_components=5)
            pca.fit(data_b)
            components = pca.transform(data_b)

            ax[0, n_p].scatter(components[:,0],components[:,1], color = c_meta_hist[f'code_b{b}'].astype(str).map(color_map))
            handlelist = [plt.plot([], marker="o", ls="", color=color)[0] for color in color_values]
            ax[0, n_p].legend(handlelist, color_labels, title="Updated cluster")
            ax[0, n_p].set_xlabel('pc1')
            ax[0, n_p].set_xlabel('pc2')
            ax[0, n_p].set_title(f'PCA on latent space in round {b}, (k={k})')

            data_b_sub = c_latent_list[k_sub][b]

            pca_sub = PCA(n_components=5)
            pca_sub.fit(data_b_sub)
            components_sub = pca_sub.transform(data_b_sub)

            ax[1, n_p].scatter(components_sub[:,0],components_sub[:,1], color = c_meta_hist[f'code_sub_b{b}'].astype(str).map(color_map_sub))
            handlelist = [plt.plot([], marker="o", ls="", color=color)[0] for color in color_values_sub]
            ax[1, n_p].legend(handlelist, color_labels_sub, title="Updated cluster")
            ax[1, n_p].set_xlabel('pc1')
            ax[1, n_p].set_xlabel('pc2')
            ax[1, n_p].set_title(f'PCA on latent space in round {b}, (k={k})')

    # Plot c_data

    plt.tight_layout()
    
    if plot_save_path != '':
        if not os.path.exists(os.path.dirname(plot_save_path)):
            os.makedirs(os.path.dirname(plot_save_path))
        plt.savefig(plot_save_path, dpi=300)
        
    plt.show()


def plot_c_PCA_latent(c_data, n_rounds, fit_returns, model, plots_save_path):
    _, c_meta_hist, _, _, _, _, _, c_latent_list, _ = fit_returns
    
    for k in model.which_non_empty_cluster:
        if k in model.which_non_empty_subcluster:
            k_sub = model.cluster_id_for_subcluster[model.which_non_empty_subcluster.index(k)]
            plot_c_PCA_latent_help(c_data, c_latent_list, c_meta_hist, n_rounds, k=k, k_sub = k_sub, plot_save_path=f'{plots_save_path}_k{k}.png')
        else:
            plot_c_PCA_latent_help(c_data, c_latent_list, c_meta_hist, n_rounds, k=k, plot_save_path=f'{plots_save_path}_k{k}.png')


def plot_c_PCA_latent_old(c_data, c_latent_list, c_meta_hist, n_rounds, legend_title='cluster', k=1, plot_save_path=''):
    
    color_labels = np.array(list(map(str, c_meta_hist['code'].unique())))
    for b in range(n_rounds):
        color_labels = np.union1d(color_labels, c_meta_hist[f'code_b{b}'].unique())   
    color_values = sns.color_palette("Set2", 8)
    color_map = dict(zip(color_labels, color_values))

    n_pics = n_rounds+1
    fig, ax = plt.subplots(nrows=1, ncols=n_pics, figsize=(n_pics*5, 5))

    # Plot c_data
    pca = PCA(n_components=5)
    pca.fit(c_data)
    components = pca.transform(c_data)

    ax[0].scatter(components[:,0],components[:,1], color=c_meta_hist['code'].astype(str).map(color_map))
    handlelist = [plt.plot([], marker="o", ls="", color=color)[0] for color in color_values]
    ax[0].legend(handlelist, color_labels, title=legend_title)
    ax[0].set_xlabel('pc1')
    ax[0].set_ylabel('pc2')
    ax[0].set_title(f'Original clustering (k={k})')

    for b in range(n_rounds):
        n_p = b + 1

        data_b = c_latent_list[k][b]
        pca = PCA(n_components=5)
        pca.fit(data_b)
        components = pca.transform(data_b)

        ax[n_p].scatter(components[:,0],components[:,1], color = c_meta_hist[f'code_b{b}'].astype(str).map(color_map))
        handlelist = [plt.plot([], marker="o", ls="", color=color)[0] for color in color_values]
        ax[n_p].legend(handlelist, color_labels, title=legend_title)
        ax[n_p].set_xlabel('pc1')
        ax[n_p].set_xlabel('pc2')
        ax[n_p].set_title(f'Round {b + 1} updated clustering (k={k})')

    plt.tight_layout()
    
    if plot_save_path != '':
        plt.savefig(plot_save_path, dpi=300)
        
    plt.show()


def plot_c_PCA_latent_test(model, device, n_rounds, c_latent_list, c_train, c_test, cdr_train_rslt_cluster, cdr_test_rslt_cluster, k=1, plot_save_path=''):

    if k == -1:
        # prepare data
        c_data = pd.concat([c_train, c_test])
    else:
        c_latent_test = model.CDPmodel_list[k].c_VAE.encode(torch.from_numpy(c_test.values).float().to(device), repram=False)
        c_latent_test = c_latent_test.detach().numpy()
        c_latent_train = c_latent_list[k][n_rounds-1]

        c_latent = np.vstack((c_latent_train, c_latent_test))
        c_latent = pd.DataFrame(c_latent)
        c_latent.index = np.concatenate((c_train.index.values, c_test.index.values))
    

    cdr_train_rslt_cluster.index = cdr_train_rslt_cluster['c_name']
    c_meta_train_tmp = cdr_train_rslt_cluster
    c_meta_train_tmp['split'] = 'train' 

    cdr_test_rslt_cluster.index = cdr_test_rslt_cluster['c_name']
    c_meta_test_tmp = cdr_test_rslt_cluster
    c_meta_test_tmp['split'] = 'test'

    c_meta_tmp = pd.concat([c_meta_train_tmp, c_meta_test_tmp])
    if k == -1:
        c_meta_tmp = c_meta_tmp.loc[c_data.index.values,]
    else:
        c_meta_tmp = c_meta_tmp.loc[c_latent.index.values,]

    color_labels = np.array(c_meta_tmp['c_cluster'].astype(str).unique())
    color_labels = np.unique(np.concatenate((color_labels, np.array(['2']))))
    color_values = sns.color_palette("Set2", 8)
    color_map = dict(zip(color_labels, color_values))

    marker_map = {'train': 'o', 'test': '*'}

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    if k == -1:
        # 0. Plot c_data
        pca = PCA(n_components=5)
        pca.fit(c_data)
        components = pca.transform(c_data)

        x = components[:,0]
        y = components[:,1]
        m = c_meta_tmp['split'].map(marker_map)

        handlelist = [plt.plot([], marker="o", ls="", color=color)[0] for color in color_values]

        ## 0.0  color initial cluster
        c = c_meta_tmp['cluster_init'].astype(str).map(color_map).values
        for _m, _c, _x, _y in zip(m, c, x, y):
            ax[0].scatter(_x, _y, color=_c, marker=_m)
        ax[0].legend(handlelist, color_labels, title="cluster")
        ax[0].set_xlabel('pc1')
        ax[0].set_ylabel('pc2')
        ax[0].set_title(f'PCA on Data, initial cluster (k={k})')

        ## 0.1  color update cluster
        c = c_meta_tmp['c_cluster'].astype(str).map(color_map).values
        for _m, _c, _x, _y in zip(m, c, x, y):
            ax[1].scatter(_x, _y, color=_c, marker=_m)
        ax[1].legend(handlelist, color_labels, title="cluster")
        ax[1].set_xlabel('pc1')
        ax[1].set_ylabel('pc2')
        ax[1].set_title(f'PCA on Data, updated cluster (k={k})')

        ## 0.0  color true cluster
        c = c_meta_tmp['cluster_true'].astype(str).map(color_map).values
        for _m, _c, _x, _y in zip(m, c, x, y):
            ax[2].scatter(_x, _y, color=_c, marker=_m)
        ax[2].legend(handlelist, color_labels, title="cluster")
        ax[2].set_xlabel('pc1')
        ax[2].set_ylabel('pc2')
        ax[2].set_title(f'PCA on Data, true cluster (k={k})')

    else:
        # Plot c_latent
        pca = PCA(n_components=5)
        pca.fit(c_latent)
        components = pca.transform(c_latent)

        x = components[:,0]
        y = components[:,1]
        m = c_meta_tmp['split'].map(marker_map)

        handlelist = [plt.plot([], marker="o", ls="", color=color)[0] for color in color_values]

        ## 0.0  color initial cluster
        c = c_meta_tmp['cluster_init'].astype(str).map(color_map).values
        for _m, _c, _x, _y in zip(m, c, x, y):
            ax[0].scatter(_x, _y, color=_c, marker=_m)
        ax[0].legend(handlelist, color_labels, title="cluster")
        ax[0].set_xlabel('pc1')
        ax[0].set_ylabel('pc2')
        ax[0].set_title(f'PCA on Latent, initial cluster (k={k})')

        ## 0.1  color update cluster
        c = c_meta_tmp['c_cluster'].astype(str).map(color_map).values
        for _m, _c, _x, _y in zip(m, c, x, y):
            ax[1].scatter(_x, _y, color=_c, marker=_m)
        ax[1].legend(handlelist, color_labels, title="cluster")
        ax[1].set_xlabel('pc1')
        ax[1].set_ylabel('pc2')
        ax[1].set_title(f'PCA on Latent, updated cluster (k={k})')

        ## 0.0  color true cluster
        c = c_meta_tmp['cluster_true'].astype(str).map(color_map).values
        for _m, _c, _x, _y in zip(m, c, x, y):
            ax[2].scatter(_x, _y, color=_c, marker=_m)
        ax[2].legend(handlelist, color_labels, title="cluster")
        ax[2].set_xlabel('pc1')
        ax[2].set_ylabel('pc2')
        ax[2].set_title(f'PCA on Latent, true cluster (k={k})')

    plt.tight_layout()
    
    if plot_save_path != '':
        plt.savefig(plot_save_path, dpi=300)
        
    plt.show()



def plot_d_PCA_latent_help(d_data, d_latent_list, d_sens_hist, n_rounds, k=1, k_sub=None, plot_save_path=''):
    
    color_labels = np.array(list(map(str, d_sens_hist['sensitive_k'].unique())))
    for b in range(n_rounds):
        color_labels = np.union1d(color_labels, d_sens_hist[f'sensitive_k_b{b}'].unique())   
    color_values = sns.color_palette("Set2", 8)
    color_map = dict(zip(color_labels, color_values))

    if  k_sub != None:
        color_labels_sub = d_sens_hist[f'sensitive_k_b{n_rounds - 1}'].astype(str).values
        for b in range(n_rounds):
            color_labels_sub = np.union1d(color_labels_sub, d_sens_hist[f'sensitive_k_sub_b{b}'].unique())   
        color_values_sub = sns.color_palette("Set2", 8)
        color_map_sub = dict(zip(color_labels_sub, color_values_sub))


    n_pics = n_rounds+1

    pca = PCA(n_components=5)
    pca.fit(d_data)
    components = pca.transform(d_data)

    if k_sub == None:
        fig, ax = plt.subplots(nrows=1, ncols=n_pics, figsize=(n_pics*5, 5))

        ax[0].scatter(components[:,0],components[:,1], color=d_sens_hist['sensitive_k'].astype(str).map(color_map))
        handlelist = [plt.plot([], marker="o", ls="", color=color)[0] for color in color_values]
        ax[0].legend(handlelist, color_labels, title="Prior cluster")
        ax[0].set_xlabel('pc1')
        ax[0].set_ylabel('pc2')
        ax[0].set_title(f'PCA on input data (k={k})')


        for b in range(n_rounds):
            n_p = b + 1

            data_b = d_latent_list[k][b]
            pca = PCA(n_components=5)
            pca.fit(data_b)
            components = pca.transform(data_b)

            ax[n_p].scatter(components[:,0],components[:,1], color = d_sens_hist[f'sensitive_k_b{b}'].astype(str).map(color_map))
            handlelist = [plt.plot([], marker="o", ls="", color=color)[0] for color in color_values]
            ax[n_p].legend(handlelist, color_labels, title="Updated cluster")
            ax[n_p].set_xlabel('pc1')
            ax[n_p].set_xlabel('pc2')
            ax[n_p].set_title(f'PCA on latent space in round {b}, (k={k})')

    else:
        fig, ax = plt.subplots(nrows=2, ncols=n_pics, figsize=(n_pics*5, 2*5))

        ax[0,0].scatter(components[:,0],components[:,1], color=d_sens_hist['sensitive_k'].astype(str).map(color_map))
        handlelist = [plt.plot([], marker="o", ls="", color=color)[0] for color in color_values]
        ax[0,0].legend(handlelist, color_labels, title="Prior cluster")
        ax[0,0].set_xlabel('pc1')
        ax[0,0].set_ylabel('pc2')
        ax[0,0].set_title(f'PCA on input data (k={k})')

        ax[1,0].scatter(components[:,0],components[:,1], color=d_sens_hist[f'sensitive_k_b{n_rounds-1}'].astype(str).map(color_map_sub))
        handlelist = [plt.plot([], marker="o", ls="", color=color)[0] for color in color_values_sub]
        ax[1,0].legend(handlelist, color_labels_sub, title="Prior cluster")
        ax[1,0].set_xlabel('pc1')
        ax[1,0].set_ylabel('pc2')
        ax[1,0].set_title(f'PCA on input data (k={k_sub})')

        for b in range(n_rounds):
            n_p = b + 1

            data_b = d_latent_list[k][b]

            pca = PCA(n_components=5)
            pca.fit(data_b)
            components = pca.transform(data_b)

            ax[0, n_p].scatter(components[:,0],components[:,1], color = d_sens_hist[f'sensitive_k_b{b}'].astype(str).map(color_map))
            handlelist = [plt.plot([], marker="o", ls="", color=color)[0] for color in color_values]
            ax[0, n_p].legend(handlelist, color_labels, title="Updated cluster")
            ax[0, n_p].set_xlabel('pc1')
            ax[0, n_p].set_xlabel('pc2')
            ax[0, n_p].set_title(f'PCA on latent space in round {b}, (k={k})')

            data_b_sub = d_latent_list[k_sub][b]

            pca_sub = PCA(n_components=5)
            pca_sub.fit(data_b_sub)
            components_sub = pca_sub.transform(data_b_sub)

            ax[1, n_p].scatter(components_sub[:,0],components_sub[:,1], color = d_sens_hist[f'sensitive_k_sub_b{b}'].astype(str).map(color_map_sub))
            handlelist = [plt.plot([], marker="o", ls="", color=color)[0] for color in color_values_sub]
            ax[1, n_p].legend(handlelist, color_labels_sub, title="Updated cluster")
            ax[1, n_p].set_xlabel('pc1')
            ax[1, n_p].set_xlabel('pc2')
            ax[1, n_p].set_title(f'PCA on latent space in round {b}, (k={k_sub})')

    # Plot c_data

    plt.tight_layout()
    
    if plot_save_path != '':
        plt.savefig(plot_save_path, dpi=300)
        
    plt.show()


def plot_d_PCA_latent(d_data, n_rounds, fit_returns, model, plots_save_path):
    _, _, d_sens_hist, _, _, _, _, _, d_latent_list = fit_returns
    
    for k in model.which_non_empty_cluster:
        if k in model.which_non_empty_subcluster:
            k_sub = model.cluster_id_for_subcluster[model.which_non_empty_subcluster.index(k)]
            plot_d_PCA_latent_help(d_data, d_latent_list, d_sens_hist, n_rounds, k=k, k_sub = k_sub, plot_save_path=f'{plots_save_path}_k{k}.png')
        else:
            plot_d_PCA_latent_help(d_data, d_latent_list, d_sens_hist, n_rounds, k=k, plot_save_path=f'{plots_save_path}_k{k}.png')


def plot_d_PCA_latent_old(d_data, d_latent_list, d_sens_hist, n_rounds, legend_title='cluster', k=0, plot_save_path=''):
    
    color_labels = np.array(list(map(str, d_sens_hist['sensitive_k'].unique())))
    for b in range(n_rounds):
        color_labels = np.union1d(color_labels, d_sens_hist[f'sensitive_k_b{b}'].unique())   
    color_values = sns.color_palette("Set2", 8)
    color_map = dict(zip(color_labels, color_values))

    n_pics = n_rounds+1
    fig, ax = plt.subplots(nrows=1, ncols=n_pics, figsize=(n_pics*5, 5))

    # Plot c_data
    pca = PCA(n_components=5)
    pca.fit(d_data)
    components = pca.transform(d_data)

    ax[0].scatter(components[:,0],components[:,1], color=d_sens_hist['sensitive_k'].astype(str).map(color_map))
    handlelist = [plt.plot([], marker="o", ls="", color=color)[0] for color in color_values]
    ax[0].legend(handlelist, color_labels, title=legend_title)
    ax[0].set_xlabel('pc1')
    ax[0].set_ylabel('pc2')
    ax[0].set_title(f'Original clustering (k={k})')

    for b in range(n_rounds):
        n_p = b + 1

        data_b = d_latent_list[k][b]
        pca = PCA(n_components=5)
        pca.fit(data_b)
        components = pca.transform(data_b)

        ax[n_p].scatter(components[:,0],components[:,1], color = d_sens_hist[f'sensitive_k_b{b}'].astype(str).map(color_map))
        handlelist = [plt.plot([], marker="o", ls="", color=color)[0] for color in color_values]
        ax[n_p].legend(handlelist, color_labels, title=legend_title)
        ax[n_p].set_xlabel('pc1')
        ax[n_p].set_xlabel('pc2')
        ax[n_p].set_title(f'Round {b + 1} updated clustering (k={k})')

    plt.tight_layout()
    
    if plot_save_path != '':
        plt.savefig(plot_save_path, dpi=300)
        
    plt.show()




def plot_training_losses(losses_train_hist_list_1round, train_hist = True, test_hist = True, best_epoch_1round = []):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('')
    
    losses = losses_train_hist_list_1round[0]
    if train_hist:
        axs[0,0].plot(np.array(losses["epoch"]), np.array(losses["loss_train"]), label = "total loss");
        axs[0,0].plot(np.array(losses["epoch"]), np.array(losses["vae_loss_train"]), label = "vae loss");
        axs[0,0].plot(np.array(losses["epoch"]), np.array(losses["recon_loss_train"]), label = "recon loss");
        axs[0,0].plot(np.array(losses["epoch"]), np.array(losses["kld_train"]), label = "kld");
        axs[0,0].plot(np.array(losses["epoch"]), np.array(losses["latent_d_loss_train"]), label = "latent distance loss");
        axs[0,0].plot(np.array(losses["epoch"]), np.array(losses["prediction_loss_train"]), label = "prediction loss");
    if test_hist:
        axs[0,0].plot(np.array(losses["epoch"]), np.array(losses["loss_test"]), label = "total loss");
        axs[0,0].plot(np.array(losses["epoch"]), np.array(losses["vae_loss_test"]), label = "vae loss");
        axs[0,0].plot(np.array(losses["epoch"]), np.array(losses["recon_loss_test"]), label = "recon loss");
        axs[0,0].plot(np.array(losses["epoch"]), np.array(losses["kld_test"]), label = "kld");
        axs[0,0].plot(np.array(losses["epoch"]), np.array(losses["latent_d_loss_test"]), label = "latent distance loss");
        axs[0,0].plot(np.array(losses["epoch"]), np.array(losses["prediction_loss_test"]), label = "prediction loss");
    if best_epoch_1round != []:
        axs[0,0].axvline(x=best_epoch_1round[0], color='r', linestyle='--')
    axs[0,0].set_title('(a) C-VAE losses')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    losses = losses_train_hist_list_1round[1]
    if train_hist:
        axs[1,0].plot(np.array(losses["epoch"]), np.array(losses["loss_train"]), label = "total loss");
        axs[1,0].plot(np.array(losses["epoch"]), np.array(losses["vae_loss_train"]), label = "vae loss");
        axs[1,0].plot(np.array(losses["epoch"]), np.array(losses["recon_loss_train"]), label = "recon loss");
        axs[1,0].plot(np.array(losses["epoch"]), np.array(losses["kld_train"]), label = "kld");
        axs[1,0].plot(np.array(losses["epoch"]), np.array(losses["latent_d_loss_train"]), label = "latent distance loss");
        axs[1,0].plot(np.array(losses["epoch"]), np.array(losses["prediction_loss_train"]), label = "prediction loss");

    if test_hist:
        axs[1,0].plot(np.array(losses["epoch"]), np.array(losses["loss_test"]), label = "total loss");
        axs[1,0].plot(np.array(losses["epoch"]), np.array(losses["vae_loss_test"]), label = "vae loss");
        axs[1,0].plot(np.array(losses["epoch"]), np.array(losses["recon_loss_test"]), label = "recon loss");
        axs[1,0].plot(np.array(losses["epoch"]), np.array(losses["kld_test"]), label = "kld");
        axs[1,0].plot(np.array(losses["epoch"]), np.array(losses["latent_d_loss_test"]), label = "latent distance loss");
        axs[1,0].plot(np.array(losses["epoch"]), np.array(losses["prediction_loss_test"]), label = "prediction loss");
    if best_epoch_1round != []:
        axs[1,0].axvline(x=best_epoch_1round[1], color='r', linestyle='--')
    axs[1,0].set_title('(c) D-VAE & Predictor losses')
    # axs[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.show()


def plot_training_losses_train_test_2cols(losses_train_hist_list_1round, best_epoch_1round = [], plot_save_path=''):
    
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 5))
    fig.suptitle('')
    
    losses = losses_train_hist_list_1round[0]
    axs[0,0].plot(np.array(losses["epoch"]), np.array(losses["loss_train"]), label = "total loss");
    axs[0,0].plot(np.array(losses["epoch"]), np.array(losses["prediction_loss_train"]), label = "prediction loss");
    axs[0,0].plot(np.array(losses["epoch"]), np.array(losses["update_overlap_train"]), label = "cluster overlap loss");
    axs[0,0].plot(np.array(losses["epoch"]), np.array(losses["vae_loss_train"]), label = "vae loss");
    axs[0,0].plot(np.array(losses["epoch"]), np.array(losses["recon_loss_train"]), label = "recon loss");
    axs[0,0].plot(np.array(losses["epoch"]), np.array(losses["kld_train"]), label = "kld");
    axs[0,0].plot(np.array(losses["epoch"]), np.array(losses["latent_dist_loss_train"]), label = "latent distance loss");

    if best_epoch_1round != []:
        axs[0,0].axvline(x=best_epoch_1round[0], color='r', linestyle='--')
    axs[0,0].set_title('(a) D-VAE & Predictor losses [train]')
    
    axs[0,1].plot(np.array(losses["epoch"]), np.array(losses["loss_test"]), label = "total loss");
    axs[0,1].plot(np.array(losses["epoch"]), np.array(losses["prediction_loss_test"]), label = "prediction loss");
    axs[0,1].plot(np.array(losses["epoch"]), np.array(losses["update_overlap_test"]), label = "cluster overlap loss");
    axs[0,1].plot(np.array(losses["epoch"]), np.array(losses["vae_loss_test"]), label = "vae loss");
    axs[0,1].plot(np.array(losses["epoch"]), np.array(losses["recon_loss_test"]), label = "recon loss");
    axs[0,1].plot(np.array(losses["epoch"]), np.array(losses["kld_test"]), label = "kld");
    axs[0,1].plot(np.array(losses["epoch"]), np.array(losses["latent_dist_loss_test"]), label = "latent distance loss");
    if best_epoch_1round != []:
        axs[0,1].axvline(x=best_epoch_1round[0], color='r', linestyle='--')
    axs[0,1].set_title('(a) D-VAE + Predictor losses [test]')
    axs[0,1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    losses = losses_train_hist_list_1round[1]
    axs[1,0].plot(np.array(losses["epoch"]), np.array(losses["loss_train"]), label = "total loss");
    axs[1,0].plot(np.array(losses["epoch"]), np.array(losses["prediction_loss_train"]), label = "prediction loss");
    axs[1,0].plot(np.array(losses["epoch"]), np.array(losses["update_overlap_train"]), label = "cluster overlap loss");
    axs[1,0].plot(np.array(losses["epoch"]), np.array(losses["vae_loss_train"]), label = "vae loss");
    axs[1,0].plot(np.array(losses["epoch"]), np.array(losses["recon_loss_train"]), label = "recon loss");
    axs[1,0].plot(np.array(losses["epoch"]), np.array(losses["kld_train"]), label = "kld");
    axs[1,0].plot(np.array(losses["epoch"]), np.array(losses["latent_dist_loss_train"]), label = "latent distance loss");
    if best_epoch_1round != []:
            axs[1,0].axvline(x=best_epoch_1round[1], color='r', linestyle='--')
    axs[1,0].set_title('(c) C-VAE & Predictor losses [train]')
    
    axs[1,1].plot(np.array(losses["epoch"]), np.array(losses["loss_test"]), label = "total loss");
    axs[1,1].plot(np.array(losses["epoch"]), np.array(losses["prediction_loss_test"]), label = "prediction loss");
    axs[1,1].plot(np.array(losses["epoch"]), np.array(losses["update_overlap_test"]), label = "cluster overlap loss");
    axs[1,1].plot(np.array(losses["epoch"]), np.array(losses["vae_loss_test"]), label = "vae loss");
    axs[1,1].plot(np.array(losses["epoch"]), np.array(losses["recon_loss_test"]), label = "recon loss");
    axs[1,1].plot(np.array(losses["epoch"]), np.array(losses["kld_test"]), label = "kld");
    axs[1,1].plot(np.array(losses["epoch"]), np.array(losses["latent_dist_loss_test"]), label = "latent distance loss");
    if best_epoch_1round != []:
        axs[1,1].axvline(x=best_epoch_1round[1], color='r', linestyle='--')
    axs[1,1].set_title('(c) C-VAE & Predictor losses [test]')
    axs[1,1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()

    if plot_save_path != '':
        plt.savefig(plot_save_path, dpi=300)

    plt.show()


def plot_predict_training_losses_train_test_2cols(losses_train_hist_list_1round, best_epoch_1round = [], plot_save_path=''):
    
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    fig.suptitle('')
    
    losses = losses_train_hist_list_1round[0]
    axs[0,0].plot(np.array(losses["epoch"]), np.array(losses["prediction_loss_train"]), label = "prediction loss (train)");
    if best_epoch_1round != []:
            axs[0,0].axvline(x=best_epoch_1round[0], color='r', linestyle='--')
    axs[0,0].set_title('(a) D-VAE & Predictor losses [train]')

    
    axs[0,1].plot(np.array(losses["epoch"]), np.array(losses["prediction_loss_test"]), label = "prediction loss (test)");
    if best_epoch_1round != []:
        axs[0,1].axvline(x=best_epoch_1round[0], color='r', linestyle='--')
    axs[0,1].set_title('(a) D-VAE & Predictor losses [test]')
    axs[0,1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    

    losses = losses_train_hist_list_1round[1]
    axs[1,0].plot(np.array(losses["epoch"]), np.array(losses["prediction_loss_train"]), label = "prediction loss (train)");
    if best_epoch_1round != []:
            axs[1,0].axvline(x=best_epoch_1round[1], color='r', linestyle='--')
    axs[1,0].set_title('(c) C-VAE & Predictor losses [train]')
    
    axs[1,1].plot(np.array(losses["epoch"]), np.array(losses["prediction_loss_test"]), label = "prediction loss (test)");
    if best_epoch_1round != []:
        axs[1,1].axvline(x=best_epoch_1round[1], color='r', linestyle='--')
    axs[1,1].set_title('(c) C-VAE & Predictor losses [test]')
    axs[1,1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()

    if plot_save_path != '':
        plt.savefig(plot_save_path, dpi=300)

    plt.show()






