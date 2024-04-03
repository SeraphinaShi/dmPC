from cmath import nan
# import selectors
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.utils
import torch.distributions
import numpy as np

import pandas as pd
import itertools

import traceback

import copy
from copy import deepcopy
import logging
import os

import random


import multiprocessing

from statistics import mean

from trainers import *

class CDPmodel(nn.Module):
    def __init__(self, K, params):                
        super(CDPmodel, self).__init__()

        self.K = K
        self.original_K = K
        self.n_cluser = K

        CDPmodel_list = []
        for k in range(0,self.K):
            CDPmodel_list.append(CDPmodel_sub(params)) 
        
        CDPmodel_list_sub = []
        for k in range(0,self.K):
            CDPmodel_list_sub.append(CDPmodel_sub(params)) 
        
        self.CDPmodel_list = CDPmodel_list
        self.CDPmodel_list_sub = CDPmodel_list_sub
        
        self.which_non_empty_cluster = []
        self.subcluster_map_dict = MyMap()

        self.c_centroids = [None] * self.K
        self.c_sds = [None] * self.K
        self.c_in_trainnig = []
        self.c_name_clusters_in_trainnig = [None] * self.K
        self.d_centroids = [None] * self.K
        self.d_sds = [None] * self.K
        self.d_in_trainnig = []
        self.d_name_clusters_in_trainnig = [None] * self.K

        self.sens_cutoff = params['sens_cutoff']
        

    def forward(self, c_X, d_X, c_name: object = '', d_name: object = '', k: int = -1, sd_scale: float = 3, device = 'cpu'):
        if not isinstance(c_X, torch.Tensor):
                c_X = torch.FloatTensor(c_X.values)
        if not isinstance(c_X, torch.Tensor):
                c_X = torch.FloatTensor(c_X.values)
        if k == -1:
            c_cluster = []
            d_cluster = []
            cluster = []

            # find the cluster of C
            if c_name in self.c_in_trainnig:
                for k_itr in self.which_non_empty_cluster :
                    if c_name in self.c_name_clusters_in_trainnig[k_itr]:
                        c_cluster.append(k_itr)
        
            else:
                # search through clusters
                c_dists = []
                for k_itr in self.which_non_empty_cluster:
                    c_mu = self.CDPmodel_list[k_itr].c_VAE.encode(c_X, repram=False)
                    dist = c_mu.to(device) - self.c_centroids[k_itr].to(device)
                    is_outlier = (dist.abs() > sd_scale * self.c_sds[k_itr].to(device)).any().item()
                    if is_outlier:
                        c_dist_k = torch.tensor(float('inf'))
                    else: 
                        c_dist_k = (dist / (self.c_sds[k_itr].to(device))).norm()
                    c_dists.append(c_dist_k)

                stacked_c_dists = torch.stack(c_dists)

                if stacked_c_dists.min().item() < float('inf'):
                    c_cluster_tep = self.which_non_empty_cluster[torch.argmin(stacked_c_dists).item()]
                    c_cluster.append(c_cluster_tep)

                         
            # find the cluster of D
            if d_name in self.d_in_trainnig:

                for k_itr in self.which_non_empty_cluster:
                    if d_name in self.d_name_clusters_in_trainnig[k_itr]:
                        d_cluster.append(k_itr)
                
            else:
                # search through clusters
                d_dists = []
                for k_itr in self.which_non_empty_cluster:
                    d_mu = self.CDPmodel_list[k_itr].d_VAE.encode(d_X, repram=False).to(device).to(device)
                    dist = d_mu - self.d_centroids[k_itr].to(device)
                    is_outlier = (dist.abs() > sd_scale * self.d_sds[k_itr].to(device)).any().item()
                    if is_outlier:
                        d_dist_k = torch.tensor(float('inf'))
                    else: 
                        d_dist_k = (dist / (self.d_sds[k_itr].to(device))).norm()
                    d_dists.append(d_dist_k)

                stacked_d_dists = torch.stack(d_dists)

                if stacked_d_dists.min().item() < float('inf'):
                    d_cluster_tep = self.which_non_empty_cluster[torch.argmin(stacked_d_dists).item()]
                    d_cluster.append(d_cluster_tep)


                
            if len(d_cluster) == 0 and len(c_cluster) == 0:
                d_cluster.append(-1)    # check through subclusters
                c_cluster.append(-1)    # check through subclusters
                CDR = 0
                cluster = -1
            else: 
                # Loop through clusters of C and D to find CD bicluster
                CDR_tmp_list = []
                
                cd_clusters = list(set(c_cluster).intersection(set(d_cluster)))
                    
                if len(cd_clusters) == 0:
                    cd_clusters = list(set(c_cluster + d_cluster))
                    print(f"find c OR d cluster: {cd_clusters}, where c cluster {c_cluster} c name {c_name} and d {d_cluster} {d_name}")
                else:
                    print(f"find cd cluster: {cd_clusters}, where c cluster {c_cluster} c name {c_name} and d {d_cluster} {d_name}")

                
                if len(c_cluster) == 0:
                    c_cluster.append(-1)
                if len(d_cluster) == 0:
                    d_cluster.append(-1)

                for k in cd_clusters:
                    if k != -1:
                        CDR_temp = self.predict_given_model(self.CDPmodel_list[k], c_X, d_X, device)
                        CDR_tmp_list.append(CDR_temp)
                
                if len(CDR_tmp_list) == 0:
                    CDR_tmp_list.append(0)
            
                CDR = format_list_as_string(CDR_tmp_list)
                cluster = format_list_as_string(cd_clusters)

            c_cluster = format_list_as_string(c_cluster)
            d_cluster = format_list_as_string(d_cluster)
            
        else:
            cluster = k
            c_cluster = k
            d_cluster = k
            CDR = self.predict_given_model(self.CDPmodel_list[cluster], c_X, d_X, device)

        return CDR, cluster, c_cluster, d_cluster


    def predict_given_model(self, local_model, c_X: pd.DataFrame, d_X: pd.DataFrame, device = 'cpu'):
        local_model = local_model.to(device)
        c_mu = local_model.c_VAE.encode(c_X.to(device), repram=False).to(device)
        d_mu = local_model.d_VAE.encode(d_X.to(device), repram=False).to(device)
        CDR_temp = local_model.predictor(c_mu, d_mu).to(device)
        CDR_temp = round(CDR_temp.item(), 6)
            
        return CDR_temp


    def predict(self, c_X: pd.DataFrame, d_X: pd.DataFrame, k: int = -1, sd_scale: float = 3, device = 'cpu'):
        c_names = c_X.index.values
        d_names = d_X.index.values
        combinations = list(itertools.product(c_names, d_names))
        CDR_df = pd.DataFrame(combinations, columns=['c_name', 'd_name'])
        CDR_df['cdr_hat'] = None
        CDR_df['cdr_all'] = None
        CDR_df['cluster'] = k
        CDR_df['c_cluster'] = k
        CDR_df['d_cluster'] = k

        for index, row in CDR_df.iterrows():
            c_name = row['c_name']
            d_name = row['d_name']
            k = int(row['cluster'])
            c_X_tensor = torch.from_numpy(c_X.loc[c_name].values).float().view(1, -1)
            d_X_tensor = torch.from_numpy(d_X.loc[d_name].values).float().view(1, -1)

            cdr_hat, cluster,c_cluster, d_cluster = self(c_X_tensor, d_X_tensor, c_name, d_name, k, sd_scale = sd_scale, device = device)

            if isinstance(cdr_hat, str):
                numbers = [float(num.strip()) for num in cdr_hat.split(' & ')]
                cdr_mean = sum(numbers) / len(numbers)
            else:
                cdr_mean = cdr_hat
            
            CDR_df.at[index, 'cdr_all'] = cdr_hat
            CDR_df.at[index, 'cdr_hat'] = cdr_mean
            CDR_df.at[index, 'cluster'] = cluster
            CDR_df.at[index, 'c_cluster'] = c_cluster
            CDR_df.at[index, 'd_cluster'] = d_cluster
            
        return CDR_df


    def fit(self, c_data, c_meta, d_data, cdr, train_params, n_rounds=3, search_subcluster = True, device='cpu', seed = 42):
        
        random.seed(seed)
        np.random.seed(seed)
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        c_meta_hist = c_meta.copy()
        d_sens_hist = pd.DataFrame() 
        losses_train_hist_list = []
        best_epos_list = []

        return_latents = True
        if return_latents:
            c_latent_list = []
            d_latent_list = []

        self.c_in_trainnig = c_data.index.values
        self.d_in_trainnig = d_data.index.values

        # 1. Pre-train a C-VAE and D-VAE on all cells and compounds (no clustering loss).  Copy each k-times as each initial bi-cluster VAE.
        print(f"=> Initialize C-VAE:")
        C_VAE, C_VAE_init_losses = train_VAE(self.CDPmodel_list[0].c_VAE, device, c_data, vae_type = "C", save_path = train_params['cVAE_save_path'], params=train_params)
        print(f"=> Initialize D-VAE:")
        D_VAE, D_VAE_init_losses = train_VAE(self.CDPmodel_list[0].d_VAE, device, d_data, vae_type = "D", save_path = train_params['dVAE_save_path'], params=train_params)

        # Assign C-VAE and D-VAE to each CDP model
        # Copy over the parameters
        for k in range(0,self.K):
            self.CDPmodel_list[k].c_VAE.load_state_dict(C_VAE.state_dict())
            self.CDPmodel_list[k].d_VAE.load_state_dict(D_VAE.state_dict())

        # def train_k():
        for k in range(0,self.K):
            print(f"########################################################")
            print(f"#### {k}. k = {k}                                     ")      
            print(f"########################################################")
            print(f"  ===================================")
            print(f"  === {k}.1. Training local CDP model ")
            print(f"  ===================================")

            losses_train_hist_list_k = []
            best_epos_k = []

            if return_latents:
                c_latent_k = []
                d_latnet_k = []

            meta_key = "k" + str(k)
            c_meta_k = c_meta[[meta_key]].rename(columns={meta_key:'key'})

            # 1. Run the dual loop to train local models
            for b in range(0, n_rounds):
                print(f"     -- round {b} -------------")    
 
                if b == 0:
                    d_sens_hist[f'sensitive_k{k}'] = (cdr.loc[c_meta_k.index.values[c_meta_k.key == 1]].mean(axis=0) > self.sens_cutoff).astype(int)
                    d_names_k_init = d_sens_hist.index.values[d_sens_hist[f'sensitive_k{k}']==1]

                    sensitive_cut_off = 0.35
                else:
                    d_names_k_init = d_sens_hist.index.values[d_sens_hist[f'sensitive_k{k}_b{b-1}']==1]

                    sensitive_cut_off = self.sens_cutoff
                
                c_names_k_init = c_meta_k.index.values[c_meta_k.key == 1]                  

                zero_cluster, c_centroid, d_centroid, c_sd, d_sd, c_name_cluster_k, d_name_sensitive_k, losses_train_hist, \
                    best_epos, c_meta_k, d_sens_k = train_CDPmodel_local_1round(self.CDPmodel_list[k], device, False, c_data, c_meta_k, d_data, \
                                                                                cdr, c_names_k_init, d_names_k_init, sensitive_cut_off, k, train_params)
                
                if zero_cluster:
                    # store/update the centroids
                    self.c_centroids[k] = None
                    self.c_sds[k] = None
                    self.d_centroids[k] = None
                    self.d_sds[k] = None
                    self.c_name_clusters_in_trainnig[k] = None
                    self.d_name_clusters_in_trainnig[k] = None

                    # returns
                    c_meta_hist[f'k{k}_b{b}'] = None
                    d_sens_hist[f'sensitive_k{k}_b{b}'] = None

                    losses_train_hist_list.append(None)
                    best_epos_list.append(None)

                    if return_latents:
                        c_latent_list.append(None)
                        d_latent_list.append(None)

                    break

                else:
                    # store the centroids
                    self.c_centroids[k] = c_centroid
                    self.c_sds[k] = c_sd
                    self.d_centroids[k] = d_centroid
                    self.d_sds[k] = d_sd
                    self.c_name_clusters_in_trainnig[k] = c_name_cluster_k
                    self.d_name_clusters_in_trainnig[k] = d_name_sensitive_k

                    # returns
                    c_meta[meta_key] = c_meta_k.key
                    c_meta_hist[f'k{k}_b{b}'] = c_meta_k.key
                    d_sens_hist[f'sensitive_k{k}_b{b}'] = d_sens_k.sensitive
                    losses_train_hist_list_k.append(losses_train_hist)
                    best_epos_k.append(best_epos)

                    if return_latents:
                        c_latent = self.CDPmodel_list[k].c_VAE.encode(torch.from_numpy(c_data.values).float().to(device), repram=False)
                        c_latent_k.append(c_latent.detach().cpu().numpy())
                        d_latent = self.CDPmodel_list[k].d_VAE.encode(torch.from_numpy(d_data.values).float().to(device), repram=False)
                        d_latnet_k.append(d_latent.detach().cpu().numpy())

                    if b == n_rounds - 1:
                        self.which_non_empty_cluster.append(k)

                        losses_train_hist_list.append(losses_train_hist_list_k)
                        best_epos_list.append(best_epos_k)

                        if return_latents:
                            c_latent_list.append(c_latent_k)
                            d_latent_list.append(d_latnet_k)
                
                ## Use multiprocessing to run the loop in parallel for each k
                # with multiprocessing.Pool() as pool:
                # s    pool.map(train_k, range(0, self.K))
            

            if search_subcluster and not zero_cluster:
                # ---------------------------------------------
                # 2. Run the dual loop again to find subclusters
                print(f"  ===================================")
                print(f"  === {k}.2. sub local CDP model      ")
                print(f"  ===================================")

                zero_cluster_sub, c_meta_hist, d_sens_hist = self.find_subcluster(
                    k, c_data, d_data, cdr, c_meta_k,
                    n_rounds, train_params, return_latents, 
                    c_meta_hist, d_sens_hist, losses_train_hist_list, best_epos_list,
                    device, c_latent_list, d_latent_list)
                

                while not zero_cluster_sub:
                    print(f"   ---------------------------------")
                    print(f"   try to find another subcluster")
                    print(f"   ---------------------------------")
                    zero_cluster_sub, c_meta_hist, d_sens_hist = self.find_subcluster(
                        k, c_data, d_data, cdr, c_meta_k,
                        n_rounds, train_params, return_latents, 
                        c_meta_hist, d_sens_hist, losses_train_hist_list, best_epos_list,
                        device, c_latent_list, d_latent_list)


        c_meta_hist = add_meta_code(c_meta_hist, self.K, n_rounds)
        d_sens_hist = add_sensk_to_d_sens_init(d_sens_hist, self.original_K)
        d_sens_hist = add_sensk_to_d_sens_hist(d_sens_hist, self.K, n_rounds)
        
        c_meta_hist['code_latest'] = c_meta_hist[f'code_b{n_rounds-1}']
        d_sens_hist['sensitive_k_latest'] = d_sens_hist[f'sensitive_k_b{n_rounds-1}']

        if search_subcluster:
            print(f"########################################################")
            print(f"#### Check all subclusters                              ")      
            print(f"########################################################")
            all_k_with_subcuster_str = self.subcluster_map_dict.get_all_keys()
            all_k_with_subcuster = [int(x) for x in all_k_with_subcuster_str]

            for k in all_k_with_subcuster:
                subcluster_id = self.subcluster_map_dict.get_from_map(str(k))
                for k_sub in subcluster_id:

                    print(f" - Cluster {k} found a subcluster with cluster ID: {k_sub}")
                    
        #             for b in range(0, n_rounds):
        #                 c_meta_hist[f'k{k_sub}_b{b}'] = c_meta_hist[f'k{k}_sub_b{b}']
        #                 d_sens_hist[f'sensitive_k{k_sub}_b{b}'] = d_sens_hist[f'sensitive_k{k}_sub_b{b}']
                
            
        #     c_meta_hist = add_meta_code_with_subcluster(c_meta_hist, self.K, n_rounds)
        #     d_sens_hist = add_sensk_to_d_sens_hist_with_subcluster(d_sens_hist, self.K, n_rounds)
        
        #     c_meta_hist['code_sub_latest'] = c_meta_hist[f'code_sub_b{n_rounds-1}']
        #     d_sens_hist['sensitive_k_sub_latest'] = d_sens_hist[f'sensitive_k_sub_b{n_rounds-1}']

        ## Use multiprocessing to run the loop in parallel for each k
        # with multiprocessing.Pool() as pool:
        # s    pool.map(train_k, range(0, self.K))

        if return_latents:
            return c_meta, c_meta_hist, d_sens_hist, losses_train_hist_list, best_epos_list, C_VAE_init_losses, D_VAE_init_losses, c_latent_list, d_latent_list
        else: 
            return c_meta, c_meta_hist, d_sens_hist, losses_train_hist_list, best_epos_list, C_VAE_init_losses, D_VAE_init_losses

    def find_subcluster(
        self, 
        k, c_data, d_data, cdr, c_meta_k,
        n_rounds, train_params, return_latents, 
        c_meta_hist, d_sens_hist, losses_train_hist_list, best_epos_list,
        device, c_latent_list, d_latent_list):

        self.CDPmodel_list_sub[k].load_state_dict(self.CDPmodel_list[k].state_dict())

        c_name_k_1 = self.c_name_clusters_in_trainnig[k]
        d_name_k_1 = self.d_name_clusters_in_trainnig[k]
        
        found_subcluster_id = self.subcluster_map_dict.get_from_map(str(k))
        if len(found_subcluster_id) >= 1:
            for found_k_sub in found_subcluster_id:
                d_name_k_before_sub = self.d_name_clusters_in_trainnig[found_k_sub]
                d_name_k_1 = np.concatenate((d_name_k_1, d_name_k_before_sub))

        d_data_1 = d_data.drop(d_name_k_1)
        cdr_1 = cdr.drop(columns=d_name_k_1)

        d_sens_hist_1 = pd.DataFrame() 

        losses_train_hist_list_k_1 = []
        best_epos_k_1 = []

        if return_latents:
            c_latent_k_1 = []
            d_latnet_k_1 = []

        for b in range(0, n_rounds):
            print(f"     -- round {b} -------------")     

            if b == 0:
                d_sens_hist_1[f'sensitive_k{k}'] = (cdr_1.loc[c_meta_k.index.values[c_meta_k.key == 1]].mean(axis=0) > self.sens_cutoff).astype(int)
                d_names_k_init_1 = d_sens_hist_1.index.values[d_sens_hist_1[f'sensitive_k{k}']==1]

                c_names_k_init_1 = c_name_k_1

                sensitive_cut_off = self.sens_cutoff / 1.5
            else:
                d_names_k_init_1 = d_sens_hist.index.values[d_sens_hist[f'sensitive_k{k}_sub_b{b-1}']==1]
                c_names_k_init_1 = c_meta_hist.index.values[c_meta_hist[f'k{k}_sub_b{b-1}']==1]

                sensitive_cut_off = self.sens_cutoff

            train_results = train_CDPmodel_local_1round(self.CDPmodel_list_sub[k], device, True, c_data, c_meta_k, d_data_1, \
                                                                            cdr_1, c_names_k_init_1, d_names_k_init_1, sensitive_cut_off, k, train_params)
            zero_cluster_sub, c_centroid_1, d_centroid_1, c_sd_1, d_sd_1, c_name_cluster_k_1, d_name_sensitive_k_1, losses_train_hist_1, best_epos_1, c_meta_k_1, d_sens_k_1 = train_results
            # print(c_meta_k_1)

            if zero_cluster_sub:
                print("  No subcluster found")

                return zero_cluster_sub, c_meta_hist, d_sens_hist

            else:
                c_meta_hist[f'k{k}_sub_b{b}'] = c_meta_k_1.key
                d_sens_hist[f'sensitive_k{k}_sub_b{b}'] = d_sens_k_1.sensitive
                losses_train_hist_list_k_1.append(losses_train_hist_1)
                best_epos_k_1.append(best_epos_1)

                if return_latents:
                    c_latent_1 = self.CDPmodel_list_sub[k].c_VAE.encode(torch.from_numpy(c_data.values).float().to(device), repram=False)
                    c_latent_k_1.append(c_latent_1.detach().cpu().numpy())
                    d_latent_1 = self.CDPmodel_list_sub[k].d_VAE.encode(torch.from_numpy(d_data.values).float().to(device), repram=False)
                    d_latnet_k_1.append(d_latent_1.detach().cpu().numpy())
                
                if b == n_rounds - 1:

                    k_sub = self.K
                    self.K += 1
                    self.subcluster_map_dict.add_to_map(str(k), k_sub)
                    self.which_non_empty_cluster.append(k_sub)

                    # store/update the centroids
                    self.c_centroids.append(c_centroid_1)
                    self.c_sds.append(c_sd_1)
                    self.d_centroids.append(d_centroid_1)
                    self.d_sds.append(d_sd_1)
                    self.c_name_clusters_in_trainnig.append(c_name_cluster_k_1)
                    self.d_name_clusters_in_trainnig.append(d_name_sensitive_k_1)

                    losses_train_hist_list.append(losses_train_hist_list_k_1)
                    best_epos_list.append(best_epos_k_1)

                    new_colnames = {}
                    for b_temp in range(n_rounds):
                        old_col = f'k{k}_sub_b{b_temp}'
                        new_col = f'k{k_sub}_b{b_temp}'
                        new_colnames[old_col] = new_col
                    c_meta_hist.rename(columns=new_colnames, inplace=True)

                    new_colnames = {}
                    for b_temp in range(n_rounds):
                        old_col = f'sensitive_k{k}_sub_b{b_temp}'
                        new_col = f'sensitive_k{k_sub}_b{b_temp}'
                        new_colnames[old_col] = new_col
                    d_sens_hist.rename(columns=new_colnames, inplace=True)

                    if return_latents:
                        c_latent_list.append(c_latent_k_1)
                        d_latent_list.append(d_latnet_k_1)

                    self.CDPmodel_list.append(self.CDPmodel_list_sub[k])

                    print(f"  Cluster {k} found a subcluster with cluster ID: {k_sub}")
        
        return zero_cluster_sub, c_meta_hist, d_sens_hist



class CDPmodel_sub(nn.Module):
    def __init__(self, params):                
        super(CDPmodel_sub, self).__init__()

        c_input_dim = params['c_input_dim']
        c_h_dims = params['c_h_dims']
        c_latent_dim = params['c_latent_dim']
        d_input_dim = params['d_input_dim']
        d_h_dims = params['d_h_dims']
        d_latent_dim = params['d_latent_dim']
        p_sec_dim = params['p_sec_dim']
        p_h_dims = params['p_h_dims']
        drop_out = params['drop_out']

        self.c_VAE = VAE(input_dim=c_input_dim, h_dims=c_h_dims, latent_dim=c_latent_dim, drop_out=drop_out)
        self.d_VAE = VAE(input_dim=d_input_dim, h_dims=d_h_dims, latent_dim=d_latent_dim, drop_out=drop_out)
        self.predictor = Predictor(c_input_dim=c_latent_dim, d_input_dim=d_latent_dim, sec_dim = p_sec_dim, h_dims=p_h_dims, drop_out=drop_out)

        self.c_VAE.apply(weights_init_uniform_rule)
        self.d_VAE.apply(weights_init_uniform_rule)
        self.predictor.apply(weights_init_uniform_rule)

    def forward(self, c_X=None, c_latent=None, d_X=None, d_latent=None, device='cpu'):    
        if c_X is None and c_latent is None:
            raise ValueError("At least one of c_X and c_latent must not be None.")
        if d_X is None and d_latent is None:
            raise ValueError("At least one of d_X and d_latent must not be None.")

        if c_latent is None:
            c_X = c_X.to(device)
            self.c_VAE = self.c_VAE.to(device)
            _, c_mu, c_log_var, c_Z, c_X_rec = self.c_VAE(c_X)
            c_mu = c_mu.to(device)
        else:
            c_mu = c_latent.to(device)
            c_log_var = None
            c_X_rec = None


        if d_latent is None: 
            d_X = d_X.to(device)
            self.d_VAE = self.d_VAE.to(device)
            _, d_mu, d_log_var, d_Z, d_X_rec = self.d_VAE(d_X)
            d_mu = d_mu.to(device)
        else:
            d_mu = d_latent.to(device)
            d_log_var = None
            d_X_rec = None

        self.predictor = self.predictor.to(device)

        CDR = self.predictor(c_mu, d_mu)
        
        return c_mu, c_log_var, c_X_rec, d_mu, d_log_var, d_X_rec, CDR



class VAE(nn.Module):
    def __init__(self,
                 input_dim,
                 h_dims=[512],
                 latent_dim = 128,
                 drop_out=0):                
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        hidden_dims = deepcopy(h_dims)
        hidden_dims.insert(0, input_dim)

        # Encoder
        modules_e = []
        for i in range(1, len(hidden_dims)):
            i_dim = hidden_dims[i-1]
            o_dim = hidden_dims[i]

            modules_e.append(
                nn.Sequential(
                    nn.Linear(i_dim, o_dim),
                    nn.BatchNorm1d(o_dim),
                    nn.Dropout(drop_out),
                    nn.LeakyReLU())
                )

        self.encoder_body = nn.Sequential(*modules_e)
        self.encoder_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        hidden_dims.reverse()

        modules_d = []

        self.decoder_first_layer = nn.Linear(latent_dim, hidden_dims[0])

        for i in range(len(hidden_dims) - 2):
            i_dim = hidden_dims[i]
            o_dim = hidden_dims[i + 1]

            modules_d.append(
                nn.Sequential(
                    nn.Linear(i_dim, o_dim),
                    nn.BatchNorm1d(o_dim),
                    nn.Dropout(drop_out),
                    nn.LeakyReLU())
            )
        
        self.decoder_body = nn.Sequential(*modules_d)

        self.decoder_last_layer = nn.Sequential(
            nn.Linear(hidden_dims[-2], hidden_dims[-1]),
            nn.LeakyReLU() # Sigmoid()
        )
            

    def encode_(self, X: Tensor):
        """
        Encodes the inputed tensor X by passing through the encoder network.
        Return a list with two tensors, mu and log_variance of the latent space. 
        """
        # print(f"result = self.encoder_body(X), X.size(): {X.size()}")
        # print(f"X dim: {X.dim}")
        # print(X)
        result = self.encoder_body(X)
        mu = self.encoder_mu(result)
        log_var = self.encoder_logvar(result)
       
        return [mu, log_var]

    def encode(self, X: Tensor, repram=True):
        """
        Encodes the inputed tensor X by passing through the encoder network.
        Returns the reparameterized latent space Z if reparameterization == True;
        otherwise returns the mu of the latent space.
        """
        mu, log_var = self.encode_(X)

        if(repram==True):
            Z = self.reparameterize(mu, log_var)
            return Z
        else: 
            return mu

    def decode(self, Z: Tensor):
        """
        Decodes the inputed tensor Z by passing through the decoder network.
        Returns the reconstructed X (the decoded Z) as a tensor. 
        """
        result = self.decoder_first_layer(Z)
        result = self.decoder_body(result)
        X_rec = self.decoder_last_layer(result)

        return X_rec

    def forward(self, X: Tensor):
        """
        Passes X through the encoder and decoder networks.
        Returns the reconstructed X. 
        """
        mu, log_var = self.encode_(X)
        Z = self.reparameterize(mu, log_var)
        X_rec = self.decode(Z)

        return [X, mu, log_var, Z, X_rec]
    
    def reparameterize(self, mu: Tensor, log_var: Tensor):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1)
        """
        std = torch.exp(0.5 * log_var)
        z = mu + std * torch.randn_like(std)

        return z


class Predictor(nn.Module):
    def __init__(self,
                 c_input_dim,
                 d_input_dim,
                 sec_dim = 16,
                 h_dims=[16],
                 drop_out=0):                
        super(Predictor, self).__init__()

        hidden_dims = deepcopy(h_dims)
        hidden_dims.insert(0, 2*sec_dim)
    
        self.cell_line_layer = nn.Sequential(
            nn.Linear(c_input_dim, sec_dim),
            nn.Dropout(drop_out),
            nn.ReLU()
        )

        self.drug_layer = nn.Sequential(
            nn.Linear(d_input_dim, sec_dim),
            nn.Dropout(drop_out),
            nn.ReLU()
        )

        # Predictor
        modules_e = []
        for i in range(2, len(hidden_dims)):
            i_dim = hidden_dims[i-1]
            o_dim = hidden_dims[i]

            modules_e.append(
                nn.Sequential(
                    nn.Linear(i_dim, o_dim),
                    # nn.BatchNorm1d(o_dim),
                    nn.Dropout(drop_out),
                    nn.ReLU())
                )
        modules_e.append(
            nn.Sequential(
                nn.Linear(hidden_dims[-1], 1),
                nn.Sigmoid()
            )
        )
        
        self.predictor_body = nn.Sequential(*modules_e)
            

    def forward(self, c_latent: Tensor, d_latent: Tensor):
        c = self.cell_line_layer(c_latent)
        d = self.drug_layer(d_latent)
        combination = torch.cat([c, d], dim=1)
        CDR = self.predictor_body(combination)
        return CDR



def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)

