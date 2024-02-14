from cmath import nan
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.utils
import torch.distributions
import numpy as np
from typing import Optional

import pandas as pd
import itertools

import traceback

import copy
from copy import deepcopy
import logging
import os

import multiprocessing

from statistics import mean

from trainers import *

## TODO: remove after debugging
torch.autograd.set_detect_anomaly(True)

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
        self.which_non_empty_subcluster = []
        self.cluster_id_for_subcluster = []

        self.c_centroids = [None] * self.K
        self.c_sds = [None] * self.K
        self.c_in_trainnig = []
        self.c_name_clusters_in_trainnig = [None] * self.K
        self.d_centroids = [None] * self.K
        self.d_sds = [None] * self.K
        self.d_in_trainnig = []
        self.d_name_clusters_in_trainnig = [None] * self.K

        self.sens_cutoff = params['sens_cutoff']
        

    def forward(self, c_X, d_X, c_name: object = '', d_name: object = '', k: int = -1, sd_scale: float = 3):
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
                for k_itr in (self.which_non_empty_cluster + self.cluster_id_for_subcluster):
                    if c_name in self.c_name_clusters_in_trainnig[k_itr]:
                        c_cluster.append(k_itr)
        
            else:
                # search through clusters
                c_dists = []
                for k_itr in self.which_non_empty_cluster:
                    c_mu = self.CDPmodel_list[k_itr].c_VAE.encode(c_X, repram=False)
                    is_outlier = ((c_mu - self.c_centroids[k_itr]).abs() > sd_scale * self.c_sds[k_itr]).any().item()
                    if is_outlier:
                        c_dist_k = torch.tensor(float('inf'))
                    else: 
                        c_dist_k = ((c_mu - self.c_centroids[k_itr]) / (self.c_sds[k_itr])).norm()
                    c_dists.append(c_dist_k)

                stacked_c_dists = torch.stack(c_dists)

                if stacked_c_dists.min().item() < float('inf'):
                    c_cluster_tep = self.which_non_empty_cluster[torch.argmin(stacked_c_dists).item()]
                    c_cluster.append(c_cluster_tep)

                # search through subclusters
                if len(self.cluster_id_for_subcluster) > 0:
                    c_sub_dists = []
                    for k_sub_itr in self.cluster_id_for_subcluster:
                        c_sub_mu = self.CDPmodel_list[k_sub_itr].c_VAE.encode(c_X, repram=False)
                        is_outlier = ((c_sub_mu - self.c_centroids[k_sub_itr]).abs() > sd_scale * self.c_sds[k_sub_itr]).any().item()
                        if is_outlier:
                            c_sub_dist_k = torch.tensor(float('inf'))
                        else: 
                            c_sub_dist_k = ((c_sub_mu - self.c_centroids[k_sub_itr]) / (self.c_sds[k_sub_itr])).norm()
                        c_sub_dists.append(c_sub_dist_k)

                    stacked_c_sub_dists = torch.stack(c_sub_dists)
                    
                    if stacked_c_sub_dists.min().item() < float('inf'):
                        c_cluster_tep = self.cluster_id_for_subcluster[torch.argmin(stacked_c_sub_dists).item()]
                        c_cluster.append(c_cluster_tep)
                
            if len(c_cluster) == 0:
                c_cluster.append(-1)    # check through subclusters

                         
            # find the cluster of D
            if d_name in self.d_in_trainnig:

                for k_itr in (self.which_non_empty_cluster + self.cluster_id_for_subcluster):
                    if d_name in self.d_name_clusters_in_trainnig[k_itr]:
                        d_cluster.append(k_itr)
                
            else:
                # search through clusters
                d_dists = []
                for k_itr in self.which_non_empty_cluster:
                    d_mu = self.CDPmodel_list[k_itr].d_VAE.encode(d_X, repram=False)
                    is_outlier = ((d_mu - self.d_centroids[k_itr]).abs() > sd_scale * self.d_sds[k_itr]).any().item()
                    if is_outlier:
                        d_dist_k = torch.tensor(float('inf'))
                    else: 
                        d_dist_k = ((d_mu - self.d_centroids[k_itr]) / (self.d_sds[k_itr])).norm()
                    d_dists.append(d_dist_k)

                stacked_d_dists = torch.stack(d_dists)

                if stacked_d_dists.min().item() < float('inf'):
                    d_cluster_tep = self.which_non_empty_cluster[torch.argmin(stacked_d_dists).item()]
                    d_cluster.append(d_cluster_tep)

                # search through subclusters
                d_sub_dists = []
                for k_sub_itr in self.cluster_id_for_subcluster:
                    d_sub_mu = self.CDPmodel_list[k_sub_itr].d_VAE.encode(d_X, repram=False)
                    is_outlier = ((d_sub_mu - self.d_centroids[k_sub_itr]).abs() > sd_scale * self.d_sds[k_sub_itr]).any().item()
                    if is_outlier:
                        d_sub_dist_k = torch.tensor(float('inf'))
                    else: 
                        d_sub_dist_k = ((d_sub_mu - self.d_centroids[k_sub_itr]) / (self.d_sds[k_sub_itr])).norm()
                    d_sub_dists.append(d_sub_dist_k)

                stacked_d_sub_dists = torch.stack(d_sub_dists)
                
                if stacked_d_sub_dists.min().item() < float('inf'):
                    d_cluster_tep = self.cluster_id_for_subcluster[torch.argmin(stacked_d_sub_dists).item()]
                    d_cluster.append(d_cluster_tep)
                
            if len(d_cluster) == 0:
                d_cluster.append(-1)    # check through subclusters
    
            # Loop through clusters of C and D to find CDR
            CDR_tmp_list = []
            cluster_tmp_list =[]
            
            if c_name in self.c_in_trainnig and d_name in self.d_in_trainnig: 
                cd_clusters = list(set(c_cluster).intersection(set(d_cluster)))
            else:
                cd_clusters = list(set(c_cluster + d_cluster))
            
            for k in cd_clusters:
                if k != -1:
                    CDR_temp = self.predict_given_model(self.CDPmodel_list[k], c_X, d_X)
                    CDR_tmp_list.append(CDR_temp)
                    
                    if (k in c_cluster and k in d_cluster) or CDR_temp >= 0.5:
                        cluster_tmp_list.append(k)
                    else:
                        cluster_tmp_list.append(None)
            
            idx_notNone = [i for i, item in enumerate(cluster_tmp_list) if item is not None]
            
            if len(idx_notNone) > 0:
                CDR = [CDR_tmp_list[i] for i in idx_notNone]
                cluster = [cluster_tmp_list[i] for i in idx_notNone]

                CDR = format_list_as_string(CDR)
                cluster = format_list_as_string(cluster)
            else:
                CDR = 0
                cluster = -1

            c_cluster = format_list_as_string(c_cluster)
            d_cluster = format_list_as_string(d_cluster)
            
 
        else:
            cluster = k
            c_cluster = k
            d_cluster = k
            CDR = self.predict_given_model(self.CDPmodel_list[cluster], c_X, d_X)

        return CDR, cluster, c_cluster, d_cluster


    def predict_given_model(self, local_model, c_X: pd.DataFrame, d_X: pd.DataFrame):
        c_mu = local_model.c_VAE.encode(c_X, repram=False)
        d_mu = local_model.d_VAE.encode(d_X, repram=False)
        CDR_temp = local_model.predictor(c_mu, d_mu)
        CDR_temp = round(CDR_temp.item(), 6)
            
        return CDR_temp


    def predict(self, c_X: pd.DataFrame, d_X: pd.DataFrame, k: int = -1, sd_scale: float = 3):
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

            cdr_hat, cluster,c_cluster, d_cluster = self(c_X_tensor, d_X_tensor, c_name, d_name, k, sd_scale = sd_scale)

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


    def fit(self, c_data, c_meta, d_data, cdr, train_params, n_rounds=3, search_subcluster = True, device='cpu'):
        
        c_meta_hist = c_meta.copy()
        d_sens_hist = pd.DataFrame() 
        losses_train_hist_list = []
        best_epos_list = []
        self.nonzero_clusters = 0

        return_latents = True
        if return_latents:
            c_latent_list = []
            d_latent_list = []

        self.c_in_trainnig = c_data.index.values
        self.d_in_trainnig = d_data.index.values

        if search_subcluster:
            losses_train_hist_list_sub = []
            best_epos_list_sub = []

            c_centroids_sub = [None] * self.K
            c_sds_sub = [None] * self.K
            d_centroids_sub = [None] * self.K
            d_sds_sub = [None] * self.K

            c_name_clusters_in_trainnig_sub = [None] * self.K
            d_name_clusters_in_trainnig_sub = [None] * self.K

            if return_latents:
                c_latent_list_sub = []
                d_latent_list_sub = []


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
                d_latent_k = []

            if search_subcluster:
                losses_train_hist_list_k_1 = []
                best_epos_k_1 = []

                if return_latents:
                    c_latent_k_1 = []
                    d_latent_k_1 = []

            meta_key = "k" + str(k)
            c_meta_k = c_meta[[meta_key]].rename(columns={meta_key:'key'})

            # 1. Run the dual loop to train local models
            for b in range(0, n_rounds):
                print(f"     -- round {b} -------------")    
 
                if b == 0:
                    d_sens_hist[f'sensitive_k{k}'] = (cdr.loc[c_meta_k.index.values[c_meta_k.key == 1]].mean(axis=0) > self.sens_cutoff).astype(int)
                    d_names_k_init = d_sens_hist.index.values[d_sens_hist[f'sensitive_k{k}']==1]
                else:
                    d_names_k_init = d_sens_hist.index.values[d_sens_hist[f'sensitive_k{k}_b{b-1}']==1]
                
                c_names_k_init = c_meta_k.index.values[c_meta_k.key == 1] 


                (zero_cluster, self.CDPmodel_list[k], 
                 c_centroid, d_centroid, c_sd, d_sd, 
                 c_name_cluster_k, d_name_sensitive_k, 
                 losses_train_hist, best_epos,) = train_CDPmodel_local_1round(
                     self.CDPmodel_list[k], device, 
                     ifsubmodel = False, 
                     c_data = c_data, d_data = d_data, cdr_org = cdr, 
                     c_names_k_init = c_names_k_init, d_names_k_init = d_names_k_init, 
                     sens_cutoff = self.sens_cutoff, 
                     group_id = k, 
                     params = train_params
                     )
                
                ## update binarized column vectors of cell and drug sensitivity based on results.
                c_meta_k, d_sens_k = create_bin_sensitive_dfs(
                    c_data, d_data, c_name_cluster_k, d_name_sensitive_k
                )

                
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
                    losses_train_hist_list_k.append(None)
                    best_epos_k.append(None)

                    if return_latents:
                        c_latent_k.append(None)
                        d_latent_k.append(None)

                    break

                else:
                    if b == n_rounds - 1:
                        self.which_non_empty_cluster.append(k)
                        self.nonzero_clusters += 1

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
                        c_latent_k.append(c_latent.detach().numpy())
                        d_latent = self.CDPmodel_list[k].d_VAE.encode(torch.from_numpy(d_data.values).float().to(device), repram=False)
                        d_latent_k.append(d_latent.detach().numpy())
                
                ## Use multiprocessing to run the loop in parallel for each k
                # with multiprocessing.Pool() as pool:
                # s    pool.map(train_k, range(0, self.K))
            
            if k in self.which_non_empty_cluster:                    
                losses_train_hist_list.append(losses_train_hist_list_k)
                best_epos_list.append(best_epos_k)

                if return_latents:
                    c_latent_list.append(c_latent_k)
                    d_latent_list.append(d_latent_k)
            else:
                losses_train_hist_list.append(None)
                best_epos_list.append(None)

                if return_latents:
                    c_latent_list.append(None)
                    d_latent_list.append(None)

            if zero_cluster:
                break
            
            if search_subcluster:
                # ---------------------------------------------
                # 2. Run the dual loop again to find subclusters
                print(f"  ===================================")
                print(f"  === {k}.2. sub local CDP model      ")
                print(f"  ===================================")

                self.CDPmodel_list_sub[k].load_state_dict(
                    self.CDPmodel_list[k].state_dict())

                c_name_k_1 = self.c_name_clusters_in_trainnig[k]
                d_name_k_1 = self.d_name_clusters_in_trainnig[k]

                d_data_1 = d_data.drop(d_name_k_1)
                cdr_1 = cdr.drop(columns=d_name_k_1)

                d_sens_hist_1 = pd.DataFrame() 

                for b in range(0, n_rounds):
                    print(f"     -- round {b} -------------")     

                    if b == 0:
                        d_sens_hist_1[f'sensitive_k{k}'] = (cdr_1.loc[c_meta_k.index.values[c_meta_k.key == 1]].mean(axis=0) > 0.5).astype(int)
                        d_names_k_init_1 = d_sens_hist_1.index.values[d_sens_hist_1[f'sensitive_k{k}']==1]

                        c_names_k_init_1 = c_name_k_1

                        ## TODO: this feels quite arbitrary.
                        sensitive_cut_off = 0.2
                    else:
                        d_names_k_init_1 = d_sens_hist.index.values[d_sens_hist[f'sensitive_k{k}_sub_b{b-1}']==1]
                        c_names_k_init_1 = c_meta_hist.index.values[c_meta_hist[f'k{k}_sub_b{b-1}']==1]

                        sensitive_cut_off = self.sens_cutoff

                    (zero_cluster_sub, self.CDPmodel_list_sub[k], 
                     c_centroid_1, d_centroid_1, c_sd_1, d_sd_1, 
                     c_name_cluster_k_1, d_name_sensitive_k_1, 
                     losses_train_hist_1,
                     best_epos_1,) = train_CDPmodel_local_1round(
                            self.CDPmodel_list_sub[k], device, 
                            ifsubmodel = True, 
                            c_data = c_data, d_data = d_data_1, cdr_org = cdr_1, 
                            c_names_k_init = c_names_k_init_1, 
                            d_names_k_init = d_names_k_init_1, 
                            sens_cutoff = sensitive_cut_off, 
                            group_id = k, params = train_params)

                    c_meta_k_1, d_sens_k_1 = create_bin_sensitive_dfs(
                        c_data, d_data_1, 
                        c_name_cluster_k_1, d_name_sensitive_k_1
                    )

                    if(zero_cluster_sub):
                        print("  No subcluster found")

                        losses_train_hist_list_k_1.append(None)
                        best_epos_k.append(None)
                        
                        if return_latents:
                            c_latent_k_1.append(None)
                            d_latent_k_1.append(None)

                        break
                    else:

                        if b == n_rounds - 1:
                            self.which_non_empty_subcluster.append(k)

                        # store/update the centroids
                        c_centroids_sub[k] = c_centroid_1
                        c_sds_sub[k] = c_sd_1
                        d_centroids_sub[k] = d_centroid_1
                        d_sds_sub[k] = d_sd_1
                        c_name_clusters_in_trainnig_sub[k] = c_name_cluster_k_1
                        d_name_clusters_in_trainnig_sub[k] = d_name_sensitive_k_1

                        c_meta_hist[f'k{k}_sub_b{b}'] = c_meta_k_1.key
                        d_sens_hist[f'sensitive_k{k}_sub_b{b}'] = d_sens_k_1.sensitive
                        losses_train_hist_list_k_1.append(losses_train_hist_1)
                        best_epos_k_1.append(best_epos_1)

                        if return_latents:
                            c_latent_1 = self.CDPmodel_list_sub[k].c_VAE.encode(torch.from_numpy(c_data.values).float().to(device), repram=False)
                            c_latent_k_1.append(c_latent_1.detach().numpy())
                            d_latent_1 = self.CDPmodel_list_sub[k].d_VAE.encode(torch.from_numpy(d_data.values).float().to(device), repram=False)
                            d_latent_k_1.append(d_latent_1.detach().numpy())
                
                if k in self.which_non_empty_subcluster:
                    print(f"Subcluster found as cluster {self.original_K + self.which_non_empty_subcluster.index(k)}")
                    
                    losses_train_hist_list_sub.append(losses_train_hist_list_k_1)
                    best_epos_list_sub.append(best_epos_k_1)

                    if return_latents:
                        c_latent_list_sub.append(c_latent_k_1)
                        d_latent_list_sub.append(d_latent_k_1) 
                else:
                    losses_train_hist_list_sub.append(None)
                    best_epos_list_sub.append(None)

                    if return_latents:
                        c_latent_list_sub.append(None)
                        d_latent_list_sub.append(None)

        if self.nonzero_clusters == 0:
            raise ValueError("No Biclusters found for these data. Consider raising " +
                             "the number of epochs, to allow the predictor time to " + 
                             "fit small sensitive groups.")
        c_meta_hist = add_meta_code(c_meta_hist, self.K, n_rounds)
        d_sens_hist = add_sensk_to_d_sens_init(d_sens_hist, self.original_K)
        d_sens_hist = add_sensk_to_d_sens_hist(d_sens_hist, self.K, n_rounds)
        
        c_meta_hist['code_latest'] = c_meta_hist[f'code_b{n_rounds-1}']
        d_sens_hist['sensitive_k_latest'] = d_sens_hist[f'sensitive_k_b{n_rounds-1}']

        if search_subcluster:
            print(f"########################################################")
            print(f"#### Check all subclusters                              ")      
            print(f"########################################################")
            for k in range(0, self.original_K):
                if k in self.which_non_empty_subcluster:
                    print("update")
                    sub_cluster_id = self.K # + 1 - 1
                    self.K = sub_cluster_id + 1

                    self.cluster_id_for_subcluster.append(sub_cluster_id)

                    self.CDPmodel_list.append(self.CDPmodel_list_sub[k]) 

                    print(f" - Cluster {k} found a subcluster with cluster ID: {sub_cluster_id}")
                    print(f"   - Now we have {self.K} clusters")

                    self.c_centroids.append(c_centroids_sub[k])
                    self.c_sds.append(c_sds_sub[k])
                    self.d_centroids.append(d_centroids_sub[k])
                    self.d_sds.append(d_sds_sub[k])
                    self.c_name_clusters_in_trainnig.append(c_name_clusters_in_trainnig_sub[k])
                    self.d_name_clusters_in_trainnig.append(d_name_clusters_in_trainnig_sub[k])

                    for b in range(0, n_rounds):
                        c_meta_hist[f'k{sub_cluster_id}_b{b}'] = c_meta_hist[f'k{k}_sub_b{b}']
                        d_sens_hist[f'sensitive_k{sub_cluster_id}_b{b}'] = d_sens_hist[f'sensitive_k{k}_sub_b{b}']
                
                    losses_train_hist_list.append(losses_train_hist_list_sub[k])
                    best_epos_list.append(best_epos_list_sub[k])

                    if return_latents:
                        c_latent_list.append(c_latent_list_sub[k])
                        d_latent_list.append(d_latent_list_sub[k])
            
            c_meta_hist = add_meta_code_with_subcluster(c_meta_hist, self.K, n_rounds)
            d_sens_hist = add_sensk_to_d_sens_hist_with_subcluster(d_sens_hist, self.K, n_rounds)
        
            c_meta_hist['code_sub_latest'] = c_meta_hist[f'code_sub_b{n_rounds-1}']
            d_sens_hist['sensitive_k_sub_latest'] = d_sens_hist[f'sensitive_k_sub_b{n_rounds-1}']

        ## Use multiprocessing to run the loop in parallel for each k
        # with multiprocessing.Pool() as pool:
        # s    pool.map(train_k, range(0, self.K))

        if return_latents:
            return c_meta, c_meta_hist, d_sens_hist, losses_train_hist_list, best_epos_list, C_VAE_init_losses, D_VAE_init_losses, c_latent_list, d_latent_list
        else: 
            return c_meta, c_meta_hist, d_sens_hist, losses_train_hist_list, best_epos_list, C_VAE_init_losses, D_VAE_init_losses




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

        self.c_mu_fixed = None
        self.d_mu_fixed = None

    def forward(
            self, 
            c_X: Optional[torch.Tensor] = None, 
            d_X: Optional[torch.Tensor] = None, 
            ):
        ## Compute the needed embedding/s
        if c_X is None:
            c_mu = self.c_mu_fixed
            c_log_var = self.c_log_var_fixed
            c_X_rec = self.c_X_rec_fixed
        else:
            _, c_mu, c_log_var, c_Z, c_X_rec = self.c_VAE(c_X)

        if d_X is None:
            d_mu = self.d_mu_fixed
            d_log_var = self.d_log_var_fixed
            d_X_rec = self.d_X_rec_fixed
        else:
            _, d_mu, d_log_var, d_Z, d_X_rec = self.d_VAE(d_X)
            

        ## Run the predictor:
        ## TODO: remove after debugging!
        if torch.sum(torch.isnan(d_mu)).item() > 0:
            print('check model')
            _, d_mu, d_log_var, d_Z, d_X_rec = self.d_VAE(d_X)
            traceback.print_stack()
            
        CDR = self.predictor(c_mu, d_mu)
        return c_mu, c_log_var, c_X_rec, d_mu, d_log_var, d_X_rec, CDR
    
    def update_fixed_encoding(
            self, 
            c_X: Optional[torch.Tensor] = None, 
            d_X: Optional[torch.Tensor] = None,
            ):
        if c_X is not None:
            with torch.no_grad():
                _, self.c_mu_fixed, self.c_log_var_fixed, _, self.c_X_rec_fixed = self.c_VAE(c_X)
        if d_X is not None:
            with torch.no_grad():
                _, self.d_mu_fixed, self.d_log_var_fixed, _, self.d_X_rec_fixed = self.d_VAE(d_X)



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
        
        if torch.sum(torch.isnan(mu)).item() > 0:
            print("check encoder!")
            mu, log_var = self.encode_(X)

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

        ### concatenate the vectors of each possible combination, in cell line order.
        c_d1, c_d2 = c.shape
        d_d1, d_d2 = d.shape

        combination = torch.cat(
            [torch.repeat_interleave(c, repeats=d_d1, dim=0), d.repeat(c_d1, 1), ]
            , dim=-1,
        )

        #combination = torch.cat([c, d], dim=1)

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

