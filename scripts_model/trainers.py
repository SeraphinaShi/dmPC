from email import header
from tkinter import Y
import torch
from torch import nn, optim, Tensor
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
import torch.distributions
import torch.utils

# import ray.train.torch
# from ray.train.torch import TorchTrainer
# from ray.train import ScalingConfig

import torch.multiprocessing
torch.multiprocessing.set_start_method('spawn', force=True)

import logging
import os
import sys
import time
import warnings
import copy
from copy import deepcopy

import numpy as np
import pandas as pd
from statistics import mean
from sklearn.model_selection import train_test_split
import math

from losses import *
from utils import *

# logging.getLogger().setLevel(logging.INFO)


def train_CDPmodel_local_1round(model, device, ifsubmodel,
                              c_data, c_meta_k, d_data, cdr_org, 
                              c_names_k_init, d_names_k_init, 
                              sens_cutoff, k, params):
    valid_size = params['valid_size']
    n_epochs = params['n_epochs']
    batch_size = params['batch_size']
    num_workers = params['num_workers']
    lr =  params['lr']
    C_VAE_loss_weight = params['C_VAE_loss_weight']
    C_recon_loss_weight = params['C_recon_loss_weight']
    C_kld_weight = params['C_kld_weight']
    C_cluster_distance_weight = params['C_cluster_distance_weight']
    C_update_ratio_weight = params['C_update_ratio_weight']
    D_VAE_loss_weight = params['D_VAE_loss_weight']
    D_recon_loss_weight = params['D_recon_loss_weight']
    D_kld_weight = params['D_kld_weight']
    D_cluster_distance_weight = params['D_cluster_distance_weight']
    D_update_ratio_weight = params['D_update_ratio_weight']
    predict_loss_weight = params['predict_loss_weight']
    if ifsubmodel == False:
        c_p_save_path = f"{params['c_p_save_path']}{'_'}{k}{'.pkl'}"
        d_p_save_path = f"{params['d_p_save_path']}{'_'}{k}{'.pkl'}"
    else:
        c_p_save_path = f"{params['c_p_save_path']}{'_sub_'}{k}{'.pkl'}"
        d_p_save_path = f"{params['d_p_save_path']}{'_sub_'}{k}{'.pkl'}"

    # Prepare and wrap your model with DistributedDataParallel
    # Move the model the correct GPU/CPU device
    # model = ray.train.torch.prepare_model(model)

    # -- clean data -- 
    cdr = cdr_org.copy()

    cdr['c_name'] = cdr.index.values
    cdr = pd.melt(cdr, id_vars='c_name', value_vars=None, var_name=None, value_name='value', col_level=None)
    cdr = cdr.rename(columns={'variable':'d_name', 'value':'cdr'})
    cdr_all = cdr.copy()

    cdr_all = cdr_all[~cdr_all['cdr'].isnull()] # remove NA values

    c_name_encoded, c_name_encode_map = one_hot_encode(cdr_all['c_name'])
    cdr_all['c_name_encoded'] = c_name_encoded
    d_name_encoded, d_name_encode_map = one_hot_encode(cdr_all['d_name'])
    cdr_all['d_name_encoded'] = d_name_encoded

    c_names_encoded_k_init = [c_name_encode_map[string] for string in c_names_k_init]
    d_names_encoded_k_init = [d_name_encode_map[string] for string in d_names_k_init]


    #a=================================================================================
    # Train D_VAE and predictor
    print(f"       a. Training D_VAE and Predictor")

    ##---------------------
    ## prepare data
    dataloaders_DP, cdr_k = prepare_dataloaders(model, c_data, c_meta_k, d_data, None, cdr_all, valid_size, batch_size, within_C_cluster=True, within_D_cluster=False, device=device)

    ##---------------------
    ## define optimizer
    optimizer_e = optim.Adam(model.parameters(), lr=lr)
    exp_lr_scheduler_e = lr_scheduler.ReduceLROnPlateau(optimizer_e)

    ##---------------------
    ## update D_VAE and predictor

    start = time.time()

    loss_train_hist, best_epo_a = train_CDPmodel_local(
        model=model, device = device,
        data_loaders=dataloaders_DP,
        c_names_k_old = c_names_encoded_k_init, 
        d_names_k_old = d_names_encoded_k_init,
        C_VAE_loss_weight = 0,
        C_recon_loss_weight = C_recon_loss_weight,
        C_kld_weight = C_kld_weight,
        C_cluster_distance_weight = C_cluster_distance_weight,
        C_update_ratio_weight = C_update_ratio_weight,
        D_VAE_loss_weight = 1 * D_VAE_loss_weight,
        D_recon_loss_weight = D_recon_loss_weight,
        D_kld_weight = D_kld_weight,
        D_cluster_distance_weight = D_cluster_distance_weight,
        D_update_ratio_weight = D_update_ratio_weight,
        predict_loss_weight = predict_loss_weight,
        sens_cutoff = sens_cutoff,
        optimizer=optimizer_e,
        n_epochs=n_epochs,
        scheduler=exp_lr_scheduler_e,
        within_C_cluster = True,
        within_D_cluster = False,
        save_path = d_p_save_path)
    end = time.time()
    print(f"            Running time: {end - start}")

    a_losses = get_train_VAE_predictor_hist_df(loss_train_hist, n_epochs, vae_type="D")

    #b=================================================================================
    # Drugs with predicted sensitive outcome is assigned to the K-th cluster. Then drugs in cluster k with latent space that is not close to the centroid will be dropped from the cluster.
    
    # get predicted CDR
    c_latentTensor = model.c_VAE.encode(torch.from_numpy(c_data.loc[cdr_k.c_name].values).float().to(device), repram=False)
    d_latentTensor = model.d_VAE.encode(torch.from_numpy(d_data.loc[cdr_k.d_name].values).float().to(device), repram=False)
    y_hatTensor = model.predictor(c_latentTensor, d_latentTensor)

    y_hat = y_hatTensor.detach().view(-1).cpu()
    y_hat = y_hat.numpy()
    cdr_k_hat = pd.DataFrame({'d_name':cdr_k.d_name, 'c_name':cdr_k.c_name, 'cdr':y_hat})

    # find sensitive drugs according predicted CDR
    d_sens_k = get_D_sensitive_codes(cdr_k_hat, sens_cutoff)
    d_name_sensitive_k = d_sens_k.index.values[d_sens_k.sensitive == 1]

    # remove outliers 
    d_sensitive_latent = model.d_VAE.encode(torch.from_numpy(d_data.loc[d_name_sensitive_k].values).float().to(device), repram=False)
    d_centroid = d_sensitive_latent.mean(dim=0)

    # d_sensitive_distances = torch.cdist(d_sensitive_latent, d_centroid.view(1, -1))
    d_is_outlier = find_outliers_3sd(d_sensitive_latent)

    if len(d_is_outlier) > 0:
        # d_sens_k.at[d_is_outlier, 'sensitive'] = 0
        for index in d_is_outlier:
            d_sens_k.at[index, 'sensitive'] = 0

    print(f"       b. {sum(d_sens_k.sensitive)} sensitive drug(s)")
     
    if sum(d_sens_k.sensitive) <= 1:
        return True, None, None, None, None, None, None, None, None, None, None

    #c=================================================================================
    # Train C_VAE and predictor 

    ##---------------------
    ## prepare data
    dataloaders_CP, cdr_k = prepare_dataloaders(model, c_data, None, d_data, d_sens_k, cdr_all, valid_size, batch_size, within_C_cluster=False, within_D_cluster=True, device=device)

    ##---------------------
    ## define optimizer
    optimizer_e = optim.Adam(model.parameters(), lr=lr)
    exp_lr_scheduler_e = lr_scheduler.ReduceLROnPlateau(optimizer_e)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    ##---------------------
    ## update C_VAE and predictor
    print(f"       c. Training C_VAE and Predictor")
    start = time.time()

    loss_train_hist, best_epo_c = train_CDPmodel_local(
        model=model, device = device,
        data_loaders=dataloaders_CP,
        c_names_k_old = c_names_encoded_k_init, 
        d_names_k_old = d_names_encoded_k_init,
        C_VAE_loss_weight = 1 * C_VAE_loss_weight,
        C_recon_loss_weight = C_recon_loss_weight,
        C_kld_weight = C_kld_weight,
        C_cluster_distance_weight = C_cluster_distance_weight,
        C_update_ratio_weight = C_update_ratio_weight,
        D_VAE_loss_weight = 0,
        D_recon_loss_weight = D_recon_loss_weight,
        D_kld_weight = D_kld_weight,
        D_cluster_distance_weight = D_cluster_distance_weight,
        D_update_ratio_weight = D_update_ratio_weight,
        predict_loss_weight = predict_loss_weight,
        optimizer=optimizer_e,
        n_epochs=n_epochs,
        scheduler=exp_lr_scheduler_e,
        within_C_cluster = False,
        within_D_cluster = True,
        save_path = c_p_save_path)
    end = time.time()
    print(f"            Running time: {end - start}")

    c_losses = get_train_VAE_predictor_hist_df(loss_train_hist, n_epochs, vae_type="C")


    #d=================================================================================
    # Cell lines with predicted sensitive outcome is assigned to the K-th cluster. Again, cell lines in cluster k with latent space that is not close to the centroid will be dropped from the cluster.
    
    # get predicted CDR
    c_latentTensor = model.c_VAE.encode(torch.from_numpy(c_data.loc[cdr_k.c_name].values).float().to(device), repram=False)
    d_latentTensor = model.d_VAE.encode(torch.from_numpy(d_data.loc[cdr_k.d_name].values).float().to(device), repram=False)
    y_hatTensor = model.predictor(c_latentTensor, d_latentTensor)
    # _,_,_,_,_,_,y_hatTensor = model(torch.from_numpy(c_data.loc[cdr_k.c_name].values).float().to(device),
    #                                 torch.from_numpy(d_data.loc[cdr_k.d_name].values).float().to(device))

    y_hat = y_hatTensor.detach().view(-1).cpu()
    y_hat = y_hat.numpy()
    cdr_k_hat = pd.DataFrame({'d_name':cdr_k.d_name, 'c_name':cdr_k.c_name, 'cdr':y_hat})

    # find sensitive cell lines according predicted CDR
    c_sens_k = get_C_sensitive_codes(cdr_k_hat, sens_cutoff)
    c_name_sensitive_k = c_sens_k.index.values[c_sens_k.sensitive == 1]

    # remove outliers 
    c_sensitive_latent = model.c_VAE.encode(torch.from_numpy(c_data.loc[c_name_sensitive_k].values).float().to(device), repram=False)
    c_centroid = c_sensitive_latent.mean(dim=0)

    # c_sensitive_distances = torch.cdist(c_sensitive_latent, c_centroid.view(1, -1))
    c_outlier_idx = find_outliers_3sd(c_sensitive_latent)

    if len(c_outlier_idx) > 0:
        for index in c_outlier_idx:
            c_sens_k.at[index, 'sensitive'] = 0

    old_1_boo = c_meta_k['key'] == 1
    c_meta_k.loc[old_1_boo, 'key'] = 0
    
    sens_boo = c_sens_k['sensitive'] == 1
    c_meta_k.loc[sens_boo, 'key'] = 1
    # c_meta_k.loc[c_meta_k.index[idx_cluster_updated], 'key'] = 1

    sensitive_count = c_sens_k.sensitive.sum()
    print(f"       d. {sensitive_count} cancer cell line(s) in the cluster")

    losses_train_hist = [a_losses, c_losses]
    best_epos = [best_epo_a, best_epo_c]

    c_name_cluster_k = c_meta_k.index.values[c_meta_k.key == 1]
    c_cluster_latent = model.c_VAE.encode(torch.from_numpy(c_data.loc[c_name_cluster_k].values).float().to(device), repram=False)
    c_centroid = c_cluster_latent.mean(dim=0)
    c_sd = c_cluster_latent.std(dim=0)
    d_name_sensitive_k = d_sens_k.index.values[d_sens_k.sensitive == 1]
    d_cluster_latent = model.d_VAE.encode(torch.from_numpy(d_data.loc[d_name_sensitive_k].values).float().to(device), repram=False)
    d_centroid = d_cluster_latent.mean(dim=0)
    d_sd = d_cluster_latent.std(dim=0)

    return False, c_centroid, d_centroid, c_sd, d_sd, c_name_cluster_k, d_name_sensitive_k, losses_train_hist, best_epos, c_meta_k, d_sens_k



def train_CDPmodel_local(model, device, data_loaders={}, c_names_k_old=None, d_names_k_old=None, 
                         C_VAE_loss_weight = 1, C_recon_loss_weight = 1, C_kld_weight = None, C_cluster_distance_weight=100, C_update_ratio_weight = 100, 
                         D_VAE_loss_weight = 1, D_recon_loss_weight = 1, D_kld_weight = None, D_cluster_distance_weight=100, D_update_ratio_weight = 100, 
                         predict_loss_weight = 1, 
                         sens_cutoff = 0.5,
                         optimizer=None, n_epochs=100, scheduler=None,
                         within_C_cluster = False, within_D_cluster = False,
                         load=False, save_path="model.pkl",  best_model_cache = "drive"):
    
    if(load!=False):
        if(os.path.exists(save_path)):
            model.load_state_dict(torch.load(save_path))           
            return model, 0, 0
        else:
            logging.warning("Failed to load existing model file, proceed to the trainning process.")
    
    dataset_sizes = {x: data_loaders[x].dataset.tensors[0].shape[0] for x in ['train', 'val']}
    loss_hist = {}
    c_vae_loss_hist = {}
    c_recon_loss_hist = {}
    c_kld_hist = {}
    c_cluster_dist_hist = {}
    c_update_overlap_hist = {}
    d_vae_loss_hist = {}
    d_recon_loss_hist = {}
    d_kld_hist = {}
    d_cluster_dist_hist = {}
    d_update_overlap_hist = {}
    prediction_loss_hist = {}
    
    if best_model_cache == "memory":
        best_model_wts = copy.deepcopy(model.state_dict())
    else:
        torch.save(model.state_dict(), save_path+"_bestcahce.pkl")
    
    best_loss = np.inf
    best_epoch = -1

    for epoch in range(n_epochs):
        logging.info('Epoch {}/{}'.format(epoch, n_epochs - 1))
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # optimizer = scheduler(optimizer, epoch)
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_prediction_loss = 0.0
            running_c_update_overlap = 0.0
            running_d_update_overlap = 0.0

            running_c_vae_loss = 0.0
            running_c_recon_loss = 0.0
            running_c_kld_loss = 0.0
            running_c_cluster_d = 0.0
            
            running_d_vae_loss = 0.0
            running_d_recon_loss = 0.0
            running_d_kld_loss = 0.0
            running_d_cluster_d = 0.0
            

            n_iters = len(data_loaders[phase])

            # Iterate over data.
            # for data in data_loaders[phase]:
            for batchidx, (y, c_data, d_data, c_name, d_name) in enumerate(data_loaders[phase]):
                
                y.requires_grad_(True)
                d_data.requires_grad_(True)

                # encode and decode 
                if within_C_cluster:
                    c_mu, c_log_var, c_X_rec, d_mu, d_log_var, d_X_rec, y_hat = model(c_latent = c_data, d_X = d_data, device = device)
                if within_D_cluster: 
                    c_mu, c_log_var, c_X_rec, d_mu, d_log_var, d_X_rec, y_hat = model(c_X = c_data, d_latent = d_data, device = device)

                sensitive = y_hat > sens_cutoff
                sensitive = sensitive.long()

                # compute loss

                mse = nn.MSELoss(reduction="sum")

                #   1. Prediction loss: 
                bce = nn.BCELoss()
                try:
                    prediction_loss = bce(y_hat, y)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    print(f"y_hat passed into the function: {y_hat}")
                    print(f"y passed into the function: {y}")
                
                if C_VAE_loss_weight > 0:
                    # 2. C_VAE:
                    # 2.1. VAE loss: reconstruction loss & kld
                    C_recon_loss, C_kld = custom_vae_loss(c_data, c_mu, c_log_var, c_X_rec, mse)

                    # 2.2. the loss of latent spaces distances:
                    #     - distances of cells in the cluster to the cluster centroid 
                    #     + distances of cells outside the cluster to the cluster centroid 
                    C_latent_dist_loss = cluster_mu_distance(c_mu, sensitive, device)
                    
                    # adding all up
                    if C_kld_weight is None:
                        C_kld_weight = data_loaders[phase].batch_size/dataset_sizes[phase]

                    C_VAE_loss = C_recon_loss_weight * C_recon_loss + C_kld_weight * C_kld + C_cluster_distance_weight * C_latent_dist_loss # + C_update_ratio_weight * C_overlap_loss


                    # 3. requiring the updated cluster overlaping with the old clustering
                    cdr_k_hat = pd.DataFrame({'d_name':d_name.detach().cpu().numpy(), 'c_name':c_name.detach().cpu().numpy(), 'cdr':y_hat.detach().cpu().numpy().flatten()})

                    if len(set(c_names_k_old)) > 0:
                        c_sens_k = get_C_sensitive_codes(cdr_k_hat, sens_cutoff)
                        c_assignments = torch.tensor(c_sens_k['sensitive'].values == 1, dtype=torch.float32)
                        
                        prev_assignments = torch.zeros(c_assignments.shape, dtype=torch.float32)
                        for i, name in enumerate(c_sens_k.index.values):
                            if name in c_names_k_old:
                                prev_assignments[i] = 1.0

                        overlap = torch.dot(c_assignments, prev_assignments) / c_assignments.numel()

                        # Define the loss as negative overlap to maximize it
                        C_overlap_loss = -overlap
                        
                    else:
                        C_overlap_loss = 0
                
                else:
                    C_VAE_loss = 0
                    C_overlap_loss = 0


                
                if D_VAE_loss_weight > 0:
                    # 4. D_VAE
                    # 4.1. VAE loss: reconstruction loss & kld
                    D_recon_loss, D_kld = custom_vae_loss(d_data, d_mu, d_log_var, d_X_rec, mse)

                    # 4.2. the loss of latent spaces distances:
                    #     - distances of drugs in the cluster to the cluster centroid 
                    #     + distances of drugs outside the cluster to the cluster centroid 
                    D_latent_dist_loss = cluster_mu_distance(d_mu, sensitive, device)
                    
                    # adding all up
                    if D_kld_weight is None:
                        D_kld_weight = data_loaders[phase].batch_size/dataset_sizes[phase]

                    D_VAE_loss = D_recon_loss_weight * D_recon_loss + D_kld_weight * D_kld + D_cluster_distance_weight * D_latent_dist_loss # + D_update_ratio_weight * D_overlap_loss


                    # 5 requiring the updated cluster overlaping with the old clustering
                    cdr_k_hat = pd.DataFrame({'d_name':d_name.cpu().detach().numpy(), 'c_name':c_name.cpu().detach().numpy(), 'cdr':y_hat.cpu().detach().numpy().flatten()})

                    if len(set(d_names_k_old)) > 0:
                        d_sens_k = get_D_sensitive_codes(cdr_k_hat, sens_cutoff)

                        d_assignments = torch.tensor(d_sens_k['sensitive'].values == 1, dtype=torch.float32)
                        
                        d_prev_assignments = torch.zeros(d_assignments.shape, dtype=torch.float32)
                        for i, name in enumerate(d_sens_k.index.values):
                            if name in d_names_k_old:
                                d_prev_assignments[i] = 1.0

                        overlap = torch.dot(d_assignments, d_prev_assignments) / d_assignments.numel()

                        # Define the loss as negative overlap to maximize it
                        D_overlap_loss = -overlap
                    else:
                        D_overlap_loss = 0
                
                else:
                    D_VAE_loss = 0
                    D_overlap_loss = 0

                
                # Add up three losses:
                loss =  C_VAE_loss_weight * C_VAE_loss + D_VAE_loss_weight * D_VAE_loss + predict_loss_weight * prediction_loss + C_update_ratio_weight * C_overlap_loss + D_update_ratio_weight * D_overlap_loss


                optimizer.zero_grad()

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()

                    optimizer.step()

                # print loss statistics
                running_loss += loss.item()
                running_prediction_loss += (predict_loss_weight * prediction_loss).item()   
                running_c_update_overlap += (C_update_ratio_weight * C_overlap_loss)
                running_d_update_overlap += (D_update_ratio_weight * D_overlap_loss)
                # print(f"   running_d_update_overlap: {running_d_update_overlap}")

                if C_VAE_loss_weight > 0:
                    running_c_vae_loss += (C_VAE_loss_weight * C_VAE_loss).item()
                    running_c_recon_loss += (C_VAE_loss_weight * C_recon_loss_weight * C_recon_loss).item()
                    running_c_kld_loss += (C_VAE_loss_weight * C_kld_weight * C_kld).item()
                    running_c_cluster_d += (C_VAE_loss_weight * C_cluster_distance_weight * C_latent_dist_loss).item()
                    
                if D_VAE_loss_weight > 0:            
                    running_d_vae_loss += (D_VAE_loss_weight * D_VAE_loss).item()
                    running_d_recon_loss += (D_VAE_loss_weight * D_recon_loss_weight * D_recon_loss).item()
                    running_d_kld_loss += (D_VAE_loss_weight * D_kld_weight * D_kld).item()
                    running_d_cluster_d += (D_VAE_loss_weight * D_cluster_distance_weight * D_latent_dist_loss).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_prediction_loss = running_prediction_loss / dataset_sizes[phase]
            epoch_c_update_overlap = running_c_update_overlap / (batchidx + 1)
            epoch_d_update_overlap = running_d_update_overlap / (batchidx + 1)

            epoch_c_vae_loss = running_c_vae_loss / dataset_sizes[phase]
            epoch_c_recon_loss = running_c_recon_loss / dataset_sizes[phase]
            epoch_c_kld_loss = running_c_kld_loss / dataset_sizes[phase]
            epoch_c_cluster_d = running_c_cluster_d / dataset_sizes[phase]

            epoch_d_vae_loss = running_d_vae_loss / dataset_sizes[phase]
            epoch_d_recon_loss = running_d_recon_loss / dataset_sizes[phase]
            epoch_d_kld_loss = running_d_kld_loss / dataset_sizes[phase]
            epoch_d_cluster_d = running_d_cluster_d / dataset_sizes[phase]
            
            loss_hist[epoch,phase] = epoch_loss
            c_vae_loss_hist[epoch,phase] = epoch_c_vae_loss
            c_recon_loss_hist[epoch,phase] = epoch_c_recon_loss
            c_kld_hist[epoch,phase] = epoch_c_kld_loss
            c_cluster_dist_hist[epoch,phase] = epoch_c_cluster_d
            c_update_overlap_hist[epoch,phase] = epoch_c_update_overlap
            d_vae_loss_hist[epoch,phase] = epoch_d_vae_loss
            d_recon_loss_hist[epoch,phase] = epoch_d_recon_loss
            d_kld_hist[epoch,phase] = epoch_d_kld_loss
            d_cluster_dist_hist[epoch,phase] = epoch_d_cluster_d
            d_update_overlap_hist[epoch,phase] = epoch_d_update_overlap
            prediction_loss_hist[epoch,phase] = epoch_prediction_loss
            
            if phase == 'val' and epoch >= 5 and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch

                if best_model_cache == "memory":
                    best_model_wts = copy.deepcopy(model.state_dict())
                else:
                    torch.save(model.state_dict(), save_path+"_bestcahce.pkl")
            elif phase == 'val' and best_loss == np.inf and epoch == round(n_epochs/2):
                if best_model_cache == "memory":
                    best_model_wts = copy.deepcopy(model.state_dict())
                else:
                    print(f'          save model half way (epoch {epoch}) since testing loss is NaN')
                    torch.save(model.state_dict(), save_path+"_bestcahce.pkl")
                    best_epoch = epoch
                
    train_hist = [loss_hist, c_vae_loss_hist, c_recon_loss_hist, c_kld_hist, c_cluster_dist_hist, c_update_overlap_hist, d_vae_loss_hist, d_recon_loss_hist, d_kld_hist, d_cluster_dist_hist, d_update_overlap_hist, prediction_loss_hist]

    # Select best model wts if use memory to cahce models
    if best_model_cache == "memory":
        torch.save(best_model_wts, save_path)
        model.load_state_dict(best_model_wts)  
    else:
        print(f'            Best epoc with test loss: epoch {best_epoch}')
        model.load_state_dict((torch.load(save_path+"_bestcahce.pkl")))
        torch.save(model.state_dict(), save_path)

    return train_hist, best_epoch





def train_VAE_train(vae, device, data_loaders={}, recon_loss_weight=1, kld_weight = None, optimizer=None, n_epochs=100, scheduler=None, load=False, save_path="vae.pkl", best_model_cache = "drive"):
    
    if(load!=False):
        if(os.path.exists(save_path)):
            vae.load_state_dict(torch.load(save_path))           
            return vae, 0, 0
        else:
            logging.warning("Failed to load existing file, proceed to the trainning process.")

    dataset_sizes = {x: data_loaders[x].dataset.tensors[0].shape[0] for x in ['train', 'val']}
    loss_train = {}
    vae_loss_train = {}
    recon_loss_train = {}
    kld_train = {}
    
    best_loss = np.inf
    best_epoch = -1
    
    vae.to(device)

    for epoch in range(n_epochs):
        logging.info('Epoch {}/{}'.format(epoch, n_epochs - 1))
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #optimizer = scheduler(optimizer, epoch)
                vae.train()  # Set model to training mode
            else:
                vae.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_recon_loss = 0.0
            running_kld_loss = 0.0

            n_iters = len(data_loaders[phase])

            # Iterate over data.
            # for data in data_loaders[phase]:
            for batchidx, (x, _) in enumerate(data_loaders[phase]):

                x.requires_grad_(True)
                x = x.to(device)
                
                # encode and decode 
                X, mu, log_var, Z, X_rec = vae(x)
                
                # compute loss
                mse = nn.MSELoss(reduction="sum")

                recon_loss, kld = custom_vae_loss(X, mu, log_var, X_rec, mse)

                if kld_weight is None:
                    kld_weight = data_loaders[phase].batch_size/dataset_sizes[phase]

                loss = recon_loss_weight*recon_loss + kld_weight * kld

                # zero the parameter (weight) gradients
                optimizer.zero_grad()

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    # update the weights
                    optimizer.step()

                # print loss statistics
                running_loss += loss.item()
                running_recon_loss += (recon_loss_weight*recon_loss).item()
                running_kld_loss += (kld_weight*kld).item()
            
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_recon_loss = running_recon_loss / dataset_sizes[phase]
            epoch_kld_loss = running_kld_loss / dataset_sizes[phase]
            
            #if phase == 'train':
            #    scheduler.step(epoch_loss)
                
            #last_lr = scheduler.optimizer.param_groups[0]['lr']
            loss_train[epoch,phase] = epoch_loss
            recon_loss_train[epoch,phase] = epoch_recon_loss
            kld_train[epoch,phase] = epoch_kld_loss
            train_hist = [loss_train, recon_loss_train, kld_train]
            #logging.info('{} Loss: {:.8f}. Learning rate = {}'.format(phase, epoch_loss,last_lr))
            
            if phase == 'val' and epoch >= 5 and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch

                if best_model_cache == "memory":
                    best_model_wts = copy.deepcopy(vae.state_dict())
                else:
                    torch.save(vae.state_dict(), save_path+"_bestcahce.pkl")
            elif phase == 'val' and best_loss == np.inf and epoch == round(n_epochs/2):
                if best_model_cache == "memory":
                    best_model_wts = copy.deepcopy(vae.state_dict())
                else:
                    print(f'          save model half way (epoch {epoch}) since testing loss is NaN')
                    torch.save(vae.state_dict(), save_path+"_bestcahce.pkl")
                    best_epoch = epoch
    
    # Select best model wts if use memory to cahce models
    if best_model_cache == "memory":
        torch.save(best_model_wts, save_path)
        vae.load_state_dict(best_model_wts)  
    else:
        print(f'        Best epoc with test loss: epoch {best_epoch}')
        vae.load_state_dict((torch.load(save_path+"_bestcahce.pkl")))
        torch.save(vae.state_dict(), save_path)

    return vae, train_hist, best_epoch



def train_VAE(VAE_model, device, data, vae_type, save_path, params, num_workers = 2):
    valid_size = params['valid_size']
    # n_epochs = params['n_epochs']
    n_epochs = 150
    batch_size = params['batch_size']
    lr =  params['lr']
    if vae_type == "C":
        recon_loss_weight = params['C_recon_loss_weight']
        kld_weight = params['C_kld_weight']
    if vae_type == "D":
        recon_loss_weight = params['D_recon_loss_weight']
        kld_weight = params['D_kld_weight']

    if torch.cuda.device_count() > 1:
        VAE_model = torch.nn.DataParallel(VAE_model)
    VAE_model.to(device)

    ##---------------------
    ## prepare data 
    X_train, X_valid = train_test_split(data, test_size=valid_size)

    last_batch_size = X_train.shape[0] % batch_size
    if last_batch_size < 3:
        sampled_rows = X_train.sample(n=last_batch_size)
        X_train = X_train.drop(sampled_rows.index)
        X_valid = pd.concat([X_valid, sampled_rows], ignore_index=True)

    X_trainTensor = torch.FloatTensor(X_train.values).to(device)
    X_validTensor = torch.FloatTensor(X_valid.values).to(device)

    train_dataset = TensorDataset(X_trainTensor, X_trainTensor)
    valid_dataset = TensorDataset(X_validTensor, X_validTensor)

    X_trainDataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    X_validDataLoader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
    # X_trainDataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # X_validDataLoader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    dataloaders = {'train':X_trainDataLoader,'val':X_validDataLoader}
    ##---------------------
    ## define optimizer
    optimizer_e = optim.Adam(VAE_model.parameters(), lr=lr)
    # exp_lr_scheduler_e = lr_scheduler.ReduceLROnPlateau(optimizer_e)

    ##---------------------
    ## update VAE
    start = time.time()
    VAE_model, train_hist, best_epo_cVAE = train_VAE_train(
        vae=VAE_model,
        device = device,
        data_loaders=dataloaders,
        recon_loss_weight = recon_loss_weight,
        kld_weight = kld_weight,
        optimizer=optimizer_e,
        n_epochs=n_epochs, 
        # scheduler=exp_lr_scheduler_e
        save_path = save_path
        )
    end = time.time()
    print(f"        Running time: {end - start}")

    VAE_losses = get_train_VAE_hist_df(train_hist, n_epochs)

    return VAE_model, VAE_losses
