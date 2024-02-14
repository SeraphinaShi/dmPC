from email import header
from tkinter import Y
import torch
from torch import nn, optim, Tensor
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
import torch.distributions
import torch.utils

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
import math

from losses import *
from utils import *

# logging.getLogger().setLevel(logging.INFO)


def train_CDPmodel_local_1round(
        model, device, ifsubmodel,
        c_data, d_data, cdr_org, 
        c_names_k_init, d_names_k_init, 
        sens_cutoff, group_id, 
        params
        ):
    valid_size = params['valid_size']
    n_epochs = params['n_epochs']
    batch_size = params['batch_size']
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
    rm_cluster_outliers = params['rm_cluster_outliers']
    use_mixture_kld = params['use_mixture_kld'] ## whether to swap to mixture of Gaussians KLD during bicluster VAE training.
    use_weighted_bce = params['use_weighted_bce']
    
    if ifsubmodel == False:
        c_p_save_path = f"{params['c_p_save_path']}{'_'}{group_id}{'.pkl'}"
        d_p_save_path = f"{params['d_p_save_path']}{'_'}{group_id}{'.pkl'}"
    else:
        c_p_save_path = f"{params['c_p_save_path']}{'_sub_'}{group_id}{'.pkl'}"
        d_p_save_path = f"{params['d_p_save_path']}{'_sub_'}{group_id}{'.pkl'}"

    ### Prep CDR Data:
    cdr_all = clean_cdr_for_trainers(cdr_data = cdr_org)

    #a=================================================================================
    # Train D_VAE and predictor
    ##---------------------
    
    ## prepare data

    ## sort objects:
    c_data, d_data, cdr_all, c_orig_idx_map, d_orig_idx_map = sort_cdr_data(
        c_data, d_data, cdr_all
    )

    ## data wrangling wrt subgroup and torch loader prep.
    c_data_k, dataloaders_DP, cdr_outputs_dict = prepare_model_IO(
        train_data_type = "drug",
        c_data = c_data,
        d_data = d_data,
        cdr_all = cdr_all,
        c_names_k = c_names_k_init,
        d_names_k = d_names_k_init,
        valid_size = valid_size, batch_size = batch_size,
        device = device
    )

    ##---------------------
    ## define optimizer
    optimizer_e = optim.Adam(model.parameters(), lr=lr)
    exp_lr_scheduler_e = lr_scheduler.ReduceLROnPlateau(optimizer_e)

    ##---------------------
    ## update D_VAE and predictor
    print(f"       a. Training D_VAE and Predictor")
    start = time.time()

    model.update_fixed_encoding(
        c_X = torch.FloatTensor(c_data_k.values).to(device), 
        d_X = torch.FloatTensor(d_data.values).to(device))

    model, loss_train_hist, best_epo_a = train_CDPmodel_local(
        model=model,
        data_loaders=dataloaders_DP,
        outputs_dict = cdr_outputs_dict,
        VAE_loss_weight = D_VAE_loss_weight,
        recon_loss_weight = D_recon_loss_weight,
        kld_weight = D_kld_weight,
        cluster_distance_weight = D_cluster_distance_weight,
        update_ratio_weight = D_update_ratio_weight,
        predict_loss_weight = predict_loss_weight,
        sens_cutoff = sens_cutoff,
        optimizer=optimizer_e,
        n_epochs=n_epochs,
        scheduler=exp_lr_scheduler_e,
        save_path = d_p_save_path,
        use_mixture_kld = use_mixture_kld,
        use_weighted_bce = use_weighted_bce)
    end = time.time()
    print(f"            Running time: {end - start}")


    a_losses = get_train_VAE_predictor_hist_df(loss_train_hist, n_epochs, )

    #=================================================================================
    # Drugs with predicted sensitive outcome is assigned to the K-th cluster. Then drugs in cluster k with latent space that is not close to the centroid will be dropped from the cluster.
    
    # get predicted CDR
    model.update_fixed_encoding(
        c_X = torch.FloatTensor(c_data_k.values).to(device), 
        d_X = torch.FloatTensor(d_data.values).to(device))

    ## generate full list of predictions and update drug sensitivity list.
    with torch.no_grad():
        y_hatTensor = model.predictor(model.c_mu_fixed, model.d_mu_fixed)

    ### Get sensitive drugs:
    d_sens_k = torch.zeros(d_data.shape[0])
    for i in range(d_data.shape[0]):
        d_sens_k[i] = torch.mean(y_hatTensor[cdr_outputs_dict['drug_idx'] == i]) > sens_cutoff
    
    # find sensitive drugs according predicted CDR
    d_name_sensitive_k = d_data.index.values[d_sens_k == 1]

    # remove outliers
    if rm_cluster_outliers:
        d_sensitive_latent = model.d_mu_fixed[d_sens_k == 1]
        d_centroid = d_sensitive_latent.mean(dim=0)

        ## TODO: I think I'm a little unclear on the logic of this step.
        d_outlier_idx = find_outliers_3sd(d_sensitive_latent)
        d_name_outlier = d_data.index.values[d_outlier_idx]

        if len(d_outlier_idx) > 0:
            d_name_sensitive_k = np.array(
                [item for item in d_name_sensitive_k if 
                    item not in d_name_outlier]
            )

    print(f"       b. {d_name_sensitive_k.shape[0]} sensitive drug(s)")
    
    if d_name_sensitive_k.shape[0] <= 1:
        return True, None, None, None, None, None, None, None, None, None

    #c=================================================================================
    # Train C_VAE and predictor 

    ## prepare data

    ## data wrangling wrt subgroup and torch loader prep.
    d_data_k, dataloaders_CP, cdr_outputs_dict = prepare_model_IO(
        train_data_type = "cell",
        c_data = c_data,
        d_data = d_data,
        cdr_all = cdr_all,
        c_names_k = c_names_k_init,
        d_names_k = d_name_sensitive_k,
        valid_size = valid_size, batch_size = batch_size,
        device = device
    )

    ##---------------------
    ## define optimizer
    optimizer_e = optim.Adam(model.parameters(), lr=lr)
    exp_lr_scheduler_e = lr_scheduler.ReduceLROnPlateau(optimizer_e)

    ##---------------------
    ## update C_VAE and predictor
    print(f"       c. Training C_VAE and Predictor")
    start = time.time()

    ## Update the Drug fixed encoding:
    model.update_fixed_encoding(
        c_X = torch.FloatTensor(c_data.values).to(device), 
        d_X = torch.FloatTensor(d_data_k.values).to(device))
    

    model, loss_train_hist, best_epo_c = train_CDPmodel_local(
        model=model,
        data_loaders=dataloaders_CP,
        outputs_dict = cdr_outputs_dict,
        VAE_loss_weight = C_VAE_loss_weight,
        recon_loss_weight = C_recon_loss_weight,
        kld_weight = C_kld_weight,
        cluster_distance_weight = C_cluster_distance_weight,
        update_ratio_weight = C_update_ratio_weight,
        predict_loss_weight = predict_loss_weight,
        optimizer=optimizer_e,
        n_epochs=n_epochs,
        scheduler=exp_lr_scheduler_e,
        save_path = c_p_save_path,
        use_mixture_kld = use_mixture_kld,
        use_weighted_bce = use_weighted_bce)
    
    end = time.time()
    print(f"            Running time: {end - start}")

    c_losses = get_train_VAE_predictor_hist_df(loss_train_hist, n_epochs, )


    #d=================================================================================
    # Cell lines with predicted sensitive outcome is assigned to the K-th cluster. Again, cell lines in cluster k with latent space that is not close to the centroid will be dropped from the cluster.
    
    # get predicted CDR
    model.update_fixed_encoding(
        c_X = torch.FloatTensor(c_data.values).to(device), 
        d_X = torch.FloatTensor(d_data_k.values).to(device))
    
    ## generate full list of predictions and update drug sensitivity list.
    with torch.no_grad():
        y_hatTensor = model.predictor(model.c_mu_fixed, model.d_mu_fixed)

    ### Get sensitive cells:
    c_sens_k = torch.zeros(c_data.shape[0])
    for i in range(c_data.shape[0]):
        c_sens_k[i] = torch.mean(y_hatTensor[cdr_outputs_dict['cell_idx'] == i]) > sens_cutoff
    # find sensitive drugs according predicted CDR
    c_name_sensitive_k = c_data.index.values[c_sens_k == 1]

    # remove outliers 
    if rm_cluster_outliers:
        c_sensitive_latent = model.c_mu_fixed[c_sens_k == 1]
        c_centroid = c_sensitive_latent.mean(dim=0)

        c_outlier_idx = find_outliers_3sd(c_sensitive_latent)
        c_name_outlier = c_data.index.values[c_outlier_idx]

        if len(c_outlier_idx) > 0:
            c_name_sensitive_k = np.array(
                [item for item in c_name_sensitive_k if 
                    item not in c_name_outlier]
            )

    print(f"       d. {c_name_sensitive_k.shape[0]} cancer cell line(s) in the cluster")

    if c_name_sensitive_k.shape[0] <= 1:
        return True, None, None, None, None, None, None, None, None, None

    losses_train_hist = [a_losses, c_losses]
    best_epos = [best_epo_a, best_epo_c]

    ## Collect the current centroid of both clusters.
    model.update_fixed_encoding(
        c_X = torch.FloatTensor(c_data_k.values).to(device), 
        d_X = torch.FloatTensor(d_data_k.values).to(device))
    
    c_centroid = torch.mean(model.c_mu_fixed, dim=0)
    d_centroid = torch.mean(model.d_mu_fixed, dim=0)
    
    c_sd = torch.std(model.c_mu_fixed, dim=0)
    d_sd = torch.std(model.d_mu_fixed, dim=0)

    return (
        False, 
        model, 
        c_centroid, d_centroid, 
        c_sd, d_sd, 
        c_name_sensitive_k, d_name_sensitive_k, 
        losses_train_hist, best_epos,
        )



def train_CDPmodel_local(
        model, 
        data_loaders, 
        outputs_dict, 
        VAE_loss_weight = 1, 
        recon_loss_weight = 1, 
        kld_weight = None, 
        cluster_distance_weight=100, 
        update_ratio_weight = 100,
        predict_loss_weight = 1, 
        sens_cutoff = 0.5,
        optimizer = None, 
        n_epochs=100, 
        scheduler=None,
        load=False, 
        save_path="model.pkl", 
        best_model_cache = "drive",
        use_mixture_kld = True,
        use_weighted_bce = False):
    
    if(load!=False):
        if(os.path.exists(save_path)):
            model.load_state_dict(torch.load(save_path))           
            return model, 0, 0
        else:
            logging.warning("Failed to load existing model file, proceed to the trainning process.")
    
    dataset_sizes = {x: data_loaders[x].dataset.data.shape[0] for x in ['train', 'val']}
    loss_hist = {}
    vae_loss_hist = {}
    recon_loss_hist = {}
    kld_hist = {}
    cluster_dist_hist = {}
    update_overlap_hist = {}
    
    prediction_loss_hist = {}

    n_current = data_loaders['train'].dataset.data.numpy().shape[0]
        
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
            running_update_overlap = 0.0
            running_vae_loss = 0.0
            running_recon_loss = 0.0
            running_kld_loss = 0.0
            running_cluster_d = 0.0
            

            n_iters = len(data_loaders[phase])

            batch_size = data_loaders[phase].batch_size

            # Iterate over data.
            # for data in data_loaders[phase]:
            for batchidx, (data, data_idx) in enumerate(data_loaders[phase]):

                ## presort the data to make subsequent operations simpler.
                
                # returns the indices of smallest to largest values:
                batch_sort_res = torch.sort(data_idx) 
                sort_idx = batch_sort_res[1]

                ## updated data and data_idx (sorted)
                data = data[sort_idx.numpy(), :]
                data_idx = batch_sort_res[0]

                ## Create mutable storage objects that correspond to full data sets 
                ## (where we store relevant batched model outputs)
                y_hat_mutable = torch.zeros(outputs_dict['cdr_vals'].shape)

                # pass data through the full model:
                c_mu, c_log_var, c_X_rec, d_mu, d_log_var, d_X_rec, y_hat_mod = model(
                    c_X = None if data_loaders['data_type'] == 'drug' else data, 
                    d_X = None if data_loaders['data_type'] == 'cell' else data, 
                    )
                

                ## collect the overall out_idx by train/valid split:
                if phase ==  'train':
                    out_idx = outputs_dict['cdr_idx_train']
                else:
                    out_idx = outputs_dict['cdr_idx_valid']
                
                ## this locates which of overall outputs in present in the current batch.
                if data_loaders['data_type'] == 'drug':
                    cdr_in_batch = np.isin(outputs_dict['drug_idx'], data_idx)
                else:
                    cdr_in_batch = np.isin(outputs_dict['cell_idx'], data_idx)

                ## TODO: In general we can write "sort-free" method path that is fast if batch-size == data-size.
                ## e.g:
                # if batch_size == n_current:
                #   cdr_in_batch = [True] * len(outputs_dict['drug_idx'])

                ## this identifies which of train/valid split is in ths current batch.
                out_idx_in_batch = out_idx[[cdr_in_batch[out_idx[i, 0]] for i in range(out_idx.shape[0])]]

                ## update a mutable y-hat object with the batch-updates outputs:
                y_hat_mutable[cdr_in_batch] = y_hat_mod.squeeze(-1)


                y = outputs_dict['cdr_vals'][out_idx_in_batch[:,0]]
                
                y_hat = y_hat_mutable[out_idx_in_batch[:,0]]
                ## TODO: remove after done debugging.
                # print("YHAT MAX AND MIN========================================")
                # print(torch.max(y_hat))
                # print(torch.min(y_hat))

                # compute loss
                mse = nn.MSELoss(reduction="sum")

                #   1. Prediction loss:
                bce = nn.BCELoss(reduction="none")
                prediction_losses = bce(y_hat, torch.tensor(y, dtype=torch.float32))
                if use_weighted_bce:
                    ## TODO: check approaches for weight values.
                    weight_zero = np.mean(y)
                    weight_one = 1 - weight_zero
                    weights = np.zeros(prediction_losses.shape[0])
                    sens = np.where(y)[0]
                    weights[sens] = weight_one
                    weights[weights==0] = weight_zero
                    weights = torch.Tensor(weights)
                    
                    prediction_loss = torch.sum(prediction_losses * weights) / torch.sum(weights)

                else:
                    prediction_loss = torch.mean(prediction_losses)


                ### Prepare various loss-related objects depending on the update modality:
                data_idx_np = data_idx.numpy()
                value_mapping = {data_idx_np[i]: i for i in range(data_idx_np.shape[0])}

                if data_loaders['data_type'] == 'drug':
                    mu, log_var, X_rec = d_mu, d_log_var, d_X_rec
                    idx_for_mu_repeats_if_full_batch = outputs_dict['drug_idx'][out_idx_in_batch]
                    idx_for_mu_repeats = np.vectorize(value_mapping.get)(idx_for_mu_repeats_if_full_batch)[:,0]
                    idx_k_old = outputs_dict['drug_idx_k_init']
                else:
                    mu, log_var, X_rec = c_mu, c_log_var, c_X_rec
                    idx_for_mu_repeats_if_full_batch = outputs_dict['cell_idx'][out_idx_in_batch]
                    idx_for_mu_repeats = np.vectorize(value_mapping.get)(idx_for_mu_repeats_if_full_batch)[:,0]
                    idx_k_old = outputs_dict['cell_idx_k_init']
                    
                # 2. the loss of latent spaces distances:
                #     - distances of cells/drugs in the predicted sensitive group to the cluster centroid 
                #     + distances of cells/drugs outside the predicted sensitive group to the cluster centroid 

                sensitive = y_hat > sens_cutoff
                sensitive = sensitive.long()
                
                ## TODO: remove below when done debugging.                
                # print("Sensitive Proportion:::::::")
                # print(torch.sum(sensitive) / sensitive.shape[0])
                # print(torch.mean(y_hat))
                # sens_k_act = np.zeros(len(data_idx))
                # for i in range(len(data_idx)):
                #         hat_mean = np.mean(y[idx_for_mu_repeats == i])
                #         sens_k_act[i] = hat_mean > sens_cutoff
                # print(np.sum(sens_k_act))
                ## TODO: remove above when done debugging.
                
                
                mu_full = mu[idx_for_mu_repeats].squeeze(1)
                latent_dist_loss = cluster_mu_distance(mu_full, sensitive)
                
                idx_of_sensitive = np.unique(idx_for_mu_repeats[np.where(sensitive)[0]])
                
                # 3. VAE loss: reconstruction loss & kld (can be contingent on the clusters.)
                if use_mixture_kld is False:
                    recon_loss, kld = custom_vae_loss(
                        data, mu, log_var, X_rec, mse)
                else:
                    ## TODO: pull Z from its original VAE generation
                    std_temp = torch.exp(0.5 * log_var)
                    z = mu + std_temp * torch.randn_like(std_temp)
                    z_full = z[idx_for_mu_repeats].squeeze(1)
                    log_var_full = log_var[idx_for_mu_repeats].squeeze(1)
                    recon_loss, kld = bicluster_gauss_vae_loss(
                        data, mu, log_var, X_rec, sensitive, idx_of_sensitive, mu_full, log_var_full, z_full,
                        mse)

                    
                ## default KLD weight if none is provided:
                if kld_weight is None:
                    kld_weight = batch_size / dataset_sizes[phase]

                # adding all up
                full_loss_no_overlap = (
                    VAE_loss_weight * (recon_loss_weight * recon_loss + kld_weight * kld) + 
                    cluster_distance_weight * latent_dist_loss + 
                    predict_loss_weight * prediction_loss
                    )


                # 4. requiring the updated cluster overlaping with the old clustering

                if update_ratio_weight != 0:
                    sens_k = torch.zeros(len(data_idx))
                    for i in range(len(data_idx)):
                        hat_mean = torch.mean(y_hat[idx_for_mu_repeats == i])
                        sens_k[i] = hat_mean > sens_cutoff
                    
                    #print("Num sensitive items::")
                    #print(torch.sum(sens_k).item())
                    ## total batch overlap with old cluster:
                    n_old_batch_olap = sum([c in data_idx for c in idx_k_old])

                    ## sensitive in-batch that exist in the old IDX group
                    n_sens_of_old_k = torch.sum(
                        sens_k[np.where(np.isin(list(value_mapping.keys()), idx_k_old))[0]]
                        )
                    
                    ## batch ID's for the old cluster items
                    value_mapping.items
                    if n_old_batch_olap > 0:
                        overlap_ratio = n_sens_of_old_k / n_old_batch_olap
                        overlap_loss = -overlap_ratio # at least 50% overlapping, the more overlap the better?
                    else:
                        overlap_loss = 0
                else:
                    overlap_loss = 0


                # Add up three losses:
                loss =  (
                    full_loss_no_overlap + 
                    update_ratio_weight * overlap_loss
                    )

                optimizer.zero_grad()

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    # addressing the instability by lowering the learning rate or use gradient clipping
                    # torch.nn.utils.clip_grad_norm_(d_vae_predictor.parameters(), 0.01)
                    # update the weights
                    optimizer.step()

                # compute loss statistics in-batch
                running_loss += loss.item()
                running_prediction_loss += (predict_loss_weight * prediction_loss).item()   
                running_update_overlap += (update_ratio_weight * overlap_loss)
                
                running_vae_loss += (recon_loss_weight * recon_loss + kld_weight * kld).item()
                running_recon_loss += (recon_loss_weight * recon_loss).item()
                running_kld_loss += (kld_weight * kld).item()
                running_cluster_d += (cluster_distance_weight * latent_dist_loss).item()

            ## Post-epoch averages
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_prediction_loss = running_prediction_loss / dataset_sizes[phase]
            epoch_update_overlap = running_update_overlap / (batchidx + 1)
        
            epoch_vae_loss = running_vae_loss / dataset_sizes[phase]
            epoch_recon_loss = running_recon_loss / dataset_sizes[phase]
            epoch_kld_loss = running_kld_loss / dataset_sizes[phase]
            epoch_cluster_d = running_cluster_d / dataset_sizes[phase]
            
            #if phase == 'train':
            #    scheduler.step(epoch_loss)
                
            # last_lr = scheduler.optimizer.param_groups[0]['lr']
            loss_hist[epoch,phase] = epoch_loss
            vae_loss_hist[epoch,phase] = epoch_vae_loss
            recon_loss_hist[epoch,phase] = epoch_recon_loss
            kld_hist[epoch,phase] = epoch_kld_loss
            cluster_dist_hist[epoch,phase] = epoch_cluster_d
            update_overlap_hist[epoch,phase] = epoch_update_overlap
            
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
                
    train_hist = [
        loss_hist, 
        vae_loss_hist, 
        recon_loss_hist, 
        kld_hist, 
        cluster_dist_hist, 
        update_overlap_hist, 
        prediction_loss_hist,
        ]

    # Select best model wts if use memory to cahce models
    if best_model_cache == "memory":
        torch.save(best_model_wts, save_path)
        model.load_state_dict(best_model_wts)  
    else:
        print(f'            Best epoc with test loss: epoch {best_epoch}')
        model.load_state_dict((torch.load(save_path+"_bestcahce.pkl")))
        torch.save(model.state_dict(), save_path)

    return model, train_hist, best_epoch

def train_VAE_train(vae, data_loaders={}, recon_loss_weight=1, kld_weight = None, optimizer=None, n_epochs=100, scheduler=None, load=False, save_path="vae.pkl", best_model_cache = "drive"):
    
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



def train_VAE(VAE_model, device, data, vae_type, save_path, params):
    valid_size = params['valid_size']
    n_epochs = params['n_epochs']
    batch_size = params['batch_size']
    if batch_size is None:
        batch_size = data.shape[0]
    lr =  params['lr']
    if vae_type == "C":
        recon_loss_weight = params['C_recon_loss_weight']
        kld_weight = params['C_kld_weight']
    if vae_type == "D":
        recon_loss_weight = params['D_recon_loss_weight']
        kld_weight = params['D_kld_weight']


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
