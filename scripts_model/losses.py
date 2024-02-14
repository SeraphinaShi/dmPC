import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.utils
import torch.distributions
import numpy as np

import copy
from copy import deepcopy
import logging
import os

from statistics import mean


def custom_vae_loss(X, mu, log_var, X_rec, reconstruction_loss_function):
    """
    recon_x: regenerated X
    x: origin X
    mu: latent mean
    logvar: latent log variance
    """
    recon_loss = reconstruction_loss_function(X_rec, X)

    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
    kld = torch.sum(kld_element).mul_(-0.5)

    return recon_loss, kld

def bicluster_gauss_vae_loss(
    X, mu, log_var, X_rec, 
    sensitive_vec, idx_of_sensitive,
    mu_full, log_var_full, z_full,
    reconstruction_loss_function,
    divergence_tol = 1e12,
    ):
    ## This 'mixture of Gaussians' KLD is taken from https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009086 
    ## It is essentially just ignoring that the means are all equal to 0.
    recon_loss = reconstruction_loss_function(X_rec, X) ## same as before:
    
    ## assumes \mu_0 = \mu_1
    ## $$loss_{KLD} = 0.5 * {tr(\Sigma_1^{-1} \Sigma_0) - k + ln\frac{|\Sigma_1|}{|\Sigma_0|}}$$
    ## estimated variances 
    
    # code_size = 20
    # ## TODO: check with TFD implementation that the authors use.
    # batch_var_avgs = torch.mean(log_var.exp(), dim=0)
    # ones_latent_dim = torch.ones(mu.shape[1])
    
    # kld = 0.5 * (torch.sum(ones_latent_dim / batch_var_avgs) - code_size + torch.log(torch.prod(batch_var_avgs) / torch.prod(ones_latent_dim)))
    rev_sensitive_vec = 1 - sensitive_vec
    
    if idx_of_sensitive.shape[0] <= 1:
        in_cluster_kld = 0
        out_clust_samp_dist_sq = torch.square(z_full)
        out_clust_samp_var = torch.sum(out_clust_samp_dist_sq) / (z_full.shape[0] - 1)
    else:
        _, _, _, _, centroid = (
            compute_centroid_and_dists(mu_full, sensitive_vec)
            )
    
        z_full_in_clust = z_full.index_select(0, sensitive_vec.nonzero().squeeze())
        z_full_out_clust = z_full.index_select(0, rev_sensitive_vec.nonzero().squeeze())

        in_clust_samp_dist_sq = torch.square(z_full_in_clust - centroid)
        out_clust_samp_dist_sq = torch.square(z_full_out_clust)

        in_clust_samp_var = torch.sum(in_clust_samp_dist_sq, axis=0) / (in_clust_samp_dist_sq.shape[0] - 1)
        out_clust_samp_var = torch.sum(out_clust_samp_dist_sq, axis=0) / (out_clust_samp_dist_sq.shape[0] - 1)

        if np.sum(np.logical_and(
            in_clust_samp_var.detach().numpy() > divergence_tol,
            ~np.isinf(in_clust_samp_var.detach().numpy()))) > 0:
            raise ValueError("Variance along at least one latent dimension for the sensitive group has diverged.")
        ## In-Cluster KLD:
        in_cluster_kld = 0.5 * (torch.sum(in_clust_samp_var) + torch.sum(torch.log(in_clust_samp_var)))
    
    ## Out-Cluster KLD:
    if (torch.sum(torch.isinf(out_clust_samp_var)).item() > 0 or 
        torch.sum(rev_sensitive_vec).item() == 0): 
        ## occasionally (w/ small batch size), less than two nonsensitive items are processed.
        out_cluster_kld = 0
    else:
        out_cluster_kld = 0.5 * (torch.sum(out_clust_samp_var) + torch.sum(torch.log(out_clust_samp_var)))

    kld = in_cluster_kld + out_cluster_kld
    
    if torch.isinf(kld).item() is True:
        print('stop check')
    
    # centroids_expanded = tf.expand_dims(centroids, axis=0)
    # code_expanded = tf.expand_dims(code, axis=1)
    # dist_samples = code_expanded - centroids_expanded
    # dist_samples_squared = tf.square(dist_samples)
    # gates_assignments_expanded = tf.expand_dims(gates_assignments, axis=-1)
    # dist_samples_mask = dist_samples_squared * gates_assignments_expanded

    # std_unnormalized = tf.reduce_sum(dist_samples_mask, axis=0)
    # normalizing_factors = tf.clip_by_value(tf.reduce_sum(gates_assignments_expanded, axis=0), 1, 1e20)
    # std_normalized = std_unnormalized / normalizing_factors
    # std_normalized = tf.where(tf.equal(std_normalized, 0.), tf.ones_like(std_normalized), std_normalized)
    # kl_divergence_code_standard_gaussian = 0.5 * (
    #     tf.reduce_sum(std_normalized, axis=1) - 
    #     code_size * tf.ones(model_hparams['num_experts']) - 
    #     tf.reduce_sum(tf.log(std_normalized), axis=1))
    
    # loss_kl_divergence_code_standard_gaussian = tf.reduce_mean(kl_divergence_code_standard_gaussian)

    return recon_loss, kld

def c_cluster_mu_distance(mu, cluster_key):
    """
    mu: latent mean
    labels: sample labels
    """
    # Cluster distance loss
    # k cluster centroid
    cluster_key = cluster_key.view(-1)
    cluster_mu = mu[cluster_key == 1]
    other_mu = mu[cluster_key != 1]

    if(sum(cluster_key == 1) == 0):
        centroid = torch.zeros(1, mu.shape[1])
    else:
        centroid = cluster_mu.mean(dim=0)

    # within cluster distance
    cluster_distances = torch.cdist(cluster_mu, centroid.view(1, -1))
    cluster_distances = torch.abs(cluster_distances)
    within_cluster_distance = cluster_distances.mean()
    
    # other samples distance to this centroid
    if(sum(cluster_key == 0) == 0):
        out_cluster_distance = 0
    else:
        out_cluster_d = torch.cdist(other_mu, centroid.view(1, -1))
        out_cluster_d = torch.abs(out_cluster_d)
        out_cluster_distance = out_cluster_d.mean()
    
    d_loss = within_cluster_distance - out_cluster_distance

    return d_loss


def compute_centroid_and_dists(mu, sensitive):
    # Cluster distance loss
    # sensitive cluster centroid
    sensitive = sensitive.view(-1)
    cluster_mu = mu[sensitive==1]
    other_mu = mu[sensitive==0]
    
    if(sum(sensitive) == 0):
        centroid = torch.zeros(1, mu.shape[1])
    else:
        centroid = cluster_mu.mean(dim=0)

    # within cluster distance
    cluster_distances = torch.cdist(cluster_mu, centroid.view(1, -1))
    cluster_distances = torch.abs(cluster_distances)
    within_cluster_distance = cluster_distances.mean()

    # outsiders' distances to this centroid
    out_cluster_d = torch.cdist(other_mu, centroid.view(1, -1))
    out_cluster_d = torch.abs(out_cluster_d)
    out_cluster_distance = out_cluster_d.mean()
    
    return within_cluster_distance, out_cluster_distance, cluster_distances, out_cluster_d, centroid

def cluster_mu_distance(mu, sensitive):
    """
    recon_x: regenerated X
    x: origin X
    mu: latent mean
    logvar: latent log variance
    labels: sample labels
    """
    
    within_cluster_distance, out_cluster_distance, _, _, _ = (
        compute_centroid_and_dists(mu, sensitive)
        )

    d_loss = within_cluster_distance - out_cluster_distance
    if torch.isnan(d_loss).any().item():
        d_loss = torch.zeros_like(d_loss)
    return d_loss

