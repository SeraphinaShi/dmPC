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

def custom_vae_loss(X, mu, log_var, X_rec, reconstruction_loss_function, cluster_key, use_mixture_kld=False, epsilon=1e-8):
    """
    Calculate the custom VAE loss, optionally using a mixture model for KLD.
    
    Parameters:
    - X: original input data
    - mu: latent mean
    - log_var: logarithm of latent variance
    - X_rec: reconstructed input
    - reconstruction_loss_function: function to compute reconstruction loss
    - cluster_key: indicates cluster membership
    - use_mixture_kld: flag to use a mixture model for KLD calculation
    - epsilon: small constant to avoid division by zero or log(0)
    
    Returns:
    - Reconstruction loss and KLD loss.
    """
    # Reconstruction loss
    recon_loss = reconstruction_loss_function(X_rec, X)

    if not use_mixture_kld:
        kld_element = 1 + log_var - mu.pow(2) - log_var.exp()
        kld = -0.5 * torch.sum(kld_element)        
    else:
        cluster_key = cluster_key.view(-1)
        mu_in_cluster = mu[cluster_key == 1]
        mu_out_cluster = mu[cluster_key == 0]

        # Initialize variables to ensure they are defined before use
        in_cluster_kld = out_cluster_kld = 0

        # Compute in-cluster variance if applicable
        if mu_in_cluster.shape[0] > 1:
            in_clust_var = safe_variance(mu_in_cluster, epsilon)
            in_cluster_kld = 0.5 * (torch.sum(in_clust_var) + torch.sum(torch.log(in_clust_var + epsilon)))
        
        # Compute out-cluster variance if applicable
        if mu_out_cluster.shape[0] > 1:
            out_clust_var = safe_variance(mu_out_cluster, epsilon)
            out_cluster_kld = 0.5 * (torch.sum(out_clust_var) + torch.sum(torch.log(out_clust_var + epsilon)))
        
        kld = in_cluster_kld + out_cluster_kld

    return recon_loss, kld

def safe_variance(samples, epsilon=1e-8):
    """
    Computes the variance of samples, ensuring numerical stability.

    Parameters:
    - samples: Tensor of samples
    - epsilon: Small constant to ensure non-zero division

    Returns:
    - Variance of samples with epsilon added for stability
    """
    if samples.shape[0] > 1:
        dist_sq = torch.square(samples - samples.mean(dim=0))
        var = torch.sum(dist_sq, axis=0) / (dist_sq.shape[0] - 1)
        return var + epsilon
    else:
        # Return a small value (epsilon) if not enough samples to compute variance
        return torch.tensor(epsilon, device=samples.device)

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


def cluster_mu_distance(mu, sensitive, device = 'cpu'):
    """
    recon_x: regenerated X
    x: origin X
    mu: latent mean
    logvar: latent log variance
    labels: sample labels
    """
    # Cluster distance loss
    # sensitive cluster centroid
    sensitive = sensitive.view(-1).to(device)
    cluster_mu = mu[sensitive==1].to(device)
    other_mu = mu[sensitive==0].to(device)
    
    if(sum(sensitive) == 0):
        centroid = torch.zeros(1, mu.shape[1]).to(device)
    else:
        centroid = cluster_mu.mean(dim=0).to(device)

    # within cluster distance
    cluster_distances = torch.cdist(cluster_mu, centroid.view(1, -1)).to(device)
    cluster_distances = torch.abs(cluster_distances).to(device)
    within_cluster_distance = cluster_distances.mean().to(device)

    # outsiders' distances to this centroid
    out_cluster_d = torch.cdist(other_mu, centroid.view(1, -1)).to(device)
    out_cluster_d = torch.abs(out_cluster_d).to(device)
    out_cluster_distance = out_cluster_d.mean().to(device)

    d_loss = within_cluster_distance - out_cluster_distance
    if torch.isnan(d_loss).any().item():
        d_loss = torch.zeros_like(d_loss)
    return d_loss

