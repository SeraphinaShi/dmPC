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

