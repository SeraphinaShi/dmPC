import os
import git 
from pathlib import Path

def get_project_root():
    return Path(git.Repo('.', search_parent_directories=True).working_tree_dir)

root = get_project_root()

os.chdir(root)
os.getcwd()

plot_folder = "results/images/simulations/"

import argparse
import logging
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200

import torch
from torch import nn, optim, Tensor
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

print('pytorch version:', torch.__version__)
print('orig num threads:', torch.get_num_threads())

sys.path.append(os.getcwd() + "/scripts_model")

from models import *
from trainers import *
from losses import *
from utils import *
from visuals import *


import random
seed=42

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


simu_folder = "data/simulations"
RNAseq = pd.read_csv(os.path.join(simu_folder, "simu2_RNAseq.csv"), index_col = 0)
RNAseq_meta = pd.read_csv(os.path.join(simu_folder, "simu2_RNAseq_meta.csv"), index_col = 0)
d_fp = pd.read_csv(os.path.join(simu_folder, "simu2_d_fp.csv"), index_col = 0)
# cdr = pd.read_csv(os.path.join(simu_folder, "simu2_cdr.csv"), index_col = 0)
cdr = pd.read_csv(os.path.join(simu_folder, "simu2.2_cdr_noise.csv"), index_col = 0)


RNAseq_meta['C_type'] = RNAseq_meta['C_type'].replace('grp3', 'grp0')


c_data = RNAseq.T

# originally
c_meta, meta_map = get_CCL_meta_codes(RNAseq.columns.values, RNAseq_meta)

print(f"Cancer type coding map: ")
print(meta_map)

d_data = d_fp.T

cdr = cdr
cdr.index = cdr.index.astype("str")


num_cluster = 3

# only two groups
two_grp = True
if two_grp:
    num_cluster = 2
    RNAseq_meta.loc[RNAseq_meta.C_type=='grp2', 'C_type'] = 'grp1'
    
    c_meta_true = c_meta
    c_meta, meta_map = get_CCL_meta_codes(RNAseq.columns.values, RNAseq_meta)
    print(f"Cancer type coding map: ")
    print(meta_map)



c_train, c_test = train_test_split(c_data, test_size=0.15)

c_meta_train = get_CCL_meta(c_train.index.values, c_meta)
c_meta_test = get_CCL_meta(c_test.index.values, c_meta)

cdr_train_idx = np.isin(cdr.index.values, c_train.index.values)
cdr_train = cdr[cdr_train_idx]
cdr_test = cdr[~cdr_train_idx]

if two_grp:
    c_meta_train_true = get_CCL_meta(c_train.index.values, c_meta_true)
    c_meta_test_true = get_CCL_meta(c_test.index.values, c_meta_true)


print(f"Training data: \n   cdr: {cdr_train.shape}\n   c_data: {c_train.shape} \n   d_data: {d_data.shape}")
print(f"   Number of each initial cancer clusters: \n")
if two_grp:
    print(pd.crosstab(c_meta_train_true['code'], c_meta_train['code'], margins=True, margins_name="Total"))
else:
    print(c_meta_train['code'].value_counts())

print(f"\nTesting data:  \n   cdr: {cdr_test.shape}\n   c_data: {c_test.shape} \n   d_data: {d_data.shape}")
print(f"   Number of each initial cancer clusters: \n")
if two_grp:
    print(pd.crosstab(c_meta_test_true['code'], c_meta_test['code'], margins=True, margins_name="Total"))
else:
    print(c_meta_test['code'].value_counts())



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_device(device)
print(device)


class Train_Args:
    def __getitem__(self, key):
        return getattr(self, key)
    def __setitem__(self, key, val):
        setattr(self, key, val)
    def __contains__(self, key):
        return hasattr(self, key)

    valid_size = 0.2 #@param {type: "float"}

    n_epochs = 100 #@param {type: "integer"}
    batch_size = None #@param {type: "integer"}
    lr = 0.01 #@param {type: "float"}
    
    C_VAE_loss_weight = 0.1 #@param {type: "float"}
    C_recon_loss_weight = 1 #@param {type: "float"}
    C_kld_weight = 0.5 #@param {type: "float"}
    C_cluster_distance_weight = 150 #@param {type: "float"}  
    C_update_ratio_weight = 0 #@param {type: "float"}
    
    D_VAE_loss_weight = 0.5 #@param {type: "float"}
    D_recon_loss_weight = 1 #@param {type: "float"}
    D_kld_weight = 0.25 #@param {type: "float"}
    D_cluster_distance_weight = 50 #@param {type: "float"}
    D_update_ratio_weight = 0 #@param {type: "float"}
    
    predict_loss_weight = 2500 #@param {type: "float"}

    rm_cluster_outliers = True #@param {type: "bool"}
    
    cVAE_save_path = 'data/model_fits/GDSC_simu2.2_c_vae' #@param
    dVAE_save_path = 'data/model_fits/GDSC_simu2.2_d_vae' #@param
    
    c_p_save_path = 'data/model_fits/GDSC_simu2.2_c_vae_predictor' #@param
    d_p_save_path = 'data/model_fits/GDSC_simu2.2_d_vae_predictor' #@param
    

class CDPModel_sub_Args:
    def __getitem__(self, key):
        return getattr(self, key)
    def __setitem__(self, key, val):
        setattr(self, key, val)
    def __contains__(self, key):
        return hasattr(self, key)

    # c_VAE
    c_input_dim = 0 #@param {type: "integer"}
    c_h_dims = [64] #@param {type: "vactor"}
    c_latent_dim = 32 #@param {type: "integer"}

    # d_VAE
    d_input_dim = 0 #@param {type: "integer"}
    d_h_dims = [64]  #@param {type: "vactor"}
    d_latent_dim = 32 #@param {type: "integer"}

    # predictor
    p_sec_dim = 32 #@param {type: "integer"}
    p_h_dims = [p_sec_dim*2, 16]  #@param {type: "vactor"}
    
    # all
    drop_out = 0  #@param {type: "float"}
    
    # sensitive threshold
    sens_cutoff = 0.5

train_args = Train_Args()

K = len(c_meta['code'].unique())

CDPmodel_args = CDPModel_sub_Args()
CDPmodel_args['c_input_dim'] = c_data.shape[1] 
CDPmodel_args['d_input_dim'] = d_data.shape[1]


if CDPmodel_args['c_input_dim'] <= 0:
  warnings.warn(
      '''\nCancer Cell line feat222iiZZXCx9MWWMW                           ure number not specified''')
if CDPmodel_args['d_input_dim'] <= 0:
  warnings.warn(
      '''\nDrug feature number not specified''')
  



CDPmodel = CDPmodel(K, CDPmodel_args)
n_rounds = 5
fit_returns = CDPmodel.fit(c_train, c_meta_train, d_data, cdr_train, train_args, n_rounds=n_rounds, search_subcluster=True, device = device)
c_meta, c_meta_hist, d_sens_hist, losses_train_hist_list, best_epos_list, C_VAE_init_losses, D_VAE_init_losses, c_latent_list, d_latent_list = fit_returns



