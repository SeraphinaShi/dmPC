    def fit_k_nrounds(
        self, k, n_rounds, cdr, c_data, c_meta, d_data, train_params, 
        c_meta_hist, d_sens_hist, return_latents, device,
        subcluster=False,
        c_centroids_sub=None, c_sds_sub=None, d_centroids_sub=None, d_sds_sub=None, 
        c_name_clusters_in_trainnig_sub=None, d_name_clusters_in_trainnig_sub=None):

        meta_key = "k" + str(k)
        c_meta_k = c_meta[[meta_key]].rename(columns={meta_key:'key'})

        losses_train_hist_list_k = []
        best_epos_k = []
            
        if return_latents:
            c_latent_k = []
            d_latent_k = []

        if subcluster:
            d_sens_hist_1 = pd.DataFrame()

	    # 1. Run the dual loop to train local models
        for b in range(0, n_rounds):
            print(f"     -- round {b} -------------")    
 
            if b == 0:
                if not subcluster:
                    d_sens_hist[f'sensitive_k{k}'] = (cdr.loc[c_meta_k.index.values[c_meta_k.key == 1]].mean(axis=0) > self.sens_cutoff).astype(int)
                    d_names_k_init = d_sens_hist.index.values[d_sens_hist[f'sensitive_k{k}']==1]
                    c_names_k_init = c_meta_k.index.values[c_meta_k.key == 1]
                    sensitive_cut_off = self.sens_cutoff

                else:
                    d_sens_hist_1[f'sensitive_k{k}'] = (cdr.loc[c_meta_k.index.values[c_meta_k.key == 1]].mean(axis=0) > 0.5).astype(int)
                    d_names_k_init = d_sens_hist_1.index.values[d_sens_hist_1[f'sensitive_k{k}']==1]
                    c_names_k_init = self.c_name_clusters_in_trainnig[k]
                    sensitive_cut_off = self.sens_cutoff/2
            else:
                d_names_k_init = d_sens_hist.index.values[d_sens_hist[f'sensitive_k{k}_b{b-1}']==1]
                if not subcluster:
                    c_names_k_init = c_meta_k.index.values[c_meta_k.key == 1] 
                else:
                    c_names_k_init = c_meta_hist.index.values[c_meta_hist[f'k{k}_sub_b{b-1}']==1]
                sensitive_cut_off = self.sens_cutoff

            if not subcluster:
                (zero_cluster, self.CDPmodel_list[k], 
                c_centroid, d_centroid, c_sd, d_sd, 
                c_name_cluster_k, d_name_sensitive_k, 
                losses_train_hist, best_epos,) = train_CDPmodel_local_1round(
                    self.CDPmodel_list[k], device, 
                    ifsubmodel = False, 
                    c_data = c_data, d_data = d_data, cdr_org = cdr, 
                    c_names_k_init = c_names_k_init, d_names_k_init = d_names_k_init, 
                    sens_cutoff = sensitive_cut_off, 
                    group_id = k, 
                    params = train_params
                    )
            else:
                (zero_cluster, self.CDPmodel_list_sub[k], 
                c_centroid, d_centroid, c_sd, d_sd, 
                c_name_cluster_k, d_name_sensitive_k, 
                losses_train_hist, best_epos,) = train_CDPmodel_local_1round(
                    self.CDPmodel_list_sub[k], device, 
                    ifsubmodel = True, 
                    c_data = c_data, d_data = d_data, cdr_org = cdr, 
                    c_names_k_init = c_names_k_init, 
                    d_names_k_init = d_names_k_init, 
                    sens_cutoff = sensitive_cut_off, 
                    group_id = k, params = train_params)
                
            ## update binarized column vectors of cell and drug sensitivity based on results.
            c_meta_k, d_sens_k = create_bin_sensitive_dfs(
                c_data, d_data, c_name_cluster_k, d_name_sensitive_k
            )
                
            if zero_cluster:
                if not subcluster:
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
                else:
                    print("  No subcluster found")
                        
                losses_train_hist_list_k.append(None)
                best_epos_k.append(None)

                if return_latents:
                    c_latent_k.append(None)
                    d_latent_k.append(None)

                break

            else:
                if b == n_rounds - 1:
                    if not subcluster:
                        self.which_non_empty_cluster.append(k)
                        self.nonzero_clusters += 1
                    else:
                        self.which_non_empty_subcluster.append(k)

                if not subcluster:
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
                else:
                    # store/update the centroids
                    c_centroids_sub[k] = c_centroid
                    c_sds_sub[k] = c_sd
                    d_centroids_sub[k] = d_centroid
                    d_sds_sub[k] = d_sd
                    c_name_clusters_in_trainnig_sub[k] = c_name_cluster_k
                    d_name_clusters_in_trainnig_sub[k] = d_name_sensitive_k

                    c_meta_hist[f'k{k}_sub_b{b}'] = c_meta_k.key
                    d_sens_hist[f'sensitive_k{k}_sub_b{b}'] = d_sens_k.sensitive
                    losses_train_hist_list_k.append(losses_train_hist)
                    best_epos_k.append(best_epos)

                    if return_latents:
                        c_latent = self.CDPmodel_list_sub[k].c_VAE.encode(torch.from_numpy(c_data.values).float().to(device), repram=False)
                        c_latent_k.append(c_latent.detach().numpy())
                        d_latent = self.CDPmodel_list_sub[k].d_VAE.encode(torch.from_numpy(d_data.values).float().to(device), repram=False)
                        d_latent_k.append(d_latent.detach().numpy())
            
        return (zero_cluster, c_latent_k, d_latent_k, 
        d_sens_hist, c_meta_hist, losses_train_hist_list_k, best_epos_k, 
        c_centroids_sub, c_sds_sub, d_centroids_sub, d_sds_sub, 
        c_name_clusters_in_trainnig_sub, d_name_clusters_in_trainnig_sub)
