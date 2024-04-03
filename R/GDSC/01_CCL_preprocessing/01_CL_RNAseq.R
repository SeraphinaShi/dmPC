
###############################################################################
## Cell Model Passports
###############################################################################
rna_rpkm_org <- read.csv(file=here(GDSC_raw_files, "CellModelPassports", "rnaseq_all_20220624", "rnaseq_fpkm_20220624.csv"))
rna_rpkm_org <- fread(file=here(GDSC_raw_files, "CellModelPassports", "rnaseq_all_20220624", "rnaseq_fpkm_20220624.csv"),
                      header = TRUE)

rna_rpkm <- rna_rpkm_org[-c(1:4),]
names(rna_rpkm)[1:2] <- c("gene_id","symbol")

# get rowname <- gene_id
row.names(rna_rpkm) <- rna_rpkm$gene_id
rna_rpkm <- rna_rpkm[,-c(1,2)]
dim(rna_rpkm) #[1] 37602  1431

# remove cell lines that are not in the cleaned meta data
table(names(rna_rpkm) %in% cl_meta$model_id) # FALSE: 308
rna_rpkm <- data.frame(rna_rpkm)
rna_rpkm <- rna_rpkm[, names(rna_rpkm) %in% cl_meta$model_id]
dim(rna_rpkm) # [1] 37602  1123

# change model_id to cosmic_id
ind <- match(names(rna_rpkm), cl_meta$model_id)

identical(cl_meta$model_id[ind], names(rna_rpkm)) # check

old_names <- names(rna_rpkm)
new_names <- cl_meta$COSMIC_ID[ind]
names(rna_rpkm) <- new_names

# some cell lines don't have COSMIC_ID
table(is.na(names(rna_rpkm))) # TRUE: 224
names(rna_rpkm)[is.na(names(rna_rpkm))] <- old_names[is.na(names(rna_rpkm))]
table(is.na(names(rna_rpkm))) # TRUE: 0

rna_rpkm <- rna_rpkm %>% 
  mutate_if(sapply(., is.character), as.numeric)

dim(rna_rpkm) #[1] 37602  1123


#############################
## Output big file
#############################

write.csv(rna_rpkm, file = here(dataOutFolder, "GDSC_RNAseq_rpkm_allGenes.csv"))



#############################
## Filtering + Scaling
#############################

### remove lowly expressed genes
row_means <- apply(rna_rpkm, 1, mean)
rna_rpkm <- rna_rpkm[which(row_means > 1), ]
dim(rna_rpkm) # 14257  1123

### log2-transform CCLE data
log2_rna_rpkm = log2(rna_rpkm + 1)
dim(log2_rna_rpkm) #[1] 14257  1123

### choose variably expressed genes
row_vars <- apply(log2_rna_rpkm, 1, var)
variable_genes <- names(which(row_vars > quantile(row_vars, 0.6)))

log2_rna_rpkm_var_genes = log2_rna_rpkm[variable_genes,]
dim(log2_rna_rpkm_var_genes) #[1] 5703 1123

### scaled 
scaled_rpkm_var_genes = apply(log2_rna_rpkm_var_genes, 1, scale) %>% 
  t() %>% 
  as.data.frame()
names(scaled_rpkm_var_genes) <- colnames(log2_rna_rpkm_var_genes)
dim(scaled_rpkm_var_genes) # [1] 5703 1123
range(scaled_rpkm_var_genes)

### sample subset (keep the cell lines with COSMIC-ID)
scaled_rpkm_var_genes_cosmicID <- scaled_rpkm_var_genes[, names(scaled_rpkm_var_genes) %in% cl_meta$COSMIC_ID]
dim(scaled_rpkm_var_genes_cosmicID) # [1] 5703  899

write.csv(scaled_rpkm_var_genes, file = here(dataOutFolder, "GDSC_RNAseq_rpkm_all_cl.csv"))
write.csv(scaled_rpkm_var_genes_cosmicID, file = here(dataOutFolder, "GDSC_RNAseq_rpkm_cosmic_cl.csv"))



scaled_rpkm_var_genes_lung <- scaled_rpkm_var_genes[, names(scaled_rpkm_var_genes) %in% cl_meta_lung$COSMIC_ID]
write.csv(scaled_rpkm_var_genes_lung, file = here(dataOutFolder, "GDSC_lung_RNAseq_rpkm.csv"))

#############################
## Summary
#############################
rna_seq <- scaled_rpkm_var_genes_cosmicID
dim(rna_seq) #[1] 5703 899

rna_seq[1:10, 1:6]


#############################
## Some visual
#############################
library(umap)
library(ggplot2)

set.seed(123)

# UMAP
rna_seq_umap <- 
  uwot::umap(t(rna_seq),
             pca = 25,
             n_neighbors = 10)

rna_seq_umap_df <- 
  data.frame(rna_seq_umap) %>% 
  mutate(COSMIC_ID = colnames(rna_seq)) 

names(rna_seq_umap_df) <- c("UMAP1", "UMAP2", "COSMIC_ID")
rna_seq_umap_df <- merge(rna_seq_umap_df, cl_meta, by="COSMIC_ID", all.x = T)

ggplot(rna_seq_umap_df) + 
  geom_point(aes(x=UMAP1,y=UMAP2, color=gender)) + 
  labs(x="UMAP1", y="UMAP2") + 
  theme_bw() +
  ggtitle("UMAP of RNAseq_rpkm\ncolored by gender")

ggplot(rna_seq_umap_df) + 
  geom_point(aes(x=UMAP1,y=UMAP2, color=age_at_sampling)) + 
  labs(x="UMAP1", y="UMAP2") + 
  theme_bw() +
  ggtitle("UMAP of RNAseq_rpkm\ncolored by age_at_sampling")

ggplot(rna_seq_umap_df) + 
  geom_point(aes(x=UMAP1,y=UMAP2, color=tissue)) + 
  labs(x="UMAP1", y="UMAP2") + 
  theme_bw() +
  ggtitle("UMAP of RNAseq_rpkm\ncolored by tissue")

ggplot(rna_seq_umap_df) + 
  geom_point(aes(x=UMAP1,y=UMAP2, color=cancer_type)) + 
  labs(x="UMAP1", y="UMAP2") + 
  theme_bw() +
  ggtitle("UMAP of RNAseq_rpkm\ncolored by cancer type")



