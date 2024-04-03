
###############################################################################
## GDSC1000
### RMA Normalized
###############################################################################
rma_org <- fread(here(GDSC_raw_files, "GDSC1000",  "Cell_line_RMA_proc_basalExp.txt"))

rma <- rma_org %>% as.data.frame()
colnames(rma) <- gsub("DATA.", "", colnames(rma))
dim(rma) # [1] 17737  1020

table(duplicated(rma$GENE_SYMBOLS))
dup_gene <- rma$GENE_SYMBOL[duplicated(rma$GENE_SYMBOLS)] 
unique(dup_gene) # ""

# remove genes without name/SYMBOLS
rma <- rma %>% filter(! GENE_SYMBOLS %in% dup_gene)
dim(rma) # [1] 17419  1020

# get rowname <- GENE_SYMBOL
rma_gene_names <- rma$GENE_SYMBOLS
rma_gene_titles <- rma$GENE_title
row.names(rma) <- rma_gene_names
rma <- rma[,-c(1,2)]

dim(rma) # [1] 17419  1018

# remove cell lines that are not in the cleaned meta data
table(names(rma) %in% cl_meta$COSMIC_ID) # FALSE: 144
rma <- rma[, names(rma) %in% cl_meta$COSMIC_ID]
dim(rma) # [1] 17419   874

#############################
## Output big file
#############################

write.csv(rma, file = here(dataOutFolder, "GDSC_RMA_allGenes.csv"))



#############################
## Filtering 
#############################
# 
# ### remove lowly expressed genes
# row_means <- apply(rma, 1, mean)
# rma <- rma[which(row_means > 1), ]
# dim(rma)
# 

### choose variably expressed genes
row_vars <- apply(rma, 1, var)
variable_genes <- names(which(row_vars > quantile(row_vars, 0.6)))

rma_var_genes = rma[variable_genes,]
dim(rma_var_genes) #[1] 6968  874
range(rma_var_genes) # [1]  2.094251 13.770006

write.csv(rma_var_genes, file = here(dataOutFolder, "GDSC_RMA.csv"))

rma_var_genes_lung <- rma_var_genes[, names(rma_var_genes) %in% cl_meta_lung$COSMIC_ID]
write.csv(rma_var_genes_lung, file = here(dataOutFolder, "GDSC_RMA_lung.csv"))

#############################
## Summary
#############################
rma_seq <- rma_var_genes
dim(rma_seq) #[1] 6968  874

rma_seq[1:10, 1:6]


#############################
## Some visual
#############################
library(umap)
library(ggplot2)

set.seed(123)

# UMAP
rma_seq_umap <- 
  uwot::umap(t(rma_seq),
             pca = 25,
             n_neighbors = 10)

rma_seq_umap_df <- 
  data.frame(rma_seq_umap) %>% 
  mutate(COSMIC_ID = colnames(rma_seq)) 

names(rma_seq_umap_df) <- c("UMAP1", "UMAP2", "COSMIC_ID")
rma_seq_umap_df <- merge(rma_seq_umap_df, cl_meta, by="COSMIC_ID", all.x = T)

ggplot(rma_seq_umap_df) + 
  geom_point(aes(x=UMAP1,y=UMAP2, color=gender)) + 
  labs(x="UMAP1", y="UMAP2") + 
  theme_bw() +
  ggtitle("UMAP of RMA\ncolored by gender")

ggplot(rma_seq_umap_df) + 
  geom_point(aes(x=UMAP1,y=UMAP2, color=age_at_sampling)) + 
  labs(x="UMAP1", y="UMAP2") + 
  theme_bw() +
  ggtitle("UMAP of RMA\ncolored by age_at_sampling")

ggplot(rma_seq_umap_df) + 
  geom_point(aes(x=UMAP1,y=UMAP2, color=tissue)) + 
  labs(x="UMAP1", y="UMAP2") + 
  theme_bw() +
  ggtitle("UMAP of RMA\ncolored by tissue")

ggplot(rma_seq_umap_df) + 
  geom_point(aes(x=UMAP1,y=UMAP2, color=cancer_type)) + 
  labs(x="UMAP1", y="UMAP2") + 
  theme_bw() +
  ggtitle("UMAP of RMA\ncolored by cancer type")










