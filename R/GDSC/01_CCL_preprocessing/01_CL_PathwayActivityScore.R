
###############################################################################
## GDSC1000
### SPEED Pathway activity scores
###############################################################################
path_score_org <- read_excel(here(GDSC_raw_files, "GDSC1000",  "TableS5A.xlsx"), skip=4)

path_score <- path_score_org %>% as.data.frame()
names(path_score)[1] <- "sample_name"

dim(path_score) # [1] 968  12

# get cosmic_id for each cell line sample name
ind <- match(path_score$sample_name, cl_meta$sample_name)

identical(cl_meta$sample_name[ind], path_score$sample_name) # check, we have some NA values
path_score$COSMIC_ID <- cl_meta$COSMIC_ID[ind]
table(is.na(path_score$COSMIC_ID))
# FALSE  TRUE 
# 849   119

# remove cell lines without COSMIC_ID
path_score <- path_score[!is.na(path_score$COSMIC_ID),]
row.names(path_score) <- path_score$COSMIC_ID
path_score <- path_score %>% select(-sample_name, -COSMIC_ID)
dim(path_score) # [1] 849  11

path_score <- t(path_score) %>% as.data.frame()
dim(path_score) # [1]  11 849
range(path_score)

#############################
## Output file
#############################
write.csv(path_score, file = here(dataOutFolder, "GDSC_pathway_activity_score.csv"))




#############################
## Summary
#############################
dim(path_score) #11 849

path_score[, 1:6]


#############################
## Some visual
#############################
library(umap)
library(ggplot2)

set.seed(123)

# UMAP
path_score_umap <- 
  uwot::umap(t(path_score),
             pca = 3,
             n_neighbors = 10)

path_score_umap_df <- 
  data.frame(path_score_umap) %>% 
  mutate(COSMIC_ID = colnames(path_score)) 

names(path_score_umap_df) <- c("UMAP1", "UMAP2", "COSMIC_ID")
path_score_umap_df <- merge(path_score_umap_df, cl_meta, by="COSMIC_ID", all.x = T)

ggplot(path_score_umap_df) + 
  geom_point(aes(x=UMAP1,y=UMAP2, color=gender)) + 
  labs(x="UMAP1", y="UMAP2") + 
  theme_bw() +
  ggtitle("UMAP of Pathway Activity Score\ncolored by gender")

ggplot(path_score_umap_df) + 
  geom_point(aes(x=UMAP1,y=UMAP2, color=age_at_sampling)) + 
  labs(x="UMAP1", y="UMAP2") + 
  theme_bw() +
  ggtitle("UMAP of Pathway Activity Score\ncolored by age_at_sampling")

ggplot(path_score_umap_df) + 
  geom_point(aes(x=UMAP1,y=UMAP2, color=tissue)) + 
  labs(x="UMAP1", y="UMAP2") + 
  theme_bw() +
  ggtitle("UMAP of Pathway Activity Score\ncolored by tissue")

ggplot(path_score_umap_df) + 
  geom_point(aes(x=UMAP1,y=UMAP2, color=cancer_type)) + 
  labs(x="UMAP1", y="UMAP2") + 
  theme_bw() +
  ggtitle("UMAP of Pathway Activity Score\ncolored by cancer type")


