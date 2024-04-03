###############################################################################
## GDSC1000 source
###############################################################################
#------------------------------------------------------------------------------
# TableS2C.xlsx
#------------------------------------------------------------------------------
CL_sequence_variants <- read_excel(here(GDSC_raw_files, "GDSC1000", "TableS2C.xlsx"),
                                   skip = 20)
CL_variants <- CL_sequence_variants %>% 
  select(COSMIC_ID, Gene) 

table((CL_variants$COSMIC_ID == ""))

CL_variants <- CL_variants %>%
  filter(!is.na(COSMIC_ID)) %>%
  unique() 

# count the samples and variants in the table
length(unique(CL_variants$COSMIC_ID)) # [1] 1001
length(unique(CL_variants$Gene)) # [1] 19100


# keep cell lines in the cleaned meta data
CL_variants <- CL_variants[CL_variants$COSMIC_ID %in% cl_meta$COSMIC_ID, ]
length(unique(CL_variants$COSMIC_ID)) # [1] 877
length(unique(CL_variants$Gene)) # [1] 19053

# Convert to binary event matrix
library(reshape2)
CL_variants <- dcast(CL_variants, 
                     formula = Gene ~ COSMIC_ID, fun.aggregate = length)

row.names(CL_variants) <- CL_variants$Gene
CL_variants <- CL_variants[, -1]

dim(CL_variants) # [1] 19053   877



# summary
variant_counts_each_cl <- apply(CL_variants, MARGIN = 2, FUN = sum)
summary(variant_counts_each_cl)
hist(variant_counts_each_cl)

cl_counts_each_variant <- apply(CL_variants, MARGIN = 1, FUN = sum)
summary(cl_counts_each_variant)
hist(cl_counts_each_variant)

#############################
## Output file
#############################
write.csv(CL_variants, file = here(dataOutFolder, "GDSC_variants_all_mutations.csv"))

#############################
## Output file for lung cancer
#############################
CL_lung_variants <- CL_variants[,names(CL_variants) %in% cl_meta_lung$COSMIC_ID]
write.csv(CL_lung_variants, file = here(dataOutFolder, "GDSC_lung_variants_all_mutations.csv"))

#############################
## Filtering 
#############################

# Load the identified 358 cancer driver genes
cgg <- read_excel(here(GDSC_raw_files, "GDSC1000", "TableS2B.xlsx"), skip=2)
cancer_driven_gene <- cgg$Gene %>% unique()
length(cancer_driven_gene)

table(cancer_driven_gene %in% rownames(CL_variants))

drive_variant_counts_each_cl <- apply(CL_variants[rownames(CL_variants) %in% cancer_driven_gene, ], MARGIN = 2, FUN = sum)
summary(drive_variant_counts_each_cl)
hist(drive_variant_counts_each_cl)


cl_counts_each_drive_variant <- apply(CL_variants[rownames(CL_variants) %in% cancer_driven_gene, ], MARGIN = 1, FUN = sum)
summary(cl_counts_each_drive_variant)
hist(cl_counts_each_drive_variant)


# Note that the CellLines_CG_BEMs got a smaller sample than and TableS2C.xlsx,
# and files from cell Model Passports are messy. So we will use TableS2C.xlsx
# #------------------------------------------------------------------------------
# # CellLines_CG_BEMs
# #------------------------------------------------------------------------------
# bem_file_names <- list.files(here(GDSC_raw_files, "GDSC1000", "CellLines_CG_BEMs"),
#                              full.names = T)
# 
# bem_files <- lapply(bem_file_names, fread)
# genes <- c()
# for (i in 1:length(bem_files)) {
#   genes <- c(genes, bem_files[[i]]$CG)
#   if(sum(names(bem_files[[i]])=="CG") > 1){
#     bem_files[[i]] = bem_files[[i]][,-c(1,2)]
#   }
# }
# names(bem_files[[25]]) <- unlist(unname(bem_files[[25]][1,]))
# bem_files[[25]] <- bem_files[[25]][-1,] %>% mutate(cancer_type_specific = NA)
# 
# library(tidyverse)
# bem_file <- bem_files %>% 
#   reduce(full_join, by = c("CG", "cancer_type_specific")) %>%
#   unique()
# 
# 
# samples <- gsub("\\.[xy]$", "", names(bem_file)) %>% unique()
# length(samples)
# genes <- bem_file$CG %>% unique()
# length(genes)
#                                              
# 
# ###############################################################################
# ## Cell Model Passports
# ###############################################################################
# mutations_tgs_files_names <- list.files(here(GDSC_raw_files, "CellModelPassports", "mutations_tgs_vcf_20211124"),
#                                   full.names = T)
# 
# library(vcfR)
# mutations_tgs_files <- lapply(mutations_tgs_files_names, read.vcfR, verbose = FALSE)
# 
# View(mutations_tgs_files[[1]])



