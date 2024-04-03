###############################################################################
## GDSC1000 source
###############################################################################
# List of copy number altered regions of focal amplification/deletion (RACSs) found in cell-lines
CL_cnv_org <- read_excel(here(GDSC_raw_files, "GDSC1000", "TableS2G.xlsx"), skip=2)

CL_cnv <- CL_cnv_org %>% select(Sample, 'Region identifier') %>% as.data.frame()
names(CL_cnv) <- c("sample_name", "RAC")


# count the samples and variants in the table
length(unique(CL_cnv$sample_name)) # [1] 539
length(unique(CL_cnv$RAC)) # [1] 558

# get cosmic_id for each cell line sample name
ind <- match(CL_cnv$sample_name, cl_meta$sample_name)

identical(cl_meta$sample_name[ind], CL_cnv$sample_name) # check, we have some NA values
CL_cnv$COSMIC_ID <- cl_meta$COSMIC_ID[ind]
table(is.na(CL_cnv$COSMIC_ID))

# keep cell lines in the cleaned meta data
CL_cnv <- CL_cnv[CL_cnv$COSMIC_ID %in% cl_meta$COSMIC_ID, ]
length(unique(CL_cnv$COSMIC_ID)) # [1] 530
length(unique(CL_cnv$RAC)) # [1] 558

# Convert to binary event matrix
library(reshape2)
CL_cnv <- dcast(CL_cnv, 
                     formula = RAC ~ COSMIC_ID, fun.aggregate = length)

row.names(CL_cnv) <- CL_cnv$RAC
CL_cnv <- CL_cnv[, -1]

dim(CL_cnv) #  [1] 558 530



# summary
RAC_counts_each_cl <- apply(CL_cnv, MARGIN = 2, FUN = sum)
summary(RAC_counts_each_cl)
hist(RAC_counts_each_cl)

cl_counts_each_RAC <- apply(CL_cnv, MARGIN = 1, FUN = sum)
summary(cl_counts_each_RAC)
hist(cl_counts_each_RAC)

#############################
## Output file
#############################
write.csv(CL_cnv, file = here(dataOutFolder, "GDSC_RAC.csv"))




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

