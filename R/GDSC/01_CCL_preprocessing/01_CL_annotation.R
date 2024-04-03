# Annotation
## Preprocessing

###############################################################################
## GDSC1000
###############################################################################
## https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources/Home.html 

cell_line_metadata <- 
  readxl::read_xlsx(here(GDSC_raw_files, "GDSC1000", "TableS1E.xlsx") ,
                    range = "B5:N1005", col_names = FALSE)


names(cell_line_metadata) <-
  c("sample_name", "COSMIC_ID", "whole_exome_seq", "cna", "gene_expression",
    "methylation", "drug_response", 
    "tissue_descriptor_1", "tissue_descriptor_2", "cancer_type",
    "ms_instability", "screen_medium", "growth_props")

cell_line_metadata <- 
  cell_line_metadata %>% 
  mutate(COSMIC_ID = gsub("COSMIC_ID.", "", COSMIC_ID)) %>% 
  mutate(tissue_descriptor = 
           case_when(grepl("lung", tissue_descriptor_1) ~ "lung", 
                     TRUE ~ tissue_descriptor_1))

length(unique(cell_line_metadata$cancer_type))
cancer_type_cnt <- cell_line_metadata %>% select(cancer_type) %>% group_by(cancer_type) %>% mutate(n = n()) %>% unique()

# number of cell lines:
n_cl <- length(unique(cell_line_metadata$COSMIC_ID))
n_cl

# print("Counts for each data type: \n")
cl_data_type_cont <- cell_line_metadata %>% 
  group_by(whole_exome_seq, cna, gene_expression, methylation) %>% 
  mutate(n = n()) %>%
  select(whole_exome_seq, cna, gene_expression, methylation, n) %>%
  unique()
cl_data_type_cont


###############################################################################
## Cell Model Passports
###############################################################################
cell_model_anno <- read.csv(file=here(GDSC_raw_files, "CellModelPassports", "model_list_20220810.csv"))
names(cell_model_anno)

cell_model_anno <- cell_model_anno %>% 
  select(sample_id,model_id,model_name,patient_id,COSMIC_ID, BROAD_ID,CCLE_ID, 
         synonyms,tissue,tissue_status,sample_site,
         gender,ethnicity,age_at_sampling,cancer_type,
         family_history_of_cancer, smoking_status, alcohol_exposure_intensity, 
         alcohol_consumption_per_week, history_diabetes, diabetes_treatment, colorectal_cancer_risk_factors) %>% 
  rename(cancer_type_passports=cancer_type)


###############################################################################
## merge both together
###############################################################################
cell_line_metadata$GDSC_1000 <- TRUE

cl_meta <- merge(cell_line_metadata, cell_model_anno,
                 by = "COSMIC_ID", all=T)

cl_meta <- cl_meta[! cl_meta$COSMIC_ID %in% c("SIDM00001", "SIDM00002"), ]
# View(cl_meta %>% select(COSMIC_ID, model_id, cancer_type, cancer_type_passports) %>% filter(cancer_type_passports == "Non-Cancerous"))

table(is.na(cl_meta$COSMIC_ID))
table(cl_meta$COSMIC_ID == "")
cl_meta$COSMIC_ID[cl_meta$COSMIC_ID == ""] = NA


# remove non-cancerous cell lines 
dim(cl_meta)
cl_meta <- cl_meta %>%
  filter(! cancer_type_passports %in% c("", "Unknown", "UNABLE TO CLASSIFY", "Non-Cancerous"),
         ! (cancer_type) %in% c("", "Unknown", "UNABLE TO CLASSIFY", "Non-Cancerous"))  
dim(cl_meta) # 2082 -> 1975

# compare cancer_type in two data sets
t1 <- table(cl_meta$cancer_type, cl_meta$cancer_type_passports) %>% 
  as.data.frame() %>% 
  filter(Freq != 0) %>%
  rename(GDSC1000 = Var1, Passport = Var2)

cl_meta <- cl_meta %>%
  filter(!(cancer_type == "CESC" & cancer_type_passports == "Other Solid Cancers"),
         !(cancer_type == "CLL" & cancer_type_passports == "B-Cell Non-Hodgkin's Lymphoma"),
         !(cancer_type == "DLBC" & cancer_type_passports %in% c("B-Lymphoblastic Leukemia", "Acute Myeloid Leukemia")),
         !(cancer_type == "LAML" & cancer_type_passports == "T-Lymphoblastic Leukemia"),
         !(cancer_type == "LUAD" & cancer_type_passports == "Squamous Cell Lung Carcinoma"),
         !(cancer_type == "LUSC" & cancer_type_passports %in% c("Mesothelioma", "Other Solid Cancers")),
         !(cancer_type == "MESO" & cancer_type_passports == "Non-Small Cell Lung Carcinoma"),
         !(cancer_type == "NB" & cancer_type_passports == "Ewing's Sarcoma"),
         !(cancer_type == "OV" & cancer_type_passports == "Colorectal Carcinoma"),
         !(cancer_type == "STAD" & cancer_type_passports %in% c("B-Cell Non-Hodgkin's Lymphoma", "Squamous Cell Lung Carcinoma")))
dim(cl_meta) # 1975 -> 1525


# compare tissue in two data sets
t1 <- table(cl_meta$tissue_descriptor_1, cl_meta$tissue) %>% 
  as.data.frame() %>% 
  filter(Freq != 0) %>%
  rename(GDSC1000 = Var1, Passport = Var2)

cl_meta <- cl_meta %>%
  filter(!(tissue_descriptor_1 == "kidney" & tissue == "Adrenal Gland"),
         !(tissue_descriptor_1 == "soft_tissue" & tissue %in% c("Stomach", "Uterus")),
         !(tissue_descriptor_1 == "urogenital_system" & tissue == "Soft Tissue"))

dim(cl_meta) # 1525 -> 1479


cl_meta <- cl_meta %>%
  select(COSMIC_ID, sample_name, sample_id, model_id, model_name, patient_id, BROAD_ID, CCLE_ID,
         tissue_descriptor_1, tissue_descriptor_2, tissue_descriptor, tissue, tissue_status, sample_site, 
         cancer_type, cancer_type_passports, 
         gender, ethnicity, age_at_sampling, 
         family_history_of_cancer, smoking_status, alcohol_exposure_intensity, 
         alcohol_consumption_per_week, history_diabetes, diabetes_treatment, colorectal_cancer_risk_factors,
         GDSC_1000, whole_exome_seq, cna, gene_expression, methylation, drug_response)

# View data info
table(cl_meta$GDSC_1000)

cl_data_type_cont <- cl_meta %>% 
  group_by(whole_exome_seq, cna, gene_expression, methylation) %>% 
  mutate(n = n()) %>%
  select(whole_exome_seq, cna, gene_expression, methylation, n) %>%
  unique()
cl_data_type_cont


tissue <- c("Bone", "Haematopoietic and Lymphoid", 
            "Central Nervous System", "Peripheral Nervous System", 
            "Head and Neck", "Eye", "Thyroid",
            "Breast", "Pancreas", 
            "Esophagus","Stomach",  "Biliary Tract", # digestive_system
            "Kidney", "Lung", "Liver",
            "Large Intestine",  
            "Endometrium",  "Ovary", "Cervix", "Uterus", "Bladder", "Prostate",
            "Skin", "Soft Tissue"
            )
cl_meta$tissue = factor(cl_meta$tissue, levels = tissue)


write.csv(cl_meta, file = here(dataOutFolder, "GDSC_CellLine_annotation.csv"))

rm(cell_line_metadata, cell_model_anno)



# compare tissue in two data sets
t1 <- table(cl_meta$tissue_descriptor_1, cl_meta$cancer_type) %>% 
  as.data.frame() %>% 
  filter(Freq != 0) %>%
  rename(tissue = Var1, cancer_type = Var2)


###################################
# Lung
###################################
cl_meta_lung <- cl_meta[cl_meta$tissue=="Lung",]
cl_meta_lung <- cl_meta_lung %>% 
  select(COSMIC_ID, sample_name, sample_id, model_id, model_name,patient_id, BROAD_ID, CCLE_ID,
         tissue, sample_site, cancer_type, cancer_type_passports,
         gender, ethnicity, ethnicity, smoking_status,
         family_history_of_cancer, alcohol_exposure_intensity, history_diabetes,
         whole_exome_seq, cna, gene_expression, methylation, drug_response)

table(cl_meta_lung$cancer_type, cl_meta_lung$cancer_type_passports)
# unify cancer types 

cl_meta_lung$cancer_type_1 = cl_meta_lung$cancer_type
cl_meta_lung$cancer_type_1[cl_meta_lung$cancer_type_1 %in% c("LUAD", "LUSC")] = "NSCLC"
cl_meta_lung$cancer_type_1[cl_meta_lung$cancer_type_1 == "SCLC" &  cl_meta_lung$cancer_type_passports == "Squamous Cell Lung Carcinoma"] = "NSCLC"

table(cl_meta_lung$cancer_type_1, cl_meta_lung$cancer_type_passports)

cl_meta_lung <- cl_meta_lung %>%
  select(COSMIC_ID, sample_name, sample_id, model_id, model_name,patient_id,BROAD_ID, CCLE_ID,
         tissue, sample_site, cancer_type_1,
         gender, ethnicity, ethnicity, smoking_status,
         family_history_of_cancer, alcohol_exposure_intensity, history_diabetes,
         whole_exome_seq, cna, gene_expression, methylation, drug_response) %>%
  rename(cancer_type = cancer_type_1)

write.csv(cl_meta_lung, file = here(dataOutFolder, "GDSC_CellLine_lung_annotation.csv"))

