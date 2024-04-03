library(here)
library(tidyr)
library(dplyr)
library(readxl)

#===============================================================================
# prepare the burglarizing function for area under the dose-response curve (AUC); 
# the curves were fitted using he viability assays
binarize_sensitive_dr <- function(auc_dr){
  # 1. sorts cell lines according to their AUC values in descending order
  auc_dr_sorted <- sort(auc_dr, decreasing = T)
  orders <- length(auc_dr_sorted):1
  # 2. generates an AUC-cell line curve in which the x-axis represents cell lines and the y-axis represents AUC values
  # plot(orders, auc_dr_sorted)
  # 3. generate the cutoff of AUC values
  cor_pearson <- cor(orders, auc_dr_sorted, method = "pearson")
  if(cor_pearson > 0.95){
    # 3.1. for linear curves (whose regression line fitting has a Pearson correlation >0.95), 
    #      the sensitive/resistant cutoff of AUC values is the median among all cell lines
    cutoff <- median(auc_dr, na.rm = T)
  } else {
    # 3.2 otherwise, the cut off is the AUC value of a specific boundary data point. 
    #     It has the largest distance to a line linking two data points having the largest and smallest AUC values
    cutoff <- max_dist_MinMaxLine(points = auc_dr_sorted)
  }
  binarized_dr <- auc_dr
  binarized_dr[auc_dr < cutoff] = 1
  binarized_dr[auc_dr >= cutoff] = 0
  return(binarized_dr)
}


max_dist_MinMaxLine <- function(points){
  min = c(1, min(points))
  max = c(length(points), max(points))
  dists <- apply(cbind(length(points):1, points), 1, dist_MinMaxLine, v1=max - min, max=max)
  return(points[dists==max(dists)])
}

dist_MinMaxLine <- function(x, v1, max) {
  v2 <- x - max
  m <- cbind(v1,v2)
  d <- abs(det(m))/sqrt(sum(v1*v1))
}

#===============================================================================
# load and binarize drug response auc data
GDSC2_fitted_dose_response <- read_excel(here("data", "raw_data", "GDSC", "web", "GDSC2_fitted_dose_response_24Jul22.xlsx"))
length(unique(GDSC2_fitted_dose_response$COSMIC_ID))
length(unique(GDSC2_fitted_dose_response$DRUG_ID))

dr_auc <- GDSC2_fitted_dose_response %>% 
  select(COSMIC_ID, DRUG_ID, AUC) %>%
  pivot_wider(names_from = DRUG_ID, values_from = AUC) %>%
  as.data.frame()
row.names(dr_auc) <- dr_auc$COSMIC_ID
dr_auc <- dr_auc %>% select(-COSMIC_ID)

write.csv(dr_auc, file = here(dataOutFolder, "GDSC_DrugResponse.csv"))

dr_binary <- apply(dr_auc, 2, binarize_sensitive_dr) %>% as.data.frame()
write.csv(dr_binary, file = here(dataOutFolder, "GDSC_DrugResponse_binarized.csv"))

#===============================================================================
# checkout the binarized drug response data
dim(dr_binary)

mean(is.na(dr_binary))
hist((colMeans(is.na(dr_binary)))*100,
     main = "Percentage of missing values for each drug",
     ylab = "number of drugss",
     xlab = "% missing values")

hist(apply(dr_binary, 1, mean, na.rm=T), 
     main = "Percentage of sensitive drugs",
     ylab = "number of cancer cell lines",
     xlab = "sensitive to % drugs")

hist(apply(dr_binary, 2, mean, na.rm=T), 
     main = "Percentage of sensitive cancer cell lines",
     ylab = "number of drugss",
     xlab = "% cancer cell lines experimented that is sensitive to the drug")

# source(here("scripts_GDSC", "01_CCL_preprocessing", "01_CL_annotation.R"))
sum(rownames(dr_binary) %in% cl_meta$COSMIC_ID)
mean(rownames(dr_binary) %in% cl_meta$COSMIC_ID)

dr_binary_sub <- dr_binary[rownames(dr_binary) %in% cl_meta$COSMIC_ID, ]
dim(dr_binary_sub)

dr_auc_sub <- dr_auc[rownames(dr_auc) %in% cl_meta$COSMIC_ID, ]

write.csv(dr_binary_sub, file = here(dataOutFolder, "GDSC_DrugResponse_cleaned.csv"))
write.csv(dr_binary_sub, file = here(dataOutFolder, "GDSC_DrugResponse_binarized_cleaned.csv"))




