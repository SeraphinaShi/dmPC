impact_A <- c("AZ6102", "AZD2014", "CCT-018159", "Dabrafenib", "Dinaciclib", "Elesclomol", "Epirubicin", "LMP744", "Mirin", "Mycophenolic acid", "PD0325901", "PLX-4720", "Piperlongumine", "Refametinib", "SB590885", "SCH772984", "Selumetinib", "Trametinib", "UMI-77", "Ulixertinib")

impact_B <- c("AZD5153", "AZD5438", "AZD7762", "AZD8055", "BX795", "Buparlisib", "Camptothecin", "Dactolisib", "Docetaxel", "Elephantin", "GNE-317", "MK-1775", "Pevonedistat", "Romidepsin", "Sabutoclax", "Tanespimycin", "Telomerase", "Inhibitor IX", "Topotecan", "Vinblastine", "WZ4003", "Wee1 Inhibitor", "YK-4-279")

plaid_A <- c("AZD2014", "AZD8055", "CCT-018159", "Dabrafenib", "Dactolisib", "Dinaciclib", "Epirubicin","Mirin",  "Mycophenolic acid", "PD0325901", "PLX-4720", "Refametinib", "Romidepsin", "SB590885", "SCH772984", "Selumetinib", "Tanespimycin", "Topotecan", "Trametinib", "UMI-77", "Ulixertinib")

plaid_B <- c("AZ6102",   "Buparlisib",  "Camptothecin",  "Docetaxel",   "GNE-317",  "Pevonedistat", "Sabutoclax",  "Telomerase Inhibitor IX", "Vinblastine",   "Wee1 Inhibitor",   "YK-4-279")


plaid_C <- c("AZD5438",  "AZD7762",  "Elephantin",    "Pictilisib",  "WZ4003")

library(VennDiagram)


venn.plot <- draw.triple.venn(
  area1 = length(impact_A),
  area2 = length(impact_B),
  area3 = length(plaid_A),
  n12 = length(intersect(impact_A, impact_B)),
  n23 = length(intersect(impact_B, plaid_A)),
  n13 = length(intersect(impact_A, plaid_A)),
  n123 = length(intersect(intersect(impact_A, impact_B), plaid_A)),
  category = c("impactA", "impactB", "plaid_A"), 
  fill = c("salmon", "lightgreen", "lightblue"),
  lty = "blank",
  cex = 2,
  cat.cex = 2,
  cat.col = c("salmon", "lightgreen", "lightblue")
)

grid.draw(venn.plot)


venn.plot <- draw.triple.venn(
  area1 = length(impact_A),
  area2 = length(impact_B),
  area3 = length(plaid_B),
  n12 = length(intersect(impact_A, impact_B)),
  n23 = length(intersect(impact_B, plaid_B)),
  n13 = length(intersect(impact_A, plaid_B)),
  n123 = length(intersect(intersect(impact_A, impact_B), plaid_B)),
  category = c("impactA", "impactB", "plaid_B"), 
  fill = c("salmon", "lightgreen", "mediumpurple1"),
  lty = "blank",
  cex = 2,
  cat.cex = 2,
  cat.col = c("salmon", "lightgreen", "mediumpurple1")
)

grid.draw(venn.plot)


venn.plot <- draw.triple.venn(
  area1 = length(impact_A),
  area2 = length(impact_B),
  area3 = length(plaid_C),
  n12 = length(intersect(impact_A, impact_B)),
  n23 = length(intersect(impact_B, plaid_C)),
  n13 = length(intersect(impact_A, plaid_C)),
  n123 = length(intersect(intersect(impact_A, impact_B), plaid_C)),
  category = c("impactA", "impactB", "plaid_C"), 
  fill = c("salmon", "lightgreen", "orange"),
  lty = "blank",
  cex = 2,
  cat.cex = 2,
  cat.col = c("salmon", "lightgreen", "orange")
)

