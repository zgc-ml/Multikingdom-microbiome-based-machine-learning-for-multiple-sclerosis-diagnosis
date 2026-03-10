setwd("~/iMSMS/")

library(Maaslin2)

fungi_css <- read.csv("~/iMSMS/fungi_css.csv", row.names = 1)

feature_table <- fungi_css

metadata_fungi <- read.csv("~/iMSMS/metadata.csv", row.names = 1, check.names = F)

dim(feature_table)
dim(metadata_fungi)

common_samples <- intersect(colnames(feature_table), rownames(metadata_fungi))

feature_table <- feature_table[, common_samples]
metadata_fungi <- metadata_fungi[common_samples, ]

colnames(metadata_fungi) <- make.names(colnames(metadata_fungi))
print(colnames(metadata_fungi))

fit_data_ms <- Maaslin2(
  input_data = feature_table,
  input_metadata = metadata_fungi,
  min_prevalence = 0,
  output = "~/iMSMS/fungi_ms",
  fixed_effects = c("MS", "site", "treatment",
                    "Beta.carotene", "Bread..pasta..rice", "Good.oils", 
                    "without.potatoes", "Vitamin.E"),
  normalization = "NONE",
  transform = "NONE",
  analysis_method = "LM",
  standardize = FALSE,
  plot_heatmap = FALSE,
  plot_scatter = FALSE
)

fit_data <- Maaslin2(
  input_data = feature_table,
  input_metadata = metadata_fungi,
  min_prevalence = 0,
  output = "~/iMSMS/fungi",
  fixed_effects = c("site", "treatment",
                    "Beta.carotene", "Bread..pasta..rice", "Good.oils", 
                    "without.potatoes", "Vitamin.E"),
  normalization = "NONE",
  transform = "NONE",
  analysis_method = "LM",
  standardize = FALSE,
  plot_heatmap = FALSE,
  plot_scatter = FALSE
)

adjusted_expr_data <- fit_data$residuals
write.csv(adjusted_expr_data, "~/iMSMS/fungi_residual.csv")
