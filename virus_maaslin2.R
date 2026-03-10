setwd("~/iMSMS/")

library(Maaslin2)

virus_css <- read.csv("~/iMSMS/virus_css.csv", row.names = 1)
feature_table <- virus_css

metadata_virus <- read.csv("~/iMSMS/metadata.csv", row.names = 1, check.names = F)

dim(feature_table)
dim(metadata_virus)

common_samples <- intersect(colnames(feature_table), rownames(metadata_virus))

feature_table <- feature_table[, common_samples]
metadata_virus <- metadata_virus[common_samples, ]

colnames(metadata_virus) <- make.names(colnames(metadata_virus))
print(colnames(metadata_virus))

fit_data_ms <- Maaslin2(
  input_data = feature_table,
  input_metadata = metadata_virus,
  min_prevalence = 0,
  output = "~/iMSMS/virus_ms",
  fixed_effects = c("MS", "age", "height", "site", "treatment",
                    "B1..B2", "Beta.carotene", "Dietary.Fiber", "Fat", "Fat.as...of.cals", "Good.oils", "Potassium",
                    "Vitamdietpairsumin.A"),
  normalization = "NONE",
  transform = "NONE",
  analysis_method = "LM",
  standardize = FALSE,
  plot_heatmap = FALSE,
  plot_scatter = FALSE
)

fit_data <- Maaslin2(
  input_data = feature_table,
  input_metadata = metadata_virus,
  min_prevalence = 0,
  output = "~/iMSMS/virus",
  fixed_effects = c("age", "height", "site", "treatment",
                    "B1..B2", "Beta.carotene", "Dietary.Fiber", "Fat", "Fat.as...of.cals", "Good.oils", "Potassium",
                    "Vitamdietpairsumin.A"),
  normalization = "NONE",
  transform = "NONE",
  analysis_method = "LM",
  standardize = FALSE,
  plot_heatmap = FALSE,
  plot_scatter = FALSE
)

adjusted_expr_data <- fit_data$residuals
write.csv(adjusted_expr_data, "~/iMSMS/virus_residual.csv")
