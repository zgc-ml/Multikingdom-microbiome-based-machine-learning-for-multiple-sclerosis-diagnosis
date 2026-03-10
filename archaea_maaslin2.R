setwd("~/iMSMS/")

library(Maaslin2)
library(ggplot2)

archaea_css <- read.csv("~/iMSMS/archaea_css.csv", row.names = 1)

feature_table <- archaea_css

metadata_archaea <- read.csv("~/iMSMS/metadata.csv", row.names = 1, check.names = F)

dim(feature_table)
dim(metadata_archaea)

common_samples <- intersect(colnames(feature_table), rownames(metadata_archaea))

feature_table <- feature_table[, common_samples]
metadata_archaea <- metadata_archaea[common_samples, ]

colnames(metadata_archaea) <- make.names(colnames(metadata_archaea))
print(colnames(metadata_archaea))

fit_data_ms <- Maaslin2(
  input_data = feature_table,
  input_metadata = metadata_archaea,
  min_prevalence = 0,
  output = "~/iMSMS/archaea_ms",
  fixed_effects = c("MS", "sex", "age", "diet_no_special_needs", "site", "treatment",
                    "Beta.carotene", "Calories", "Dietary.Fiber", "Fat", "Fruits..fruit.juices",
                    "Magnesium", "Calcium"),
  normalization = "NONE",
  transform = "NONE",
  analysis_method = "LM",
  standardize = FALSE,
  plot_heatmap = FALSE,
  plot_scatter = FALSE
)

fit_data <- Maaslin2(
  input_data = feature_table,
  input_metadata = metadata_archaea,
  min_prevalence = 0,
  output = "~/iMSMS/archaea",
  fixed_effects = c("age", "diet_no_special_needs", "site", "treatment",
                    "Beta.carotene", "Calories", "Dietary.Fiber", "Fat", "Fruits..fruit.juices",
                    "Magnesium", "Calcium"),
  normalization = "NONE",
  transform = "NONE",
  analysis_method = "LM",
  standardize = FALSE,
  plot_heatmap = FALSE,
  plot_scatter = FALSE
)


# 提取调整后的表达矩阵
adjusted_expr_data <- fit_data$residuals
write.csv(adjusted_expr_data, "~/iMSMS/archaea_residual.csv")
