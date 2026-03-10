setwd("~/iMSMS/")

library(Maaslin2)

bacteria_css <- read.csv("~/iMSMS/bacteria_css.csv", row.names = 1)

feature_table <- bacteria_css

metadata_bacteria <- read.csv("~/iMSMS/metadata.csv", row.names = 1, check.names = F)

dim(feature_table)
dim(metadata_bacteria)

common_samples <- intersect(colnames(feature_table), rownames(metadata_bacteria))

feature_table <- feature_table[, common_samples]
metadata_bacteria <- metadata_bacteria[common_samples, ]

colnames(metadata_bacteria) <- make.names(colnames(metadata_bacteria))
print(colnames(metadata_bacteria))

fit_data_ms <- Maaslin2(
  input_data = feature_table,
  input_metadata = metadata_bacteria,
  min_prevalence = 0,
  output = "~/iMSMS/bacteria_ms",
  fixed_effects = c("MS", "sex", "age", "weight", "height", "allergy","diet_no_special_needs", "site", "smoke", "treatment",
                    "Beta.carotene", "Bread..pasta..rice", "Calories", "Carbohydrate", "Cholesterol", "Dietary.Fiber", "Fat", 
                    "Fruits..fruit.juices", "Good.oils", "Magnesium", "Milk..cheese..yogurt", "Potassium", "Vegetables.group",
                    "Vitamdietpairsumin.A", "Calcium"),
  normalization = "NONE",
  transform = "NONE",
  analysis_method = "LM",
  standardize = FALSE,
  plot_heatmap = FALSE,
  plot_scatter = FALSE
)

fit_data <- Maaslin2(
  input_data = feature_table,
  input_metadata = metadata_bacteria,
  min_prevalence = 0,
  output = "~/iMSMS/bacteria",
  fixed_effects = c("age", "weight", "height", "allergy","diet_no_special_needs", "site", "smoke", "treatment",
                    "Beta.carotene", "Bread..pasta..rice", "Calories", "Carbohydrate", "Cholesterol", "Dietary.Fiber", "Fat", 
                    "Fruits..fruit.juices", "Good.oils", "Magnesium", "Milk..cheese..yogurt", "Potassium", "Vegetables.group",
                    "Vitamdietpairsumin.A", "Calcium"),
  normalization = "NONE",
  transform = "NONE",
  analysis_method = "LM",
  standardize = FALSE,
  plot_heatmap = FALSE,
  plot_scatter = FALSE
)

adjusted_expr_data <- fit_data$residuals
write.csv(adjusted_expr_data, "~/iMSMS/bacteria_residual.csv")
