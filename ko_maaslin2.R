setwd("~/iMSMS/")

library(Maaslin2)

control_u_ko <- read.csv("~/iMSMS/control_U/ko_cpm_unstratified.csv", row.names = 1)
control_t_ko <- read.csv("~/iMSMS/control_T/ko_cpm_unstratified.csv", row.names = 1)
rrms_u_ko <- read.csv("~/iMSMS/RRMS_U/ko_cpm_unstratified.csv", row.names = 1)
rrms_t_ko <- read.csv("~/iMSMS/RRMS_T/ko_cpm_unstratified.csv", row.names = 1)
ppms_u_ko <- read.csv("~/iMSMS/PPMS_U/ko_cpm_unstratified.csv", row.names = 1)
ppms_t_ko <- read.csv("~/iMSMS/PPMS_T/ko_cpm_unstratified.csv", row.names = 1)
spms_u_ko <- read.csv("~/iMSMS/SPMS_U/ko_cpm_unstratified.csv", row.names = 1)
spms_t_ko <- read.csv("~/iMSMS/SPMS_T/ko_cpm_unstratified.csv", row.names = 1)

ko <- merge(control_u_ko, control_t_ko, by = 0)
ko <- merge(ko, rrms_u_ko, by.x = 1, by.y = 0)
ko <- merge(ko, rrms_t_ko, by.x = 1, by.y = 0)
ko <- merge(ko, ppms_u_ko, by.x = 1, by.y = 0)
ko <- merge(ko, ppms_t_ko, by.x = 1, by.y = 0)
ko <- merge(ko, spms_u_ko, by.x = 1, by.y = 0)
ko <- merge(ko, spms_t_ko, by.x = 1, by.y = 0)
rownames(ko) <- ko$Row.names
ko <- ko[,-1]

filter_threshold <- 0.1
keep_rows <- rowSums(ko > 0) >= (filter_threshold * ncol(ko))
ko_filtered <- ko[keep_rows, ]
feature_table <- ko_filtered

metadata_ko <- read.csv("~/iMSMS/metadata.csv", row.names = 1, check.names = F)

dim(feature_table)
dim(metadata_ko)

common_samples <- intersect(colnames(feature_table), rownames(metadata_ko))

feature_table <- feature_table[, common_samples]
metadata_ko <- metadata_ko[common_samples, ]

colnames(metadata_ko) <- make.names(colnames(metadata_ko))
print(colnames(metadata_ko))

fit_data_ms <- Maaslin2(
  input_data = feature_table,
  input_metadata = metadata_ko,
  min_prevalence = 0,
  output = "~/iMSMS/ko_ms",
  fixed_effects = c("MS", "sex", "age", "weight", "allergy","diet_no_special_needs", "site", "treatment",
                    "Bread..pasta..rice", "Carbohydrate", "Cholesterol", "Dietary.Fiber", "Fat", "Fruits..fruit.juices",
                    "Niacin", "Potassium"),
  normalization = "NONE",
  transform = "NONE",
  analysis_method = "LM",
  standardize = FALSE,
  plot_heatmap = FALSE,
  plot_scatter = FALSE
)

fit_data <- Maaslin2(
  input_data = feature_table,
  input_metadata = metadata_ko,
  min_prevalence = 0,
  output = "~/iMSMS/ko",
  fixed_effects = c("age", "weight", "allergy","diet_no_special_needs", "site", "treatment",
                    "Bread..pasta..rice", "Carbohydrate", "Cholesterol", "Dietary.Fiber", "Fat", "Fruits..fruit.juices",
                    "Niacin", "Potassium"),
  normalization = "NONE",
  transform = "NONE",
  analysis_method = "LM",
  standardize = FALSE,
  plot_heatmap = FALSE,
  plot_scatter = FALSE
)

adjusted_expr_data <- fit_data$residuals
write.csv(adjusted_expr_data, "~/iMSMS/ko_residual.csv")
