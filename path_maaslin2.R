setwd("~/iMSMS/")

library(Maaslin2)

control_u_path <- read.csv("~/iMSMS/control_U/pathabundance_cpm_unstratified.csv", row.names = 1)
control_t_path <- read.csv("~/iMSMS/control_T/pathabundance_cpm_unstratified.csv", row.names = 1)
rrms_u_path <- read.csv("~/iMSMS/RRMS_U/pathabundance_cpm_unstratified.csv", row.names = 1)
rrms_t_path <- read.csv("~/iMSMS/RRMS_T/pathabundance_cpm_unstratified.csv", row.names = 1)
ppms_u_path <- read.csv("~/iMSMS/PPMS_U/pathabundance_cpm_unstratified.csv", row.names = 1)
ppms_t_path <- read.csv("~/iMSMS/PPMS_T/pathabundance_cpm_unstratified.csv", row.names = 1)
spms_u_path <- read.csv("~/iMSMS/SPMS_U/pathabundance_cpm_unstratified.csv", row.names = 1)
spms_t_path <- read.csv("~/iMSMS/SPMS_T/pathabundance_cpm_unstratified.csv", row.names = 1)

path <- merge(control_u_path, control_t_path, by = 0)
path <- merge(path, rrms_u_path, by.x = 1, by.y = 0)
path <- merge(path, rrms_t_path, by.x = 1, by.y = 0)
path <- merge(path, ppms_u_path, by.x = 1, by.y = 0)
path <- merge(path, ppms_t_path, by.x = 1, by.y = 0)
path <- merge(path, spms_u_path, by.x = 1, by.y = 0)
path <- merge(path, spms_t_path, by.x = 1, by.y = 0)
rownames(path) <- path$Row.names
path <- path[,-1]

filter_threshold <- 0.1
keep_rows <- rowSums(path > 0) >= (filter_threshold * ncol(path))
path_filtered <- path[keep_rows, ]
feature_table <- path_filtered

colnames(metadata_path) <- make.names(colnames(metadata_path))
print(colnames(metadata_path))

fit_data_ms <- Maaslin2(
  input_data = feature_table,
  input_metadata = metadata_path,
  min_prevalence = 0,
  output = "~/iMSMS/path_ms",
  fixed_effects = c("MS", "age", "weight", "height", "allergy","diet_no_special_needs", "site", "treatment",
                    "Calories", "Cholesterol", "Dietary.Fiber", "Fat.as...of.cals", 
                    "Potassium", "Vitamdietpairsumin.A", "Calcium"),
  normalization = "NONE",
  transform = "NONE",
  analysis_method = "LM",
  standardize = FALSE,
  plot_heatmap = FALSE,
  plot_scatter = FALSE
)

fit_data <- Maaslin2(
  input_data = feature_table,
  input_metadata = metadata_path,
  min_prevalence = 0,
  output = "~/iMSMS/path",
  fixed_effects = c("age", "weight", "height", "allergy","diet_no_special_needs", "site", "treatment",
                    "Calories", "Cholesterol", "Dietary.Fiber", "Fat.as...of.cals", 
                    "Potassium", "Vitamdietpairsumin.A", "Calcium"),
  normalization = "NONE",
  transform = "NONE",
  analysis_method = "LM",
  standardize = FALSE,
  plot_heatmap = FALSE,
  plot_scatter = FALSE
)

adjusted_expr_data <- fit_data$residuals
write.csv(adjusted_expr_data, "~/iMSMS/path_residual.csv")