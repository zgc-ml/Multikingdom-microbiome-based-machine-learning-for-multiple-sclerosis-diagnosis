library(edgeR)
library(DESeq2)
library(compositions)
library(metagenomeSeq)  
library(GMPR)

setwd("~/iMSMS/")

# 读取数据
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

metadata_path <- read.csv("~/iMSMS/metadata.csv", row.names = 1)

phenotype_vars <- c("MS", "sex", "age", "weight", "height", "bmi", "allergy", "diet_no_special_needs", "site", "smoke",
                    "weight_change", "pets", "treatment")

library(vegan)
library(ggplot2)

data_norm_t <- t(path_filtered)
data_norm_t <- data_norm_t[rownames(metadata_path), ]

bray_curtis_dist <- vegdist(data_norm_t, method = "bray")

adonis_result <- adonis(bray_curtis_dist ~ ., data = metadata_path[, phenotype_vars], permutations = 9999)

phenotype_r2 <- adonis_result$aov.tab[1:length(phenotype_vars), "R2"]
names(phenotype_r2) <- phenotype_vars

phenotype_p <- adonis_result$aov.tab[1:length(phenotype_vars), "Pr(>F)"]
names(phenotype_p) <- phenotype_vars

phenotype_r2_pct <- phenotype_r2 * 100

df <- data.frame(
  Phenotype = names(phenotype_r2_pct),
  VarianceExplained = phenotype_r2_pct,
  p = phenotype_p
)

df$Phenotype <- factor(
  df$Phenotype, 
  levels = df$Phenotype[order(df$VarianceExplained, decreasing = TRUE)]
)  

p <- ggplot(df, aes(x = Phenotype, y = VarianceExplained)) +
  geom_bar(stat = "identity", fill = "#1f78b4", width = 0.7) +
  coord_cartesian(ylim = c(0, max(df$VarianceExplained) * 1.2)) +
  labs(
    title = paste("Variance Explained by Phenotypes (", "CSS", ")", sep = ""),
    x = NULL,
    y = "Variance Explained (%)"
  ) +
  theme_bw(base_size = 18) +
  theme(
    plot.title = element_text(
      hjust = 0.5, 
      face = "bold", 
      size = 20
    ),
    axis.title.y = element_text(
      face = "bold", 
      size = 18
    ),
    axis.text.x = element_text(
      angle = 45, 
      hjust = 1, 
      size = 16
    ),
    axis.text.y = element_text(size = 16),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank()
  )  

print(adonis_result$aov.tab)
write.csv(df, "~/iMSMS/path_importance.csv", row.names = F)

# Diet
metadata_path_diet <- read.csv("~/iMSMS/metadata.csv", row.names = 1, check.names = F)
print(colnames(metadata_path_diet))

phenotype_vars <- c("Alcohol % of cals", "B1, B2", "Beta-carotene", "Bread, pasta, rice", "Calories", "Carbohydrate", 
                    "Carbohydrate as % of cals", "Cholesterol", "Dietary Fiber", "Fat", "Fat as % of cals", "Fruits, fruit juices", 
                    "Good oils", "Magnesium", "Meat, eggs, or beans", "Milk, cheese, yogurt", "Monounsaturated fat", "Niacin",
                    "Polyunsaturated fat", "Potassium", "Protein", "Protein as % of cals", "Saturated fat", "Saturated fat as % of cals",
                    "Sodium", "Sweets % of cals", "Vegetables group", "Vitamin B6", "Whole grains", "without potatoes", 
                    "Vitamdietpairsumin A", "Vitamin C", "Vitamin E", "Folate", "Calcium", "Iron", "Zinc")

library(vegan)
library(ggplot2)

data_norm_t <- t(path_filtered)
common <- intersect(rownames(data_norm_t), rownames(metadata_path_diet))
data_norm_t <- data_norm_t[common, ]
metadata_path_diet <- metadata_path_diet[common, ]
data_norm_t <- data_norm_t[rownames(metadata_path_diet), ]

bray_curtis_dist <- vegdist(data_norm_t, method = "bray")

adonis_result_diet <- adonis(bray_curtis_dist ~ ., data = metadata_path_diet[, phenotype_vars], permutations = 9999)

phenotype_r2 <- adonis_result_diet$aov.tab[1:length(phenotype_vars), "R2"]
names(phenotype_r2) <- phenotype_vars

phenotype_p <- adonis_result_diet$aov.tab[1:length(phenotype_vars), "Pr(>F)"]
names(phenotype_p) <- phenotype_vars

phenotype_r2_pct <- phenotype_r2 * 100

df <- data.frame(
  Phenotype = names(phenotype_r2_pct),
  VarianceExplained = phenotype_r2_pct,
  p = phenotype_p
)

df$Phenotype <- factor(
  df$Phenotype, 
  levels = df$Phenotype[order(df$VarianceExplained, decreasing = TRUE)]
)  

p <- ggplot(df, aes(x = Phenotype, y = VarianceExplained)) +
  geom_bar(stat = "identity", fill = "#1f78b4", width = 0.7) +
  coord_cartesian(ylim = c(0, max(df$VarianceExplained) * 1.2)) +
  labs(
    title = paste("Variance Explained by Phenotypes (", "CSS", ")", sep = ""),
    x = NULL,
    y = "Variance Explained (%)"
  ) +
  theme_bw(base_size = 18) +
  theme(
    plot.title = element_text(
      hjust = 0.5, 
      face = "bold", 
      size = 20
    ),
    axis.title.y = element_text(
      face = "bold", 
      size = 18
    ),
    axis.text.x = element_text(
      angle = 45, 
      hjust = 1, 
      size = 16
    ),
    axis.text.y = element_text(size = 16),
    panel.grid.major.x = element_blank(),
    
    panel.grid.minor = element_blank()
  )  

print(adonis_result_diet$aov.tab)
write.csv(df, "~/iMSMS/path_diet_importance.csv", row.names = F)
