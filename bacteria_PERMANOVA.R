library(edgeR)
library(DESeq2)
library(compositions)
library(metagenomeSeq)  
library(GMPR)

setwd("~/iMSMS/")

# 读取数据
control_u_bacteria <- read.csv("~/iMSMS/control_U/bracken_bacteria_species.csv", row.names = 1)
control_t_bacteria <- read.csv("~/iMSMS/control_T/bracken_bacteria_species.csv", row.names = 1)
rrms_u_bacteria <- read.csv("~/iMSMS/RRMS_U/bracken_bacteria_species.csv", row.names = 1)
rrms_t_bacteria <- read.csv("~/iMSMS/RRMS_T/bracken_bacteria_species.csv", row.names = 1)
ppms_u_bacteria <- read.csv("~/iMSMS/PPMS_U/bracken_bacteria_species.csv", row.names = 1)
ppms_t_bacteria <- read.csv("~/iMSMS/PPMS_T/bracken_bacteria_species.csv", row.names = 1)
spms_u_bacteria <- read.csv("~/iMSMS/SPMS_U/bracken_bacteria_species.csv", row.names = 1)
spms_t_bacteria <- read.csv("~/iMSMS/SPMS_T/bracken_bacteria_species.csv", row.names = 1)

bacteria <- merge(control_u_bacteria, control_t_bacteria, by = 0)
bacteria <- merge(bacteria, rrms_u_bacteria, by.x = 1, by.y = 0)
bacteria <- merge(bacteria, rrms_t_bacteria, by.x = 1, by.y = 0)
bacteria <- merge(bacteria, ppms_u_bacteria, by.x = 1, by.y = 0)
bacteria <- merge(bacteria, ppms_t_bacteria, by.x = 1, by.y = 0)
bacteria <- merge(bacteria, spms_u_bacteria, by.x = 1, by.y = 0)
bacteria <- merge(bacteria, spms_t_bacteria, by.x = 1, by.y = 0)
rownames(bacteria) <- bacteria$Row.names
bacteria <- bacteria[,-1]

filter_threshold <- 0.1
keep_rows <- rowSums(bacteria > 0) >= (filter_threshold * ncol(bacteria))
bacteria_filtered <- bacteria[keep_rows, ]

meta_zero <- data.frame(sample = colnames(bacteria_filtered))
rownames(meta_zero) <- colnames(bacteria_filtered)
obj <- newMRexperiment(bacteria_filtered, phenoData = AnnotatedDataFrame(meta_zero))

obj <- cumNorm(obj, p = cumNormStatFast(obj))
css_counts <- MRcounts(obj, norm = TRUE, log = FALSE)

write.csv(css_counts, "~/iMSMS/bacteria_css.csv")

metadata_bacteria <- read.csv("~/iMSMS/all_metadata.csv", row.names = 1)

phenotype_vars <- c("MS", "sex", "age", "weight", "height", "bmi", "allergy", "diet_no_special_needs", "site", "smoke",
                    "weight_change", "pets", "treatment")

library(vegan)
library(ggplot2)

data_norm_t <- t(css_counts)

data_norm_t <- data_norm_t[rownames(metadata_bacteria), ]

bray_curtis_dist <- vegdist(data_norm_t, method = "bray")

adonis_result <- adonis(bray_curtis_dist ~ ., data = metadata_bacteria[, phenotype_vars], permutations = 9999, by = "margin")

phenotype_r2 <- adonis_result$aov.tab[1:length(phenotype_vars), "R2"]
names(phenotype_r2) <- phenotype_vars

phenotype_p <- adonis_result$aov.tab[1:length(phenotype_vars), "Pr(>F)"]
names(phenotype_p) <- phenotype_vars

phenotype_p_adj <- p.adjust(phenotype_p, method = "BH")
names(phenotype_p_adj) <- phenotype_vars

phenotype_r2_pct <- phenotype_r2 * 100

df <- data.frame(
  Phenotype = names(phenotype_r2_pct),
  VarianceExplained = phenotype_r2_pct,
  p = as.numeric(phenotype_p),
  p_adj = as.numeric(phenotype_p_adj)
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
write.csv(df, "~/iMSMS/bacteria_importance.csv", row.names = F)

# Diet
metadata_bacteria_diet <- read.csv("~/iMSMS/metadata.csv", row.names = 1, check.names = F)
print(colnames(metadata_bacteria_diet))

library(vegan)
library(ggplot2)

data_norm_t <- t(css_counts)

common <- intersect(rownames(data_norm_t), rownames(metadata_bacteria_diet))
data_norm_t <- data_norm_t[common, ]
metadata_bacteria_diet <- metadata_bacteria_diet[common, ]

data_norm_t <- data_norm_t[rownames(metadata_bacteria_diet), ]

bray_curtis_dist <- vegdist(data_norm_t, method = "bray")

adonis_result_diet <- adonis(bray_curtis_dist ~ ., data = metadata_bacteria_diet[, phenotype_vars], permutations = 9999, by = "margin")

phenotype_r2 <- adonis_result_diet$aov.tab[1:length(phenotype_vars), "R2"]
names(phenotype_r2) <- phenotype_vars

phenotype_p <- adonis_result_diet$aov.tab[1:length(phenotype_vars), "Pr(>F)"]
names(phenotype_p) <- phenotype_vars

phenotype_p_adj <- p.adjust(phenotype_p, method = "BH")
names(phenotype_p_adj) <- phenotype_vars

phenotype_r2_pct <- phenotype_r2 * 100

df <- data.frame(
  Phenotype = names(phenotype_r2_pct),
  VarianceExplained = phenotype_r2_pct,
  p = as.numeric(phenotype_p),
  p_adj = as.numeric(phenotype_p_adj)
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
write.csv(df, "~/iMSMS/bacteria_diet_importance.csv", row.names = F)
