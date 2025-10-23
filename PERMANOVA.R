library(edgeR)
library(DESeq2)
library(compositions)
library(metagenomeSeq)  
library(GMPR)
library(vegan)
library(ggplot2)

# Load data
archaea_species <- read.csv("bracken_archaea_species.csv", row.names = 1)
# bacteria_species <- read.csv("bracken_bacteria_species.csv", row.names = 1)
# fungi_species <- read.csv("bracken_fungi_species.csv", row.names = 1)
# virus_species <- read.csv("bracken_virus_species.csv", row.names = 1)

# Check samples
sample_sums <- colSums(archaea_species)
zero_samples <- names(sample_sums[sample_sums == 0])
if (length(zero_samples) > 0) {
  warning(paste("Total abundance = 0, will be removed:", paste(zero_samples, collapse = ", ")))
  archaea_species <- archaea_species[, !colnames(archaea_species) %in% zero_samples]
}

# CSS normalization
filter_threshold <- 0.001
keep_rows <- rowSums(archaea_species > 0) >= (filter_threshold * ncol(archaea_species))
archaea_species_filtered <- archaea_species[keep_rows, ]
non_zero_features_per_sample <- colSums(archaea_species_filtered > 0)

if (any(colSums(archaea_species_filtered) == 0)) {
  stop("Exist samples total abundance = 0, please check.")
}

meta_zero <- data.frame(sample = colnames(archaea_species_filtered))
rownames(meta_zero) <- colnames(archaea_species_filtered)
obj <- newMRexperiment(archaea_species_filtered, phenoData = AnnotatedDataFrame(meta_zero))

obj <- cumNorm(obj, p = cumNormStatFast(obj))
css_counts <- MRcounts(obj, norm = TRUE, log = FALSE)

# Save normalized abundance
write.csv(css_counts, "archaea_species_css.csv")

# Load metadata
metadata_archaea_species <- read.csv("metadata.csv", row.names = 1)
metadata_archaea_species <- metadata_archaea_species[!rownames(metadata_archaea_species) %in% zero_samples, ]

# Define phenotypes
phenotype_vars <- c("MS", "sex", "age", "weight", "height", "bmi", "allergy", "diet", "site", "smoke",
                    "weight_change", "pets", "Treatment")

# Diet components
# phenotype_vars <- c("Alcohol % of cals", "B1, B2", "Beta-carotene", "Bread, pasta, rice", "Calories", "Carbohydrate", 
#                     "Carbohydrate as % of cals", "Cholesterol", "Dietary Fiber", "Fat", "Fat as % of cals", "Fruits, fruit juices", 
#                     "Good oils", "Magnesium", "Meat, eggs, or beans", "Milk, cheese, yogurt", "Monounsaturated fat", "Niacin",
#                     "Polyunsaturated fat", "Potassium", "Protein", "Protein as % of cals", "Saturated fat", "Saturated fat as % of cals",
#                     "Sodium", "Sweets % of cals", "Vegetables group", "Vitamin B6", "Whole grains", "without potatoes", 
#                     "Vitamdietpairsumin A", "Vitamin C", "Vitamin E", "Folate", "Calcium", "Iron", "Zinc")

# Transpose
data_norm_t <- t(css_counts)
data_norm_t <- data_norm_t[rownames(metadata_archaea_species), ]

# Calculate Bray Curtis distance
bray_curtis_dist <- vegdist(data_norm_t, method = "bray")

# PERMANOVA analysis
adonis_result <- adonis(bray_curtis_dist ~ ., data = metadata_archaea_species[, phenotype_vars], permutations = 9999)
phenotype_r2 <- adonis_result$aov.tab[1:length(phenotype_vars), "R2"]
names(phenotype_r2) <- phenotype_vars

phenotype_r2_pct <- phenotype_r2 * 100
df <- data.frame(
  Phenotype = names(phenotype_r2_pct),
  VarianceExplained = phenotype_r2_pct
)
df$Phenotype <- factor(
  df$Phenotype, 
  levels = df$Phenotype[order(df$VarianceExplained, decreasing = TRUE)]
)  

# Save results
print(adonis_result$aov.tab)
write.csv(df, "archaea_importance.csv")