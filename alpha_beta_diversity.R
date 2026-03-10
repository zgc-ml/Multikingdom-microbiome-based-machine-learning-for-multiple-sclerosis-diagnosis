setwd("~/iMSMS/")
library(vegan)
library(compositions)
library(metagenomeSeq)  

### archaea
archaea <- read.csv("~/iMSMS/archaea_species.csv", row.names = 1)

otu <- t(archaea)

metadata <- read.csv("~/iMSMS/metadata.csv", row.names = 1, stringsAsFactors = FALSE, check.names = FALSE)

common_samples <- intersect(rownames(otu), rownames(metadata))
otu <- otu[common_samples, ]
metadata <- metadata[common_samples, ]

richness <- rowSums(otu > 0)

shannon_index <- diversity(otu, index = 'shannon', base = 2)

result <- data.frame(richness, shannon_index)

group <- metadata$Disease
result$group <- group

bray_curtis_dist <- vegdist(otu, method = "bray", na.rm = TRUE)
bray_curtis_mat <- as.matrix(bray_curtis_dist)

n_samples <- nrow(bray_curtis_mat)
mean_bray_curtis <- rowSums(bray_curtis_mat) / (n_samples - 1)
result$mean_bray_curtis <- mean_bray_curtis

write.csv(result, '~/iMSMS/archaea_species_diversity.csv', quote = FALSE)


### bacteria
bacteria <- read.csv("~/iMSMS/bacteria_species.csv", row.names = 1)

otu <- t(bacteria)

metadata <- read.csv("~/iMSMS/metadata.csv", row.names = 1, stringsAsFactors = FALSE, check.names = FALSE)

common_samples <- intersect(rownames(otu), rownames(metadata))
otu <- otu[common_samples, ]
metadata <- metadata[common_samples, ]

richness <- rowSums(otu > 0)

shannon_index <- diversity(otu, index = 'shannon', base = 2)

result <- data.frame(richness, shannon_index)

group <- metadata$Disease
result$group <- group

bray_curtis_dist <- vegdist(otu, method = "bray", na.rm = TRUE)
bray_curtis_mat <- as.matrix(bray_curtis_dist)

n_samples <- nrow(bray_curtis_mat)
mean_bray_curtis <- rowSums(bray_curtis_mat) / (n_samples - 1)
result$mean_bray_curtis <- mean_bray_curtis

write.csv(result, '~/iMSMS/bacteria_species_diversity.csv', quote = FALSE)

### fungi
fungi <- read.csv("~/iMSMS/fungi_species.csv", row.names = 1)

otu <- t(fungi)

metadata <- read.csv("~/iMSMS/metadata.csv", row.names = 1, stringsAsFactors = FALSE, check.names = FALSE)

common_samples <- intersect(rownames(otu), rownames(metadata))
otu <- otu[common_samples, ]
metadata <- metadata[common_samples, ]

richness <- rowSums(otu > 0)

shannon_index <- diversity(otu, index = 'shannon', base = 2)

result <- data.frame(richness, shannon_index)

group <- metadata$Disease
result$group <- group

bray_curtis_dist <- vegdist(otu, method = "bray", na.rm = TRUE)
bray_curtis_mat <- as.matrix(bray_curtis_dist)

n_samples <- nrow(bray_curtis_mat)
mean_bray_curtis <- rowSums(bray_curtis_mat) / (n_samples - 1)
result$mean_bray_curtis <- mean_bray_curtis

write.csv(result, '~/iMSMS/fungi_species_diversity.csv', quote = FALSE)

### virus
virus <- read.csv("~/iMSMS/virus_species.csv", row.names = 1)

otu <- t(virus)

metadata <- read.csv("~/iMSMS/metadata.csv", row.names = 1, stringsAsFactors = FALSE, check.names = FALSE)

common_samples <- intersect(rownames(otu), rownames(metadata))
otu <- otu[common_samples, ]
metadata <- metadata[common_samples, ]

richness <- rowSums(otu > 0)

shannon_index <- diversity(otu, index = 'shannon', base = 2)

result <- data.frame(richness, shannon_index)

group <- metadata$Disease
result$group <- group

bray_curtis_dist <- vegdist(otu, method = "bray", na.rm = TRUE)
bray_curtis_mat <- as.matrix(bray_curtis_dist)

n_samples <- nrow(bray_curtis_mat)
mean_bray_curtis <- rowSums(bray_curtis_mat) / (n_samples - 1)
result$mean_bray_curtis <- mean_bray_curtis

write.csv(result, '~/iMSMS/virus_species_diversity.csv', quote = FALSE)
