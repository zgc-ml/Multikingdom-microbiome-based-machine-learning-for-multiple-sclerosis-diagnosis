library(vegan)
library(compositions)
library(metagenomeSeq)  

# Load data
otu <- read.csv("archaea_species_css.csv",row.names = 1,stringsAsFactors = FALSE, check.names = FALSE)
# otu <- read.csv("bacteria_species_css.csv",row.names = 1,stringsAsFactors = FALSE, check.names = FALSE)
# otu <- read.csv("fungi_species_css.csv",row.names = 1,stringsAsFactors = FALSE, check.names = FALSE)
# otu <- read.csv("virus_species_css.csv",row.names = 1,stringsAsFactors = FALSE, check.names = FALSE)
otu <- t(otu)
metadata <- read.csv("metadata.csv",row.names = 1,stringsAsFactors = FALSE, check.names = FALSE)

# Richness 
richness <- rowSums(otu > 0)

#Shannon index
shannon_index <- diversity(otu, index = 'shannon', base = 2)

# Save results
result <- data.frame(richness, shannon_index)
group_data <- read.csv("group.csv", stringsAsFactors = FALSE)
rownames(group_data) <- group_data$iMSMS_ID
group <- group_data[metadata$iMSMS_ID, "Group"]
result$group <- group
write.csv(result, 'archaea_alpha_diversity.csv', quote = FALSE)