library(Maaslin2)

# Load metadata
metadata_archaea_species <- read.csv("PMS_metadata.csv", row.names = 1, check.names = F)

# Archaea
archaea_species <- read.csv("archaea_species_css.csv", row.names = 1)
feature_table <- archaea_species
dim(feature_table)
dim(metadata_archaea_species)

# Obtain common samples
common_samples <- intersect(colnames(feature_table), rownames(metadata_archaea_species))
feature_table <- feature_table[, common_samples]
metadata_archaea_species <- metadata_archaea_species[common_samples, ]

colnames(metadata_archaea_species) <- make.names(colnames(metadata_archaea_species))

# Maaslin
fit_data_ms <- Maaslin2(
  input_data = feature_table,
  input_metadata = metadata_archaea_species,
  output = "archaea_ms",
  fixed_effects = c("MS", "diet", "site", "Dietary.Fiber"),
  normalization = "NONE",
  transform = "NONE",
  analysis_method = "LM",
  standardize = FALSE
)

# Bacteria
bacteria_species <- read.csv("bacteria_species_css.csv", row.names = 1)
feature_table <- bacteria_species
dim(feature_table)
dim(metadata_bacteria_species)

# Obtain common samples
common_samples <- intersect(colnames(feature_table), rownames(metadata_bacteria_species))
feature_table <- feature_table[, common_samples]
metadata_bacteria_species <- metadata_bacteria_species[common_samples, ]

colnames(metadata_bacteria_species) <- make.names(colnames(metadata_bacteria_species))

# Maaslin
fit_data_ms <- Maaslin2(
  input_data = feature_table,
  input_metadata = metadata_bacteria_species,
  output = "bacteria_ms",
  fixed_effects = c("MS", "sex", "age", "height", "allergy", "diet", "weight_change", "Treatment",
                    "Bread..pasta..rice", "Fruits..fruit.juices", "Milk..cheese..yogurt", "Protein", "Calcium"),
  normalization = "NONE",
  transform = "NONE",
  analysis_method = "LM",
  standardize = FALSE
)

# Fungi
fungi_species <- read.csv("fungi_species_css.csv", row.names = 1)
feature_table <- fungi_species
dim(feature_table)
dim(metadata_fungi_species)

# Obtain common samples
common_samples <- intersect(colnames(feature_table), rownames(metadata_fungi_species))
feature_table <- feature_table[, common_samples]
metadata_fungi_species <- metadata_fungi_species[common_samples, ]

colnames(metadata_fungi_species) <- make.names(colnames(metadata_fungi_species))

# Maaslin
fit_data_ms <- Maaslin2(
  input_data = feature_table,
  input_metadata = metadata_fungi_species,
  output = "fungi_ms",
  fixed_effects = c("MS", "weight_change", "Treatment", 
                    "Fat.as...of.cals", "Fruits..fruit.juices", "Niacin", "Zinc"),
  normalization = "NONE",
  transform = "NONE",
  analysis_method = "LM",
  standardize = FALSE
)

# Virus
virus_species <- read.csv("virus_species_css.csv", row.names = 1)
feature_table <- virus_species
dim(feature_table)
dim(metadata_virus_species)

# Obtain common samples
common_samples <- intersect(colnames(feature_table), rownames(metadata_virus_species))
feature_table <- feature_table[, common_samples]
metadata_virus_species <- metadata_virus_species[common_samples, ]

colnames(metadata_virus_species) <- make.names(colnames(metadata_virus_species))

# Maaslin
fit_data_ms <- Maaslin2(
  input_data = feature_table,
  input_metadata = metadata_virus_species,
  output = "virus_ms",
  fixed_effects = c("MS", "age", "site", "Treatment",
                    "Dietary.Fiber", "Fat.as...of.cals" ),
  normalization = "NONE",
  transform = "NONE",
  analysis_method = "LM",
  standardize = FALSE
)

# KO
ko <- read.csv("KO.csv", row.names = 1)
feature_table <- t(ko)
dim(feature_table)
dim(metadata_ko)

# Obtain common samples
common_samples <- intersect(colnames(feature_table), rownames(metadata_ko))
feature_table <- feature_table[, common_samples]
metadata_ko <- metadata_ko[common_samples, ]

colnames(metadata_ko) <- make.names(colnames(metadata_ko))

# Maaslin
fit_data_ms <- Maaslin2(
  input_data = feature_table,
  input_metadata = metadata_ko,
  output = "ko_ms",
  fixed_effects = c("MS", "sex", "age", "allergy", "weight_change", "Treatment",
                    "Bread..pasta..rice", "Niacin", "Calcium"),
  normalization = "NONE",
  transform = "NONE",
  analysis_method = "LM",
  standardize = FALSE
)

# Path
path <- read.csv("Path.csv", row.names = 1)
feature_table <- t(path)
dim(feature_table)
dim(metadata_path)

# Obtain common samples
common_samples <- intersect(colnames(feature_table), rownames(metadata_path))
feature_table <- feature_table[, common_samples]
metadata_path <- metadata_path[common_samples, ]

colnames(metadata_path) <- make.names(colnames(metadata_path))

# Maaslin
fit_data_ms <- Maaslin2(
  input_data = feature_table,
  input_metadata = metadata_path,
  output = "path_ms",
  fixed_effects = c("MS", "sex", "age", "allergy", "diet", "weight_change", "Treatment",
                    "Bread..pasta..rice", "Zinc"),
  normalization = "NONE",
  transform = "NONE",
  analysis_method = "LM",
  standardize = FALSE
)