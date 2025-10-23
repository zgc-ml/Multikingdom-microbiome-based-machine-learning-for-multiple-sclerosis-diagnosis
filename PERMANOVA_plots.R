library(ggplot2)
library(dplyr)
library(viridis)
library(RColorBrewer)

# Load data
archaea <- read.csv("archaea_importance.csv", row.names = 1)
archaea$Domain <- "Archaea"
bacteria <- read.csv("bacteria_importance.csv", row.names = 1)
bacteria$Domain <- "Bacteria"
fungi <- read.csv("fungi_importance.csv", row.names = 1)
fungi$Domain <- "Fungi"
virus <- read.csv("virus_importance.csv", row.names = 1)
virus$Domain <- "Viruses"
ko <- read.csv("ko_importance.csv", row.names = 1)
ko$Domain <- "KO genes"
path <- read.csv("path_importance.csv", row.names = 1)
path$Domain <- "Pathways"

# Combine
data <- bind_rows(archaea, bacteria, fungi, virus, ko, path)
data$Category <- as.character(data$Phenotype)
data$`Dietary component` <- data$Phenotype

# Convert Domain to factor
data$Domain <- factor(data$Domain, levels = c("Archaea", "Bacteria", "Fungi", "Viruses", "KO genes", "Pathways"))

# Order Category by total VarianceExplained
category_totals <- data %>%
  group_by(Category) %>%
  summarise(TotalVariance = sum(VarianceExplained)) %>%
  arrange(desc(TotalVariance))

data$Category <- factor(data$Category, levels = category_totals$Category)

num_categories <- length(unique(data$Category))

# Set custom_theme
custom_theme <- theme_classic() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 24, color = "black"),
    axis.text.y = element_text(size = 26, color = "black"),
    axis.title.y = element_text(size = 28, color = "black", margin = margin(t = 0, r = 10, b = 0, l = 0)),
    axis.title.x = element_text(size = 28, color = "black"),
    legend.text = element_text(size = 28, color = "black"),
    legend.title = element_text(size = 30, color = "black"),
    legend.key.size = unit(0.8, "lines"),
    panel.grid = element_blank()
  )

# Plot VarianceExplained
colors <- colorRampPalette(brewer.pal(13, "RdYlBu"))(num_categories)
ggplot(data, aes(x = Domain, y = VarianceExplained, fill = Category)) +
  geom_bar(stat = "identity", width = 0.7) +
  labs(x = "", y = expression("Beta-diversity variance explained (%)"), fill = "Dietary Component") +
  scale_fill_manual(values = colors) +
  custom_theme +
  geom_text(
    aes(label = ifelse(VarianceExplained > 0.1, round(VarianceExplained, 2), "")),
    position = position_stack(vjust = 0.5),
    size = 6,
    color = "black"
  ) +
  guides(fill = guide_legend(ncol = 1)) +  
  theme(
    legend.title = element_text(size = 22, face = "bold"),
    legend.text = element_text(size = 20),
    axis.title = element_text(size = 24, face = "bold"),
    axis.text = element_text(size = 20),
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 14, hjust = 0.5),
    plot.caption = element_text(size = 10, hjust = 1, face = "italic")
  )


# clear working space
rm(list = ls())

library(tidyverse)
library(ComplexHeatmap)
library(circlize)
library(RColorBrewer)
library(grid)
library(dplyr)  

# Load data
archaea <- read.csv('archaea_sig.csv')
bacteria <- read.csv('bacteria_sig.csv')
fungi <- read.csv('fungi_sig.csv')
virus <- read.csv('virus_sig.csv')
ko <- read.csv('ko_sig.csv')
path <- read.csv('path_sig.csv')

# Merge all data
archaea_p <- archaea %>% dplyr::select(Phenotype, P) %>% dplyr::rename(Archaea = P)
bacteria_p <- bacteria %>% dplyr::select(Phenotype, P) %>% dplyr::rename(Bacteria = P)
fungi_p <- fungi %>% dplyr::select(Phenotype, P) %>% dplyr::rename(Fungi = P)
virus_p <- virus %>% dplyr::select(Phenotype, P) %>% dplyr::rename(Viruses = P)
ko_p <- ko %>% dplyr::select(Phenotype, P) %>% dplyr::rename(KO_genes = P)
path_p <- path %>% dplyr::select(Phenotype, P) %>% dplyr::rename(Pathways = P)
merged_data <- purrr::reduce(list(archaea_p, bacteria_p, fungi_p, virus_p, ko_p, path_p), 
                             function(x, y) full_join(x, y, by='Phenotype'))

# Ensure order: Phenotype, Archaea, Bacteria, Fungi, Viruses, KO_genes, Pathways
merged_data <- merged_data %>% dplyr::select(Phenotype, Archaea, Bacteria, Fungi, Viruses, KO_genes, Pathways)

rownames(merged_data) <- merged_data$Phenotype
p_values <- merged_data %>% dplyr::select(-Phenotype)
p_values[p_values == 0] <- 1e-10 
r <- -log10(p_values)

# Significance marker matrix
p <- as.matrix(p_values)
p_chars <- matrix("", nrow = nrow(p), ncol = ncol(p), dimnames = dimnames(p))
p_chars[p >= 0 & p < 0.001] <- "***"
p_chars[p >= 0.001 & p < 0.01] <- "**"
p_chars[p >= 0.01 & p < 0.05] <- "*"
p_chars[p >= 0.05 & p <= 1] <- ""


range(r, na.rm=TRUE)

# Define colors
col_fun <- colorRamp2(c(min(r, na.rm=TRUE), 
                        (min(r, na.rm=TRUE) + max(r, na.rm=TRUE))/2, 
                        max(r, na.rm=TRUE)), 
                      c("#A6CEE3", "white", "#FB9A99")) 

# Row annotations
row_anno <- rowAnnotation(
  Phenotype = anno_text(rownames(r), 
                        location = 1,  
                        just = "right",  
                        gp = gpar(fontsize = 20)),  
  annotation_width = unit(6, "cm"), 
  show_annotation_name = FALSE
)

# Heatmap
Heatmap(
  r, 
  name = "-log10(P)", 
  col = col_fun,
  rect_gp = gpar(col = "white", lwd = 1.5),
  border_gp = gpar(col = "#1F78B4", lty = 1, lwd = 1),
  cluster_columns = FALSE,
  cluster_rows = FALSE,
  column_order = c("Archaea", "Bacteria", "Fungi", "Viruses", "KO_genes", "Pathways"),
  row_title = NULL, 
  column_title = NULL,
  column_names_gp = gpar(fontsize = 20),
  column_names_rot = 45,
  row_names_gp = gpar(fontsize = 20), 
  row_names_side = "left",
  left_annotation = NULL,
  row_names_max_width = unit(10, "cm"),  
  heatmap_legend_param = list(
    title = "-log10(P)", 
    legend_height = unit(4, "cm"), 
    grid_width = unit(0.5, "cm"),
    labels_gp = gpar(fontsize = 12),
    title_gp = gpar(fontsize = 14, fontface = "bold"),
    title_gap = unit(6, "pt") 
  ),
  cell_fun = function(j, i, x, y, width, height, fill) (
    # 将星号精确放在单元格中心
    grid.text(
      p_chars[i, j],
      x = x + unit(0.5, "npc") - unit(0.5, "npc"),
      y = y + unit(0.5, "npc") - unit(0.5, "npc"),
      just = "center",
      gp = gpar(fontsize = 24, col = "black")
    )
  )
)

