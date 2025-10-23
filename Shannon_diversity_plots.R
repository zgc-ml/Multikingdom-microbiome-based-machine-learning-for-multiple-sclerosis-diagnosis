library(ggplot2)
library(ggpubr)
library(gghalves)
library(ggrepel)
library(survminer) 

# Colors
ordercolors<-c("#377EB8","#E41A1C")

data$Group <- factor(data$group,
                     levels = c('Control','RRMS'))
# data$Group <- factor(data$group,
#                      levels = c('Control','PMS'))

# Define comparisons
my_comparisons <- list( c("Control","RRMS"))

# Archaea
data <- read.csv("archaea_alpha_diversity.csv", row.names = 1)

# Plot for shannon_index
ggplot(data = data, aes(x = Group, y = shannon_index, fill = Group)) +
  geom_half_violin(side = "r", color = NA, alpha = 0.4) +
  geom_half_boxplot(side = "r", errorbar.draw = FALSE, width = 0.2, linewidth = 0.8) +
  geom_half_point_panel(side = "l", shape = 21, size = 3, color = "white") +
  scale_fill_manual(values = ordercolors,
                    labels = c('Control', 'RRMS')) +  
  scale_y_continuous(limits = c(0, 10.5), expand = c(0, 0)) +
  scale_x_discrete(labels = c('Control' = 'Control', 'RRMS' = 'RRMS')) +  
  labs(y = "Shannon Diversity", x = NULL, title = "") +
  rotate_x_text(angle = 0) +
  geom_hline(yintercept = mean(data$shannon_index), linetype = 2) +
  theme(plot.title = element_text(hjust = 0.5, size = 36, face = "bold"),
        legend.position = "none",
        legend.text = element_text(size = 24),
        legend.title = element_blank(), 
        panel.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text = element_text(size = 32, color = "black"),
        axis.title.y = element_text(color = "black", size = 30),
        axis.line = element_line(size = 0.8, colour = "black")) +
  stat_compare_means(comparisons = my_comparisons, label = "p.format", 
                     method = "wilcox.test", label.y = c(8, 8),
                     hide.ns = TRUE, size = 12, p.adjust.method = "fdr")

## Bacteria
data <- read.csv("bacteria_alpha_diversity.csv", row.names = 1)

# Plot for shannon_index
ggplot(data = data, aes(x = Group, y = shannon_index, fill = Group)) +
  geom_half_violin(side = "r", color = NA, alpha = 0.4) +
  geom_half_boxplot(side = "r", errorbar.draw = FALSE, width = 0.2, linewidth = 0.8) +
  geom_half_point_panel(side = "l", shape = 21, size = 3, color = "white") +
  scale_fill_manual(values = ordercolors,
                    labels = c('Control', 'RRMS')) +  
  scale_y_continuous(limits = c(0, 11), expand = c(0, 0)) +
  labs(y = "Shannon Diversity", x = NULL, title = "") +
  rotate_x_text(angle = 0) +
  geom_hline(yintercept = mean(data$shannon_index), linetype = 2) +
  theme(plot.title = element_text(hjust = 0.5, size = 30),
        legend.position = "none",
        legend.text = element_text(size = 24),
        legend.title = element_blank(),  
        panel.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text = element_text(size = 32, color = "black"),
        axis.title.y = element_text(color = "black", size = 30),
        axis.line = element_line(size = 0.8, colour = "black")) +
  stat_compare_means(comparisons = my_comparisons, label = "p.format", 
                     method = "wilcox.test", label.y = c(8, 8),
                     hide.ns = TRUE, size = 12, p.adjust.method = "fdr")

## Fungi
data <- read.csv("fungi_alpha_diversity.csv", row.names = 1)

# Plot for shannon_index
ggplot(data = data, aes(x = Group, y = shannon_index, fill = Group)) +
  geom_half_violin(side = "r", color = NA, alpha = 0.4) +
  geom_half_boxplot(side = "r", errorbar.draw = FALSE, width = 0.2, linewidth = 0.8) +
  geom_half_point_panel(side = "l", shape = 21, size = 3, color = "white") +
  scale_fill_manual(values = ordercolors,
                    labels = c('Control', 'RRMS')) +  
  scale_y_continuous(limits = c(0, 10), expand = c(0, 0)) +
  labs(y = "Shannon Diversity", x = NULL, title = "") +
  rotate_x_text(angle = 0) +
  geom_hline(yintercept = mean(data$shannon_index), linetype = 2) +
  theme(plot.title = element_text(hjust = 0.5, size = 36, face = "bold"),
        legend.position = "none",
        legend.text = element_text(size = 24),
        legend.title = element_blank(),  
        panel.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text = element_text(size = 32, color = "black"),
        axis.title.y = element_text(color = "black", size = 30),
        axis.line = element_line(size = 0.8, colour = "black")) +
  stat_compare_means(comparisons = my_comparisons, label = "p.format", 
                     method = "wilcox.test", label.y = c(7, 7),
                     hide.ns = TRUE, size = 12, p.adjust.method = "fdr")

## Virus
data <- read.csv("virus_alpha_diversity.csv", row.names = 1)
# Plot for shannon_index
ggplot(data = data, aes(x = Group, y = shannon_index, fill = Group)) +
  geom_half_violin(side = "r", color = NA, alpha = 0.4) +
  geom_half_boxplot(side = "r", errorbar.draw = FALSE, width = 0.2, linewidth = 0.8) +
  geom_half_point_panel(side = "l", shape = 21, size = 3, color = "white") +
  scale_fill_manual(values = ordercolors,
                    labels = c('Control', 'RRMS')) + 
  scale_y_continuous(limits = c(0, 8), expand = c(0, 0)) +
  labs(y = "Shannon Diversity", x = NULL, title = "") +
  rotate_x_text(angle = 0) +
  geom_hline(yintercept = mean(data$shannon_index), linetype = 2) +
  theme(plot.title = element_text(hjust = 0.5, size = 36, face = "bold"),
        legend.position = "none",
        legend.text = element_text(size = 24),
        legend.title = element_blank(), 
        panel.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text = element_text(size = 32, color = "black"),
        axis.title.y = element_text(color = "black", size = 30),
        axis.line = element_line(size = 0.8, colour = "black")) +
  stat_compare_means(comparisons = my_comparisons, label = "p.format", 
                     method = "wilcox.test", label.y = c(6, 6),
                     hide.ns = TRUE, size = 12, p.adjust.method = "fdr")
