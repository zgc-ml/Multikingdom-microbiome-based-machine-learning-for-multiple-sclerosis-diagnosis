library(ggplot2)
library(ggpubr)
library(gghalves)
library(ggrepel)
library(survminer)

# Colors
ordercolors<-c("#377EB8","#E41A1C")

# Adjust order of groups
data$Group <- factor(data$Group,
                     levels = c('Control','RRMS'))
# data$Group <- factor(data$Group,
#                      levels = c('Control','PMS'))

# Define comparisons
my_comparisons <- list( c("Control","RRMS"))
# Archaea
data <- read.csv("archaea_alpha_diversity.csv", row.names = 1)

# Plot for richness
ggplot(data = data, aes(x = Group, y = richness, fill = Group)) +
  geom_half_violin(side = "r", color = NA, alpha = 0.4) +
  geom_half_boxplot(side = "r", errorbar.draw = FALSE, width = 0.2, linewidth = 0.8) +
  geom_half_point_panel(side = "l", shape = 21, size = 3, color = "white") +
  scale_fill_manual(values = ordercolors,
                    labels = c('Control', 'RRMS')) +  
  scale_y_continuous(limits = c(0, 480), expand = c(0, 0)) +
  labs(y = "Richness", x = NULL, title = "") +
  rotate_x_text(angle = 0) +
  geom_hline(yintercept = mean(data$richness), linetype = 2) +
  theme(plot.title = element_text(hjust = 0.5, size = 36, face = "bold"),
        legend.position = "none",
        legend.text = element_text(size = 24),
        legend.title = element_blank(),
        panel.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text = element_text(size = 32, color = "black"),
        axis.title.y = element_text(color = "black", size = 36, face = "bold"),
        axis.line = element_line(size = 0.8, colour = "black")) +
  stat_compare_means(comparisons = my_comparisons, label = "p.format", 
                     method = "wilcox.test", label.y = c(360, 360),
                     hide.ns = TRUE, size = 12, , p.adjust.method = "fdr")

# Bacteria
data <- read.csv("bacteria_alpha_diversity.csv", row.names = 1)

# Plot for richness
ggplot(data = data, aes(x = Group, y = richness, fill = Group)) +
  geom_half_violin(side = "r", color = NA, alpha = 0.4) +
  geom_half_boxplot(side = "r", errorbar.draw = FALSE, width = 0.2, linewidth = 0.8) +
  geom_half_point_panel(side = "l", shape = 21, size = 3, color = "white") +
  scale_fill_manual(values = ordercolors,
                    labels = c('Control', 'RRMS')) + 
  scale_y_continuous(limits = c(0, 11000), expand = c(0, 0)) +
  labs(y = "Richness", x = NULL, title = "") +
  rotate_x_text(angle = 0) +
  geom_hline(yintercept = mean(data$richness), linetype = 2) +
  theme(plot.title = element_text(hjust = 0.5, size = 36, face = "bold"),
        legend.position = "none",
        legend.text = element_text(size = 24),
        legend.title = element_blank(),
        panel.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text = element_text(size = 32, color = "black"),
        axis.title.y = element_text(color = "black", size = 36, face = "bold"),
        axis.line = element_line(size = 0.8, colour = "black")) +
  stat_compare_means(comparisons = my_comparisons, label = "p.format", 
                     method = "wilcox.test", label.y = c(8400, 8400),
                     hide.ns = TRUE, size = 12, p.adjust.method = "fdr")

# Fungi
data <- read.csv("fungi_archaea_diversity.csv", row.names = 1)

# Plot for richness
ggplot(data = data, aes(x = Group, y = richness, fill = Group)) +
  geom_half_violin(side = "r", color = NA, alpha = 0.4) +
  geom_half_boxplot(side = "r", errorbar.draw = FALSE, width = 0.2, linewidth = 0.8) +
  geom_half_point_panel(side = "l", shape = 21, size = 3, color = "white") +
  scale_fill_manual(values = ordercolors,
                    labels = c('Control', 'RRMS')) +  
  scale_y_continuous(limits = c(0, 130), expand = c(0, 0)) +
  labs(y = "Richness", x = NULL, title = "") +
  rotate_x_text(angle = 0) +
  geom_hline(yintercept = mean(data$richness), linetype = 2) +
  theme(plot.title = element_text(hjust = 0.5, size = 36, face = "bold"),
        legend.position = "none",
        legend.text = element_text(size = 24),
        legend.title = element_blank(),
        panel.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text = element_text(size = 32, color = "black"),
        axis.title.y = element_text(color = "black", size = 36, face = "bold"),
        axis.line = element_line(size = 0.8, colour = "black")) +
  stat_compare_means(comparisons = my_comparisons, label = "p.format", 
                     method = "wilcox.test", label.y = c(95, 95),
                     hide.ns = TRUE, size = 12, p.adjust.method = "fdr")

# Virus
data <- read.csv("virus_alpha_diversity.csv", row.names = 1)

# Plot for richness
ggplot(data = data, aes(x = Group, y = richness, fill = Group)) +
  geom_half_violin(side = "r", color = NA, alpha = 0.4) +
  geom_half_boxplot(side = "r", errorbar.draw = FALSE, width = 0.2, linewidth = 0.8) +
  geom_half_point_panel(side = "l", shape = 21, size = 3, color = "white") +
  scale_fill_manual(values = ordercolors,
                    labels = c('Control', 'RRMS')) +  
  scale_y_continuous(limits = c(0, 255), expand = c(0, 0)) +
  labs(y = "Richness", x = NULL, title = "") +
  rotate_x_text(angle = 0) +
  geom_hline(yintercept = mean(data$richness), linetype = 2) +
  theme(plot.title = element_text(hjust = 0.5, size = 36, face = "bold"),
        legend.position = "none",
        legend.text = element_text(size = 24),
        legend.title = element_blank(),
        panel.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text = element_text(size = 32, color = "black"),
        axis.title.y = element_text(color = "black", size = 36, face = "bold"),
        axis.line = element_line(size = 0.8, colour = "black")) +
  stat_compare_means(comparisons = my_comparisons, label = "p.format", 
                     method = "wilcox.test", label.y = c(205, 205),
                     hide.ns = TRUE, size = 12, p.adjust.method = "fdr")

