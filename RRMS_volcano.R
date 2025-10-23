library(ggplot2)
library(ggrepel)

##Archaea
data1 <- read.csv('archaea_volcano.csv')
data1$group <- factor(data1$group, levels = c("Up", "NS", "Down"))

# Filter data to points within visible plot range
data_in_range <- data1[data1$coef >= -4 & data1$coef <= 4 & 
                         -log10(data1$pval) >= -0.1 & -log10(data1$pval) <= 3, ]

top_up <- data_in_range[data_in_range$group == "Up", ]
top_up <- top_up[order(-top_up$coef), ][1:min(2, nrow(top_up)), ]  
top_down <- data_in_range[data_in_range$group == "Down", ]
top_down <- top_down[order(top_down$coef), ][1:min(2, nrow(top_down)), ] 
top_genes <- rbind(top_up, top_down)  

# Create volcano plot
ggplot(data = data1, aes(x = coef, y = -log10(pval), color = group)) +
  geom_point(alpha = 0.7, size = 4.5) +
  scale_color_manual(values = c('#E41A1C', "#ADB6B6FF", '#377EB8')) +
  scale_x_continuous(limits = c(-4, 4)) +
  scale_y_continuous(expand = expansion(add = c(0.1, 0.1)), limits = c(-0.1, 3)) +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", size = 0.6, color = "gray") +
  geom_vline(xintercept = 0, linetype = "dashed", size = 0.6, color = "gray") +
  labs(x = "Coefficient (by MaAsLin2)", y = expression(-log[10](italic(p)*"-value"))) +
  theme_classic() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, color = "black"),
    axis.text = element_text(size = 22, color = "black"),
    axis.title.x = element_text(size = 26, color = "black"),
    axis.title.y = element_text(size = 26, color = "black", vjust = 1.5),
    legend.position = 'None'
  ) +
  geom_text_repel(
    data = top_genes,
    aes(label = feature),
    color = "black",
    size = 7,
    box.padding = 0.5,
    point.padding = 0.3,
    max.overlaps = Inf
  )

##Bacteria
data1 <- read.csv('bacteria_volcano.csv')
data1$group <- factor(data1$group,levels = c("Up","NS","Down"))
data_in_range <- data1[data1$coef >= -4 & data1$coef <= 4 & 
                         -log10(data1$pval) >= -0.1 & -log10(data1$pval) <= 3, ]

top_up <- data_in_range[data_in_range$group == "Up", ]
top_up <- top_up[order(-top_up$coef), ][1:min(2, nrow(top_up)), ] 
top_down <- data_in_range[data_in_range$group == "Down", ]
top_down <- top_down[order(top_down$coef), ][1:min(2, nrow(top_down)), ]  
top_genes <- rbind(top_up, top_down)  

# Create volcano plot
ggplot(data = data1, aes(x = coef, y = -log10(pval), color = group)) +
  geom_point(alpha = 0.7, size = 4.5) +
  scale_color_manual(values = c('#E41A1C', "#ADB6B6FF", '#377EB8')) +
  scale_x_continuous(limits = c(-4, 4)) +
  scale_y_continuous(expand = expansion(add = c(0.1, 0.1)), limits = c(-0.1, 3)) +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", size = 0.6, color = "gray") +
  geom_vline(xintercept = 0, linetype = "dashed", size = 0.6, color = "gray") +
  labs(x = "Coefficient (by MaAsLin2)", y = expression(-log[10](italic(p)*"-value"))) +
  theme_classic() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, color = "black"),
    axis.text = element_text(size = 22, color = "black"),
    axis.title.x = element_text(size = 26, color = "black"),
    axis.title.y = element_text(size = 26, color = "black", vjust = 1.5),
    legend.position = 'None'
  ) +
  geom_text_repel(
    data = top_genes,
    aes(label = feature),
    color = "black",
    size = 7,
    box.padding = 0.5,
    point.padding = 0.3,
    max.overlaps = Inf
  )

##Fungi
data1 <- read.csv('fungi_volcano.csv')
data1$group <- factor(data1$group,levels = c("Up","NS","Down"))
data_in_range <- data1[data1$coef >= -15 & data1$coef <= 15 & 
                         -log10(data1$pval) >= -0.1 & -log10(data1$pval) <= 3, ]

top_up <- data_in_range[data_in_range$group == "Up", ]
top_up <- top_up[order(-top_up$coef), ][1:min(2, nrow(top_up)), ] 
top_down <- data_in_range[data_in_range$group == "Down", ]
top_down <- top_down[order(top_down$coef), ][1:min(2, nrow(top_down)), ]  
top_genes <- rbind(top_up, top_down) 

# Create volcano plot
ggplot(data = data1, aes(x = coef, y = -log10(pval), color = group)) +
  geom_point(alpha = 0.7, size = 4.5) +
  scale_color_manual(values = c('#E41A1C', "#ADB6B6FF", '#377EB8')) +
  scale_x_continuous(limits = c(-15, 15)) +
  scale_y_continuous(expand = expansion(add = c(0.1, 0.1)), limits = c(-0.1, 3)) +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", size = 0.6, color = "gray") +
  geom_vline(xintercept = 0, linetype = "dashed", size = 0.6, color = "gray") +
  labs(x = "Coefficient (by MaAsLin2)", y = expression(-log[10](italic(p)*"-value"))) +
  theme_classic() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, color = "black"),
    axis.text = element_text(size = 22, color = "black"),
    axis.title.x = element_text(size = 26, color = "black"),
    axis.title.y = element_text(size = 26, color = "black", vjust = 1.5),
    legend.position = 'None'
  ) +
  geom_text_repel(
    data = top_genes,
    aes(label = feature),
    color = "black",
    size = 7,
    box.padding = 0.5,
    point.padding = 0.3,
    max.overlaps = Inf
  )

##Virus
data1 <- read.csv('virus_volcano.csv')
data1$group <- factor(data1$group,levels = c("NS","Down"))
data_in_range <- data1[data1$coef >= -2583 & data1$coef <= 15 & 
                         -log10(data1$pval) >= -0.1 & -log10(data1$pval) <= 3, ]

top_up <- data_in_range[data_in_range$group == "Up", ]
top_up <- top_up[order(-top_up$coef), ][1:min(2, nrow(top_up)), ]  
top_down <- data_in_range[data_in_range$group == "Down", ]
top_down <- top_down[order(top_down$coef), ][1:min(2, nrow(top_down)), ]  
top_genes <- rbind(top_up, top_down)  

# Create volcano plot
ggplot(data = data1, aes(x = coef, y = -log10(pval), color = group)) +
  geom_point(alpha = 0.7, size = 4.5) +
  scale_color_manual(values = c("#ADB6B6FF", '#377EB8')) +
  scale_x_continuous(limits = c(-2583, 150)) +
  scale_y_continuous(expand = expansion(add = c(0.1, 0.1)), limits = c(-0.1, 3)) +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", size = 0.6, color = "gray") +
  geom_vline(xintercept = 0, linetype = "dashed", size = 0.6, color = "gray") +
  labs(x = "Coefficient (by MaAsLin2)", y = expression(-log[10](italic(p)*"-value"))) +
  theme_classic() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, color = "black"),
    axis.text = element_text(size = 22, color = "black"),
    axis.title.x = element_text(size = 26, color = "black"),
    axis.title.y = element_text(size = 26, color = "black", vjust = 1.5),
    legend.position = 'None'
  ) +
  geom_text_repel(
    data = top_genes,
    aes(label = feature),
    color = "black",
    size = 7,
    box.padding = 0.5,
    point.padding = 0.3,
    max.overlaps = Inf
  )
ggplot(data = data1, aes(x = coef, y = -log10(pval), color = group)) +
  geom_point(alpha = 0.7, size = 4.5) +
  scale_color_manual(values = c("#ADB6B6FF", '#377EB8')) +
  scale_x_continuous(limits = c(-6, 6)) +
  scale_y_continuous(expand = expansion(add = c(0.1, 0.1)), limits = c(-0.1, 3)) +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", size = 0.6, color = "gray") +
  geom_vline(xintercept = 0, linetype = "dashed", size = 0.6, color = "gray") +
  labs(x = "Coefficient (by MaAsLin2)", y = expression(-log[10](italic(p)*"-value"))) +
  theme_classic() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, color = "black"),
    axis.text = element_text(size = 22, color = "black"),
    axis.title.x = element_text(size = 26, color = "black"),
    axis.title.y = element_text(size = 26, color = "black", vjust = 1.5),
    legend.position = 'None'
  ) +
  geom_text_repel(
    data = subset(data1, abs(coef) >= 3 & pval < 0.05),
    aes(label = feature),
    color = "black",
    size = 7
  )