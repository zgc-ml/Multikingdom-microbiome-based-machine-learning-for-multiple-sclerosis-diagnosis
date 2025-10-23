suppressPackageStartupMessages({
  library(tidyverse)
  library(readr)
  library(janitor)
  library(scales)
  library(igraph)
  library(tidygraph)
  library(ggraph)
  library(ggrepel)
  library(ggnetwork)
})

# Load data
df <- read_csv("K02926.csv", show_col_types = FALSE) %>%
  janitor::clean_names()

num_cols <- c("rho", "p", "n", "p_adj")
df <- df %>%
  mutate(across(all_of(num_cols), as.numeric))

center_ko <- "K02926"

df <- df %>% filter(ko == center_ko)


alpha <- 0.05
topN  <- 50

df_sig <- df %>%
  filter(!is.na(p_adj), p_adj < alpha) %>%
  arrange(desc(abs(rho))) %>%
  slice_head(n = topN)

# Edges
edges <- df_sig %>%
  transmute(
    from = center_ko,
    to   = species,
    rho,
    p,
    p_adj,
    group_y = group_y,
    sig = sig
  )

# Nodes
nodes_center <- tibble(
  name = center_ko,
  group_y = NA_character_,
  type = "center",
  node_color_group = "Center"
)

nodes_species <- tibble(name = unique(edges$to)) %>%
  left_join(edges %>% select(to, group_y) %>% distinct(), by = c("name" = "to")) %>%
  mutate(
    type = "bacteria",
    node_color_group = case_when(
      group_y == "Up" ~ "Up",
      group_y == "Down" ~ "Down",
      TRUE ~ "Other"
    )
  )

nodes <- bind_rows(nodes_center, nodes_species)

# Graph
graph <- tbl_graph(nodes = nodes, edges = edges, directed = FALSE)

# Colors and property
node_palette <- c(
  "Center" = "goldenrod1",   
  "Up"     = "#E41A1C",      
  "Down"   = "#377EB8",     
  "Other"  = "#9AA4B2"      
)

edge_low  <- "#5B6FD8"  
edge_mid  <- "#D1D5DB" 
edge_high <- "#F26457"  

graph <- graph %>%
  activate(edges) %>%
  mutate(
    alpha = rescale(-log10(pmax(p_adj, 1e-300)), to = c(0.35, 1)),
    edge_lbl = sprintf("ρ = %.3f\nFDR = %s", rho, pvalue(p_adj, accuracy = 0.001))
  )

# Layout
ig <- as.igraph(graph)
set.seed(123)
lay_star <- igraph::layout_as_star(ig, center = which(V(ig)$name == center_ko))
center_idx <- which(V(ig)$name == center_ko)
r_factor   <- 1.25  # 半径扩张因子，适度增加留白
lay <- lay_star
for (i in seq_len(nrow(lay))) {
  if (i != center_idx) {
    lay[i, ] <- lay[i, ] * r_factor
  }
}
coords <- as_tibble(lay, .name_repair = ~c("x","y")) %>%
  mutate(.row = row_number())
nodes_tbl <- graph %>%
  activate(nodes) %>%
  as_tibble() %>%
  mutate(.row = row_number()) %>%
  left_join(coords, by = ".row")

# Plot
p <- ggraph(graph, layout = "manual", x = lay[,1], y = lay[,2]) +
  # 边
  geom_edge_link(
    aes(
      edge_width = pmin(pmax(abs(rho), 0.05), 0.6),
      edge_alpha = alpha,
      edge_color = rho
    ),
    lineend = "round",
    show.legend = TRUE
  ) +
  scale_edge_color_gradient2(
    low = edge_low,
    mid = edge_mid,
    high = edge_high,
    midpoint = 0,
    name = "Correlation (ρ)"
  ) +
  scale_edge_width(range = c(0.6, 3.6), guide = "none") +
  scale_edge_alpha(range = c(0.35, 1), guide = "none") +
  geom_node_point(
    aes(
      color = node_color_group,
      shape = type,
      size  = case_when(
        type == "center" ~ 12.2,               
        node_color_group %in% c("Up","Down") ~ 12,  
        TRUE ~ 6.4
      )
    ),
    stroke = 0.8
  ) +
  scale_color_manual(
    values = node_palette,
    breaks = c("Up", "Down", "Other"), 
    name = "Group"
  ) +
  scale_shape_manual(
    values = c(center = 21, bacteria = 19),
    guide = "none" 
  ) +
  guides(size = "none") +
  geom_point(
    data = nodes_tbl %>% filter(type == "center"),
    aes(x = x, y = y),
    size = 36, shape = 21, stroke = 2,   
    color = "#0F172A", fill = "goldenrod1",
    inherit.aes = FALSE
  ) +
  geom_text(
    data = nodes_tbl %>% filter(type == "center"),
    aes(x = x, y = y, label = name),
    fontface = "bold",
    family = "",
    size = 7.8,   
    color = "#0B1020",
    inherit.aes = FALSE
  ) +
  ggrepel::geom_text_repel(
    data = nodes_tbl %>% filter(type == "bacteria"),
    aes(x = x, y = y, label = name, color = node_color_group),
    size = 6,
    box.padding = 0.55,
    point.padding = 0.3,
    min.segment.length = 0,
    segment.size = 0.3,
    segment.color = "grey60",
    seed = 123,
    max.overlaps = Inf,
    force = 1.1,
    force_pull = 0.9,
    nudge_x = 0,
    nudge_y = 0,
    show.legend = FALSE,
    inherit.aes = FALSE
  ) +
  theme_minimal(base_size = 28) +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.grid = element_blank(),
    legend.position = "right",
    legend.title = element_text(size = 32, face = "bold"),  # 放大图例标题
    legend.text  = element_text(size = 22),                  # 放大图例文字
    axis.text = element_blank(),
    axis.title = element_blank(),
    axis.ticks = element_blank(),
    plot.title = element_text(size = 20, face = "bold", hjust = 0),
    plot.subtitle = element_text(size = 14, color = "grey20"),
    plot.margin = margin(18, 20, 18, 20)
  ) +
  labs(
    title = paste0(center_ko, ""),
    subtitle = ""
  )

print(p)