suppressPackageStartupMessages({
  library(tidyverse)
  library(data.table)
  library(janitor)
  library(broom)
  library(ggrepel)
  library(ggtext)
  library(ggpubr)
  library(patchwork)
  library(scales)
  library(viridis)
  library(RColorBrewer)
  library(pheatmap)
  library(ggridges)
})

set.seed(1234)
options(stringsAsFactors = FALSE)

# ========= Path =========
ko_path   <- "ko.csv"
bac_path  <- "bacteria_species_css.csv"
meta_path <- "metadata.csv"

# ========= Load data =========
ko_raw   <- fread(ko_path) %>% as.data.frame()
bac_raw  <- fread(bac_path) %>% as.data.frame()
meta_raw <- fread(meta_path) %>% as.data.frame()

# Clear
ko  <- ko_raw  %>% remove_empty(which = c("rows","cols"))
bac <- bac_raw %>% remove_empty(which = c("rows","cols"))
meta <- meta_raw

if (is.null(rownames(ko)) || all(rownames(ko) == "")) {
  rn <- ko[[1]]; ko[[1]] <- NULL; rownames(ko) <- rn
}
if (is.null(rownames(bac)) || all(rownames(bac) == "")) {
  rn <- bac[[1]]; bac[[1]] <- NULL; rownames(bac) <- rn
}
rownames(ko) <- ko[,1]
ko <- ko[,-1]
rownames(bac) <- bac[,1]
bac <- bac[,-1]

if (!"SampleID" %in% names(meta)) {
  cand <- intersect(c("SampleID","sampleid","SampleId","sampleID","Sampleid"), names(meta))
  if (length(cand) == 1) colnames(meta)[match(cand, names(meta))] <- "SampleID"
}
if (!"MS" %in% names(meta)) {
  cand <- intersect(c("MS","ms","Ms"), names(meta))
  if (length(cand) == 1) colnames(meta)[match(cand, names(meta))] <- "MS"
}
stopifnot(all(c("SampleID","MS") %in% names(meta)))

target_ko <- c("K02926")
target_ko <- intersect(target_ko, rownames(ko))
common_samples <- Reduce(intersect, list(colnames(ko), colnames(bac), meta$SampleID))

ko_sub  <- ko[target_ko, common_samples, drop = FALSE]
bac_sub <- bac[, common_samples, drop = FALSE]
meta_sub <- meta %>% filter(SampleID %in% common_samples) %>%
  mutate(SampleID = factor(SampleID, levels = common_samples)) %>%
  arrange(SampleID)

# ========= Spearman correlation =========
spearman_pairwise <- function(ko_mat, bac_mat, sample_ids){
  res_list <- vector("list", length = length(rownames(ko_mat)) * length(rownames(bac_mat)))
  idx <- 0L
  for (k in rownames(ko_mat)) {
    x <- as.numeric(ko_mat[k, sample_ids])
    for (b in rownames(bac_mat)) {
      y <- as.numeric(bac_mat[b, sample_ids])
      if (all(is.na(x)) || all(is.na(y))) next
      if (sd(x, na.rm = TRUE) == 0 || sd(y, na.rm = TRUE) == 0) next
      ct <- suppressWarnings(cor.test(x, y, method = "spearman", exact = FALSE))
      idx <- idx + 1L
      res_list[[idx]] <- data.frame(
        KO = k, Species = b, rho = unname(ct$estimate), p = ct$p.value,
        n = sum(complete.cases(x, y)), stringsAsFactors = FALSE
      )
    }
  }
  res <- bind_rows(res_list)
  if (nrow(res) > 0) res <- res %>% mutate(p_adj = p.adjust(p, method = "BH"))
  res
}

all_ids <- as.character(common_samples)
rrms_ids <- meta_sub %>% filter(MS == 1) %>% pull(SampleID) %>% as.character()
ctl_ids  <- meta_sub %>% filter(MS == 0) %>% pull(SampleID) %>% as.character()

res_all <- spearman_pairwise(ko_sub, bac_sub, all_ids)     %>% mutate(group = "All")
res_rr  <- spearman_pairwise(ko_sub, bac_sub, rrms_ids)    %>% mutate(group = "RRMS")
res_ct  <- spearman_pairwise(ko_sub, bac_sub, ctl_ids)     %>% mutate(group = "Healthy")

res_all3 <- bind_rows(res_all, res_rr, res_ct) %>%
  mutate(pair = paste(KO, Species, sep = " | "),
         sig = case_when(
           p_adj < 0.001 ~ "***",
           p_adj < 0.01  ~ "**",
           p_adj < 0.05  ~ "*",
           TRUE ~ ""
         ))
fwrite(res_all3, file = "spearman_association.csv")