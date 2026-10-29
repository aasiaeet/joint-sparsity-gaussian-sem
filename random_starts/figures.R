# Supplementary Figure S1: random starts of the alternating algorithm on three vertices.
# Run from any directory: Rscript random_starts/figures.R
# Inputs: random_starts/results/{summary.csv, summary_small_starts.csv, runs.csv.gz,
# runs_small_starts.csv.gz}.  Output: random_starts/figures/random_starts.pdf.
library(ggplot2)
library(patchwork)
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)])))
summary <- read.csv(file.path(here, "results", "summary.csv"))
runs <- read.csv(file.path(here, "results", "runs.csv.gz"))
colors <- c("G1" = "#1B4F72", "G2" = "#E67E22", "G3" = "#27AE60")
labels <- c("G1" = "Chain", "G2" = "Confounded pair", "G3" = "Front-door")
theme_report <- theme_minimal(base_size = 8, base_family = "sans") +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major = element_line(color = "#F2F3F4", linewidth = .3),
        axis.title = element_text(size = 8), axis.text = element_text(size = 7),
        legend.title = element_blank(), legend.text = element_text(size = 7),
        strip.text = element_text(size = 7),
        plot.tag = element_text(face = "bold", size = 9),
        plot.margin = margin(5, 8, 5, 5),
        plot.background = element_rect(fill = "white", color = NA))
theme_set(theme_report)

# Panels A and B: best of the first K random starts ends at a minimum-edge support,
# for the main random starts (A) and for starts ten times smaller (B).
Ks <- c(1, 2, 5, 10, 20)
small <- read.csv(file.path(here, "results", "summary_small_starts.csv"))
bestof_panel <- function(sm) {
  long <- do.call(rbind, lapply(Ks, function(K) data.frame(
    design = sm$design, n = factor(sm$n), K = K,
    prop = sm[[paste0("bestof", K, "_emin")]])))
  ggplot(long, aes(K, prop, color = design, linetype = n, shape = n)) +
    geom_line(linewidth = .6) + geom_point(size = 1.2) +
    scale_x_log10(breaks = Ks) +
    scale_color_manual(values = colors, labels = labels) +
    scale_linetype_manual(values = c("250" = "dotted", "1000" = "dashed", "4000" = "solid"),
                          labels = function(x) parse(text = paste0("italic(n)==", x))) +
    scale_shape_manual(values = c("250" = 1, "1000" = 2, "4000" = 16),
                       labels = function(x) parse(text = paste0("italic(n)==", x))) +
    scale_y_continuous(limits = c(0, 1)) +
    labs(x = expression("Number of random starts"~italic(K)),
         y = "Proportion of datasets")
}
pA <- bestof_panel(summary)
pA2 <- bestof_panel(small)

# Panel C: end state of single random starts, pooled over n, for both start scales.
tr <- function(x) as.character(x) %in% c("True", "TRUE")
runs_small <- read.csv(file.path(here, "results", "runs_small_starts.csv.gz"))
end_state <- function(r, scale) {
  r <- r[r$start_type == "random", ]
  state <- ifelse(tr(r$is_emin), "emin",
           ifelse(tr(r$more_than_e_star) & tr(r$bowed), "more_bowed",
           ifelse(tr(r$more_than_e_star), "more_nobow", "other")))
  data.frame(design = r$design, scale = scale, state = state)
}
st <- rbind(end_state(runs, "sd 0.5"), end_state(runs_small, "sd 0.05"))
tab <- as.data.frame(table(design = st$design, scale = st$scale, state = st$state))
tot <- aggregate(Freq ~ design + scale, tab, sum)
names(tot)[3] <- "total"
tab <- merge(tab, tot)
tab$prop <- tab$Freq / tab$total
tab$state <- factor(tab$state, levels = c("other", "more_nobow", "more_bowed", "emin"))
tab$scale <- factor(tab$scale, levels = c("sd 0.5", "sd 0.05"),
                    labels = c("Main starts", "Small starts"))
tab$design <- factor(tab$design, levels = names(labels), labels = labels)
pB <- ggplot(tab, aes(scale, prop, fill = state)) +
  geom_col(width = .7) +
  facet_wrap(~design, nrow = 1) +
  scale_fill_manual(values = c(emin = "#27AE60", more_bowed = "#922B21", more_nobow = "#F5B041",
                               other = "#85C1E9"),
                    labels = c(emin = "Minimum-edge support",
                               more_bowed = expression("More than"~italic(e)^"*"~"edges, bowed"),
                               more_nobow = expression("More than"~italic(e)^"*"~"edges, no bow"),
                               other = expression("At most"~italic(e)^"*"~"edges, not minimum-edge")),
                    drop = FALSE) +
  scale_y_continuous(limits = c(0, 1.0001)) +
  labs(x = NULL, y = "Proportion of random starts")

fig <- ((pA | pA2) + plot_layout(guides = "collect")) / pB +
  plot_annotation(tag_levels = "A", tag_suffix = ".") &
  theme(legend.position = "right")
dir.create(file.path(here, "figures"), showWarnings = FALSE)
ggsave(file.path(here, "figures", "random_starts.pdf"), fig, width = 6.5, height = 5.4)
