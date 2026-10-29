# Figure 4 of the main article: global selection on three vertices (panels A and B)
# and random starts of the alternating algorithm at n = 1000 (panel C).
# Run from any directory: Rscript global_selection/figures.R
# Inputs: global_selection/results/summary.csv and random_starts/results/summary.csv.
# Output: global_selection/figures/global_selection.pdf (7.2 x 2.8 in).
library(ggplot2)
library(patchwork)
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)])))

sel <- read.csv(file.path(here, "results", "summary.csv"))
rs <- read.csv(file.path(here, "..", "random_starts", "results", "summary.csv"))
stopifnot(nrow(rs) == 9, all(rs$datasets == 100))

colors <- c("G1" = "#1B4F72", "G2" = "#E67E22", "G3" = "#27AE60")
labels <- c("G1" = "Chain", "G2" = "Confounded pair", "G3" = "Front door")
theme_report <- theme_minimal(base_size = 8, base_family = "sans") +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major = element_line(color = "#F2F3F4", linewidth = .3),
        axis.title = element_text(size = 8), axis.text = element_text(size = 7),
        legend.title = element_blank(), legend.text = element_text(size = 7),
        legend.key.height = unit(9, "pt"), legend.key.width = unit(16, "pt"),
        legend.spacing.y = unit(0, "pt"),
        plot.tag = element_text(face = "bold", size = 9),
        plot.margin = margin(5, 8, 5, 5),
        plot.background = element_rect(fill = "white", color = NA))
theme_set(theme_report)
cscale <- scale_color_manual(values = colors, labels = labels, breaks = names(labels))
fscale <- scale_fill_manual(values = colors, labels = labels, breaks = names(labels))
ns <- sort(unique(sel$n))
xscale_n <- scale_x_log10(breaks = ns, labels = ns)
xlab_n <- expression("Sample size"~italic(n))
yscale <- scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, .25),
                             labels = c("0", "0.25", "0.5", "0.75", "1"))
inside <- function(pos, just) theme(legend.position = "inside", legend.position.inside = pos,
                                     legend.justification.inside = just,
                                     legend.background = element_rect(fill = "white", color = NA),
                                     legend.margin = margin(1, 2, 1, 2))

# Panel A: proportion of the 200 samples per cell whose thresholded selected
# support is a minimum-edge support, with Wilson 95% bands.
pA <- ggplot(sel, aes(n, prop_emin, color = design, fill = design)) +
  geom_ribbon(aes(ymin = emin_wilson_lo, ymax = emin_wilson_hi), alpha = .15, color = NA) +
  geom_line(linewidth = .7) + geom_point(size = 1.3) +
  xscale_n + yscale + cscale + fscale +
  labs(x = xlab_n, y = "Proportion minimum-edge") +
  inside(c(.99, .02), c(1, 0))

# Panel B: proportion of samples whose selection has fewer than e* edges (one line per
# design) or more than e* edges. The second proportion is the same for every design at
# every n (zero throughout), so it is drawn once, in gray; the check below stops the
# script otherwise.
more_by_n <- tapply(sel$prop_more_than_e_star, sel$n, function(x) length(unique(x)))
stopifnot(all(more_by_n == 1))
more <- aggregate(prop_more_than_e_star ~ n, sel, max)
kind_labels <- c(fewer = expression("Fewer than"~italic(e)^"*"~"edges"),
                 more = expression("More than"~italic(e)^"*"~"edges"))
pB <- ggplot(sel, aes(n)) +
  geom_line(aes(y = prop_fewer_than_e_star, color = design, linetype = "fewer"),
            linewidth = .7) +
  geom_point(aes(y = prop_fewer_than_e_star, color = design, shape = "fewer"), size = 1.3) +
  geom_line(data = more, aes(n, prop_more_than_e_star, linetype = "more"),
            color = "#5D6D7E", linewidth = .5) +
  geom_point(data = more, aes(n, prop_more_than_e_star, shape = "more"),
             color = "#5D6D7E", fill = "white", size = 1.5, stroke = .6) +
  xscale_n + yscale + cscale +
  scale_linetype_manual(name = "kind", values = c(fewer = "solid", more = "dashed"),
                        labels = kind_labels, breaks = c("fewer", "more")) +
  scale_shape_manual(name = "kind", values = c(fewer = 16, more = 21),
                     labels = kind_labels, breaks = c("fewer", "more")) +
  guides(color = "none",
         linetype = guide_legend(override.aes = list(color = c("#34495E", "#5D6D7E")))) +
  labs(x = xlab_n, y = "Proportion of datasets") +
  inside(c(.99, .99), c(1, 1))

# Panel C: at n = 1000, proportion of the 100 samples in which the
# lowest-objective run among the first K random starts ends at a minimum-edge support,
# with Wilson 95% bands. The single-start rate a (all 2,000 random starts of the cell)
# is marked by an open diamond at K = 1; the dotted curve 1 - (1 - a)^K is what K
# independent starts with that rate would give.
n_mid <- sort(unique(rs$n))[2]
mid <- rs[rs$n == n_mid, ]
Ks <- c(1, 2, 5, 10, 20)
best <- do.call(rbind, lapply(Ks, function(K) data.frame(
  design = mid$design, K = K,
  prop = mid[[paste0("bestof", K, "_emin")]],
  lo = mid[[paste0("bestof", K, "_emin_lo")]],
  hi = mid[[paste0("bestof", K, "_emin_hi")]])))
Kgrid <- exp(seq(log(1), log(20), length.out = 60))
indep <- do.call(rbind, lapply(seq_len(nrow(mid)), function(i) data.frame(
  design = mid$design[i], K = Kgrid, prop = 1 - (1 - mid$single_start_emin[i])^Kgrid)))
# The three single-start rates differ by about 0.01, so the diamonds are spread
# horizontally to the left of K = 1 (log10 offsets -0.13, -0.08, -0.03) to stay visible.
single <- data.frame(design = mid$design, prop = mid$single_start_emin,
                     K = 10^c(G1 = -.13, G2 = -.08, G3 = -.03)[mid$design])
ref_labels <- c(best = expression("Best of"~italic(K)~"starts"),
                indep = expression("Pooled benchmark"~1-(1-italic(a))^italic(K)),
                single = expression("Single-start rate"~italic(a)))
pC <- ggplot(best, aes(K, prop, color = design, fill = design)) +
  geom_ribbon(aes(ymin = lo, ymax = hi), alpha = .12, color = NA) +
  geom_line(data = indep, aes(linetype = "indep"), linewidth = .45) +
  geom_line(aes(linetype = "best"), linewidth = .7) +
  geom_point(aes(shape = "best"), size = 1.3) +
  geom_point(data = single, aes(shape = "single"), size = 1.9, stroke = .6) +
  scale_x_log10(breaks = Ks, labels = Ks) + yscale + cscale + fscale +
  scale_linetype_manual(name = "ref", values = c(best = "solid", indep = "dotted",
                                                 single = "blank"),
                        labels = ref_labels, breaks = c("best", "indep", "single"),
                        drop = FALSE, limits = c("best", "indep", "single")) +
  scale_shape_manual(name = "ref", values = c(best = 16, indep = NA, single = 5),
                     labels = ref_labels, breaks = c("best", "indep", "single"),
                     drop = FALSE, limits = c("best", "indep", "single")) +
  guides(color = "none", fill = "none",
         linetype = guide_legend(override.aes = list(color = "#34495E"))) +
  labs(x = expression("Number of random starts"~italic(K)),
       y = "Proportion minimum-edge") +
  inside(c(.01, .99), c(0, 1))

# Each panel carries its legend inside an empty region of the panel.
fig <- wrap_plots(list(pA, pB, pC), nrow = 1) +
  plot_annotation(tag_levels = "A", tag_suffix = ".")

dir.create(file.path(here, "figures"), showWarnings = FALSE)
ggsave(file.path(here, "figures", "global_selection.pdf"), fig,
       width = 7.2, height = 2.8, device = cairo_pdf, bg = "white")
