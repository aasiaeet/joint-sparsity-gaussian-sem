# Figures 2 and 4 of the main article from the tables written by alternating.py.
# Figure 4: figures/alternating_and_refit.pdf (Section 7.4).
# Figure 2: figures/exact_fiber_penalties.pdf (Section 5.3).
# Run: Rscript figures.R   (from any directory)
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grepl("^--file=", args)])))
runs <- read.csv(file.path(here, "results", "runs.csv"))
trajectories <- read.csv(file.path(here, "results", "trajectories.csv"))
fiber <- read.csv(file.path(here, "results", "fiber.csv"))
library(ggplot2)
library(patchwork)
colors <- c("Gaussian" = "#1B4F72", "Student t7" = "#E67E22")
theme_report <- theme_minimal(base_size = 8, base_family = "sans") +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major = element_line(color = "#F2F3F4", linewidth = .3),
        plot.title = element_text(face = "bold", size = 8),
        axis.title = element_text(size = 8), axis.text = element_text(size = 7),
        legend.title = element_blank(), legend.text = element_text(size = 7),
        plot.tag = element_text(face = "bold", size = 9),
        plot.margin = margin(5, 8, 5, 5),
        plot.background = element_rect(fill = "white", color = NA))
theme_set(theme_report)
pop_trace <- subset(trajectories, run_id == "population")
p1 <- ggplot(pop_trace, aes(iteration, frobenius_error)) +
  geom_line(color = colors[1], linewidth = .7) +
  scale_y_log10(labels = scales::trans_format("log10", scales::math_format(10^.x))) +
  labs(x = "Alternating cycle", y = "Parameter error")
medians <- aggregate(refit_parameter_error ~ distribution + n, runs, median)
guide <- data.frame(n = sort(unique(runs$n)))
guide$error <- medians$refit_parameter_error[medians$distribution == "Gaussian" &
                                             medians$n == min(guide$n)] *
  sqrt(min(guide$n) / guide$n)
p2 <- ggplot(runs, aes(n, refit_parameter_error, color = distribution)) +
  geom_line(data = guide, aes(n, error), inherit.aes = FALSE,
            color = "#7F8C8D", linetype = "dashed", linewidth = .4) +
  geom_point(alpha = .6, size = 1.1,
             position = position_jitter(width = .018, height = 0, seed = 191)) +
  geom_line(data = medians, linewidth = .7) +
  geom_point(data = medians, size = 1.5, shape = 18) +
  scale_x_log10(breaks = sort(unique(runs$n)), labels = sort(unique(runs$n))) +
  scale_y_log10() + scale_color_manual(values = colors,
    breaks = names(colors), labels = expression(Gaussian, Transformed~italic(t)[7])) +
  labs(x = expression("Sample size"~italic(n)), y = "Parameter error")
main_figure <- (p1 | p2) + plot_annotation(tag_levels = "A", tag_suffix = ".") +
  plot_layout(guides = "collect") & theme(legend.position = "bottom")
dir.create(file.path(here, "figures"), showWarnings = FALSE)
ggsave(file.path(here, "figures", "alternating_and_refit.pdf"), main_figure,
       width = 7.2, height = 3.0, device = cairo_pdf, bg = "white")
witness <- fiber[fiber$t %in% c(0, .25, .5), ]
f1 <- ggplot(fiber, aes(t, l1)) +
  geom_line(color = "#1B4F72", linewidth = .7) +
  geom_point(data = witness, color = "#1B4F72", size = 1.8) +
  coord_cartesian(ylim = c(.44, .56)) +
  scale_x_continuous(breaks = c(0, .25, .5)) +
  labs(x = expression("Coefficient"~italic(t)), y = expression("\u2113"[1]~"penalty"))
f2 <- ggplot(fiber, aes(t, quasi_mcp)) +
  geom_line(color = "#E67E22", linewidth = .7) +
  geom_point(data = witness, color = "#E67E22", size = 1.8) +
  scale_x_continuous(breaks = c(0, .25, .5)) +
  scale_y_continuous(breaks = c(1, 1.5, 2), limits = c(.95, 2.05)) +
  labs(x = expression("Coefficient"~italic(t)), y = "Quasi-MCP penalty")
fiber_figure <- (f1 | f2) + plot_annotation(tag_levels = "A", tag_suffix = ".")
ggsave(file.path(here, "figures", "exact_fiber_penalties.pdf"), fiber_figure,
       width = 7.2, height = 2.6, device = cairo_pdf, bg = "white")
