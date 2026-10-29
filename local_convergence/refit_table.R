# Table S2 of the Supplementary Material: median refit error and the number of
# runs whose support was identified after one cycle, by sampling law and n.
# Run: Rscript refit_table.R   (from any directory)
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grepl("^--file=", args)])))
runs <- read.csv(file.path(here, "results", "runs.csv"))
runs$first_cycle_numerical_support <- as.logical(runs$first_cycle_numerical_support)
runs$threshold_support_match <- as.logical(runs$threshold_support_match)
tab <- do.call(rbind, lapply(split(runs, list(runs$distribution, runs$n)), function(d)
  data.frame(law = d$distribution[1], n = d$n[1], runs = nrow(d),
             median_refit_error = signif(median(d$refit_parameter_error), 5),
             one_cycle_support = sum(d$first_cycle_numerical_support),
             final_threshold_support = sum(d$threshold_support_match))))
tab <- tab[order(tab$law, tab$n), ]
rownames(tab) <- NULL
print(tab)
