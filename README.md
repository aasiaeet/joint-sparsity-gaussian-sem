# Joint sparsity in Gaussian causal models with correlated errors

Code for the numerical results of S. Pal, J. O'Quinn, J. P. Long and A. Asiaee, "Joint sparsity in Gaussian causal models with correlated errors" (submitted to the *Annals of Statistics*). The paper studies linear structural equation models with correlated errors, represented by acyclic directed mixed graphs, and shows that minimizing the joint number of directed and bidirected edges identifies the algebraic equivalence class of a generic bow-free generating graph. It analyzes a penalized likelihood estimator with a saturating penalty and an alternating algorithm for computing it. The scripts here reproduce every number, table and figure in Section 7 of the main article, Figures 1 and 2, and Sections S13 and S18 of the Supplementary Material.

## Requirements

- Python 3.10.12 with the packages in `requirements.txt` (`pip install -r requirements.txt`): numpy 1.26.4, scipy 1.15.3, pandas 1.5.3, cvxpy 1.7.5 with the Clarabel solver 0.11.1, sympy 1.14.0, mpmath 1.3.0 and tabulate 0.10.0.
- R 4.6.1 with ggplot2 4.0.3, patchwork 1.3.2 and scales 1.4.0, built with cairo support (the figure scripts use `cairo_pdf`).
- A LaTeX installation with TikZ, for Figure 1.

Every script finds its inputs relative to its own location, so it can be run from any directory. Scripts that run in parallel take `--workers W` (default: the smaller of 8 and the number of cores) and fix BLAS to one thread per process. All seeds are fixed in the code.

## Layout

| Folder | Content |
|---|---|
| `figures/` | TikZ source of Figure 1 |
| `local_convergence/` | Alternating algorithm near a fixed four-variable representation; two-variable exact-fiber penalties |
| `global_selection/` | Exhaustive global selection on three vertices, Table 1, Figure 3, and an independent check |
| `certify_global_selection/` | Interval-arithmetic certification of the support fits and of the criterion on 86 samples |
| `verma_four_vertices/` | The Verma constraint and the population fits of all 34,752 supports on four vertices |
| `random_starts/` | Random starts of the alternating algorithm, with `continuation/`, `bootstrap/`, `conic_steps/` and `reimplementation/` |
| `l1_standardized/` | Search for standardized counterexamples to minimum-l1 selection |
| `posterior_certificate/` | The a posteriori certificate in a two-variable example |

Each folder has a `results/` subfolder with the tables that the paper reports or that the figure scripts read, so that every figure and table can be redrawn without rerunning the computations.

## Reproducing the paper

Commands are given from the repository root. Run times are wall-clock times of the runs reported in the paper, with the number of worker processes used; CPU time is given for the longer jobs.

| Paper item | Script | Command | Run time |
|---|---|---|---|
| Figure 1 | `figures/admg_example.tex` | `cd figures && pdflatex admg_example.tex` | seconds |
| Figure 2 (Section 5.3) | `local_convergence/alternating.py`, `local_convergence/figures.R` | `python3 local_convergence/alternating.py`, then `Rscript local_convergence/figures.R` | 30 s, 1 core |
| Section 7.1, Table 1 | `global_selection/global_selection.py`, `global_selection/table1.py` | `python3 global_selection/global_selection.py --workers 40`, then `python3 global_selection/table1.py` | 6 min, 40 cores (3.1 CPU h) |
| Section 7.1, certified fits and criterion bounds (also S18) | `certify_global_selection/` | in that folder, in order: `python3 support_refits.py`, `python3 certify_supports.py`, `python3 certify_supports_spectral.py`, `python3 certify_criterion.py --skip 49,57`, `python3 certify_criterion_regions.py --only 49,57`, `python3 summarize.py`, `python3 list_uncertified.py` | about 2.5 h on up to 20 cores in total; the two criterion steps take 41 min (19 cores) and 95 min (18 cores) |
| Figure 3 | `global_selection/figures.R` | `Rscript global_selection/figures.R` (panels A and B from `global_selection/results`, panel C from `random_starts/results`) | seconds |
| Section 7.2 (also S18) | `verma_four_vertices/` | in that folder, in order: `python3 verma_symbolic.py`, `python3 verma_ideal.py`, `python3 fit_supports.py full`, `python3 classify_supports.py`, `python3 high_precision_recheck.py`, `python3 analyze.py`, `python3 near_limit_scan.py`, `python3 analyze.py`, `python3 extra_starts.py` | fits 27 min on 16 cores (6.7 CPU h); recheck 18 min on 16 cores; ideal 9 min on 1 core; other steps under 1 min |
| Section 7.3 (also S18) | `random_starts/random_starts.py` | `python3 random_starts/random_starts.py --workers 40`, then `python3 random_starts/random_starts.py --small-starts --workers 40` | 44 min on 40 cores (23.9 CPU h); 6.5 min on 40 cores |
| Section 7.3 and S18, capped runs continued to 400,000 cycles | `random_starts/continuation/` | in that folder: `python3 continue_capped.py --starts --check --run`, then `python3 analyze_continuation.py` | 13 min on 14 cores (3.0 CPU h); analysis 12 s |
| S18, clustered intervals for the single-start rates | `random_starts/bootstrap/bootstrap_rates.py` | `python3 random_starts/bootstrap/bootstrap_rates.py original` and `... continued` (the second needs the continuation analysis) | 5 s each |
| S18, conic Omega-steps and Proposition 15 | `random_starts/conic_steps/` | `python3 random_starts/conic_steps/conic_steps.py`; `python3 random_starts/conic_steps/continued_conic_runs.py` | 7 s; 15 min on 2 cores |
| S18, independent reimplementation of the random-start study | `random_starts/reimplementation/` | in that folder: `python3 test_cycle.py`; `python3 rerun.py full G1,G2,G3 250,1000,4000 0:100`; `python3 rerun.py small G1,G2,G3 250,1000,4000 0:30 --small`; `python3 compare_rerun.py full`; `python3 compare_rerun.py small --small`; `python3 local_minima.py full 10` | 70 s; 24 min and 7 min on 40 cores |
| Supplementary Figure S1 | `random_starts/figures.R` | `Rscript random_starts/figures.R` | seconds |
| S18, independent check of the global-selection study | `global_selection/independent_check.py` | `python3 global_selection/independent_check.py all --workers 40` | population 1.5 min on 3 cores; samples 13 min and regional bounds 3.5 min on 40 cores |
| Section 7.4, Figure 4, Table S2 | `local_convergence/` | `python3 local_convergence/alternating.py`, `Rscript local_convergence/figures.R`, `Rscript local_convergence/refit_table.R` | 30 s, 1 core |
| Table S3 (a posteriori certificate, S18) | `posterior_certificate/certificate.py` | `python3 posterior_certificate/certificate.py` | 1 s |
| Section S13, standardized l1 examples | `l1_standardized/` | in that folder: `python3 search_three_vertices.py`, `python3 search_four_vertices.py`, `python3 verify_four_vertices.py`, `python3 analytic_family.py`, `python3 verify_examples.py`, `python3 summarize.py > results/summary_tables.txt` | 42 min on 11 cores (7.7 CPU h); 66 min on 12 cores (13.2 CPU h); 49 min on 12 cores (9.2 CPU h); 11 s; 13 min on 1 core |

The global-selection study must be run before the random-start study, which reads `global_selection/results/population.json`. The certification reads `global_selection/results/lmin_all_datasets.npz` and the 86-sample subset `global_selection/results/independent_check_certify_subset.csv`.

Several scripts have fast modes for checking an installation against the stored results without rerunning a study: `global_selection.py --quick --out DIR` (44 of the 4,200 samples, about 1 min on 8 cores), `random_starts.py --reuse-steps --datasets 1 --out DIR` (the first sample of each cell with the stored step sizes, about 1.5 min on 8 cores), `continue_capped.py --limit N`, `independent_check.py samples --limit N`, and the `--only` options of the certification, three-vertex search and example verification scripts.

## Stored results and excluded files

The `results/` folders contain the run-level tables of the global-selection study (`runs.csv.gz`, one row per sample, and `lmin_all_datasets.npz`, the constrained likelihood minimum of every support in every sample) and of the random-start study (`runs.csv.gz` and `runs_small_starts.csv.gz`, one row per run), with all summary tables. Large intermediate outputs are not included; they are written by the scripts above when rerun:

- `random_starts/continuation/results/runs_with_continuation.csv.gz` (all 63,900 runs with outcomes before and after continuation) is written by `analyze_continuation.py` in 12 s; the per-run endpoints and starts of the continued runs are written by `continue_capped.py`.
- `global_selection/results/independent_check_samples_raw.json`, needed for the regional bounds of `independent_check.py`, is written by its `samples` stage.
- `certify_global_selection/results/support_refits.npz` is written by `support_refits.py` in about 20 s.
- The per-support fits, classes and recheck tables of `verma_four_vertices/` (`fits.csv`, `classes.csv`, `support_table.csv`, about 30 MB) are written by `fit_supports.py`, `classify_supports.py`, `high_precision_recheck.py` and `analyze.py`; `near_limit_scan.py` and `extra_starts.py` need them.
- The run tables and endpoints of the reimplementation of the random-start study are written by `rerun.py`; `local_minima.py` needs the endpoints of the full rerun.
- The invariant tables of `l1_standardized/` are rebuilt automatically (about 1 min). `results/p4_search.csv.gz` omits the per-row parameter and class-value columns; `results/p4_verify.csv.gz` covers 967 of the 1,614 four-vertex candidates, at least one for each of the 89 graphs.

## License

MIT; see `LICENSE`.
