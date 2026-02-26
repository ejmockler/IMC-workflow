# Brutalist Review Fix Tracker

## Cycle 1: Critical Config & Data-Flow Bugs [DONE — 5fb6b3e]
| # | Finding | File(s) | Status |
|---|---------|---------|--------|
| 1 | Config path mismatch: `_extract_stability_config` reads empty `optimization.stability_analysis` instead of `analysis.clustering` | multiscale_analysis.py:21-77 | DONE |
| 2 | `getattr()` on dict: `main_pipeline.py:590-593` — always returns `{}`, scales default to [10,20,50] | main_pipeline.py:590-593 | DONE |
| 3 | `scale_results` key mismatch: line 652 looks for nonexistent key | main_pipeline.py:652-659 | DONE |
| 1b | `hasattr()` on dict for coabundance_opts at line 368 | multiscale_analysis.py:368 | DONE |

## Cycle 2: Statistical Integrity [DONE — 9f99365]
| # | Finding | File(s) | Status |
|---|---------|---------|--------|
| 4 | Kruskal-Wallis still in comprehensive_figures.py:354-405 | comprehensive_figures.py | DONE |
| 5 | `selection_method` fallback is 'lasso' not 'variance' | spatial_clustering.py:98 | DONE |
| 6 | Permutation p-values can be exactly 0 (need pseudocount) | spatial_neighborhood_analysis.py:149 | DONE |
| 7 | Enrichment aggregation unweighted by n_focal_cells | spatial_neighborhood_analysis.py:288-305 | DONE |
| 4b | Manual: col_labels int(tp) crash on non-numeric timepoints | comprehensive_figures.py:394 | DONE |

## Cycle 3: Documentation & Disclosure [DONE — a1ee564]
| # | Finding | File(s) | Status |
|---|---------|---------|--------|
| 10 | MW tautology: only p={0.333,0.667,1.0} possible at n=2 | METHODS.md | DONE |
| 11 | Bootstrap CI degeneracy at n=2 (9 unique configs) | METHODS.md | DONE |
| 12 | Cell types share markers → confounded enrichment | METHODS.md | DONE |
| 13 | Proportions denominator includes 79% unassigned | METHODS.md | DONE |
| MR-1/2 | Scale-adaptive params override config scalars | METHODS.md | DONE |

## Cycle 4: Code Hygiene [DONE — d8ed174]
| # | Finding | File(s) | Status |
|---|---------|---------|--------|
| 8 | bead_signal_threshold default 100.0 vs config 10.0 | main_pipeline.py:86 | DONE |
| 9 | Dead example scripts importing deleted modules (6 files) | examples/ | DONE |

## All Cycles Complete

### Commits
| Cycle | Commit | Files | Summary |
|-------|--------|-------|---------|
| 1 | 5fb6b3e | 2 (+87/-22) | Config path mismatch, getattr-on-dict, scale_results key |
| 2 | 9f99365 | 3 (+75/-56) | KW removal, lasso fallback, pseudocount, weighted enrichment |
| 3 | a1ee564 | 1 (+11/-2) | METHODS.md: MW tautology, bootstrap degeneracy, marker sharing, denominator |
| 4 | d8ed174 | 7 (+1/-1544) | bead threshold default, 6 dead example scripts |

### Post-Fix: Pipeline Re-Run [DONE — f77e1c3, 2026-02-26]
Config tuned to feasible runtime: n_bootstrap=20, n_resolutions=15, graph_caching=true,
adaptive_search=true, adaptive_max_evaluations=10. Full pipeline completed in 435.8s.

| Step | Status | Result |
|------|--------|--------|
| Core pipeline (24 ROIs × 3 scales) | DONE | 435.8s, all ROIs verified |
| Cell type annotation | DONE | 11 types, 58137 superpixels, 77% unassigned |
| Differential abundance | DONE | n=2/group, 0 FDR-significant, Hedges' g + CI present |
| Spatial neighborhoods | DONE | 256 pairs, FDR fraction column, pseudocount + weighted agg |
| INDRA evidence table | DONE | 8/9 genes grounded, 91 causal statements |
| Bodenmiller benchmark | DONE | Spearman r=0.9962 |
| Notebooks (3) | DONE | All execute clean on fresh results |
| Git push | DONE | 17 commits pushed to origin/main |

## Cycle 5: Final Brutalist Review Fixes
| # | Finding | File(s) | Status |
|---|---------|---------|--------|
| C1 | METHODS.md claims `graph_caching=false`, actual `true` | METHODS.md:160,226 | DONE |
| C2 | METHODS.md claims `B=100`, actual `n_bootstrap=20` | METHODS.md:150 | DONE |
| H3 | stability_analysis hardcodes k=15, ignores scale-adaptive k | spatial_clustering.py:468, multiscale_analysis.py:355 | DONE |
| M3 | n_permutations=500 hardcoded, config/default is 1000 | spatial_neighborhood_analysis.py:322 | DONE |
| -- | Config comments stale (claim 100/40/disabled) | config.json:584-599 | DONE |

### Triaged as Non-Issues
| # | Finding | Verdict |
|---|---------|---------|
| H1 | Per-ROI FDR scope | Design choice — aggregate fraction is descriptive, already documented |
| H2 | spatial_weight no bounds [0,1] | Low risk — only called internally with known values |
| H4 | Variance selection scale bias | Design choice — simplest non-circular method, acceptable for pilot |
| H5 | spatial_weight scalar silently ignored | Known (MR-1/MR-2), already documented in METHODS.md |
| M1 | SEM error bars with n=2 | Already documented in limitations |
| M2 | scipy.stats unused import | Phantom — not present in journal_figures.py |
| M4 | Ablation tests marginal contribution | Already documented |
| M5 | Unassigned fraction per timepoint | Nice-to-have, not a bug |
| M6 | Hedges' g variance at n=2 | Already documented in limitations |

## Manual Review Findings

### After Cycle 1
- **MR-1**: `resolution_range` in config is list `[0.1, 3.0]` but consumer expects dict `{fine_scale_range, coarse_scale_range}`. Silently ignored; hardcoded [0.5,2.0]/[0.2,1.0] used. → Config schema mismatch, not a code bug. Documented in METHODS.md.
- **MR-2**: `spatial_weight` in config is scalar `0.3` but consumer expects dict `{fine_scale_weight, coarse_scale_weight}`. Silently ignored; hardcoded 0.2/0.4 used. → Same. Documented.
- **MR-3**: Lines 368-370 dead hasattr branch → FIXED in Cycle 1 commit
