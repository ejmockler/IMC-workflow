# Materials and Methods — spatial proteomic analysis

Draft text for the manuscript, followed by the full parameter table and the disclosures
that should accompany it. Prepared by Eric Mockler (eric@commons.email).

Two versions of the Methods paragraph are given: a **full version** and a **condensed
version** for journals with tight Methods limits. Use one or the other, not both.

---

## A. Methods paragraph — full version

> **Imaging mass cytometry data processing and spatial analysis.**
> Imaging mass cytometry data (1 µm nominal pixel resolution; nine protein markers — CD45,
> CD11b, Ly6G, CD140a, CD140b, CD31, CD34, CD206 and CD44 — plus two DNA channels) were
> analysed for 24 regions of interest from 8 mice (n = 2 animals per timepoint: Sham,
> Day 1, Day 3 and Day 7 following unilateral ureteral obstruction; 3 images per animal,
> each a single 500 × 500 µm field). One additional acquisition was excluded prior to
> analysis as a calibration test. The design is cross-sectional: different animals were
> sampled at each timepoint. Each image was annotated as cortex or medulla, and every
> animal contributed images of both.
>
> Because the antibody panel lacked the membrane and nuclear markers required for reliable
> single-cell segmentation, images were segmented into superpixels — small contiguous
> groups of pixels of similar intensity — using the SLIC algorithm (Achanta et al., *IEEE
> Trans Pattern Anal Mach Intell* 2012) applied to the combined DNA1/DNA2 signal, so that
> segmentation was driven by tissue content rather than by any marker subsequently
> quantified. Superpixels were generated at a target diameter of 10 µm (compactness 10.0,
> Gaussian smoothing σ = 1.5), yielding a mean of 2,422 superpixels per image
> (range 2,328–2,490; 58,137 in total). Analyses were repeated at 20 µm and 40 µm to test
> sensitivity to this choice. All measurements are therefore per tissue region rather than
> per cell, and all proportions are reported as fractions of tissue regions.
>
> Ion counts were summed within each superpixel and transformed as
> arcsinh(count / cofactor), with the cofactor set to the fifth percentile of positive
> counts for each marker in each image. Marker positivity was called relative to each
> image's own intensity distribution, at the 60th percentile by default, with one
> pre-specified exception — Ly6G at the 70th percentile, applying greater stringency to a
> sparse marker — and one post-hoc adjustment, CD206 at the 50th percentile. The CD206
> threshold was lowered from the 65th percentile because at the 65th percentile no tissue
> region in any image satisfied the M2-like macrophage rule, inconsistent with published
> unilateral ureteral obstruction time courses; CD206 is used by 14 of the 15 cell-type
> rules — as a required-positive marker in three and a required-negative marker in eleven —
> so this adjustment affects every cell-type assignment except neutrophil. Because
> thresholds are within-image percentiles, the proportion of regions called positive for a
> given marker is fixed by construction across images (40% at the 60th percentile, 30% for
> Ly6G, 50% for CD206). Discrete cell-type proportions therefore describe how markers
> co-occur and redistribute across the tissue, not changes in absolute marker abundance.
>
> Each superpixel was annotated in two complementary ways. First, a discrete cell-type
> label was assigned from 15 rules, each requiring a defined set of markers to be positive
> and a further set to be negative, applied in fixed priority order with the first match
> retained; superpixels satisfying no rule were labelled unassigned. A mean of 13.6% of
> superpixels per image received a label (range 8.9–18.8%); the large unassigned fraction
> reflects the absence of tubular epithelial markers from the panel together with the
> negative-marker requirements, which exclude regions co-expressing markers of more than
> one lineage. Second, three continuous, non-exclusive lineage scores (immune, endothelial
> and stromal, each on a 0–1 scale referenced to the Sham distribution) were computed for
> every superpixel, allowing description of regions carrying signal from more than one
> lineage.
>
> For the compartment-level CD44 analysis, positivity was instead called using a single
> fixed threshold per marker derived from the Sham animals only (75th percentile of the
> pooled Sham distribution) and applied unchanged to every image, so that compartment rates
> are referenced to uninjured tissue and are comparable across timepoints in absolute terms.
>
> Spatial organisation was assessed by comparing the label composition of each
> superpixel's 10 nearest neighbouring superpixels against 1,000 random permutations of
> labels within the same image. Sensitivity of these results to the neighbourhood size was
> tested at 5, 10 and 20 neighbours.
>
> Superpixel proportions were averaged within each animal before comparison across
> timepoints, so that the animal rather than the superpixel is the unit of analysis.
> Differences between timepoints are reported as Hedges' *g* standardised effect sizes.
> With two animals per group, inferential testing is uninformative by construction: the
> smallest attainable two-sided Mann–Whitney p-value is 0.33 before correction (smallest
> observed Benjamini–Hochberg-adjusted value, 0.58), so p-values are not reported. All
> results are descriptive and hypothesis-generating, and are intended to identify
> candidates for testing in an adequately powered cohort.
>
> Analyses were performed in Python 3.12.10 using scikit-image 0.25.0 (SLIC segmentation),
> scikit-learn 1.7.2, SciPy 1.16.3, NumPy 2.3.4, pandas 2.3.0 and statsmodels 0.14.5.

---

## B. Methods paragraph — condensed version

> **Imaging mass cytometry analysis.** Twenty-four imaging mass cytometry acquisitions
> (1 µm pixels, 500 × 500 µm fields; nine protein markers plus two DNA channels) from
> 8 mice (n = 2 per timepoint: Sham, Day 1, Day 3, Day 7 after unilateral ureteral
> obstruction) were analysed. As the panel lacked markers for single-cell segmentation,
> images were divided into ~10 µm superpixels using SLIC (Achanta et al., 2012) on the
> combined DNA signal (mean 2,422 per image); all measurements are per tissue region, not
> per cell. Ion counts were summed per superpixel and arcsinh-transformed (cofactor = fifth
> percentile of positive counts per marker per image). Marker positivity was called
> relative to each image's own distribution at the 60th percentile (Ly6G 70th,
> pre-specified; CD206 50th, a post-hoc change from the 65th, at which no region satisfied
> the M2-like macrophage rule; CD206 enters 14 of the 15 rules). Because thresholds are
> within-image percentiles, discrete results describe marker co-occurrence rather than
> absolute abundance. Superpixels received both a discrete cell-type label from 15
> positive/negative marker rules (mean 13.6% labelled) and three continuous, non-exclusive
> lineage scores (immune, endothelial, stromal). Compartment-level CD44 rates used a fixed
> Sham-derived threshold (75th percentile) applied to all images. Spatial organisation was
> assessed against 1,000 within-image label permutations using 10 nearest neighbours.
> Proportions were averaged within each animal and differences reported as Hedges' *g*
> effect sizes; with two animals per group the smallest attainable p-value is 0.33, so no
> p-values are reported and all findings are descriptive and hypothesis-generating.
> Analyses used Python 3.12.10 with scikit-image 0.25.0, scikit-learn 1.7.2, SciPy 1.16.3,
> NumPy 2.3.4, pandas 2.3.0 and statsmodels 0.14.5.

---

## C. Full parameter table

Include as a supplementary table if the journal allows; otherwise keep for the response to
reviewers.

| Step | Parameter | Value | Basis for the choice |
|---|---|---|---|
| Segmentation | Algorithm | SLIC | Standard superpixel method |
| | Input signal | DNA1 + DNA2 composite | Marker-independent, so segmentation is not biased toward any quantified marker |
| | Target region size | 10 µm (also 20, 40) | Set in advance from kidney anatomy (≈ peritubular capillary at 10 µm; ≈ glomerular diameter at 40 µm) |
| | Compactness | 10.0 | Library default; not tuned |
| | Gaussian smoothing σ | 1.5 | Fixed in advance; not tuned |
| Quantification | Aggregation | Sum of ion counts per region | — |
| Transformation | Function | arcsinh(count / cofactor) | Standard cytometry transform; defined at zero, near-linear at low counts, logarithmic at high counts |
| | Cofactor | Fifth percentile of positive counts, per marker per image | Rule fixed in advance; value data-derived |
| Positivity (labels) | Default | 60th percentile **within each image** | Convention; not optimised. Fixes positive fraction at 40% per image by construction |
| | Ly6G | 70th percentile | Pre-specified: greater stringency for a sparse marker |
| | CD206 | 50th percentile | **Post-hoc**: lowered from the 65th, at which no region satisfied the M2-like rule. CD206 enters 14 of 15 rules (positive in 3, negative in 11) — see §D |
| Positivity (CD44 compartments) | Sham-derived fixed threshold | 75th percentile of pooled Sham distribution | Applied unchanged to all images so rates are comparable across timepoints |
| Annotation | Discrete types | 15 positive/negative marker rules, fixed priority, first match wins; else `unassigned` | Defined a priori from panel design |
| | Continuous scores | 3 non-exclusive lineage scores, 0–1, Sham-referenced | — |
| Spatial | Neighbourhood | 10 nearest regions | Convention; sensitivity subsequently tested at 5, 10, 20 |
| | Null model | 1,000 within-image label permutations | Standard permutation null |
| Statistics | Unit of analysis | Animal (images averaged within animal) | Avoids pseudoreplication |
| | Effect size | Hedges' *g* | Standardised mean difference, small-sample corrected |
| | Inference | None reported | Smallest attainable p = 0.33 at n = 2 vs 2 |

**Note on parameter selection.** With the exception of the CD206 threshold (§D), the
parameters above were set by convention or fixed in advance from anatomy, and were **not
optimised against these data**. For a cohort of this size that is the appropriate choice:
tuning parameters against 2 animals per group would fit sampling noise rather than biology.
We recommend stating this explicitly rather than implying an optimisation was performed.

---

## D. Disclosures to keep in the manuscript

Each is answerable, and each is weaker if it first appears in a response letter rather than
the manuscript.

1. **Sample size.** n = 2 animals per timepoint; significance is unattainable by design
   (smallest possible p = 0.33). Results are descriptive and hypothesis-generating. Phrase
   as "consistent with" or "suggests", not "demonstrates" or "significantly increased".
2. **Regions, not cells.** All proportions are fractions of ~10 µm tissue regions. Figure
   axes and text should say "tissue regions" or "superpixels".
3. **Positivity is relative, not absolute.** Within-image percentile thresholds fix the
   positive fraction per marker in every image (40% / 30% Ly6G / 50% CD206). Discrete
   results therefore describe **co-occurrence and redistribution**, never abundance: write
   "the fraction of regions co-expressing CD44 and endothelial markers increased", never
   "CD44 increased". The Sham-referenced compartment rates and continuous scores are not
   subject to this constraint.
4. **CD206 threshold.** Lowered from the 65th to the 50th percentile after no region
   satisfied the M2-like macrophage rule at the 65th. CD206 enters 14 of the 15 rules
   (required positive in 3, required negative in 11), so all cell-type calls except
   neutrophil are affected.
5. **Unassigned fraction.** ~86% of regions match no rule, from the missing tubular
   epithelial markers plus the rules' negative-marker requirements. State the denominator
   for any proportion.
6. **Marker specificity.** Ly6G is transiently expressed by some monocytes; CD34 also marks
   haematopoietic progenitors and some fibroblasts; CD44 rises broadly with injury and is
   expected to be pan-compartment. Labels are marker-combination definitions, not validated
   identities.
7. **Summation and region area.** Ion counts are summed rather than averaged within a
   region, so area contributes multiplicatively. Mean region area is essentially constant
   across the time course (101.9, 101.9, 101.8, 100.3 pixels for Sham, Day 1, Day 3,
   Day 7), so timepoint comparisons are unaffected; this matters only for absolute
   magnitudes compared with other datasets.
8. **Sampled area.** Three 500 × 500 µm fields per animal (~0.75 mm²), not randomly selected.
9. **A spatial result was withdrawn.** An earlier analysis suggested that among regions
   positive for two or more lineages, only the triple-positive class was spatially dispersed
   rather than clustered. Testing neighbourhood sizes of 5, 10 and 20 showed one two-lineage
   class changes classification depending on that choice, so the "only" claim was withdrawn.
   What holds at every neighbourhood size tested: triple-positive regions are dispersed, and
   the ordering among classes is unchanged. **Do not cite the earlier phrasing.**
10. **Exploratory clustering is not reported.** Unsupervised (Leiden) clustering was run
    during method development but contributes to none of the results presented and should
    not appear in the manuscript.

---

## E. Suggested figure legend boilerplate

> Values are proportions of ~10 µm tissue regions (superpixels), not of individual cells.
> Marker positivity is defined relative to each image's own intensity distribution, so
> proportions reflect marker co-occurrence rather than absolute abundance. Each point is one
> animal (n = 2 per timepoint); images were averaged within animal before plotting. Given the
> cohort size, no inferential testing was performed and differences are descriptive.
