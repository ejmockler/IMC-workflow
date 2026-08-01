# Analysis parameters — the actual values used to produce this data

`METHODS_FOR_MANUSCRIPT.md` gives the *rules* ("the cofactor is the fifth percentile of
positive counts"). This document gives the **realised values** — the specific numbers that
produced the shipped tables, including the ones computed from the data rather than set by
hand. Two companion files carry the per-image values in full:

- `data/arcsinh_cofactors_per_image.csv` — 792 rows: every cofactor actually used
  (24 images × 11 channels × 3 region sizes)
- `data/sham_reference_sigmoid_parameters.csv` — 27 rows: the centre and scale of every
  continuous score, at each region size

Everything below is read directly from the analysis configuration and outputs; nothing is
approximated.

---

## 1. Fixed parameters (set by hand, identical for every image)

| Step | Parameter | Value |
|---|---|---|
| Segmentation | Algorithm | SLIC |
| | Input | DNA1 + DNA2 composite (summed, then arcsinh-transformed) |
| | Target region size | 10 µm (primary); 20 µm and 40 µm also computed |
| | Compactness | 10.0 |
| | Gaussian smoothing σ | 1.5 |
| | Region count | Not fixed — derived per image as tissue area ÷ (target size)², from an eroded tissue mask |
| Quantification | Aggregation | **Sum** of ion counts per region (not a mean; no area normalisation) |
| Transformation | Function | arcsinh(count ÷ cofactor) |
| | Cofactor rule | 5th percentile of positive counts, **recomputed per marker per image** (values in §3) |
| Positivity (labels) | Default | 60th percentile of the image's own distribution |
| | Ly6G | 70th percentile |
| | CD206 | 50th percentile (post-hoc change from 65th) |
| Continuous scores | Normalisation | Logistic (sigmoid) function |
| | Steepness | 10.0 |
| | Centre and scale | **Derived from Sham animals** — values in §4 |
| Composite thresholds | Lineage score cut | 0.30 |
| | Activation score cut | 0.30 |
| | Dominance ratio | 2.0 (top lineage must exceed the second by 2× or the region is called `mixed`) |
| CD44 compartments | Positivity rule | Fixed threshold per marker from **Sham animals only**, 75th percentile of the pooled Sham distribution, applied unchanged to every image |
| Spatial | Neighbourhood | 10 nearest regions |
| | Permutations | 1,000 (labels shuffled within image) |
| Statistics | Unit of analysis | Animal (images averaged within animal first) |
| | Effect size | Hedges' *g* (small-sample-corrected standardised mean difference) |
| | Multiple comparisons | Benjamini–Hochberg |

---

## 2. What the three continuous lineage scores are actually made of

This is smaller than the names suggest, and it should be stated plainly in any figure legend
that uses these columns.

| Score | Markers | How combined |
|---|---|---|
| `lineage_immune` | **CD45 only** | — |
| `lineage_endothelial` | **CD31 and CD34** | mean of the two |
| `lineage_stromal` | **CD140a only** | — |
| `activation_cd44` | **CD44 only** | — |
| `activation_cd140b` | **CD140b only** | — |

So `lineage_immune` is a transformed CD45 intensity, not a multi-marker immune signature.
There are **no negative markers** in these definitions — that is deliberate, and it is why a
region can score highly on two lineages at once (which the discrete rules forbid). Describe
them as, for example, "CD45 score" or "CD45-based immune score", rather than implying an
independently validated immune classifier.

Each score is `sigmoid((transformed_intensity − centre) ÷ scale × 10.0)`, so a score of 0.5
means the region sits exactly at the Sham reference centre for that marker.

---

## 3. Arcsinh cofactors actually used

The cofactor sets where the transform bends from linear to logarithmic. It is **recomputed
for every marker in every image**, so it is data-dependent, not a fixed setting. Full table:
`data/arcsinh_cofactors_per_image.csv`.

Mean value by timepoint at 10 µm:

| Marker | Sham | D1 | D3 | D7 | Sham → D7 |
|---|---|---|---|---|---|
| CD45 | 10.47 | 8.90 | 9.83 | 14.45 | +38% |
| CD11b | 2.11 | 1.58 | 4.24 | 12.37 | +486% |
| Ly6G | 8.45 | 6.75 | 6.64 | 7.31 | −13% |
| CD140a | 7.02 | 4.28 | 6.39 | 10.27 | +46% |
| CD140b | 10.94 | 8.13 | 9.94 | 25.54 | +133% |
| CD31 | 28.47 | 21.05 | 15.16 | 19.37 | **−32%** |
| CD34 | 40.44 | 23.85 | 13.31 | 17.09 | **−58%** |
| CD206 | 3.32 | 2.03 | 4.30 | 15.47 | +365% |
| CD44 | 8.19 | 8.54 | 27.36 | 50.27 | +514% |

### Why this matters — and where it does not

Because the cofactor is the **denominator**, a *smaller* cofactor makes the transformed
values *larger*.

- **It does not affect the discrete cell-type labels.** Those use a percentile of each
  image's own transformed values, and dividing by a constant does not change the rank order
  within an image. Cell-type proportions are unaffected by cofactor drift.
- **It does affect the continuous scores** (§2), because those compare each image's
  transformed values against a **fixed** Sham-derived centre. If an image's cofactor is
  lower, its scores shift upward regardless of biology.

**Direction of the resulting bias, Sham → D7:**

| Score | Cofactor change | Bias on the score at D7 | Observed change | Reading |
|---|---|---|---|---|
| `lineage_endothelial` | CD31 −32%, CD34 −58% | **upward** | +0.180 | Same direction as the bias — **part of this rise may be an artefact of normalisation**. Treat with caution. |
| `lineage_immune` | CD45 +38% | downward | +0.220 | Opposite direction to the bias — the rise occurs *despite* a normalisation that works against it, so it is conservative. |
| `activation_cd44` | CD44 +514% | downward | — | Any observed CD44 increase is conservative for the same reason. |
| `lineage_stromal` | CD140a +46% | downward | +0.144 | Also conservative. |

We have not corrected for this. The honest statement for the manuscript is that the
continuous endothelial trajectory is the one endpoint whose direction coincides with a known
normalisation bias, and should not be presented as a standalone quantitative result.

---

## 4. Continuous-score centres and scales actually used

Derived once from the Sham animals and applied to every image. Values at 10 µm (all three
region sizes are in `data/sham_reference_sigmoid_parameters.csv`), on the arcsinh scale:

| Marker | Centre | Scale |
|---|---|---|
| CD45 | 1.6331 | 0.7093 |
| CD11b | 2.3430 | 1.3378 |
| Ly6G | 1.8649 | 0.6800 |
| CD140a | 2.2645 | 1.0754 |
| CD140b | 2.1526 | 1.2205 |
| CD31 | 2.4372 | 1.2941 |
| CD34 | 2.0439 | 1.0022 |
| CD206 | 1.9620 | 1.0840 |
| CD44 | 2.3508 | 1.3956 |

The centre is the 60th percentile of the pooled per-animal Sham distribution; the scale is
the interquartile range (the spread between the 25th and 75th percentiles) across the whole
experiment.

---

## 5. Realised segmentation

Region counts were derived per image rather than fixed:

| Region size | Regions per image (mean) | Range |
|---|---|---|
| 10 µm | 2,422 | 2,328 – 2,490 |

Field of view is 500 × 500 µm for every image, so at 10 µm the tissue is divided into
roughly 2,400 regions of ~100 pixels each (mean region area 101.9, 101.9, 101.8 and
100.3 pixels for Sham, D1, D3 and D7 — essentially constant across the time course).

---

## 6. Channels excluded from analysis

| Channel | Role |
|---|---|
| DNA1, DNA2 | Used **only** for segmentation, not as analysis features |
| 190BCKG | Background channel; excluded from features. **No background subtraction was applied** — no step in the pipeline performs it |
| 130Ba, 131Xe | Instrument calibration; excluded from cell-type and lineage definitions |
| 80ArAr | Plasma stability monitoring; excluded |

---

## 7. Parameters we did *not* tune

Stated explicitly because it is the question a methods reviewer asks. With two animals per
group, tuning against these data would fit sampling noise rather than biology, so:

- SLIC compactness (10.0) and smoothing (1.5) are library/hand defaults, never swept.
- The 60th-percentile positivity threshold is a convention, never optimised.
- The neighbourhood size (10) was a convention; it has **since** been tested at 5, 10 and 20
  (one spatial classification proved sensitive to it — see `METHODS_FOR_MANUSCRIPT.md` §D.9).
- The sigmoid steepness (10.0) was never varied.
- The **only** parameter chosen after seeing an outcome is the CD206 threshold
  (`METHODS_FOR_MANUSCRIPT.md` §D.4).

We recommend saying this plainly in the manuscript rather than implying an optimisation
procedure was performed.
