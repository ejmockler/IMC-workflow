# Spatial analysis data package — murine kidney injury (UUO) imaging mass cytometry

Prepared for quantification plots and manuscript figures.
Contact: Eric Mockler (eric@commons.email).

Everything here is derived from 24 imaging mass cytometry acquisitions. Terms are defined
on first use; no prior familiarity with the analysis is assumed.

**Three things to read before plotting:** §1 (measurements are tissue regions, not cells),
§5 (what "positive" means — it is relative, not absolute), and §6 (what the statistics can
support). Those three govern how every figure must be labelled and described.

---

## 1. The measurements are tissue regions, not cells

Imaging mass cytometry (IMC) tags antibodies with rare-earth metal isotopes, ablates the
tissue with a laser one micron at a time, and counts the metal ions by mass spectrometry.
The raw output is, for every 1 µm pixel, an ion count for each marker.

Converting pixels into **single cells** requires membrane and nuclear stains that this
9-marker panel does not include. We therefore grouped neighbouring pixels into
**superpixels**: small contiguous patches of pixels with similar signal intensity. Ours
target **10 µm across** — roughly the size of one cell, but *not guaranteed to contain
exactly one cell*. A superpixel may contain part of a cell, one cell, or a few adjacent
cells.

**What this means for figures:** every proportion in this package is a fraction of
**tissue regions**, never a fraction of *cells*. Axis labels should read
"% of tissue regions" or "fraction of superpixels". Labelling an axis "% of cells" would
be incorrect and is the first thing a reviewer would challenge.

Throughout this README, "region" means one superpixel. "Image" means one region of
interest (one acquisition).

> One label is named `immune_cells` for historical reasons. Despite the name it counts
> **tissue regions** like everything else. In figures call it "CD45-only regions".

---

## 2. Study design (needed to group the data correctly)

- **Model:** unilateral ureteral obstruction (UUO) in mice.
- **Timepoints:** Sham, Day 1, Day 3, Day 7 — **4 groups**.
- **Animals:** **2 mice per timepoint, 8 animals total.**
- **Images:** 3 per animal, **24 total** (a 25th acquisition was a calibration test and was
  excluded before any analysis).
- **Field of view:** each image is a single **500 × 500 µm** field (0.25 mm²), so 3 fields
  sample roughly 0.75 mm² per animal. Fields were not selected at random.
- **Regions:** 58,137 superpixels total; mean 2,422 per image (range 2,328–2,490).

**This is a cross-sectional design — different animals at each timepoint.** Nothing is
followed over time within an animal.

> **Important:** the `mouse` column contains only `MS1`/`MS2`, numbered *within* each
> timepoint. `MS1` at Sham and `MS1` at Day 1 are **different animals**. Always group by
> the **`animal_id`** column (e.g. `Sham_MS1`, `D1_MS1`), which is unique across the study.
> This column was added specifically to prevent that error. The CD44 files (§3.3) do not
> carry it — build it there as `timepoint + "_" + mouse`.

**Anatomical region:** each image is labelled `Cortex` or `Medulla`. Every animal
contributed **both**, so cortex-versus-medulla comparisons are paired within animal — but
unevenly (each animal has 2 of one and 1 of the other), so each animal × region cell holds
only 1–2 images.

---

## 3. Files

```
data/
  per_image_quantification.csv        24 rows    — one row per image (start here)
  per_region_annotations.csv.gz       58,137 rows — one row per superpixel
  cd44_compartment_rates_10um.csv     168 rows   — CD44 rates by compartment (primary)
  cd44_compartment_rates_20um.csv     144 rows   — the same at 20 µm regions
  cd44_compartment_rates_40um.csv     144 rows   — the same at 40 µm regions
  arcsinh_cofactors_per_image.csv     792 rows   — every transform cofactor actually used
  sham_reference_sigmoid_parameters.csv  27 rows — centre/scale of every continuous score
METHODS_FOR_MANUSCRIPT.md             paste-ready Materials & Methods text
ANALYSIS_PARAMETERS.md                the realised parameter values (see note below)
README.md                             this file
```

> **Where the actual numbers live.** `METHODS_FOR_MANUSCRIPT.md` states the *rules*;
> **`ANALYSIS_PARAMETERS.md` gives the realised values** — including the per-image arcsinh
> cofactors and the Sham-derived centre/scale of every continuous score. Read it before
> plotting any `lineage_*` or `activation_*` column: it documents which markers each score
> is actually made of (for example `lineage_immune` is **CD45 alone**) and one
> normalisation bias that affects the continuous endothelial trajectory.

### 3.1 `per_image_quantification.csv` — the main quantification matrix

One row per image, 24 rows, 55 columns.

| Column | Meaning |
|---|---|
| `roi_id` | Unique image identifier |
| `animal_id` | **Unique animal** (`timepoint_mouse`) — group by this, not `mouse` |
| `timepoint` | `Sham`, `D1`, `D3`, `D7` |
| `region` | `Cortex` or `Medulla` (a whole-image label) |
| `mouse` | `MS1`/`MS2` — **only unique within a timepoint** (see §2) |
| `replicate` | Image number within that animal |
| `n_total` | Number of tissue regions in the image |
| `n_assigned` | Number matching one of the 15 cell-type rules |
| `assignment_rate` | `n_assigned / n_total` |
| `<celltype>_count`, `<celltype>_prop` | Per cell type: count, and count ÷ `n_total` |
| `unassigned_count`, `unassigned_prop` | Regions matching no rule. **The 15 cell types + `unassigned` sum to `n_total`.** |
| `lineage_immune_mean`, `lineage_endothelial_mean`, `lineage_stromal_mean` | Image mean of each continuous 0–1 score (§4b) |
| `subtype_<x>_count`, `subtype_<x>_prop` | The five `subtype` values (§4c) — a **separate** partition that *also* sums to `n_total` on its own |
| `mixed_fraction` | Fraction of regions scoring ≥ 0.3 on **two or more** lineages |

> **Two independent partitions live in this file.** The 15 cell types + `unassigned` sum to
> `n_total`; the 5 `subtype_*` columns *separately* sum to `n_total`. **Never add a
> `subtype_*` count to a cell-type count**, and never put both in one stacked bar — you
> would double-count the tissue.

> **The denominator.** `_prop` is out of **all** regions (`n_total`), including unassigned
> ones. For "share of labelled tissue", divide the count by `n_assigned`. State which you
> used in the legend.

### 3.2 `per_region_annotations.csv.gz` — one row per superpixel

58,137 rows. Gzip-compressed CSV; opens directly (`pd.read_csv(path)` or
`readr::read_csv`).

| Column | Meaning |
|---|---|
| `roi_id`, `animal_id`, `timepoint`, `region`, `mouse`, `replicate` | Image metadata, as above |
| `superpixel_id` | Region identifier within its image |
| `x`, `y` | Region centre position in pixels (1 pixel = 1 µm) |
| `cell_type` | One of 15 rule-based labels, or `unassigned`. Labels **13.6%** of regions. |
| `cell_type_with_activation` | **A second, independent labelling — NOT `cell_type` with a suffix.** See the warning below. Labels **80.8%** of regions. |
| `subtype` | **A third, independent labelling** (§4c) — not a rollup of `cell_type`. |
| `gating_confidence` | **Binary (0.0 or 1.0), not a graded score.** It is exactly `cell_type != 'unassigned'` (1.0 for 7,931 regions, 0.0 for 50,206). Do not use it as a quality filter — it would just re-select the labelled set. |
| `lineage_immune`, `lineage_endothelial`, `lineage_stromal` | **Continuous** 0–1 scores (§4b) |
| `activation_cd44`, `activation_cd140b` | **Continuous** 0–1 scores for the two injury/activation markers |

> ### Warning: three different labellings, never to be mixed
>
> `cell_type`, `cell_type_with_activation` and `subtype` are **three independent schemes**
> computed by different rules, not refinements of one another. Pick one per figure.
>
> - `cell_type` — the 15 rules of §4a. Labels **13.6%** of regions; the rest `unassigned`.
> - `cell_type_with_activation` — derived from the **continuous scores**, not the rules. A
>   region with no lineage score ≥ 0.3 is `unassigned`; if its top lineage score is less
>   than twice the second it is `mixed`; otherwise it takes the dominant lineage, suffixed
>   with any activation marker scoring ≥ 0.3. It labels **80.8%** of regions — including
>   **39,042 regions that `cell_type` calls `unassigned`** — and its largest single class
>   is `mixed` (30,497 regions, 52% of all tissue). It has its own vocabulary: `mixed`,
>   `stromal` and `non_myeloid_immune` appear only here; `fibroblast`, `immune_cells` and
>   `activated_immune` appear only in `cell_type`.
> - `subtype` — also from the continuous scores, evaluated only where `lineage_immune` ≥ 0.3.
>
> **Consequences:** the ~86% unassigned figure applies to `cell_type` **only**. Quoting it
> alongside `cell_type_with_activation` would misstate the data. And the schemes genuinely
> disagree — 9,127 regions are `subtype = neutrophil` versus 3,300 with
> `cell_type = neutrophil`, and 1,267 regions labelled `cell_type = neutrophil` carry
> `subtype = m2_macrophage`. Cross-tabulating them produces numbers that contradict the
> per-image file.

### 3.3 `cd44_compartment_rates_{10,20,40}um.csv`

The fraction of regions positive for CD44 within each marker-defined compartment, per
timepoint × mouse × anatomical region. **Use the 10 µm file as primary**; 20 and 40 µm are
provided so a result can be checked for sensitivity to region size.

| Column | Meaning |
|---|---|
| `compartment` | The region set the rate is computed within: `CD45`, `CD31`, `CD34`, `CD140b`, `CD206` (positive for that single marker); `endothelial_cd31cd34` (positive for CD31 **AND** CD34); `neutrophil` (regions whose `cell_type` is `neutrophil` — **10 µm file only**) |
| `timepoint` | `Sham`, `D1`, `D3`, `D7` |
| `mouse` | `MS1`/`MS2` — unique only within a timepoint. **No `animal_id` here — build it as `timepoint + "_" + mouse`** |
| `region` | `cortex`, `medulla`, or `pooled` |
| `cd44_rate` | Fraction of that compartment positive for CD44 (0–1); the **unweighted mean of the per-image rates** |
| `n_support` | Total regions in that compartment, summed over the animal's images. Use to judge how thin a cell is — it is **not** the denominator of `cd44_rate`. |

> **Two traps in this file.**
> 1. **`region` includes a `pooled` level** that is the cortex+medulla aggregate of the same
>    two rows. Filter it out (`df = df[df.region != "pooled"]`) before grouping or you will
>    count every animal twice.
> 2. **Case differs from the other files** — `cortex`/`medulla` here versus
>    `Cortex`/`Medulla` elsewhere. Normalise case before merging or rows will silently drop.
>
> Support falls as low as `n_support` = 91 at 10 µm (and lower at 40 µm); treat thin cells
> as indicative only. **Positivity in these three files is called by a different rule from
> every other file here — see §5.**

---

## 4. The three labelling schemes, defined

### (a) Discrete cell types — the 15 rules

Each region is tested against 15 rules ("gating", as in flow cytometry). Each rule requires
some markers to be **positive and others to be negative** — the neutrophil rule is
CD45+ Ly6G+ CD31− CD34−. Rules are applied in the order below and the **first match wins**,
so labels are mutually exclusive by construction rather than by biology.

| # | Label | Required positive | Required negative |
|---|---|---|---|
| 1 | `neutrophil` | CD45, Ly6G | CD31, CD34 |
| 2 | `activated_m2_cd44` | CD45, CD11b, CD206, CD44 | CD34, CD31, CD140a, Ly6G, CD140b |
| 3 | `activated_m2_cd140b` | CD45, CD11b, CD206, CD140b | CD34, CD31, CD140a, Ly6G, CD44 |
| 4 | `m2_macrophage` | CD45, CD11b, CD206 | CD34, CD31, CD44, CD140b, CD140a, Ly6G |
| 5 | `activated_myeloid_cd44` | CD45, CD11b, CD44 | CD34, CD31, CD206, CD140a, Ly6G, CD140b |
| 6 | `activated_myeloid_cd140b` | CD45, CD11b, CD140b | CD34, CD31, CD206, CD140a, Ly6G, CD44 |
| 7 | `myeloid` | CD45, CD11b | CD34, CD31, CD44, CD140b, CD206, CD140a, Ly6G |
| 8 | `activated_immune` | CD45, CD44, CD140b | CD34, CD31, CD11b, CD140a, Ly6G, CD206 |
| 9 | `immune_cells` | CD45 | CD34, CD31, CD11b, CD44, CD140b, CD140a, Ly6G, CD206 |
| 10 | `activated_endothelial_cd44` | CD31, CD34, CD44 | CD45, CD140a, Ly6G, CD11b, CD140b, CD206 |
| 11 | `activated_endothelial_cd140b` | CD31, CD34, CD140b | CD45, CD140a, Ly6G, CD11b, CD206, CD44 |
| 12 | `endothelial` | CD31, CD34 | CD45, CD44, CD140b, CD140a, Ly6G, CD11b, CD206 |
| 13 | `activated_fibroblast_cd44` | CD140a, CD44 | CD45, CD31, CD34, Ly6G, CD11b, CD140b, CD206 |
| 14 | `activated_fibroblast_cd140b` | CD140a, CD140b | CD45, CD31, CD34, Ly6G, CD11b, CD206, CD44 |
| 15 | `fibroblast` | CD140a | CD45, CD31, CD34, CD44, CD140b, Ly6G, CD11b, CD206 |

**On average only 13.6% of regions match any rule** (range 8.9–18.8% across images); ~86%
are `unassigned`. Two reasons: the panel has no tubular epithelial markers (no E-cadherin,
KIM-1 or aquaporin) and tubules are most of the kidney; and the long **negative** lists
mean a region co-expressing markers from two lineages fails every rule.

### (b) Continuous lineage scores

Because that unlabelled majority is most of the tissue, each region also carries three
0–1 scores — `lineage_immune`, `lineage_endothelial`, `lineage_stromal` — for how strongly
it resembles each lineage. These are **not exclusive**: a region can score highly on more
than one, which the rules cannot represent. They are scaled against the Sham (uninjured)
distribution, so ~0.5 is typical of uninjured tissue.

### (c) Subtype

A separate immune classification from the continuous scores, evaluated only where
`lineage_immune` ≥ 0.3; the best-scoring of four definitions wins (`neutrophil` = Ly6G;
`m2_macrophage` = CD11b + CD206; `myeloid` = CD11b without CD206/Ly6G;
`non_myeloid_immune` = neither CD11b nor Ly6G), otherwise `none`.

---

## 5. What "positive" means — it is relative, not absolute

**This is the most easily misread property of the dataset.**

A marker is called positive in a region if its intensity exceeds a **percentile of that
image's own distribution** — the 60th percentile for most markers, 70th for Ly6G, 50th for
CD206.

**Therefore the same fraction of regions is called positive in every image, by
construction, regardless of injury.** Verified across all 24 images: 40.0% of regions are
CD44-positive in every image, 40.0% CD45-positive, 30.0% Ly6G-positive, 50.0%
CD206-positive.

**Consequence:** discrete cell-type proportions describe how markers **co-occur and
redistribute** across the tissue — *not* how much of a marker is present.

- ❌ "CD44 expression increased with injury" — not supportable; total CD44 positivity is
  pinned at 40% in every image.
- ✅ "the fraction of regions co-expressing CD44 and endothelial markers increased" —
  supportable.

**The one exception:** the CD44 compartment files (§3.3) use a **fixed threshold per
marker derived from the Sham animals only** (75th percentile of the pooled Sham
distribution) applied unchanged to every image. Those rates *are* referenced to uninjured
tissue and are comparable across timepoints in absolute terms. The continuous scores (§4b)
are likewise Sham-referenced.

---

## 6. Statistics: what this dataset can and cannot support

**With 2 animals per timepoint, statistical significance is unattainable by construction.**
For a 2-versus-2 comparison the smallest possible two-sided Mann–Whitney p-value is
**0.33** — before any correction. Tests were run (smallest observed p = 0.33; smallest
after Benjamini–Hochberg correction for multiple comparisons = 0.58, so nothing is
significant), but **that is a property of the sample size, not a finding about the
biology.** Do not report it as "no significant differences were found", which implies a
test that could have succeeded.

We therefore report **effect sizes**: **Hedges' *g*** is the difference between two group
means divided by their pooled standard deviation, so it is in standard-deviation units and
unitless. Conventionally 0.2 is "small", 0.5 "medium", 0.8 "large" — but at n=2 these are
**descriptive summaries only**.

**Everything here is hypothesis-generating.** Safe phrasing: "consistent with", "suggests",
"in this pilot cohort", "warrants testing in a powered cohort". Avoid: "significantly
increased", "demonstrates", "confirms".

**Average within each animal before comparing timepoints.** Three images from one animal
are not three independent observations; treating them as such (pseudoreplication) overstates
precision roughly threefold. Group by `animal_id`, then compare the 2 animal means.

---

## 7. Recipes

**Cell-type proportion across the time course**
```python
import pandas as pd
d = pd.read_csv("data/per_image_quantification.csv")
per_animal = d.groupby(["timepoint", "animal_id"])["neutrophil_prop"].mean().reset_index()
# plot the 2 animal means per timepoint; show the individual animals, not just a bar
```

**Cortex versus medulla (paired within animal)**
```python
per_animal_region = d.groupby(["animal_id", "region"])["neutrophil_prop"].mean().reset_index()
```

**Continuous score across the time course**
```python
sp = pd.read_csv("data/per_region_annotations.csv.gz")
per_animal = sp.groupby(["timepoint", "animal_id"])["lineage_immune"].mean().reset_index()
```

**CD44 compartment rates — note the two required guards**
```python
c = pd.read_csv("data/cd44_compartment_rates_10um.csv")
c = c[c.region != "pooled"]                      # else every animal counts twice
c["animal_id"] = c.timepoint + "_" + c.mouse     # no animal_id column in this file
c["region"] = c.region.str.capitalize()          # match Cortex/Medulla elsewhere
cd45 = c[c.compartment == "CD45"]
```

**Figure conventions**
- Plot the individual animal values (n=2 per group) rather than a bar with error bars — a
  standard deviation from 2 points is not informative.
- State the denominator (`n_total` or `n_assigned`) in the legend.
- Say "tissue regions" or "superpixels", never "cells".
- Order timepoints Sham → D1 → D3 → D7.

---

## 8. Limitations to carry into the manuscript

1. **n = 2 animals per timepoint.** Significance is unattainable by design (§6).
2. **~86% of regions are unassigned** under the 15 rules — a panel limitation (no tubular
   epithelial markers) compounded by the rules' negative-marker requirements. Applies to
   `cell_type` only, not to the other two labellings.
3. **Regions are not cells** (§1).
4. **Positivity is relative within each image** (§5) — the single most important constraint
   on wording. Discrete results describe co-occurrence, not abundance.
5. **The CD206 threshold was chosen after seeing an outcome.** Most markers use the 60th
   percentile; CD206 uses the 50th, lowered from the 65th because at the 65th **no region
   in any image matched the M2-like macrophage rule**, inconsistent with published UUO time
   courses. It is a defensible calibration, but post-hoc — and **CD206 enters 14 of the 15
   rules** (required positive in 3, required negative in 11), so every cell-type call except
   `neutrophil` is affected. This must stay disclosed.
6. **Marker intensities are summed within a region, not averaged**, so region area enters
   multiplicatively. Mean region area is essentially constant across the time course
   (101.9, 101.9, 101.8, 100.3 pixels for Sham/D1/D3/D7), so timepoint comparisons are
   unaffected; it matters only for absolute magnitudes compared to other datasets.
7. **Marker specificity is imperfect.** Ly6G is transiently expressed by some monocytes;
   CD34 also marks haematopoietic progenitors and some fibroblasts; CD44 rises broadly with
   injury and is expected to be pan-compartment. Labels are marker-combination definitions,
   not validated identities.
8. **Region (cortex/medulla) is a whole-image label**, assigned per acquisition, not per
   superpixel.
9. **Small, non-randomly selected sample of tissue** — 3 fields of 500 × 500 µm per animal
   (~0.75 mm²).

---

## 9. Reproducibility

Every file here derives from the analysis repository at commit `8787245`. Regenerating the
inputs requires the raw acquisitions and the pipeline; the tables in this package are
self-contained and need nothing else to plot.

Questions, or a cut of the data in a different shape — just ask.
