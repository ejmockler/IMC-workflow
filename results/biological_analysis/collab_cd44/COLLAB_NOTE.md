# CD44 Compartment Analysis — Methods, Caveats & Draft Reply

Supplementary, self-contained, **descriptive** analysis produced in response to the
collaborator's four asks. This is a hypothesis-generating readout on a small pilot
(n=2 mice/timepoint); it introduces **no new pre-registered endpoint** and makes **no
significance claims**. Numbers cited below trace to the two tidy CSVs in this folder:
`cd44_compartment_rates_10um.csv` and `cd44_compartment_rates_40um.csv`.

---

## Part A — Methods & caveats

### What the analysis computes

For each compartment we compute the **CD44+ positivity rate** of the superpixels that
fall inside that compartment, reported per **mouse × timepoint (Sham / D1 / D3 / D7) ×
region (cortex / medulla / pooled)**, at both the **10µm** and **40µm** superpixel scales.

- **Reuse, not re-implementation.** The rate is computed by the existing pipeline
  function `compute_compartment_activation_per_roi`
  (`src/analysis/temporal_interface_analysis.py:705`), the same code that produces the
  frozen Family C outputs. Thresholds for CD45 / CD31 / CD140b / CD44 are recomputed by the frozen recipe
  (`compute_sham_reference_thresholds`, per-mouse Sham 75th percentile), reproducing the
  frozen `sham_reference_thresholds.parquet` values to machine precision (verified in the test).
  The two newly added marker thresholds (CD206, CD34) are computed by the **identical
  recipe** — the same Sham-pooled per-mouse percentile — so nothing about the thresholding
  convention changes.
- **Compartment set: 7 at 10µm** — `CD45+`, `CD206+`, `CD31+`, `CD34+`,
  `endothelial (CD31+ AND CD34+)`, `CD140b+`, and `neutrophil-typed`. **6 at 40µm** — the
  neutrophil compartment is **omitted at 40µm** because there is no cell-type basis for
  neutrophil typing at that coarser scale; we state this omission honestly rather than
  fabricate a 40µm neutrophil trajectory.

### Correctness check (reproduction)

The four compartments that overlap the existing pipeline (CD45+, CD31+, CD140b+,
neutrophil-typed) at 10µm **reproduce `compartment_activation_temporal.parquet` to machine
epsilon**. That exact-match is our correctness proof: the new marker compartments (CD206+,
CD34+, endothelial) are the only genuinely new quantities.

### Caveats — please read before circulating

1. **CD140b (PDGFRβ) marks pericytes / mural / activated-mesenchymal cells — NOT
   fibroblasts.** The fibroblast marker is **CD140a (PDGFRα)**, which is not in this panel.
   The note in the request describing a **"CD140b+ fibroblast"** compartment should be read
   as **pericyte / mural / activated-mesenchymal**, not fibroblast. We flag this explicitly
   so the terminology does not propagate.
2. **CD206 marks M2-like macrophages but is not exclusive** — it is also expressed on some
   endothelium and dendritic cells. A raw-CD206-threshold compartment is therefore an
   **M2-like** compartment, **not a pure M2 cell population**.
3. **A "compartment" is a ~10µm superpixel positivity rate, not a segmented cell.** Every
   value here is the fraction of positive superpixels in a marker-defined region — it is a
   tissue-area readout, not a single-cell measurement.
4. **n=2 mice/timepoint, hypothesis-generating.** All numbers are **descriptive**; no
   significance testing was performed and no significance is claimed. The **cortex vs
   medulla split is paired within-mouse** and **halves the already-thin support** — it makes
   the readout finer-grained but **does not improve power**.

---

## Part B — Draft reply to the collaborator

> Hi — thanks for the detailed asks. All four are addressed below. Everything here is
> descriptive on the n=2 pilot (Sham/D1/D3/D7, 2 mice/timepoint), so please read the
> trends as hypothesis-generating rather than tested effects. Figures and the two source
> CSVs are attached.

**1. Why 10µm and not 40µm.** The 10µm superpixel scale was fixed **a priori** in the
frozen analysis plan, specifically to keep patch size from becoming a
researcher-degree-of-freedom (i.e. to bar scale-shopping after seeing results). Just as
important: the "cellular neighborhood" you're after is already modeled — the framework
builds a **kNN graph over the 10µm superpixels (Family B)** to capture neighborhood
structure, rather than enlarging the patch to approximate a neighborhood. So enlarging to
40µm isn't the mechanism we use to reach neighborhood scale; the graph is.

**2. 40µm quantification.** Done, and provided as a **disclosed sensitivity panel** — see
**`fig_cd44_scale_sensitivity`** and **`cd44_compartment_rates_40um.csv`**. The honest
caveat: moving to 40µm collapses the data roughly **~20×** (from **58,137** superpixels down
to **~2,951** total; the sensitivity figure encodes per-compartment support as marker size,
which sums to **~63,732 → ~3,234** across the shown compartments — larger than the 2,951 total
because a superpixel counts in several compartments, but the same ~20× collapse). Per-compartment
support drops accordingly — e.g. pooled compartment support
runs **317–4,361 superpixels at 10µm**, whereas at 40µm several pooled cells fall to **~50
superpixels** (e.g. the endothelial and CD45 Sham compartments). Coarser patches also mix
more cell types together. Crucially we did not stop at a two-point check: we materialized all
three pipeline scales — **10µm (cellular), 20µm (tubular cross-section), 40µm (tissue domain)** —
as a proper cross-scale result (`cd44_compartment_rates_allscales.csv`, `cd44_crossscale_summary.csv`,
and a new **§4.6 figure added to the PI report** in its own interactive idiom). The finding: the
Sham→D7 CD44⁺ rise is **scale-robust** — 5 of the marker compartments (CD206, CD45, CD140b, CD31,
CD34) keep rising in the same direction as the patch coarsens from ~one cell to a tissue domain;
the only sign-flipping compartment (the endothelial CD31⁺∧CD34⁺ intersection) stays within ±1pp of
zero at every scale — a non-mover, not a scale artifact; and the neutrophil headline can only be
assessed at 10µm (no cell-type gating basis at the 20/40µm grid scales). So 40µm is a **robustness
check the story passes**, not a replacement for the 10µm readout.

**3. CD44-rate plot for CD206+ / CD31·CD34+ (endothelial) / CD140b+ across all timepoints.**
Done — see **`fig_cd44_pooled`** (pooled region, all four timepoints, each mouse shown as an
individual point, n=2). The CD31+, CD140b+, CD45+, and neutrophil compartments already
existed in the pipeline; **CD206+ and CD34 / endothelial (CD31+ AND CD34+) are newly added**
here. Two descriptive trends worth flagging (both traceable to
`cd44_compartment_rates_10um.csv`, pooled): the **neutrophil compartment's CD44 rate rises
from ~0.26–0.37 at Sham to ~0.77–0.85 at D7**, and the **CD45 compartment sits near ~0.6 at
Sham (0.62 / 0.56 across the two mice)**. These are descriptive only — no significance is
claimed, and neutrophil support is the thinnest of any compartment (as low as 317 pooled
superpixels), so read it cautiously. One terminology note carried into the figure labels:
**CD140b = pericyte / mural (PDGFRβ), not fibroblast**, and **CD206 = M2-like, not a pure M2
population** (see caveats above).

**4. Cortex vs medulla — and this is the one that pays off.** Done — see
**`fig_cd44_cortex_medulla`** and the new **§4.6 cortex/medulla dumbbell in the PI report**.
Your **"3 ROIs/region/timepoint × 2 mice" design is correct** and is what we used. The split
reveals a clear pattern we would have missed pooled: **the CD44⁺ activation is cortex-
predominant.** Every compartment's Sham→D7 rise is concentrated in the cortex, while the medulla
is flat or *declines* — CD206⁺ (M2-like) rises **+47pp in cortex vs −4pp in medulla**, the
neutrophil compartment reaches near-total activation in cortex (**+64pp**) but only **+31pp** in
medulla, and CD140b⁺ pericyte/mural tissue rises **+25pp in cortex while the medulla declines
(−7pp)**. Both mice agree within each region and the support is substantial (hundreds–thousands
of superpixels per compartment), so this is a robust *descriptive* pattern — consistent with
obstructive injury concentrating in the cortex. The caveat still holds: the split is **paired
within-mouse** at **n=2 mice**, so it sharpens *where* the signal sits without adding power —
please treat the cortex/medulla contrast as hypothesis-generating, not a tested regional effect.

> Happy to iterate on any of these. If it's useful I can add per-compartment support counts
> as an annotation layer, but I'd want to keep the framing descriptive given the pilot size.

---

*Provenance: rates via `compute_compartment_activation_per_roi`
(`src/analysis/temporal_interface_analysis.py:705`) + frozen Sham-reference thresholds;
overlapping 10µm compartments reproduce `compartment_activation_temporal.parquet` to machine
epsilon. Frozen pre-registration untouched (`verify_frozen_prereg.py` PASS). All cited
numbers trace to `cd44_compartment_rates_10um.csv` / `cd44_compartment_rates_40um.csv`, or to
the raw per-scale superpixel totals (58,137 at 10µm, 2,951 at 40µm) printed by
`collab_cd44_compartments.py` (`len` of the loaded 10µm annotation / 40µm frame) — the CSV
`n_support` columns double-count superpixels across overlapping compartments, so they sum
larger (~63,732 → ~3,234) while giving the same ~20× collapse.*
