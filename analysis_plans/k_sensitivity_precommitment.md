# Pre-commitment: kNN neighborhood size (k) sensitivity sweep

**Written BEFORE the sweep was executed.** Git history is the evidence: this file is committed
in its own commit, prior to the commit that adds any sweep output. (The composite-focalization
spec was criticised for landing spec + classifier + outputs in one commit, making the freeze
unverifiable from version control; this file exists to not repeat that.)

## Why

`k = 10` is the kNN neighborhood size underlying every self-enrichment number in this pilot
(Fig 4b, the composite focalization statuses, Family B, the lineage neighborhood lens). It is
hardcoded at 8+ independent sites (`spatial_neighborhood_analysis.py:167,454,509`,
`run_composite_focalization_scale_region.py:31`, `temporal_interface_analysis.py:57`,
`run_lineage_neighborhood_lens.py:110,205`), is read from `config.json` nowhere, and was never
swept. Its only written justification is circular ("matches existing neighborhood enrichment",
`temporal_interfaces_plan.md:73`).

This is a researcher-degrees-of-freedom exposure on a published headline, so it gets tested.

## What is being varied

`k ∈ {5, 10, 20}` — one below, the incumbent, one above (same shape as the pre-registered
{0.2,0.3,0.4} lineage-threshold and {10,20,40} min_support sweeps). Everything else is held
fixed: threshold 0.3 for composite categories, 10/20/40 µm scales, mouse-of-mouse aggregation,
frozen annotation parquets as input. No re-run of the primary pipeline.

## The endpoint at risk (stated in advance)

The published discriminator is: a category is **focal** iff its D7 self-enrichment ≥ **2.0×**,
else **diffuse**. Current values at k=10, 10 µm pooled:

| category | D7 self-enrichment | status | margin to 2.0 |
|---|---|---|---|
| endothelial+stromal | 3.26× | focal | +63% (robust) |
| immune+stromal | 2.58× | focal | +29% (robust) |
| **endothelial+immune** | **2.026×** | **focal** | **+1.3% (FRAGILE)** |
| endothelial+immune+stromal (triple) | 1.35× | diffuse | −32% (robust) |

The headline claim at risk: *"among the four multi-lineage interface categories, the
triple-positive is the LONE diffuse interface; the two-lineage interfaces focalize."*

## Report-either-way commitment

We report the outcome regardless of which way it goes. Specifically:

1. **If all four statuses are stable across k ∈ {5,10,20}** — we report that the focalization
   result is k-robust, and add the sweep as a disclosed sensitivity envelope. The headline stands
   as written.

2. **If endothelial+immune flips to diffuse at any k** — we report that the "lone diffuse
   interface" headline is **k-dependent** and rewrite it. The honest replacement is that the
   triple-positive is diffuse at every k tested (it is 32% below the cutoff), while E+I sits
   *on* the cutoff and its label is not robust. We do NOT quietly keep k=10 because it gives
   the tidier sentence.

3. **If the triple-positive (the actual scientific claim) flips to focal at any k** — that
   falsifies the mixing-signature reading and we say so plainly in RESULTS.md and the report.

4. **If results are unstable in an uninterpretable way** — we report the instability itself as
   the finding: the descriptor is too sensitive to an undisclosed parameter to carry a headline.

No outcome is a reason to suppress the sweep, and no k other than 10 will be adopted as the new
primary without a separate, stated justification — k=10 remains primary; this is a sensitivity
disclosure, not a re-selection.

## Output

`results/biological_analysis/k_sensitivity_focalization.csv` — self-enrichment and focal/diffuse
status for every composite category × timepoint × scale × region × k, plus a status-stability
flag per category. Generator: `run_k_sensitivity_sweep.py`.
