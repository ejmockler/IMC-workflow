# Kidney Notebook Narrative Throughline

**Date frozen:** 2026-04-18 (revised same day after Gate A brutalist review)
**Status:** Frozen for implementation (T49). Revision incorporates critical Gate A findings: CLR compositional tautology, Part 7 pre-registration conflict, Part 5 bridge invalidity, and Codex's tier-structure architecture.

## Central question (revised post-Gate-A)

**What candidate hypothesis can we synthesize from convergent post-hoc readings of these pre-existing analyses?**

Synthesized candidate: *stromal-marker-positive tissue area appears less stromal-only and more multi-lineage by Sham→D7.* The "appears" is load-bearing — we have no lineage tracing, no object-level transition analysis, and Family A's CLR mechanically couples some of the apparent co-movements. The notebook does not test this hypothesis (it cannot — at n=2 mice/timepoint, with analyses designed for a different question, no test is possible). It synthesizes a candidate finding for an n≥10 follow-up cohort.

This framing replaces the earlier "is the redistribution true?" framing, which Gate A correctly noted creates HARKing optics. The revised arc presents convergent post-hoc signals from pre-existing analyses, not hypothesis-driven testing.

## Architecture: evidence tiers

Per Codex's Gate A recommendation, the notebook's 7 Parts now sort into three evidence tiers:

| Tier | Role | Parts |
|---|---|---|
| **Tier 1** — direct candidate evidence | The two findings that anchor the synthesis | Part 2 (Family A CLR co-headlines), Part 6 (Family C compartment trajectories) |
| **Tier 2** — compatibility checks | Independent measures consistent with (or against) the candidate | Part 2.5 (spatial coherence join-counts; non-CLR), Part 4 (Leiden cluster concordance) |
| **Tier 3** — context, not corroboration | Background biological interpretation; legacy descriptive sections; methodological caveats | Part 1 (raw protein fields), Part 3 (marker correlations), Part 5 (neutrophil paradox as negative control), Part 7 (scale legacy section) |

Tier 3 sections do NOT bear confirmatory weight. Cell 0 must state this tier structure explicitly so the reader knows what each Part is supposed to do.

## CLR compositional tautology (must be in cell 0 and cell 12)

Stromal-only CLR decrease and triple-positive CLR increase are **mathematically coupled**: on the closed simplex, if any category goes up, others must go down. They are not two independent observations. Cell 0 introduces this constraint; cell 12 reiterates it before presenting the co-headline table.

Independent evidence comes from:
- **Family C** (`triple_overlap_fraction`, `background_compartment_cd44_rate`, `CD140b+_compartment_cd44_rate`): raw-marker compartment definitions, not CLR. These are independent of Family A's compositional simplex by construction (different markers AND different math).
- **Part 2.5** (join-count statistics on binary interface indicators): non-compositional spatial coherence measures.

The candidate hypothesis is supported only to the extent that Family C and Part 2.5 agree with Family A. Family A alone is insufficient.

## Part 5 reframe (per Gate A C2-B)

Part 5 (neutrophil paradox) is NOT a focal-diffuse bridge to Part 6. The proposed bridge was logically invalid: spatial focality (Part 5) and temporal compartment activation rates (Part 6) are different measurements connected only by analogy.

Revised role of Part 5: **negative control**. If the candidate redistribution finding were a global artifact of CLR or per-ROI normalization, *all* cell types would show apparent redistribution. Neutrophils don't — they remain stable-but-focal across timepoints. Their stability under the same analytical pipeline supports a biological rather than purely artifactual origin for the Family A + Family C signal.

This is a Tier 3 context section, not a confirmatory test.

## Part 7 reframe (per Gate A F5)

The pre-registration `temporal_interfaces_plan.md` §2 explicitly states 10μm was selected a priori and 20/40μm outputs are *not analyzed in this effort* to prevent scale-as-researcher-degree-of-freedom. The original throughline proposed using scale-invariance as robustness evidence — this would reopen the scale DOF problem mid-effort.

Revised role of Part 7: **legacy descriptive context only**. The cells exist; they show cluster counts at multiple scales as a structural observation about kidney tissue organization. They are NOT redistribution-hypothesis evidence. The synthesis cell (37) must explicitly bracket Part 7 as legacy material, NOT cite scale-invariance as supporting the candidate finding.

The "Stage 1/2/3 hierarchical fate decision" model in current cell 37 must be deleted entirely (not transitioned). It violates the forbidden-language list and contains mechanistic claims unsupported by the n=2 pilot.

## Sub-question structure (revised)

| Tier | Sub-question | Existing Part | Deliverable cell |
|---|---|---|---|
| 1 | What does Family A find at the compositional lineage level? | 2 (cells 6–12) | cell 12 (already done) |
| 2 | Does spatial coherence agree (non-CLR)? | 2.5 (cells 13–15) | cell 15 |
| 3 | What raw protein patterns frame Parts 2-6? (descriptive context) | 1 (cells 2–5) | cell 5 |
| 3 | What protein co-variation runs through the data? (descriptive, not driving) | 3 (cells 16–18) | cell 18 |
| 2 | Are unsupervised clusters consistent with the candidate? (concordance, not test) | 4 (cells 19–24) | cell 22, 23 |
| 3 | Negative control: do all cell types show apparent redistribution? | 5 (cells 25–31) | cell 31 |
| 1 | Family C compartment-level trajectories | 6 (cells 32–34) | cell 34 (already done) |
| 3 | Legacy: scale-dependent cluster count observation | 7 (cells 35–38) | cell 37 (mostly delete + brief legacy framing) |
| Synthesis | Land the candidate finding + alternative-hypothesis acknowledgement | end of cell 37 / new cell | candidate + n_required + alternative-hypothesis discrimination |

## Transitional rewrite spec — 9 cells

For each transition cell, specify (a) what to delete, (b) what to add, (c) tier label.

### Cell 0 (intro)

**Delete:** "What We'll Show" list with old framing if present. Old "decision point" sentence already removed (Gate 3); confirm.

**Add:**
1. Concise UUO model description (keep existing text)
2. **Boxed methodological disclaimer** (markdown blockquote, structurally distinct):
   > **Methodological note (post-hoc reframing).** This notebook was originally designed as exploratory analysis. The candidate hypothesis below — *stromal-marker-positive tissue area appears less stromal-only and more multi-lineage at Sham→D7* — emerged post-hoc during Gate 6 normalization sensitivity (April 2026). The analyses in Parts 1–7 were designed BEFORE this hypothesis existed. This notebook synthesizes convergent signals from pre-existing analyses; it does not test the hypothesis. Testing requires an n≥10 follow-up cohort.
3. State the candidate hypothesis explicitly with the "appears" qualifier
4. State the CLR tautology constraint (Family A's stromal-only decrease and triple-positive increase are mathematically coupled; independent evidence comes from Family C and Part 2.5)
5. State the tier structure (Tier 1 / Tier 2 / Tier 3) so the reader knows what each Part contributes
6. State the alternative hypothesis (the apparent shift could be a per-ROI normalization artifact; the pilot data cannot discriminate)

**Tier label:** Setup, not analysis.

### Cell 5 (Reading Protein Fields → Part 2)

**Delete:** Nothing substantive.

**Add (transition):** "These descriptive observations frame what follows. Stromal markers (CD140a, CD140b) are stable across timepoints in raw expression, yet *compositional classification* may still shift — Part 2 quantifies that classification at the lineage level. Part 1 is Tier 3 context: it does not bear confirmatory weight on the candidate finding."

**Tier label:** Tier 3 (context).

### Cell 12 → Cell 13 (Part 2 → Part 2.5)

**Delete:** Nothing in cell 12 (the existing co-headline table + full accounting + post-hoc disclosure are correct from Gate 7). Cell 13 intro paragraph.

**Add at the very top of cell 12:** A two-sentence reminder:
> **CLR compositional constraint.** On the closed simplex of 8 interface categories, the stromal-only decrease and triple-positive increase reported below are mathematically coupled — they are one degree of freedom dressed as two observations. Independent corroboration comes from Family C (Part 6, non-CLR) and the spatial coherence below (Part 2.5, non-compositional).

**Add (cell 12 → 13 bridge):** "The next section asks whether the spatial coherence of triple-positive interfaces strengthens — a non-CLR measure that, if it agrees, provides independent corroboration of the Family A signal."

**Cell 13 (Part 2.5 intro):** Already lists the corrected framing (neighbor-minus-self, trajectory filter). Keep most existing text. Add one sentence: "Part 2.5 is Tier 2 — compatibility check, not confirmation. The join-count statistics on binary interface indicators are non-compositional spatial measures; if they agree with the Family A direction, they provide independent corroboration. If not, Family A's CLR signal is not supported by spatial coherence."

**Tier label:** Cell 12 = Tier 1; cell 13 = Tier 2.

### Cell 15 → Cell 16 (Part 2.5 → Part 3)

**Delete:** Any narrative claim that protein correlations *drive* or *cause* multi-lineage interface formation. CD44 as "bridge molecule" can stay as a covariation statement but not a mechanism statement.

**Add (cell 15 closing):** "Part 3 examines marker covariation. The correlation structure is a Tier 3 context section — it does not test the candidate hypothesis. Correlations among markers in superpixels reflect co-expression, not causal coordination."

**Tier label:** Tier 3 (context).

### Cell 18 → Cell 19 (Part 3 → Part 4)

**Delete:** Any "tests this independently" framing. Any "drives multi-lineage interface formation" claim. Any "coordinated multi-lineage response" mechanism claim.

**Add (cell 18 closing):** "CD44 covaries with markers from multiple lineages (correlations only — not driving, not coordinating, not integrating). Part 4 asks whether unsupervised Leiden clusters are *consistent with* the multi-lineage interface concept — a concordance check on derived data, not independent validation."

**Cell 19 (Part 4 intro):** Reframe: "Part 4 is Tier 2 — concordance check on the same marker panel. Leiden clusters and Family A interface categories are derived from the same underlying expression data, so cluster correspondence to interface categories is not independent corroboration; it is a re-expression of the same data through a different algorithm."

**Tier label:** Cell 18 = Tier 3; Cell 19 = Tier 2 (with caveat about derivedness).

### Cells 22, 23 (Part 4 cluster validation cells)

**Delete:** Any claim that cluster validation *independently confirms* the redistribution finding. The Moran's I on cluster labels (cell 21 code) is methodologically rejected by the pre-registration `temporal_interfaces_plan.md` §6 (categorical Moran's I is incoherent); the markdown must quarantine this with: "Note: Moran's I on cluster labels is reported here as a legacy descriptive metric; the pre-registered analysis (`temporal_interfaces_plan.md` §6) uses join-count statistics on binary indicators, which is the methodologically correct equivalent. The cluster Moran's I is presented for historical continuity, not as evidence."

**Add:** Brief concordance summary: "Leiden clusters that mix immune, stromal, and vascular markers do correspond to multi-lineage interface superpixels from Family A — this is a concordance check (same data, different algorithm), not independent validation."

**Tier label:** Tier 2 (concordance only).

### Cell 23/27 → Cell 27/28 (Part 4 → Part 5)

**Delete:** "These are decision points" if it remains; replace with the Gate 7 phrasing.

**Add (Part 5 intro, cell 27):** "Part 5 is the **negative control** of the candidate redistribution finding. If the apparent redistribution were a global artifact of CLR or per-ROI normalization, *every* cell type would show similar artifactual movement. Neutrophils don't — Ly6G expression remains FLAT across timepoints in mean, and the focal pattern below shows neutrophils are stable-but-focal rather than redistributing. Their stability under the same analytical pipeline supports a biological rather than artifactual origin for the Family A + Family C signal."

**Tier label:** Tier 3 (negative control as context).

### Cell 31 → Cell 32 (Part 5 → Part 6)

**Delete:** Existing focal-vs-diffuse bridge framing if present.

**Add (cell 31 closing):** "Neutrophils show focal accumulation that ROI-level means dilute — but they are stable-but-focal, not redistributing. They serve as a negative control for the candidate finding: the Family A + Family C signal is not a generic artifact applied uniformly to all cell types."

**Cell 32 (Part 6 intro):** "Part 6 returns to Tier 1 evidence. Family C uses raw-marker compartment definitions (CD45, CD31, CD140b for compartments; CD44 for activation; thresholds from the Sham distribution). This is methodologically distinct from Family A's compositional CLR — different markers (CD140b vs CD140a), different math (raw threshold vs CLR transform). If both families show the same direction at Sham→D7, that's *related-but-non-identical corroboration across measurement schemes*."

**Tier label:** Cell 31 = Tier 3 (negative control); Cell 32 = Tier 1.

### Cell 34 → Cell 35 (Part 6 → Part 7)

**Delete:** Any narrative use of scale as evidence.

**Add (cell 34 closing):** "The remaining Part 7 is **legacy descriptive material** — scale-dependent cluster counts at 10/20/40μm. It is NOT part of the candidate-finding synthesis. The pre-registered analysis (`temporal_interfaces_plan.md` §2) explicitly selected 10μm a priori and excluded 20/40μm to prevent scale-as-researcher-degree-of-freedom. Reading scale-invariance as redistribution support would reopen exactly that DOF. Part 7 is presented for historical continuity only."

**Cell 35 (Part 7 intro):** Reframe entirely. Old framing ("Scale-Dependent Organization → The Repair vs Fibrosis Decision") is forbidden language. New framing: "Part 7 — Legacy: Cluster counts at 10/20/40μm. This section pre-dates the temporal interface analysis and is retained as descriptive context. It does not bear on the candidate redistribution finding."

**Tier label:** Tier 3 (legacy context).

### Cell 37 (final synthesis)

**Delete:** Entire "Biological Model: Hierarchical Tissue Fate Decision" section with Stage 1/2/3 mechanistic narrative. All "fate decision," "Repair vs Fibrosis," "Optimal target: 20μm scale" content. This is ~50 lines of mechanistic speculation that violates plan §10 forbidden language.

**Add (replacement synthesis):**
1. Restate the candidate finding with the "appears" qualifier:
   > "The pilot synthesizes a candidate finding for the n≥10 follow-up: stromal-marker-positive tissue area *appears* less stromal-only (Family A) and more multi-lineage (Family A triple-positive CLR + Family C triple-overlap fraction + Family C background CD44+ rate) by Sham→D7. We use 'appears' deliberately — Family A's CLR couples its two co-movements mathematically, and Family C uses related-but-non-identical markers (CD140b vs Family A's CD140a). The convergence is consistent with redistribution but does not demonstrate it."
2. State the alternative (must do):
   > "**Alternative hypothesis to discriminate.** The apparent signal could be a per-ROI sigmoid normalization artifact (the endothelial CLR collapse from g=+2.28 to g=+0.19 between regimes shows the per-ROI normalization can manufacture compositional shifts). The pilot data cannot discriminate. A follow-up cohort that uses Sham-reference threshold normalization upstream (in the cell-type annotation engine) — rather than per-ROI sigmoid — would resolve this. n≥10 mice/timepoint; per-animal whole-tissue normalization."
3. State the headline n_required ranges with explicit prior labels (no "default" claim):
   > "Stromal-only CLR Sham→D7: n_required 11/34/244 (skeptical/neutral/optimistic priors). Triple-overlap fraction Sham→D7: n_required 4/17/158. Pick the prior matching your scepticism; the range IS the uncertainty statement."
4. Cross-reference main_narrative for the broader framework framing.

**Tier label:** Synthesis.

### Cell 38 (final code cell)

**Delete:** "KEY FINDING" or similar overclaim print statements (already scrubbed in Gate 3).

**Add:** Nothing — this is a code cell. Verify post-Gate-3 state holds.

## Forbidden language audit (per `temporal_interfaces_plan.md` §10)

These phrases must NOT appear in transitional cells: "decision point", "decision zone", "coordination" (when describing multi-lineage coordination as a demonstrated mechanism), "drives" (for any cross-marker relationship), "integrates" (for any cross-marker relationship), "confirms", "establishes", "demonstrates" (for any biological claim), "fate decision", "tissue fate", "Repair vs Fibrosis Decision", "KEY FINDING", "Optimal target".

These phrases ARE acceptable: "consistent with the candidate hypothesis", "candidate evidence", "convergent across (related-but-non-identical) measurement schemes", "concordance check on the same data", "negative control", "compatibility check", "post-hoc reframing", "Tier 1/2/3", "appears" (as in "stromal area appears less stromal-only").

The throughline document itself uses "redistribution" as shorthand. Implementation must use the full hedged form ("consistent with a redistribution hypothesis", "the candidate finding", or "stromal-marker-positive tissue area appears less stromal-only and more multi-lineage").

## What this work will NOT do

- Re-run the pipeline or change any numerical output
- Add new analyses (Bodenmiller validation, per-ROI normalization fix in annotation engine, Family A/C marker harmonization, etc.)
- Modify code cells (only markdown transitions; the cluster Moran's I code stays but is quarantined in markdown)
- Promote findings beyond what the data supports — the candidate hypothesis is a candidate hypothesis, not a demonstrated mechanism, throughout

## Pre-Gate-A-pass checklist

- [x] Throughline frozen and revised after Gate A round 1
- [x] CLR tautology caveat specified for cells 0 and 12
- [x] Part 5 reframed as negative control (not focal-diffuse bridge)
- [x] Part 7 reframed as legacy descriptive context (not scale-invariance evidence)
- [x] Cell 37 spec includes explicit deletion (Stage 1/2/3 model) + replacement prose
- [x] Tier structure (1/2/3) added to architecture
- [x] Alternative hypothesis (per-ROI normalization artifact) explicitly required in cell 0 and synthesis
- [x] Quarantine spec for Moran's I on cluster labels
- [x] Power estimates in synthesis must mention alternative + study design that discriminates
- [ ] T49 implementation pending
