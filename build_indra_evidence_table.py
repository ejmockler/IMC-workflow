"""
INDRA/CoGEx Knowledge-Grounded Evidence Framework

Provides mechanistic biological context for IMC spatial proteomics findings
by integrating multi-layer knowledge from the INDRA/CoGEx knowledge graph.

Knowledge layers (all pre-queried via CoGEx MCP tools, 2026-02-20):
  1. Per-gene: pathways, GO terms, diseases, cell types, tissue expression, phenotypes
  2. Gene pairs: INDRA causal statements (Complex, Activation, Inhibition, Phosphorylation)
  3. Shared biology: GO processes shared between co-localizing marker pairs
  4. Kidney-specific: tissue expression in kidney structures, kidney-relevant pathways
  5. Mechanistic narratives: per-finding biological interpretations

Input: Analysis outputs from differential_abundance and spatial_neighborhoods
Output:
  - results/biological_analysis/indra_evidence_table.csv  (enriched findings)
  - results/biological_analysis/indra_knowledge_base.json  (full knowledge graph)

NOTE: All INDRA data was pre-queried via the CoGEx MCP server. This script
does not make live API calls — it structures pre-queried knowledge.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from itertools import combinations


# ============================================================================
# MARKER → GENE GROUNDING (via INDRA/CoGEx ground_entity)
# ============================================================================

MARKER_GENE_MAP = {
    'CD45':  {'curie': 'hgnc:9620',  'gene_name': 'PTPRC',  'grounded': True},
    'CD11b': {'curie': 'hgnc:6148',  'gene_name': 'ITGAM',  'grounded': True},
    'Ly6G':  {'curie': None,         'gene_name': None,      'grounded': False,
              'note': 'Murine neutrophil marker (Ly6g); no human ortholog in INDRA'},
    'CD140a':{'curie': 'hgnc:8803',  'gene_name': 'PDGFRA', 'grounded': True},
    'CD140b':{'curie': 'hgnc:8804',  'gene_name': 'PDGFRB', 'grounded': True},
    'CD31':  {'curie': 'hgnc:8823',  'gene_name': 'PECAM1', 'grounded': True},
    'CD34':  {'curie': 'hgnc:1662',  'gene_name': 'CD34',   'grounded': True},
    'CD206': {'curie': 'hgnc:7228',  'gene_name': 'MRC1',   'grounded': True},
    'CD44':  {'curie': 'hgnc:1681',  'gene_name': 'CD44',   'grounded': True},
}

# Reverse lookup: gene name → marker name
GENE_TO_MARKER = {v['gene_name']: k for k, v in MARKER_GENE_MAP.items() if v['grounded']}


# ============================================================================
# LAYER 1: PER-GENE KNOWLEDGE PROFILES
# Pre-queried via batch_call("get_pathways_for_gene"), ("get_go_terms_for_gene"),
# ("get_diseases_for_gene"), ("get_cell_types_for_marker"),
# ("get_tissues_for_gene"), ("get_phenotypes_for_gene")
# ============================================================================

GENE_KNOWLEDGE = {
    'CD44': {
        'kidney_pathways': [
            'Hyaluronan metabolism (R-HSA-2160916)',
            'ECM proteoglycans (R-HSA-3000178)',
            'Interferon gamma signaling (R-HSA-877300)',
            'Neutrophil degranulation (R-HSA-6798695)',
            'Cell surface interactions at the vascular wall (R-HSA-202733)',
        ],
        'kidney_go_terms': [
            'hyaluronic acid binding (GO:0005540)',
            'inflammatory response (GO:0006954)',
            'wound healing, spreading of cells (GO:0044319)',
            'cell migration (GO:0016477)',
            'T cell activation (GO:0042110)',
            'positive regulation of ERK1 and ERK2 cascade (GO:0070374)',
            'negative regulation of apoptotic process (GO:0043066)',
            'monocyte aggregation (GO:0070487)',
            'focal adhesion (GO:0005925)',
        ],
        'kidney_tissues': [
            'kidney epithelium (UBERON:0004819)',
            'metanephros cortex (UBERON:0010533)',
        ],
        'diseases': {
            'Acute Kidney Injury': {'curie': 'MESH:D058186', 'direct': True},
            'IgA glomerulonephritis': {'curie': 'MESH:D005922', 'direct': True},
        },
        'cell_types': [
            'tissue-resident macrophage (CL:0000864)',
        ],
        'kidney_relevance': (
            'CD44 is the hyaluronan receptor — directly implicated in AKI via '
            'HA-CD44 signaling. Upregulated in tubular injury, mediates leukocyte '
            'recruitment to damaged kidney. Only panel gene with direct AKI '
            'disease association in INDRA.'
        ),
    },
    'PDGFRA': {
        'kidney_pathways': [
            'Genes controlling nephrogenesis (WP4823)',
            'PDGF receptor signaling (GO:0048008)',
            'VEGF-activated PDGF receptor signaling (GO:0038091)',
            'Positive regulation of fibroblast proliferation (GO:0048146)',
            'Mesenchymal stem cell differentiation (GO:2000739)',
        ],
        'kidney_go_terms': [
            'cellular response to reactive oxygen species (GO:0034614)',
            'positive regulation of cell migration (GO:0030335)',
            'positive regulation of ERK1 and ERK2 cascade (GO:0070374)',
            'platelet-derived growth factor receptor-alpha signaling (GO:0035790)',
            'regulation of mesenchymal stem cell differentiation (GO:2000739)',
        ],
        'kidney_tissues': [
            'cortex of kidney (UBERON:0001225)',
            'nephron tubule (UBERON:0001231)',
        ],
        'diseases': {
            'Liver Cirrhosis': {'curie': 'MESH:D008103', 'direct': True},
            'Eosinophilia': {'curie': 'D004802', 'direct': True},
        },
        'cell_types': [
            'pericyte (CL:0000669)',
            'fibroblast (CL:0000057)',
            'mesenchymal stem cell (CL:0000134)',
        ],
        'kidney_relevance': (
            'PDGFRA+ interstitial fibroblasts are the primary effector cells in '
            'kidney fibrosis. In nephrogenesis pathway (WP4823). Expressed in '
            'kidney cortex and nephron tubule. MSC differentiation regulator — '
            'key to understanding repair vs. fibrosis fate.'
        ),
    },
    'PDGFRB': {
        'kidney_pathways': [
            'Genes controlling nephrogenesis (WP4823)',
            'Markers of kidney cell lineage (WP5236)',
            'Burn wound healing (WP5055)',
            'PDGF receptor signaling (GO:0048008)',
            'Positive regulation of smooth muscle cell migration (GO:0014911)',
        ],
        'kidney_go_terms': [
            'angiogenesis (GO:0001525)',
            'positive regulation of smooth muscle cell migration (GO:0014911)',
            'positive regulation of smooth muscle cell proliferation (GO:0048661)',
            'positive regulation of PI3K/AKT signaling (GO:0051897)',
            'cell chemotaxis (GO:0060326)',
            'positive regulation of ERK1 and ERK2 cascade (GO:0070374)',
        ],
        'kidney_tissues': [
            'kidney epithelium (UBERON:0004819)',
            'metanephros cortex (UBERON:0010533)',
            'metanephric glomerulus (UBERON:0004736)',
            'blood vessel layer (UBERON:0004797)',
        ],
        'diseases': {
            'Myofibromatosis': {'curie': 'MESH:D047708', 'direct': True},
        },
        'cell_types': [
            'pericyte (CL:0000669)',
            'vascular associated smooth muscle cell (CL:0000359)',
            'fibroblast (CL:0000057)',
        ],
        'kidney_relevance': (
            'PDGFRB marks pericytes/vascular mural cells — critical for '
            'glomerular and peritubular capillary integrity. In both nephrogenesis '
            '(WP4823) and kidney cell lineage markers (WP5236). Expressed in '
            'metanephric glomerulus. Pericyte detachment is a hallmark of AKI-induced '
            'microvascular injury.'
        ),
    },
    'ITGAM': {
        'kidney_go_terms': [
            'integrin-mediated signaling pathway (GO:0007229)',
            'positive regulation of neutrophil degranulation (GO:0043315)',
            'positive regulation of superoxide anion generation (GO:0032930)',
            'cell adhesion mediated by integrin (GO:0033627)',
        ],
        'kidney_pathways': [
            'Neutrophil degranulation (R-HSA-6798695)',
            'TLR4 Cascade (R-HSA-166016)',
            'IL-4 and IL-13 signaling (R-HSA-6785807)',
            'Cell surface interactions at the vascular wall (R-HSA-202733)',
        ],
        'kidney_tissues': [
            'metanephros cortex (UBERON:0010533)',
            'metanephric glomerulus (UBERON:0004736)',
            'blood vessel layer (UBERON:0004797)',
        ],
        'diseases': {
            'Systemic Lupus Erythematosus': {'curie': 'MESH:D008180', 'direct': True},
        },
        'cell_types': [
            'monocyte (CL:0000576)',
            'macrophage (CL:0000235)',
            'microglial cell (CL:0000129)',
            'granulocyte (CL:0000094)',
        ],
        'kidney_relevance': (
            'ITGAM (CD11b) forms the Mac-1 integrin (αMβ2) — primary receptor '
            'for complement iC3b on myeloid cells. Mediates neutrophil/macrophage '
            'adhesion to injured endothelium. TLR4 cascade involvement connects '
            'to DAMPs released during tubular necrosis. SLE disease association '
            'highlights its role in autoimmune nephritis.'
        ),
        'phenotypes': [
            'Lupus nephritis (HP:0033726)',
            'Abnormal renal cortex morphology (HP:0011035)',
            'Proteinuria (D011507)',
        ],
    },
    'MRC1': {
        'kidney_go_terms': [
            'receptor-mediated endocytosis (GO:0006898)',
            'cargo receptor activity (GO:0038024)',
        ],
        'kidney_pathways': [],
        'kidney_tissues': [
            'kidney epithelium (UBERON:0004819)',
            'metanephros cortex (UBERON:0010533)',
            'metanephric glomerulus (UBERON:0004736)',
            'blood vessel layer (UBERON:0004797)',
        ],
        'diseases': {},
        'cell_types': [
            'alternatively activated macrophage (CL:0000890)',
            'tissue-resident macrophage (CL:0000864)',
        ],
        'kidney_relevance': (
            'MRC1 (CD206) is the canonical M2/alternatively activated macrophage '
            'marker — INDRA cell ontology confirms CL:0000890. In kidney, M2 '
            'macrophages mediate tissue repair and anti-inflammatory resolution '
            'after AKI. Expressed in metanephric glomerulus and kidney cortex.'
        ),
    },
    'PTPRC': {
        'kidney_go_terms': [
            'T cell receptor signaling pathway (GO:0050852)',
            'positive regulation of B cell proliferation (GO:0030890)',
            'negative regulation of receptor signaling pathway via JAK-STAT (GO:0046426)',
            'regulation of phagocytosis (GO:0050764)',
            'hematopoietic progenitor cell differentiation (GO:0002244)',
        ],
        'kidney_pathways': [],
        'kidney_tissues': [
            'kidney epithelium (UBERON:0004819)',
            'metanephros cortex (UBERON:0010533)',
            'blood vessel layer (UBERON:0004797)',
        ],
        'diseases': {},
        'cell_types': [
            'leukocyte (CL:0000738)',
            'monocyte (CL:0000576)',
            'dendritic cell (CL:0000451)',
            'T cell (CL:0000084)',
            'B cell (CL:0000236)',
        ],
        'kidney_relevance': (
            'PTPRC (CD45) is the pan-leukocyte marker. In AKI context, its '
            'expression quantifies total immune infiltration. JAK-STAT regulation '
            'connects to cytokine signaling cascades in injury response.'
        ),
    },
    'PECAM1': {
        'kidney_go_terms': [
            'glomerular endothelium development (GO:0072011)',
            'establishment of endothelial barrier (GO:0061028)',
            'monocyte extravasation (GO:0035696)',
            'neutrophil extravasation (GO:0072672)',
            'leukocyte cell-cell adhesion (GO:0007159)',
            'diapedesis (GO:0050904)',
            'detection of mechanical stimulus (GO:0050982)',
            'positive regulation of MAPK cascade (GO:0043410)',
        ],
        'kidney_pathways': [
            'Markers of kidney cell lineage (WP5236)',
            'Neutrophil degranulation (R-HSA-6798695)',
            'Cell surface interactions at the vascular wall (R-HSA-202733)',
        ],
        'kidney_tissues': [
            'kidney epithelium (UBERON:0004819)',
            'metanephros cortex (UBERON:0010533)',
            'blood vessel layer (UBERON:0004797)',
        ],
        'diseases': {
            'Coronary Artery Disease': {'curie': 'MESH:D003324', 'direct': True},
        },
        'cell_types': [
            'endothelial cell (CL:0000115)',
            'endothelial cell of vascular tree (CL:0002139)',
        ],
        'kidney_relevance': (
            'PECAM1 (CD31) is THE endothelial marker. GO:0072011 (glomerular '
            'endothelium development) directly ties it to kidney vasculature. '
            'In kidney cell lineage markers (WP5236). Mediates leukocyte '
            'transendothelial migration — the gateway for immune cell infiltration '
            'into injured kidney tissue. Also a mechanosensor for shear stress.'
        ),
    },
    'CD34': {
        'kidney_go_terms': [
            'glomerular filtration (GO:0003094)',
            'glomerular endothelium development (GO:0072011)',
            'vascular wound healing (GO:0061042)',
            'endothelial cell proliferation (GO:0001935)',
            'metanephric glomerular mesangial cell differentiation (GO:0072254)',
            'positive regulation of angiogenesis (GO:0045766)',
            'hematopoietic stem cell proliferation (GO:0071425)',
            'tissue homeostasis (GO:0001894)',
        ],
        'kidney_pathways': [],
        'kidney_tissues': [
            'kidney epithelium (UBERON:0004819)',
            'metanephros cortex (UBERON:0010533)',
            'metanephric glomerulus (UBERON:0004736)',
        ],
        'diseases': {},
        'cell_types': [
            'endothelial progenitor cell (CL:0002619)',
            'endothelial cell of vascular tree (CL:0002139)',
            'hematopoietic stem cell (CL:0000037)',
        ],
        'kidney_relevance': (
            'CD34 has the most kidney-specific GO terms in the panel: glomerular '
            'filtration (GO:0003094), glomerular endothelium development (GO:0072011), '
            'mesangial cell differentiation (GO:0072254), and vascular wound healing '
            '(GO:0061042). Marks both endothelial progenitors and peritubular '
            'capillary endothelium. Critical readout for vascular repair after AKI.'
        ),
    },
}


# ============================================================================
# LAYER 2: INTRA-PANEL INDRA CAUSAL STATEMENTS
# Pre-queried via Cypher: MATCH (g1)-[r:indra_rel]->(g2) WHERE g1.id IN [panel]
# 91 total statements between 8 grounded panel genes
# ============================================================================

INDRA_STATEMENTS = [
    # PDGFRA ↔ PDGFRB: strongest intra-panel relationship
    {'source': 'PDGFRB', 'target': 'PDGFRA', 'type': 'Complex', 'evidence': 151, 'belief': 0.888,
     'note': 'Heterodimer formation — the core PDGF receptor signaling unit'},
    {'source': 'PDGFRB', 'target': 'PDGFRA', 'type': 'Inhibition', 'evidence': 5, 'belief': 0.505,
     'note': 'Cross-inhibition between receptor isoforms'},
    {'source': 'PDGFRA', 'target': 'PDGFRB', 'type': 'Inhibition', 'evidence': 4, 'belief': 0.602,
     'note': 'Bidirectional cross-regulation'},
    {'source': 'PDGFRB', 'target': 'PDGFRA', 'type': 'Activation', 'evidence': 4, 'belief': 0.400,
     'note': 'Trans-activation of alpha by beta'},
    {'source': 'PDGFRA', 'target': 'PDGFRB', 'type': 'Activation', 'evidence': 3, 'belief': 0.366},
    {'source': 'PDGFRB', 'target': 'PDGFRA', 'type': 'Phosphorylation', 'evidence': 2, 'belief': 0.750,
     'note': 'High-confidence kinase activity: beta phosphorylates alpha'},
    {'source': 'PDGFRA', 'target': 'PDGFRB', 'type': 'IncreaseAmount', 'evidence': 3, 'belief': 0.629,
     'note': 'Alpha upregulates beta expression'},
    {'source': 'PDGFRA', 'target': 'PDGFRB', 'type': 'Phosphorylation', 'evidence': 1, 'belief': 0.532},
    # PDGFRB ↔ CD44: physical interaction
    {'source': 'PDGFRB', 'target': 'CD44', 'type': 'Complex', 'evidence': 10, 'belief': 0.742,
     'note': 'CD44 physically associates with PDGFRB — explains pericyte-CD44 co-localization'},
    {'source': 'CD44', 'target': 'PDGFRB', 'type': 'Activation', 'evidence': 1, 'belief': 0.535},
    {'source': 'CD44', 'target': 'PDGFRB', 'type': 'Inhibition', 'evidence': 1, 'belief': 0.436},
    # PDGFRA ↔ CD44
    {'source': 'PDGFRA', 'target': 'CD44', 'type': 'Complex', 'evidence': 1, 'belief': 0.830,
     'note': 'High-belief complex despite single evidence'},
    {'source': 'CD44', 'target': 'PDGFRA', 'type': 'Complex', 'evidence': 1, 'belief': 0.830},
    # PECAM1 ↔ CD34: endothelial signaling axis
    {'source': 'PECAM1', 'target': 'CD34', 'type': 'Activation', 'evidence': 5, 'belief': 0.375,
     'note': 'CD31 activates CD34 — endothelial marker cross-talk'},
    {'source': 'PECAM1', 'target': 'CD34', 'type': 'Inhibition', 'evidence': 3, 'belief': 0.476,
     'note': 'Complex bidirectional regulation'},
    {'source': 'CD34', 'target': 'PECAM1', 'type': 'Activation', 'evidence': 2, 'belief': 0.367},
    {'source': 'CD34', 'target': 'PECAM1', 'type': 'IncreaseAmount', 'evidence': 1, 'belief': 0.484},
    {'source': 'CD34', 'target': 'PECAM1', 'type': 'DecreaseAmount', 'evidence': 1, 'belief': 0.474},
    {'source': 'PECAM1', 'target': 'CD34', 'type': 'Complex', 'evidence': 2, 'belief': 0.457},
    # CD44 ↔ CD34
    {'source': 'CD44', 'target': 'CD34', 'type': 'Activation', 'evidence': 4, 'belief': 0.376,
     'note': 'CD44 activates CD34 — injury marker drives progenitor?'},
    {'source': 'CD44', 'target': 'CD34', 'type': 'Inhibition', 'evidence': 2, 'belief': 0.441},
    {'source': 'CD34', 'target': 'CD44', 'type': 'IncreaseAmount', 'evidence': 2, 'belief': 0.472,
     'note': 'CD34+ cells upregulate CD44'},
    # CD44 ↔ MRC1: macrophage biology
    {'source': 'CD44', 'target': 'MRC1', 'type': 'Complex', 'evidence': 3, 'belief': 0.648,
     'note': 'Physical interaction on macrophage surface — validates CD44+/CD206+ phenotype'},
    # PDGFRB ↔ MRC1
    {'source': 'PDGFRB', 'target': 'MRC1', 'type': 'Complex', 'evidence': 3, 'belief': 0.423,
     'note': 'Pericyte-macrophage interaction complex'},
    # PECAM1 ↔ MRC1
    {'source': 'PECAM1', 'target': 'MRC1', 'type': 'Complex', 'evidence': 2, 'belief': 0.457,
     'note': 'Endothelial-macrophage interaction'},
    # PDGFRB ↔ CD34: pericyte-endothelial
    {'source': 'PDGFRB', 'target': 'CD34', 'type': 'Activation', 'evidence': 2, 'belief': 0.367,
     'note': 'Pericyte signaling to endothelial progenitors'},
    # PDGFRA ↔ CD34
    {'source': 'CD34', 'target': 'PDGFRA', 'type': 'Complex', 'evidence': 4, 'belief': 0.409,
     'note': 'Progenitor-fibroblast physical interaction'},
    {'source': 'PDGFRA', 'target': 'CD34', 'type': 'Activation', 'evidence': 3, 'belief': 0.379},
    # CD44 ↔ PECAM1
    {'source': 'CD44', 'target': 'PECAM1', 'type': 'Activation', 'evidence': 1, 'belief': 0.417,
     'note': 'CD44 activates endothelial PECAM1 — adhesion/extravasation'},
    {'source': 'CD44', 'target': 'PECAM1', 'type': 'IncreaseAmount', 'evidence': 1, 'belief': 0.484},
    {'source': 'PECAM1', 'target': 'CD44', 'type': 'Inhibition', 'evidence': 1, 'belief': 0.436},
    # MRC1 ↔ CD34
    {'source': 'CD34', 'target': 'MRC1', 'type': 'Complex', 'evidence': 1, 'belief': 0.426},
]


# ============================================================================
# LAYER 3: SHARED BIOLOGICAL PROCESSES BETWEEN GENE PAIRS
# Pre-queried via Cypher: shared GO terms between co-localizing pairs
# Filtered to kidney-relevant processes (excluding generic "protein binding")
# ============================================================================

SHARED_BIOLOGY = {
    ('CD34', 'PECAM1'): {
        'shared_processes': [
            'glomerular endothelium development (GO:0072011)',
            'cell-cell adhesion (GO:0098609)',
        ],
        'interpretation': (
            'Both markers share the KIDNEY-SPECIFIC process "glomerular endothelium '
            'development" — their co-localization in kidney tissue is expected from '
            'ontology. CD34+/CD31+ double-positive cells ARE the glomerular and '
            'peritubular capillary endothelium.'
        ),
    },
    ('CD44', 'PDGFRA'): {
        'shared_processes': [
            'positive regulation of ERK1 and ERK2 cascade (GO:0070374)',
            'cell migration (GO:0016477)',
        ],
        'interpretation': (
            'CD44 and PDGFRA converge on ERK1/2 signaling — a central pathway '
            'in both injury response and fibroblast activation. Their co-localization '
            'may reflect coordinated ERK-driven cell migration during repair.'
        ),
    },
    ('CD44', 'PDGFRB'): {
        'shared_processes': [
            'positive regulation of ERK1 and ERK2 cascade (GO:0070374)',
        ],
        'interpretation': (
            'CD44-PDGFRB share ERK cascade activation AND form a physical complex '
            '(10 evidence, belief=0.74). Co-localization is mechanistically supported '
            'by direct protein interaction.'
        ),
    },
    ('PDGFRA', 'PDGFRB'): {
        'shared_processes': [
            'platelet-derived growth factor receptor signaling pathway (GO:0048008)',
            'positive regulation of ERK1 and ERK2 cascade (GO:0070374)',
            'positive regulation of calcium-mediated signaling (GO:0050850)',
            'VEGF binding (GO:0038085)',
            'VEGF-activated PDGFR cell proliferation (GO:0038091)',
            'positive regulation of cell migration (GO:0030335)',
            'positive regulation of cell population proliferation (GO:0008284)',
            'peptidyl-tyrosine phosphorylation (GO:0018108)',
            'receptor protein tyrosine kinase signaling (GO:0007169)',
        ],
        'interpretation': (
            'PDGFRA/PDGFRB are functional partners sharing 14+ GO processes. '
            'They form the strongest INDRA relationship in the panel (151 evidence). '
            'Their co-localization is a trivial biological expectation — these are '
            'the alpha and beta subunits of the same receptor system.'
        ),
    },
    ('ITGAM', 'PECAM1'): {
        'shared_processes': [
            'leukocyte cell-cell adhesion (GO:0007159)',
            'cell-cell adhesion (GO:0098609)',
        ],
        'interpretation': (
            'Mac-1 (CD11b) on leukocytes binds PECAM1 on endothelium during '
            'transendothelial migration. Their spatial proximity reflects the '
            'leukocyte adhesion cascade at sites of vascular inflammation.'
        ),
    },
    ('CD44', 'PECAM1'): {
        'shared_processes': [
            'transmembrane signaling receptor activity (GO:0004888)',
        ],
        'interpretation': (
            'Both are transmembrane receptors involved in cell adhesion. '
            'CD44 mediates rolling/arrest of leukocytes, PECAM1 mediates '
            'diapedesis — sequential steps in immune cell extravasation.'
        ),
    },
    ('PDGFRA', 'PECAM1'): {
        'shared_processes': [
            'positive regulation of cell migration (GO:0030335)',
            'protein homodimerization activity (GO:0042803)',
        ],
        'interpretation': (
            'Fibroblast (PDGFRA) and endothelial (PECAM1) interaction during '
            'tissue remodeling. Shared cell migration regulation supports '
            'their co-localization at repair/fibrosis sites.'
        ),
    },
}


# ============================================================================
# LAYER 4: KIDNEY-SPECIFIC CONTEXT DICTIONARY
# ============================================================================

KIDNEY_CONTEXT = {
    'nephrogenesis_genes': ['PDGFRA', 'PDGFRB'],  # WP4823
    'kidney_lineage_markers': ['PDGFRB', 'PECAM1'],  # WP5236
    'glomerular_endothelium': ['CD34', 'PECAM1'],  # GO:0072011
    'glomerular_filtration': ['CD34'],  # GO:0003094
    'vascular_wound_healing': ['CD34'],  # GO:0061042
    'mesangial_differentiation': ['CD34'],  # GO:0072254
    'neutrophil_degranulation': ['ITGAM', 'CD44', 'PECAM1'],  # R-HSA-6798695
    'leukocyte_extravasation': ['ITGAM', 'PECAM1', 'CD44'],  # diapedesis pathway
    'm2_macrophage_markers': ['MRC1'],  # CL:0000890
    'aki_direct': ['CD44'],  # MESH:D058186

    # Kidney tissue expression (genes expressed in kidney structures)
    'kidney_epithelium': ['CD44', 'PDGFRB', 'MRC1', 'PTPRC', 'PECAM1', 'CD34'],
    'metanephros_cortex': ['CD44', 'PDGFRA', 'PDGFRB', 'ITGAM', 'MRC1', 'PTPRC', 'PECAM1', 'CD34'],
    'metanephric_glomerulus': ['PDGFRB', 'ITGAM', 'MRC1', 'CD34'],
}


# ============================================================================
# MECHANISTIC NARRATIVE TEMPLATES
# For each cell type pair, a hypothesis grounded in INDRA evidence
# ============================================================================

MECHANISTIC_NARRATIVES = {
    # Self-clustering narratives
    ('endothelial', 'endothelial'): (
        'Endothelial self-clustering reflects vascular network topology. '
        'CD31+/CD34+ endothelial cells in the peritubular capillary plexus '
        'are physically connected. PECAM1 mediates homophilic adhesion '
        '(GO:0007156). Both share glomerular endothelium development (GO:0072011).'
    ),
    ('fibroblast', 'fibroblast'): (
        'PDGFRA+ fibroblast clustering reflects interstitial compartment '
        'organization. PDGFRA is expressed in kidney cortex and nephron tubule '
        'interstitium. In nephrogenesis (WP4823), fibroblasts form the '
        'structural scaffold of the kidney.'
    ),
    ('macrophage', 'macrophage'): (
        'Myeloid cell clustering reflects immune niche formation. ITGAM+ cells '
        'accumulate at sites of injury via complement-mediated adhesion. '
        'MRC1+ M2 macrophages (CL:0000890) cluster in resolution zones.'
    ),
    # Cross-type narratives
    ('endothelial', 'fibroblast'): (
        'Endothelial-fibroblast proximity reflects the vascular-interstitial '
        'interface. PDGFRB on pericytes physically contacts endothelial cells. '
        'PDGFRA/PECAM1 share cell migration regulation (GO:0030335). '
        'This interface is where capillary rarefaction and fibrosis originate in AKI.'
    ),
    ('endothelial', 'macrophage'): (
        'Endothelial-macrophage proximity reflects leukocyte extravasation. '
        'PECAM1 mediates diapedesis (GO:0050904) and neutrophil extravasation '
        '(GO:0072672). ITGAM on myeloid cells adheres to PECAM1 via leukocyte '
        'cell-cell adhesion (GO:0007159). This is the transendothelial migration axis.'
    ),
    ('fibroblast', 'macrophage'): (
        'Fibroblast-macrophage proximity reflects the repair/fibrosis niche. '
        'PDGFRB forms a physical complex with MRC1 (3 evidence). M2 macrophages '
        '(CD206+/MRC1+) secrete TGF-beta that activates fibroblasts via PDGFR '
        'signaling. This interaction determines whether repair resolves or '
        'progresses to fibrosis.'
    ),
    ('immune', 'endothelial'): (
        'Immune-endothelial proximity reflects active inflammation. CD44-PECAM1 '
        'are sequential partners in the leukocyte adhesion cascade: CD44 mediates '
        'rolling, PECAM1 mediates transmigration. PTPRC+ cells cluster at '
        'endothelial surfaces during immune infiltration.'
    ),
    ('immune', 'fibroblast'): (
        'Immune-fibroblast proximity in AKI reflects the inflammatory-fibrotic '
        'transition zone. CD44 physically complexes with PDGFRB (10 evidence, '
        'belief=0.74) — a mechanistic basis for immune-stromal interaction. '
        'CD44 also activates CD34 (4 evidence), linking injury signaling to '
        'progenitor mobilization.'
    ),
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_markers_from_cell_type(cell_type: str) -> List[str]:
    """Extract marker names from cell type labels like 'activated_immune_cd44'."""
    markers = []
    ct_lower = cell_type.lower()

    marker_map = {
        'cd44': 'CD44', 'cd140b': 'CD140b', 'cd140a': 'CD140a',
        'cd206': 'CD206', 'cd11b': 'CD11b', 'cd45': 'CD45',
        'cd31': 'CD31', 'cd34': 'CD34', 'ly6g': 'Ly6G',
    }

    for key, marker in marker_map.items():
        if key in ct_lower:
            markers.append(marker)

    # Infer from cell type lineage
    if 'immune' in ct_lower and 'CD45' not in markers:
        markers.append('CD45')
    if 'macrophage' in ct_lower and 'CD11b' not in markers:
        markers.append('CD11b')
    if 'endothelial' in ct_lower and 'CD31' not in markers:
        markers.append('CD31')
    if 'fibroblast' in ct_lower and 'CD140a' not in markers:
        markers.append('CD140a')
    if 'neutrophil' in ct_lower and 'Ly6G' not in markers:
        markers.append('Ly6G')

    return markers


def get_gene_for_marker(marker: str) -> Optional[str]:
    """Get gene name for a marker."""
    info = MARKER_GENE_MAP.get(marker)
    if info and info['grounded']:
        return info['gene_name']
    return None


def get_lineage(cell_type: str) -> str:
    """Classify cell type into a lineage category."""
    ct = cell_type.lower()
    if 'endothelial' in ct or 'cd31' in ct or 'cd34' in ct:
        return 'endothelial'
    if 'fibroblast' in ct or 'cd140a' in ct or 'stromal' in ct:
        return 'fibroblast'
    if 'macrophage' in ct or 'cd206' in ct or 'cd11b' in ct:
        return 'macrophage'
    if 'immune' in ct or 'cd45' in ct:
        return 'immune'
    if 'neutrophil' in ct or 'ly6g' in ct:
        return 'neutrophil'
    if 'pericyte' in ct or 'cd140b' in ct:
        return 'pericyte'
    return 'unknown'


def get_pairwise_indra_summary(gene1: str, gene2: str) -> Dict:
    """Aggregate INDRA statements between two genes."""
    stmts = []
    for s in INDRA_STATEMENTS:
        if (s['source'] == gene1 and s['target'] == gene2) or \
           (s['source'] == gene2 and s['target'] == gene1):
            stmts.append(s)

    if not stmts:
        return {
            'total_evidence': 0,
            'max_belief': 0,
            'statement_types': [],
            'top_note': '',
        }

    total = sum(s['evidence'] for s in stmts)
    max_belief = max(s['belief'] for s in stmts)
    types = list(set(s['type'] for s in stmts))
    notes = [s.get('note', '') for s in stmts if s.get('note')]

    return {
        'total_evidence': total,
        'max_belief': max_belief,
        'statement_types': types,
        'top_note': notes[0] if notes else '',
    }


def get_shared_biology(gene1: str, gene2: str) -> Dict:
    """Look up shared biological processes between two genes."""
    key = tuple(sorted([gene1, gene2]))
    return SHARED_BIOLOGY.get(key, {})


def get_mechanistic_narrative(focal_type: str, neighbor_type: str) -> str:
    """Get mechanistic narrative for a cell type pair."""
    focal_lin = get_lineage(focal_type)
    neighbor_lin = get_lineage(neighbor_type)

    # Try exact match
    key = tuple(sorted([focal_lin, neighbor_lin]))
    if key in MECHANISTIC_NARRATIVES:
        return MECHANISTIC_NARRATIVES[key]

    # Try with pericyte → fibroblast mapping
    mapped = tuple(sorted([
        'fibroblast' if l == 'pericyte' else l
        for l in [focal_lin, neighbor_lin]
    ]))
    if mapped in MECHANISTIC_NARRATIVES:
        return MECHANISTIC_NARRATIVES[mapped]

    return ''


def classify_evidence_tier(total_evidence: int, max_belief: float = 0) -> str:
    """Classify evidence strength into tiers."""
    if total_evidence >= 100:
        return 'high'
    elif total_evidence >= 10 or (total_evidence >= 5 and max_belief >= 0.7):
        return 'medium'
    elif total_evidence >= 1:
        return 'low'
    return 'none'


def count_kidney_contexts(genes: List[str]) -> Dict:
    """Count how many kidney-specific contexts a set of genes appears in."""
    contexts = []
    for ctx_name, ctx_genes in KIDNEY_CONTEXT.items():
        for g in genes:
            if g in ctx_genes:
                contexts.append(ctx_name)
                break
    return {
        'kidney_context_count': len(contexts),
        'kidney_contexts': contexts,
    }


# ============================================================================
# EVIDENCE TABLE BUILDERS
# ============================================================================

def build_evidence_for_da_results(da_file: Path) -> List[Dict]:
    """Build evidence rows for differential abundance results."""
    import pandas as pd
    da_df = pd.read_csv(da_file)

    rows = []
    for _, result in da_df.iterrows():
        cell_type = result['cell_type']
        markers = extract_markers_from_cell_type(cell_type)
        genes = [get_gene_for_marker(m) for m in markers]
        genes = [g for g in genes if g is not None]

        # Disease associations
        aki_genes = [g for g in genes if g in KIDNEY_CONTEXT['aki_direct']]

        # Pairwise INDRA evidence
        max_evidence = 0
        max_belief = 0
        all_stmt_types = []
        all_notes = []
        for i, g1 in enumerate(genes):
            for g2 in genes[i+1:]:
                summary = get_pairwise_indra_summary(g1, g2)
                max_evidence = max(max_evidence, summary['total_evidence'])
                max_belief = max(max_belief, summary['max_belief'])
                all_stmt_types.extend(summary['statement_types'])
                if summary['top_note']:
                    all_notes.append(summary['top_note'])

        # Kidney context
        kidney = count_kidney_contexts(genes)

        # Gene knowledge summaries
        gene_relevance = []
        for g in genes:
            info = GENE_KNOWLEDGE.get(g, {})
            if info.get('kidney_relevance'):
                gene_relevance.append(f"{g}: {info['kidney_relevance'][:100]}...")

        rows.append({
            'finding_type': 'differential_abundance',
            'finding': f"{cell_type} {result.get('comparison', '')}",
            'cell_type': cell_type,
            'markers': ', '.join(markers),
            'genes': ', '.join(genes),
            'hedges_g': result.get('hedges_g', result.get('cohens_d', None)),
            'p_value_fdr': result.get('p_value_fdr', result.get('p_value', None)),
            'aki_gene_association': len(aki_genes) > 0,
            'aki_genes': ', '.join(aki_genes),
            'indra_evidence_count': max_evidence,
            'indra_max_belief': round(max_belief, 3),
            'indra_statement_types': ', '.join(sorted(set(all_stmt_types))),
            'evidence_tier': classify_evidence_tier(max_evidence, max_belief),
            'kidney_context_count': kidney['kidney_context_count'],
            'kidney_contexts': '; '.join(kidney['kidney_contexts']),
            'mechanistic_note': all_notes[0] if all_notes else '',
        })

    return rows


def build_evidence_for_neighborhood_results(nb_file: Path) -> List[Dict]:
    """Build evidence rows for neighborhood enrichment results."""
    import pandas as pd
    nb_df = pd.read_csv(nb_file)

    rows = []
    for _, result in nb_df.iterrows():
        focal = result['focal_cell_type']
        neighbor = result['neighbor_cell_type']

        if neighbor == 'unassigned':
            continue

        focal_markers = extract_markers_from_cell_type(focal)
        neighbor_markers = extract_markers_from_cell_type(neighbor)
        focal_genes = [g for g in [get_gene_for_marker(m) for m in focal_markers] if g]
        neighbor_genes = [g for g in [get_gene_for_marker(m) for m in neighbor_markers] if g]
        all_genes = list(set(focal_genes + neighbor_genes))

        # Cross-type INDRA evidence
        max_evidence = 0
        max_belief = 0
        all_stmt_types = []
        all_notes = []
        shared_procs = []

        for g1 in focal_genes:
            for g2 in neighbor_genes:
                if g1 != g2:
                    summary = get_pairwise_indra_summary(g1, g2)
                    max_evidence = max(max_evidence, summary['total_evidence'])
                    max_belief = max(max_belief, summary['max_belief'])
                    all_stmt_types.extend(summary['statement_types'])
                    if summary['top_note']:
                        all_notes.append(summary['top_note'])

                    bio = get_shared_biology(g1, g2)
                    if bio.get('shared_processes'):
                        shared_procs.extend(bio['shared_processes'])

        enrichment = result.get('enrichment_score', 0)
        fdr_col = 'fraction_significant_fdr' if 'fraction_significant_fdr' in result.index else 'fraction_significant_raw'
        frac_sig = result.get(fdr_col, 0)

        # Get mechanistic narrative
        narrative = get_mechanistic_narrative(focal, neighbor)

        # Kidney context for all involved genes
        kidney = count_kidney_contexts(all_genes)

        # AKI genes
        aki_genes = [g for g in all_genes if g in KIDNEY_CONTEXT['aki_direct']]

        if enrichment > 1.2 or (focal == neighbor):
            rows.append({
                'finding_type': 'neighborhood_enrichment',
                'finding': f"{focal} <-> {neighbor} @ {result.get('timepoint', 'all')}",
                'cell_type': f"{focal} | {neighbor}",
                'markers': f"{', '.join(focal_markers)} | {', '.join(neighbor_markers)}",
                'genes': f"{', '.join(focal_genes)} | {', '.join(neighbor_genes)}",
                'enrichment_score': enrichment,
                'fraction_significant': frac_sig,
                'aki_gene_association': len(aki_genes) > 0,
                'aki_genes': ', '.join(aki_genes),
                'indra_evidence_count': max_evidence,
                'indra_max_belief': round(max_belief, 3),
                'indra_statement_types': ', '.join(sorted(set(all_stmt_types))),
                'evidence_tier': classify_evidence_tier(max_evidence, max_belief),
                'shared_go_processes': '; '.join(set(shared_procs)),
                'kidney_context_count': kidney['kidney_context_count'],
                'kidney_contexts': '; '.join(kidney['kidney_contexts']),
                'mechanistic_narrative': narrative[:300] if narrative else '',
                'mechanistic_note': all_notes[0] if all_notes else '',
            })

    return rows


def build_knowledge_base() -> Dict:
    """Export full INDRA knowledge base as JSON."""
    return {
        'metadata': {
            'query_date': '2026-02-20',
            'source': 'INDRA/CoGEx via MCP tools',
            'panel_size': 9,
            'grounded_genes': 8,
            'total_indra_statements': len(INDRA_STATEMENTS),
            'note': 'All data pre-queried from CoGEx knowledge graph. '
                    'This is biological CONTEXT, not validation.',
        },
        'marker_grounding': MARKER_GENE_MAP,
        'gene_knowledge': GENE_KNOWLEDGE,
        'indra_statements': INDRA_STATEMENTS,
        'shared_biology': {
            f"{k[0]}_{k[1]}": v for k, v in SHARED_BIOLOGY.items()
        },
        'kidney_context': KIDNEY_CONTEXT,
        'mechanistic_narratives': {
            f"{k[0]}_{k[1]}": v for k, v in MECHANISTIC_NARRATIVES.items()
        },
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("INDRA/CoGEx Knowledge-Grounded Evidence Framework")
    print("=" * 80)

    output_dir = Path('results/biological_analysis')

    # 1. Export full knowledge base
    kb = build_knowledge_base()
    kb_file = output_dir / 'indra_knowledge_base.json'
    with open(kb_file, 'w') as f:
        json.dump(kb, f, indent=2, default=str)
    print(f"\n  Knowledge base: {kb_file}")
    print(f"    {len(INDRA_STATEMENTS)} intra-panel INDRA statements")
    print(f"    {len(SHARED_BIOLOGY)} shared biology entries")
    print(f"    {len(MECHANISTIC_NARRATIVES)} mechanistic narratives")

    # 2. Differential abundance evidence
    da_file = output_dir / 'differential_abundance' / 'temporal_differential_abundance.csv'
    da_rows = []
    if da_file.exists():
        print(f"\n  Processing differential abundance: {da_file}")
        da_rows = build_evidence_for_da_results(da_file)
        print(f"    {len(da_rows)} findings annotated")
    else:
        print(f"\n  Skipping DA (file not found): {da_file}")

    # 3. Neighborhood enrichment evidence
    nb_file = output_dir / 'spatial_neighborhoods' / 'temporal_neighborhood_enrichments.csv'
    nb_rows = []
    if nb_file.exists():
        print(f"\n  Processing neighborhood enrichments: {nb_file}")
        nb_rows = build_evidence_for_neighborhood_results(nb_file)
        print(f"    {len(nb_rows)} findings annotated")
    else:
        print(f"\n  Skipping neighborhoods (file not found): {nb_file}")

    # 4. Combine and save evidence table
    all_rows = da_rows + nb_rows

    if all_rows:
        import pandas as pd
        evidence_df = pd.DataFrame(all_rows)
        output_file = output_dir / 'indra_evidence_table.csv'
        evidence_df.to_csv(output_file, index=False)
        print(f"\n  Evidence table: {output_file}")
        print(f"    Total findings: {len(all_rows)}")

        # Summary statistics
        print(f"\n  Evidence tier distribution:")
        for tier in ['high', 'medium', 'low', 'none']:
            count = len([r for r in all_rows if r.get('evidence_tier') == tier])
            print(f"    {tier}: {count}")

        aki_count = len([r for r in all_rows if r.get('aki_gene_association')])
        print(f"  Findings with AKI gene association: {aki_count}")

        # INDRA statement type distribution
        all_types = []
        for r in all_rows:
            if r.get('indra_statement_types'):
                all_types.extend(r['indra_statement_types'].split(', '))
        if all_types:
            from collections import Counter
            type_counts = Counter(all_types)
            print(f"\n  INDRA statement types across findings:")
            for t, c in type_counts.most_common():
                print(f"    {t}: {c}")

        # Kidney context coverage
        with_kidney = len([r for r in all_rows if r.get('kidney_context_count', 0) > 0])
        print(f"\n  Findings with kidney-specific context: {with_kidney}/{len(all_rows)}")

    else:
        print("\n  No findings to annotate.")

    # 5. Marker grounding reference
    print(f"\n{'─' * 80}")
    print("Marker Grounding Reference (INDRA/CoGEx)")
    print(f"{'─' * 80}")
    for marker, info in MARKER_GENE_MAP.items():
        if info['grounded']:
            gene = info['gene_name']
            gk = GENE_KNOWLEDGE.get(gene, {})
            n_paths = len(gk.get('kidney_pathways', []))
            n_go = len(gk.get('kidney_go_terms', []))
            n_tissues = len(gk.get('kidney_tissues', []))
            n_diseases = len(gk.get('diseases', {}))
            n_celltypes = len(gk.get('cell_types', []))
            print(f"  {marker:8s} -> {gene:8s} ({info['curie']}) | "
                  f"{n_paths} pathways, {n_go} GO, {n_tissues} tissues, "
                  f"{n_diseases} diseases, {n_celltypes} cell types")
        else:
            print(f"  {marker:8s} -> [NOT GROUNDED] {info.get('note', '')}")

    # Key biological findings summary
    print(f"\n{'─' * 80}")
    print("Key INDRA-Grounded Biological Findings")
    print(f"{'─' * 80}")
    summaries = [
        "CD44 is the ONLY panel gene with direct AKI disease association (MESH:D058186)",
        "PDGFRA and PDGFRB are BOTH in nephrogenesis pathway (WP4823)",
        "PDGFRB and PECAM1 are BOTH in kidney cell lineage markers (WP5236)",
        "CD34 + PECAM1 share glomerular endothelium development (GO:0072011)",
        "CD34 has glomerular filtration (GO:0003094) and vascular wound healing (GO:0061042)",
        "MRC1 marks 'alternatively activated macrophage' (CL:0000890) — validates M2 annotation",
        "PDGFRB-CD44 physical complex (10 evidence, belief=0.74) — explains injury-stroma interaction",
        "PDGFRA-PDGFRB complex (151 evidence, belief=0.89) — strongest intra-panel relationship",
        "ITGAM + PECAM1 share leukocyte adhesion (GO:0007159) — transendothelial migration axis",
        "All 8 grounded genes are expressed in metanephros cortex (UBERON:0010533)",
    ]
    for i, s in enumerate(summaries, 1):
        print(f"  {i:2d}. {s}")

    print(f"\n{'=' * 80}")
    print("Evidence framework complete.")
    print("This provides MECHANISTIC CONTEXT grounded in the INDRA knowledge graph.")
    print("It contextualizes spatial patterns against known biology —")
    print("it does not validate that specific patterns are statistically real.")
    print(f"{'=' * 80}\n")


if __name__ == '__main__':
    main()
