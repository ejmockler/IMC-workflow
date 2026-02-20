"""
Module 4: INDRA Evidence Table Builder

Queries INDRA/CoGEx knowledge graph to provide biological context
for spatial findings. This is NOT validation — it contextualizes
observed patterns against the existing literature.

For each marker and marker pair in the analysis outputs:
1. Grounds markers to gene CURIEs via INDRA
2. Queries relationship evidence between marker pairs
3. Queries disease associations (AKI focus)
4. Produces a structured evidence table

Input: Analysis outputs from differential_abundance and spatial_neighborhoods
Output: results/biological_analysis/indra_evidence_table.csv

NOTE: This script is designed to be run manually. It requires the INDRA/CoGEx
MCP server to be available. The evidence table can also be built interactively
using the CoGEx MCP tools in a Claude Code session.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Kidney panel marker → gene mapping (pre-grounded via INDRA/CoGEx)
# Grounding performed in prior session using mcp__indra-cogex__ground_entity
MARKER_GENE_MAP = {
    'CD45': {'curie': 'HGNC:9666', 'gene_name': 'PTPRC', 'grounded': True},
    'CD11b': {'curie': 'HGNC:6149', 'gene_name': 'ITGAM', 'grounded': True},
    'Ly6G': {'curie': None, 'gene_name': None, 'grounded': False,
             'note': 'Murine neutrophil marker; no human ortholog in INDRA'},
    'CD140a': {'curie': 'HGNC:8803', 'gene_name': 'PDGFRA', 'grounded': True},
    'CD140b': {'curie': 'HGNC:8804', 'gene_name': 'PDGFRB', 'grounded': True},
    'CD31': {'curie': 'HGNC:8823', 'gene_name': 'PECAM1', 'grounded': True},
    'CD34': {'curie': 'HGNC:1662', 'gene_name': 'CD34', 'grounded': True},
    'CD206': {'curie': 'HGNC:7228', 'gene_name': 'MRC1', 'grounded': True},
    'CD44': {'curie': 'HGNC:1681', 'gene_name': 'CD44', 'grounded': True},
}

# Pre-queried INDRA evidence counts between marker pairs
# Source: CoGEx Cypher query: MATCH (a)-[r:indra_rel]-(b) WHERE a.name < b.name
# AND names IN panel ... RETURN a.name, b.name, count(r) ORDER BY count DESC
# Queried 2026-02-19
PAIRWISE_EVIDENCE = {
    ('PDGFRA', 'PDGFRB'): {
        'evidence_count': 46,
        'note': 'Receptor tyrosine kinase heterodimerization; highest evidence in panel'
    },
    ('ITGAM', 'MRC1'): {
        'evidence_count': 15,
        'note': 'M2 macrophage co-expression (CD11b+/CD206+)'
    },
    ('CD34', 'PTPRC'): {
        'evidence_count': 9,
        'note': 'Progenitor/leukocyte co-expression'
    },
    ('CD34', 'PECAM1'): {
        'evidence_count': 9,
        'note': 'Endothelial progenitor markers'
    },
    ('ITGAM', 'PTPRC'): {
        'evidence_count': 8,
        'note': 'Myeloid integrin + pan-leukocyte antigen'
    },
    ('PECAM1', 'PTPRC'): {
        'evidence_count': 8,
        'note': 'Endothelial/leukocyte adhesion'
    },
    ('CD44', 'PTPRC'): {
        'evidence_count': 7,
        'note': 'CD44 and CD45 co-expression on activated immune cells'
    },
    ('CD34', 'CD44'): {
        'evidence_count': 6,
        'note': 'Progenitor/adhesion markers'
    },
    ('CD44', 'ITGAM'): {
        'evidence_count': 6,
        'note': 'Adhesion molecule + myeloid integrin'
    },
    ('CD34', 'PDGFRB'): {
        'evidence_count': 5,
        'note': 'Progenitor/pericyte markers'
    },
    ('CD34', 'ITGAM'): {
        'evidence_count': 5,
        'note': 'Progenitor/myeloid markers'
    },
    ('ITGAM', 'PECAM1'): {
        'evidence_count': 5,
        'note': 'Myeloid/endothelial interaction'
    },
    ('MRC1', 'PTPRC'): {
        'evidence_count': 5,
        'note': 'Macrophage/leukocyte markers'
    },
    ('CD34', 'PDGFRA'): {
        'evidence_count': 4,
        'note': 'Progenitor/fibroblast markers'
    },
    ('CD44', 'PECAM1'): {
        'evidence_count': 4,
        'note': 'Adhesion/endothelial markers'
    },
    ('CD44', 'PDGFRB'): {
        'evidence_count': 4,
        'note': 'Activation/pericyte markers'
    },
    ('CD34', 'MRC1'): {
        'evidence_count': 2,
        'note': 'Progenitor/macrophage markers'
    },
    ('CD44', 'PDGFRA'): {
        'evidence_count': 2,
        'note': 'Activation/fibroblast markers'
    },
    ('CD44', 'MRC1'): {
        'evidence_count': 2,
        'note': 'CD44 and mannose receptor on activated macrophages'
    },
    ('MRC1', 'PECAM1'): {
        'evidence_count': 2,
        'note': 'Macrophage/endothelial interaction'
    },
    ('MRC1', 'PDGFRB'): {
        'evidence_count': 2,
        'note': 'Macrophage/pericyte interaction'
    },
    ('PDGFRA', 'PTPRC'): {
        'evidence_count': 1,
        'note': 'Fibroblast/leukocyte (sparse evidence)'
    },
    ('PDGFRB', 'PTPRC'): {
        'evidence_count': 1,
        'note': 'Pericyte/leukocyte (sparse evidence)'
    },
}

# Pre-queried disease associations
DISEASE_ASSOCIATIONS = {
    'CD44': {
        'aki_association': True,
        'disease_curie': 'MESH:D058186',
        'disease_name': 'Acute Kidney Injury',
        'evidence_type': 'gene-disease association',
        'note': 'Direct gene-disease edge in INDRA/CoGEx'
    },
    'PTPRC': {
        'aki_association': False,
        'note': 'Pan-leukocyte marker; associated with many immune conditions'
    },
    'ITGAM': {
        'aki_association': False,
        'note': 'Myeloid integrin; associated with inflammatory conditions broadly'
    },
    'PDGFRA': {
        'aki_association': False,
        'note': 'Fibroblast marker; associated with fibrotic diseases'
    },
    'PDGFRB': {
        'aki_association': False,
        'note': 'Pericyte marker; associated with vascular remodeling'
    },
    'PECAM1': {
        'aki_association': False,
        'note': 'Endothelial marker; associated with vascular diseases'
    },
    'CD34': {
        'aki_association': False,
        'note': 'Progenitor/endothelial marker'
    },
    'MRC1': {
        'aki_association': False,
        'note': 'M2 macrophage marker; associated with tissue repair'
    },
}


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


def get_pairwise_evidence(gene1: str, gene2: str) -> Dict:
    """Look up pre-queried evidence between two genes."""
    key1 = (gene1, gene2)
    key2 = (gene2, gene1)

    if key1 in PAIRWISE_EVIDENCE:
        return PAIRWISE_EVIDENCE[key1]
    elif key2 in PAIRWISE_EVIDENCE:
        return PAIRWISE_EVIDENCE[key2]
    return {'evidence_count': 0, 'note': 'No evidence queried'}


def classify_evidence_tier(count: int) -> str:
    """Classify evidence depth into tiers."""
    if count >= 100:
        return 'high'
    elif count >= 10:
        return 'medium'
    elif count >= 1:
        return 'low'
    return 'none'


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
        aki_associated = any(
            DISEASE_ASSOCIATIONS.get(g, {}).get('aki_association', False)
            for g in genes
        )

        # Pairwise evidence (if multiple markers)
        max_evidence = 0
        relationship_types = []
        for i, g1 in enumerate(genes):
            for g2 in genes[i+1:]:
                ev = get_pairwise_evidence(g1, g2)
                max_evidence = max(max_evidence, ev['evidence_count'])
                relationship_types.extend(ev.get('relationship_types', []))

        rows.append({
            'finding_type': 'differential_abundance',
            'finding': f"{cell_type} {result.get('comparison', '')}",
            'cell_type': cell_type,
            'markers': ', '.join(markers),
            'genes': ', '.join(genes),
            'hedges_g': result.get('hedges_g', result.get('cohens_d', None)),
            'p_value_fdr': result.get('p_value_fdr', result.get('p_value', None)),
            'aki_gene_association': aki_associated,
            'max_pairwise_evidence': max_evidence,
            'evidence_tier': classify_evidence_tier(max_evidence),
            'relationship_types': ', '.join(set(relationship_types)) if relationship_types else '',
        })

    return rows


def build_evidence_for_neighborhood_results(nb_file: Path) -> List[Dict]:
    """Build evidence rows for neighborhood enrichment results."""
    import pandas as pd
    nb_df = pd.read_csv(nb_file)

    # Focus on self-clustering and strong cross-type enrichments
    rows = []
    for _, result in nb_df.iterrows():
        focal = result['focal_cell_type']
        neighbor = result['neighbor_cell_type']

        if neighbor == 'unassigned':
            continue

        focal_markers = extract_markers_from_cell_type(focal)
        neighbor_markers = extract_markers_from_cell_type(neighbor)

        focal_genes = [get_gene_for_marker(m) for m in focal_markers]
        neighbor_genes = [get_gene_for_marker(m) for m in neighbor_markers]
        focal_genes = [g for g in focal_genes if g]
        neighbor_genes = [g for g in neighbor_genes if g]

        # Cross-type evidence
        max_evidence = 0
        relationship_types = []
        for g1 in focal_genes:
            for g2 in neighbor_genes:
                if g1 != g2:
                    ev = get_pairwise_evidence(g1, g2)
                    max_evidence = max(max_evidence, ev['evidence_count'])
                    relationship_types.extend(ev.get('relationship_types', []))

        enrichment = result.get('enrichment_score', 0)
        fdr_col = 'fraction_significant_fdr' if 'fraction_significant_fdr' in result.index else 'fraction_significant'
        frac_sig = result.get(fdr_col, 0)

        if enrichment > 1.2 or (focal == neighbor):  # Include self-clustering
            rows.append({
                'finding_type': 'neighborhood_enrichment',
                'finding': f"{focal} <-> {neighbor} @ {result.get('timepoint', 'all')}",
                'cell_type': f"{focal} | {neighbor}",
                'markers': f"{', '.join(focal_markers)} | {', '.join(neighbor_markers)}",
                'genes': f"{', '.join(focal_genes)} | {', '.join(neighbor_genes)}",
                'enrichment_score': enrichment,
                'fraction_significant': frac_sig,
                'aki_gene_association': any(
                    DISEASE_ASSOCIATIONS.get(g, {}).get('aki_association', False)
                    for g in focal_genes + neighbor_genes
                ),
                'max_pairwise_evidence': max_evidence,
                'evidence_tier': classify_evidence_tier(max_evidence),
                'relationship_types': ', '.join(set(relationship_types)) if relationship_types else '',
            })

    return rows


def main():
    print("="*80)
    print("INDRA Evidence Table Builder")
    print("="*80)

    output_dir = Path('results/biological_analysis')

    # Differential abundance evidence
    da_file = output_dir / 'differential_abundance' / 'temporal_differential_abundance.csv'
    da_rows = []
    if da_file.exists():
        print(f"\nProcessing differential abundance: {da_file}")
        da_rows = build_evidence_for_da_results(da_file)
        print(f"  {len(da_rows)} findings annotated")
    else:
        print(f"\n  Skipping DA (file not found): {da_file}")

    # Neighborhood enrichment evidence
    nb_file = output_dir / 'spatial_neighborhoods' / 'temporal_neighborhood_enrichments.csv'
    nb_rows = []
    if nb_file.exists():
        print(f"\nProcessing neighborhood enrichments: {nb_file}")
        nb_rows = build_evidence_for_neighborhood_results(nb_file)
        print(f"  {len(nb_rows)} findings annotated")
    else:
        print(f"\n  Skipping neighborhoods (file not found): {nb_file}")

    # Combine and save
    all_rows = da_rows + nb_rows

    if all_rows:
        import pandas as pd
        evidence_df = pd.DataFrame(all_rows)
        output_file = output_dir / 'indra_evidence_table.csv'
        evidence_df.to_csv(output_file, index=False)
        print(f"\n  Saved: {output_file}")
        print(f"  Total findings: {len(all_rows)}")

        # Summary
        print(f"\nEvidence tier distribution:")
        for tier in ['high', 'medium', 'low', 'none']:
            count = len([r for r in all_rows if r.get('evidence_tier') == tier])
            print(f"  {tier}: {count}")

        aki_count = len([r for r in all_rows if r.get('aki_gene_association')])
        print(f"\nFindings with AKI gene association: {aki_count}")
    else:
        print("\nNo findings to annotate.")

    # Write marker grounding reference
    print(f"\n{'─'*80}")
    print("Marker Grounding Reference (INDRA/CoGEx)")
    print(f"{'─'*80}")
    for marker, info in MARKER_GENE_MAP.items():
        status = f"-> {info['gene_name']} ({info['curie']})" if info['grounded'] else f"[NOT GROUNDED] {info.get('note', '')}"
        print(f"  {marker:10s} {status}")

    print(f"\n{'='*80}")
    print("Evidence table complete.")
    print("Note: This provides biological CONTEXT, not validation.")
    print("INDRA evidence confirms markers are biologically relevant;")
    print("it does not confirm that specific spatial patterns are real.")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
