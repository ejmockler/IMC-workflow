#!/bin/bash
# Complete Biological Analysis Workflow
# Runs all three modules in sequence

set -e  # Exit on any error

echo "=========================================================================="
echo "BIOLOGICAL ANALYSIS WORKFLOW - Kidney Injury Spatial Proteomics"
echo "=========================================================================="
echo ""

# Module 1: Spatial Region Phenotyping (Cell Type Annotation)
echo "Module 1: Spatial Region Phenotyping"
echo "----------------------------------------------------------------------"
PYTHONPATH=. uv run python batch_annotate_all_rois.py
echo ""

# Module 2: Differential Abundance Analysis
echo "Module 2: Differential Abundance Analysis"
echo "----------------------------------------------------------------------"
PYTHONPATH=. uv run python differential_abundance_analysis.py
echo ""

# Module 3: Spatial Neighborhood Analysis
echo "Module 3: Spatial Neighborhood Analysis"
echo "----------------------------------------------------------------------"
PYTHONPATH=. uv run python spatial_neighborhood_analysis.py
echo ""

echo "=========================================================================="
echo "✅ COMPLETE ANALYSIS FINISHED"
echo "=========================================================================="
echo ""
echo "Results:"
echo "  • Cell annotations:      results/biological_analysis/cell_type_annotations/"
echo "  • Differential abundance: results/biological_analysis/differential_abundance/"
echo "  • Spatial neighborhoods:  results/biological_analysis/spatial_neighborhoods/"
echo ""
