# Script: gatedCellWorkflow.R
# Description: Master workflow for gated cell spatial analysis
# Author: Your Name
# Date: Current Date

# Load dependencies
library(SpatialExperiment)
library(imcRtools)
source("src/core/ConfigurationManager.R")
source("src/core/Logger.R")

# Function to run the complete gated cell analysis workflow
runGatedCellWorkflow <- function(
  config_file = "config/config.yml",
  output_dir = "output",
  generate_report = TRUE
) {
  # Create logger
  logger <- Logger$new("GatedCellWorkflow")
  logger$info("Starting gated cell analysis workflow")
  
  # Load configuration
  config <- ConfigurationManager$new(config_file)
  
  # Step 1: Load gated cells
  logger$info("Loading gated cell data")
  source("src/entrypoints/loadGatedCells.R")
  spe_gated <- runLoadGatedCells(config_file = config_file)
  
  # Step 2: Spatial community analysis
  logger$info("Running spatial community analysis")
  source("src/entrypoints/spatialCommunityAnalysisGated.R")
  spe_communities <- runSpatialCommunityAnalysisGated(
    input_file = file.path(output_dir, "spe_gated_cells.rds"),
    config_file = config_file
  )
  
  # Step 3: Spatial interaction analysis
  logger$info("Running spatial interaction analysis")
  source("src/entrypoints/spatialInteractionAnalysisGated.R")
  spe_interactions <- runSpatialInteractionAnalysisGated(
    input_file = file.path(output_dir, "spe_communities_gated.rds"),
    config_file = config_file
  )
  
  # Step 4: Visualization
  logger$info("Generating visualizations")
  source("src/entrypoints/visualizeGatedCells.R")
  runVisualizeGatedCells(
    input_file = file.path(output_dir, "spe_communities_gated.rds"),
    config_file = config_file
  )
  
  # Step 5: Comparative analysis (if enabled)
  if (config$get("comparative_analysis", FALSE)) {
    logger$info("Running comparative analysis")
    source("src/entrypoints/compareWorkflows.R")
    runCompareWorkflows(
      gated_file = file.path(output_dir, "spe_communities_gated.rds"),
      unsupervised_file = file.path(output_dir, "spe_communities.rds"),
      config_file = config_file
    )
  }
  
  # Step 6: Clinical correlation (if enabled)
  if (config$get("clinical_correlation", FALSE)) {
    logger$info("Running clinical correlation analysis")
    source("src/entrypoints/clinicalCorrelation.R")
    runClinicalCorrelation(
      input_file = file.path(output_dir, "spe_communities_gated.rds"),
      config_file = config_file
    )
  }
  
  # Step 7: Report generation (if enabled)
  if (generate_report) {
    logger$info("Generating analysis report")
    source("src/entrypoints/generateReports.R")
    report_path <- runGenerateReports(
      gated_file = file.path(output_dir, "spe_gated_cells.rds"),
      communities_file = file.path(output_dir, "spe_communities_gated.rds"),
      interactions_file = file.path(output_dir, "interaction_results_classic_gated.rds"),
      config_file = config_file
    )
    logger$info(paste("Report generated at:", report_path))
  }
  
  logger$info("Gated cell analysis workflow completed successfully")
  
  return(TRUE)
}
