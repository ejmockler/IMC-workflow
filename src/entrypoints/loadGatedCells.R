#' Load and Prepare Gated Cell Data for Spatial Analysis
#'
#' This entrypoint loads pre-gated cell data from RDS files, merges them if needed,
#' and prepares them for spatial analysis. This provides an alternative to the
#' unsupervised phenotyping approach in cellPhenotyping.R.
#'
#' @return A SpatialExperiment object with gated cell annotations.

source("src/core/DependencyManager.R")
source("src/core/ConfigurationManager.R")
source("src/core/Logger.R")

# Explicitly load required packages
library(SpatialExperiment)
library(SingleCellExperiment)
library(SummarizedExperiment)
library(BiocParallel)

#' @title Load Gated Cells
#' @description Main entry point for loading gated cell data
#' @param immune_cells_file Path to immune cells RDS file
#' @param m2_macrophages_file Path to M2 macrophages RDS file
#' @param non_m2_macrophages_file Path to non-M2 macrophages RDS file
#' @param fibroblasts_file Path to fibroblasts RDS file
#' @param endothelial_cells_file Path to endothelial cells RDS file
#' @param merge_gated_cells Whether to merge all gated cells into a single SPE
#' @param output_dir Directory to save outputs
#' @param memory_limit Optional memory limit in MB
#' @return SpatialExperiment object with gated cell annotations
runLoadGatedCells <- function(
  immune_cells_file = NULL,
  m2_macrophages_file = NULL,
  non_m2_macrophages_file = NULL,
  fibroblasts_file = NULL,
  endothelial_cells_file = NULL,
  merge_gated_cells = NULL,
  output_dir = NULL,
  memory_limit = 0
) {
  # Initialize configuration and logger
  configManager <- ConfigurationManager$new()
  logger <- Logger$new(log_file = "logs/loadGatedCells.log", log_level = "INFO")
  
  # Initialize dependencies
  dependencyManager <- DependencyManager$new(logger = logger)
  dependencyManager$validate_environment()
  
  # Get parameters from config or use provided values
  immune_cells_file <- immune_cells_file %||% configManager$config$gated_cell_analysis$immune_cells_file
  m2_macrophages_file <- m2_macrophages_file %||% configManager$config$gated_cell_analysis$m2_macrophages_file
  non_m2_macrophages_file <- non_m2_macrophages_file %||% configManager$config$gated_cell_analysis$non_m2_macrophages_file
  fibroblasts_file <- fibroblasts_file %||% configManager$config$gated_cell_analysis$fibroblasts_file
  endothelial_cells_file <- endothelial_cells_file %||% configManager$config$gated_cell_analysis$endothelial_cells_file
  merge_gated_cells <- merge_gated_cells %||% configManager$config$gated_cell_analysis$merge_gated_cells
  output_dir <- output_dir %||% configManager$config$output$dir
  
  # Create a list to store all loaded SPE objects
  spe_list <- list()
  
  # Load immune cells
  if (file.exists(immune_cells_file)) {
    logger$log_info("Loading immune cells from: %s", immune_cells_file)
    spe_immune <- readRDS(immune_cells_file)
    spe_immune$gated_celltype <- "Immune"
    spe_list[["immune"]] <- spe_immune
  } else {
    logger$log_warning("Immune cells file not found: %s", immune_cells_file)
  }
  
  # Load M2 macrophages
  if (file.exists(m2_macrophages_file)) {
    logger$log_info("Loading M2 macrophages from: %s", m2_macrophages_file)
    spe_m2 <- readRDS(m2_macrophages_file)
    spe_m2$gated_celltype <- "M2_Macrophage"
    spe_list[["m2_macrophage"]] <- spe_m2
  } else {
    logger$log_warning("M2 macrophages file not found: %s", m2_macrophages_file)
  }
  
  # Load non-M2 macrophages
  if (file.exists(non_m2_macrophages_file)) {
    logger$log_info("Loading non-M2 macrophages from: %s", non_m2_macrophages_file)
    spe_non_m2 <- readRDS(non_m2_macrophages_file)
    spe_non_m2$gated_celltype <- "Non_M2_Macrophage"
    spe_list[["non_m2_macrophage"]] <- spe_non_m2
  } else {
    logger$log_warning("Non-M2 macrophages file not found: %s", non_m2_macrophages_file)
  }
  
  # Load fibroblasts
  if (file.exists(fibroblasts_file)) {
    logger$log_info("Loading fibroblasts from: %s", fibroblasts_file)
    spe_fibroblasts <- readRDS(fibroblasts_file)
    spe_fibroblasts$gated_celltype <- "Fibroblast"
    spe_list[["fibroblast"]] <- spe_fibroblasts
  } else {
    logger$log_warning("Fibroblasts file not found: %s", fibroblasts_file)
  }
  
  # Load endothelial cells
  if (file.exists(endothelial_cells_file)) {
    logger$log_info("Loading endothelial cells from: %s", endothelial_cells_file)
    spe_endothelial <- readRDS(endothelial_cells_file)
    spe_endothelial$gated_celltype <- "Endothelial"
    spe_list[["endothelial"]] <- spe_endothelial
  } else {
    logger$log_warning("Endothelial cells file not found: %s", endothelial_cells_file)
  }
  
  # Check if we have any data
  if (length(spe_list) == 0) {
    stop("No gated cell data could be loaded. Please check file paths.")
  }
  
  # If merge_gated_cells is TRUE, merge all SPE objects
  if (merge_gated_cells) {
    logger$log_info("Merging all gated cell datasets...")
    
    # Use the first SPE as a base
    spe_merged <- spe_list[[1]]
    
    # If there are more SPEs, merge them
    if (length(spe_list) > 1) {
      for (i in 2:length(spe_list)) {
        # Ensure column names match
        current_spe <- spe_list[[i]]
        
        # Merge the SPEs
        spe_merged <- cbind(spe_merged, current_spe)
      }
    }
    
    # Use the merged SPE for further processing
    spe <- spe_merged
    logger$log_info("Successfully merged %d gated cell datasets with %d cells total", 
                   length(spe_list), ncol(spe))
  } else {
    # Just use the first SPE
    spe <- spe_list[[1]]
    logger$log_info("Using %s dataset with %d cells", 
                   names(spe_list)[1], ncol(spe))
  }
  
  # Save the gated cell SPE
  output_file <- file.path(output_dir, "spe_gated_cells.rds")
  saveRDS(spe, output_file)
  logger$log_info("Saved gated cell SPE to: %s", output_file)
  
  # Set memory limit if specified
  if (memory_limit > 0) {
    gc_limit <- memory_limit * 1024 * 1024  # Convert to bytes
    message(sprintf("Setting memory limit to %d MB", memory_limit))
    utils::mem.limit(gc_limit)
  }
  
  # Return the SPE object
  invisible(spe)
}

# Run the function if this script is executed directly
if (interactive() || identical(environment(), globalenv())) {
  spe_gated <- runLoadGatedCells()
}