#' Perform Spatial Interaction Analysis with Gated Cells
#'
#' This entrypoint analyzes spatial interactions between gated cell types.
#' It includes spatial context detection, patch detection, and statistical
#' interaction testing based on spatial graphs. It is a modified version of
#' spatialInteractionAnalysis.R that works with gated cell data.
#'
#' @return A SpatialExperiment object with spatial interaction annotations.

source("src/core/DependencyManager.R")
source("src/core/ConfigurationManager.R")
source("src/core/Logger.R")
source("src/analysis/SpatialInteraction.R")

# Explicitly load required packages for spatial analysis
library(SpatialExperiment)
library(SingleCellExperiment)
library(SummarizedExperiment)
library(imcRtools)
library(BiocParallel)

runSpatialInteractionAnalysisGated <- function(
  input_file = NULL,
  output_dir = NULL,
  detect_spatial_context = NULL,
  sc_k = NULL,
  sc_threshold = NULL,
  filter_sc = NULL,
  sc_group_by = NULL,
  filter_group_threshold = NULL,
  filter_cells_threshold = NULL,
  detect_patches = NULL,
  patch_celltype = NULL,
  expand_by = NULL,
  min_patch_size = NULL,
  test_interactions = NULL,
  interaction_method = NULL,
  patch_size = NULL,
  memory_limit = 0,
  n_cores = NULL
) {
  # Initialize configuration and logger
  configManager <- ConfigurationManager$new()
  logger <- Logger$new(log_file = "logs/spatialInteractionAnalysisGated.log", log_level = "INFO")
  
  # Initialize dependencies
  dependencyManager <- DependencyManager$new(logger = logger)
  dependencyManager$validate_environment()
  
  # Get parameters from config or use provided values
  input_file <- input_file %||% "output/spe_communities_gated.rds"
  output_dir <- output_dir %||% configManager$config$output$dir %||% "output"
  detect_spatial_context <- detect_spatial_context %||% configManager$config$interaction_analysis$detect_spatial_context %||% TRUE
  sc_k <- sc_k %||% configManager$config$interaction_analysis$sc_k %||% 40
  sc_threshold <- sc_threshold %||% configManager$config$interaction_analysis$sc_threshold %||% 0.9
  filter_sc <- filter_sc %||% configManager$config$interaction_analysis$filter_sc %||% TRUE
  sc_group_by <- sc_group_by %||% configManager$config$interaction_analysis$sc_group_by %||% "sample_id"
  filter_group_threshold <- filter_group_threshold %||% configManager$config$interaction_analysis$filter_group_threshold %||% 3
  filter_cells_threshold <- filter_cells_threshold %||% configManager$config$interaction_analysis$filter_cells_threshold %||% 100
  detect_patches <- detect_patches %||% configManager$config$interaction_analysis$detect_patches %||% TRUE
  patch_celltype <- patch_celltype %||% "M2_Macrophage"  # Default to M2 macrophages for patch detection
  expand_by <- expand_by %||% configManager$config$interaction_analysis$expand_by %||% 1
  min_patch_size <- min_patch_size %||% configManager$config$interaction_analysis$min_patch_size %||% 10
  test_interactions <- test_interactions %||% configManager$config$interaction_analysis$test_interactions %||% TRUE
  interaction_method <- interaction_method %||% configManager$config$interaction_analysis$method %||% c("classic", "patch")
  patch_size <- patch_size %||% configManager$config$interaction_analysis$patch_size %||% 3
  
  # Initialize cores if not specified
  if (is.null(n_cores)) {
    n_cores <- configManager$config$interaction_analysis$n_cores %||% max(1, floor(parallel::detectCores() * 0.7))
  }
  
  # Load the SpatialExperiment object with community annotations
  logger$log_info("Loading SpatialExperiment with community annotations from: %s", input_file)
  spe <- readRDS(input_file)
  
  # Check if the gated_celltype column exists
  if (!"gated_celltype" %in% colnames(colData(spe))) {
    stop("The 'gated_celltype' column is missing from the SpatialExperiment object. Please run loadGatedCells.R first.")
  }
  
  # Detect spatial contexts
  if (detect_spatial_context) {
    logger$log_info("Detecting spatial contexts...")
    
    # Check if cn_gated_celltypes exists, if not, use gated_celltype
    count_by <- if ("cn_gated_celltypes" %in% colnames(colData(spe))) {
      "cn_gated_celltypes"
    } else {
      "gated_celltype"
    }
    
    # Generate k-nearest neighbor graph for spatial context detection
    spe <- buildSpatialGraph(
      spe,
      img_id = "sample_id",
      type = "knn",
      name = "knn_spatialcontext_graph_gated",
      k = sc_k
    )
    
    # Aggregate based on cell types or neighborhoods
    spe <- aggregateNeighbors(
      spe,
      colPairName = "knn_spatialcontext_graph_gated",
      aggregate_by = "metadata",
      count_by = count_by,
      name = "aggregatedNeighborhood_gated"
    )
    
    # Detect spatial contexts
    spe <- detectSpatialContext(
      spe,
      entry = "aggregatedNeighborhood_gated",
      threshold = sc_threshold,
      name = "spatial_context_gated"
    )
    
    logger$log_info("Spatial context detection completed.")
    
    # Filter spatial contexts if requested
    if (filter_sc) {
      logger$log_info("Filtering spatial contexts...")
      
      # Filter by number of group entries
      spe <- filterSpatialContext(
        spe,
        entry = "spatial_context_gated",
        group_by = sc_group_by,
        group_threshold = filter_group_threshold
      )
      
      # Filter by number of group entries and total number of cells
      spe <- filterSpatialContext(
        spe,
        entry = "spatial_context_gated",
        group_by = sc_group_by,
        group_threshold = filter_group_threshold,
        cells_threshold = filter_cells_threshold
      )
      
      logger$log_info("Spatial context filtering completed.")
    }
  }
  
  # Detect patches
  if (detect_patches) {
    logger$log_info("Detecting patches of %s cells...", patch_celltype)
    
    # Check if the specified cell type exists
    if (!patch_celltype %in% unique(spe$gated_celltype)) {
      if (configManager$config$interaction_analysis$skip_patches_if_not_found %||% TRUE) {
        logger$log_warning("Cell type '%s' not found in gated_celltype. Skipping patch detection.", patch_celltype)
      } else {
        stop(sprintf("Cell type '%s' not found in gated_celltype.", patch_celltype))
      }
    } else {
      # Detect patches of the specified cell type
      spe <- patchDetection(
        spe,
        patch_cells = spe$gated_celltype == patch_celltype,
        img_id = "sample_id",
        expand_by = expand_by,
        min_patch_size = min_patch_size,
        colPairName = "neighborhood"
      )
      
      # Calculate minimum distance to patches
      spe <- minDistToCells(
        spe,
        x_cells = !is.na(spe$patch_id),
        img_id = "sample_id"
      )
      
      logger$log_info("Patch detection completed.")
    }
  }
  
  # Test interactions
  if (test_interactions) {
    logger$log_info("Testing cell type interactions...")
    
    # Test interactions using the classic method
    if ("classic" %in% interaction_method) {
      logger$log_info("Testing interactions using classic method...")
      
      interaction_results_classic <- testInteractions(
        spe,
        group_by = "sample_id",
        label = "gated_celltype",
        colPairName = "neighborhood",
        BPPARAM = SerialParam(RNGseed = 221029)
      )
      
      # Save interaction results
      saveRDS(
        interaction_results_classic,
        file.path(output_dir, "interaction_results_classic_gated.rds")
      )
      
      logger$log_info("Classic interaction testing completed.")
    }
    
    # Test interactions using the patch method
    if ("patch" %in% interaction_method) {
      logger$log_info("Testing interactions using patch method...")
      
      interaction_results_patch <- testInteractions(
        spe,
        group_by = "sample_id",
        label = "gated_celltype",
        colPairName = "neighborhood",
        method = "patch",
        patch_size = patch_size,
        BPPARAM = SerialParam(RNGseed = 221029)
      )
      
      # Save interaction results
      saveRDS(
        interaction_results_patch,
        file.path(output_dir, "interaction_results_patch_gated.rds")
      )
      
      logger$log_info("Patch interaction testing completed.")
    }
  }
  
  # Save the SpatialExperiment object with spatial interaction annotations
  output_file <- file.path(output_dir, "spe_interactions_gated.rds")
  saveRDS(spe, output_file)
  logger$log_info("Saved SpatialExperiment with spatial interaction annotations to: %s", output_file)
  
  # Set memory limit if specified
  if (memory_limit > 0) {
    gc_limit <- memory_limit * 1024 * 1024  # Convert to bytes
    message(sprintf("Setting memory limit to %d MB", memory_limit))
    utils::mem.limit(gc_limit)
  }
  
  # Return the SpatialExperiment object
  invisible(spe)
}

# Run the function if this script is executed directly
if (interactive() || identical(environment(), globalenv())) {
  spe_interactions <- runSpatialInteractionAnalysisGated()
} 