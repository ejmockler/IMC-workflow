#' Perform Spatial Interaction Analysis
#'
#' This entrypoint analyzes spatial interactions between cells and cell types.
#' It includes spatial context detection, patch detection, and statistical
#' interaction testing based on spatial graphs.
#'
#' @return A SpatialExperiment object with spatial interaction annotations.

source("src/core/DependencyManager.R")
source("src/core/ConfigurationManager.R")
source("src/core/Logger.R")
source("src/analysis/SpatialInteraction.R")  # New module to create

# Explicitly load required packages for spatial analysis
library(SpatialExperiment)
library(SingleCellExperiment)
library(SummarizedExperiment)
library(imcRtools)
library(BiocParallel)

runSpatialInteractionAnalysis <- function(
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
  logger <- Logger$new(log_file = "logs/spatialInteractionAnalysis.log", log_level = "INFO")
  
  # Initialize dependencies
  dependencyManager <- DependencyManager$new(logger = logger)
  dependencyManager$validate_environment()
  
  # Get parameters from config or use provided values
  input_file <- input_file %||% configManager$config$interaction_analysis$input_file %||% "output/spe_communities.rds"
  output_dir <- output_dir %||% configManager$config$output$dir %||% "output"
  detect_spatial_context <- detect_spatial_context %||% configManager$config$interaction_analysis$detect_spatial_context %||% TRUE
  sc_k <- sc_k %||% configManager$config$interaction_analysis$sc_k %||% 40
  sc_threshold <- sc_threshold %||% configManager$config$interaction_analysis$sc_threshold %||% 0.9
  filter_sc <- filter_sc %||% configManager$config$interaction_analysis$filter_sc %||% TRUE
  sc_group_by <- sc_group_by %||% configManager$config$interaction_analysis$sc_group_by %||% "sample_id"
  filter_group_threshold <- filter_group_threshold %||% configManager$config$interaction_analysis$filter_group_threshold %||% 3
  filter_cells_threshold <- filter_cells_threshold %||% configManager$config$interaction_analysis$filter_cells_threshold %||% 100
  detect_patches <- detect_patches %||% configManager$config$interaction_analysis$detect_patches %||% TRUE
  patch_celltype <- patch_celltype %||% configManager$config$interaction_analysis$patch_celltype %||% "Tumor"
  expand_by <- expand_by %||% configManager$config$interaction_analysis$expand_by %||% 1
  min_patch_size <- min_patch_size %||% configManager$config$interaction_analysis$min_patch_size %||% 10
  test_interactions <- test_interactions %||% configManager$config$interaction_analysis$test_interactions %||% TRUE
  interaction_method <- interaction_method %||% configManager$config$interaction_analysis$method %||% c("classic", "patch")
  patch_size <- patch_size %||% configManager$config$interaction_analysis$patch_size %||% 3
  
  # Initialize cores if not specified
  if (is.null(n_cores)) {
    n_cores <- configManager$config$interaction_analysis$n_cores %||% max(1, floor(parallel::detectCores() * 0.7))
  }
  
  # Load the SpatialExperiment object
  logger$log_info("Loading SpatialExperiment from: %s", input_file)
  spe <- readRDS(input_file)
  
  # Create the SpatialInteraction object
  spatialInteraction <- SpatialInteraction$new(
    spe = spe,
    logger = logger,
    n_cores = n_cores
  )
  
  # Detect spatial contexts
  if (detect_spatial_context) {
    logger$log_info("Detecting spatial contexts")
    
    # Build spatial context graph if it doesn't exist
    if (!paste0("knn_spatialcontext_graph") %in% SingleCellExperiment::colPairNames(spe)) {
      logger$log_info("Building kNN graph for spatial context detection with k = %d", sc_k)
      spe <- spatialInteraction$buildSpatialContextGraph(k = sc_k)
    }
    
    # Aggregate neighborhoods based on cellular neighborhoods
    spe <- spatialInteraction$aggregateCNNeighbors()
    
    # Detect spatial contexts
    spe <- spatialInteraction$detectSpatialContext(threshold = sc_threshold)
    
    # Filter spatial contexts if requested
    if (filter_sc) {
      logger$log_info("Filtering spatial contexts")
      spe <- spatialInteraction$filterSpatialContext(
        group_by = sc_group_by,
        group_threshold = filter_group_threshold,
        cells_threshold = filter_cells_threshold
      )
    }
  }
  
  # Detect patches
  if (detect_patches) {
    logger$log_info("Detecting patches for cell type: %s", patch_celltype)
    
    # Check if the specified cell type exists in the data
    if (!"celltype" %in% colnames(SummarizedExperiment::colData(spe))) {
      logger$log_warning("No 'celltype' column found. Cannot detect patches.")
      logger$log_info("Skipping patch detection.")
    } else {
      # Get available cell types
      available_celltypes <- unique(as.character(spe$celltype))
      logger$log_info("Available cell types: %s", paste(available_celltypes, collapse=", "))
      
      # Check if the patch_celltype exists in the data
      if (!patch_celltype %in% available_celltypes) {
        logger$log_warning("Specified patch_celltype '%s' not found in the data", patch_celltype)
        
        # Use the most abundant cell type as fallback
        cell_type_counts <- table(spe$celltype)
        most_abundant <- names(cell_type_counts)[which.max(as.numeric(cell_type_counts))]
        
        logger$log_info("Using most abundant cell type '%s' instead", most_abundant)
        patch_celltype <- most_abundant
      }
      
      # Count cells of this type
      patch_cells <- spe$celltype == patch_celltype
      num_patch_cells <- sum(patch_cells)
      logger$log_info("Found %d cells of type '%s' for patch detection", num_patch_cells, patch_celltype)
      
      if (num_patch_cells > min_patch_size) {
        # Proceed with patch detection
        logger$log_info("Detecting %s patches (expand by = %d, min size = %d)", 
                       patch_celltype, expand_by, min_patch_size)
        
        tryCatch({
          spe <- spatialInteraction$detectPatches(
            patch_cells = patch_cells,
            expand_by = expand_by,
            min_patch_size = min_patch_size
          )
          
          # Calculate minimum distance to patches
          logger$log_info("Calculating minimum distance to patches")
          spe <- spatialInteraction$calculateDistToPatches()
        }, error = function(e) {
          logger$log_error("Error during patch detection: %s", e$message)
          logger$log_info("Skipping patch detection due to error")
        })
      } else {
        logger$log_warning("Not enough cells (%d) of type '%s' for patch detection. Need at least %d cells.", 
                          num_patch_cells, patch_celltype, min_patch_size)
        logger$log_info("Skipping patch detection due to insufficient cells")
      }
    }
  }
  
  # Test interactions
  if (test_interactions) {
    logger$log_info("Testing cell type interactions")
    
    results_list <- list()
    
    if ("classic" %in% interaction_method) {
      logger$log_info("Running classic interaction testing")
      results_classic <- spatialInteraction$testCelltypeInteractions(method = "classic")
      results_list[["classic"]] <- results_classic
    }
    
    if ("patch" %in% interaction_method) {
      logger$log_info("Running patch-based interaction testing with patch size %d", patch_size)
      results_patch <- spatialInteraction$testCelltypeInteractions(
        method = "patch",
        patch_size = patch_size
      )
      results_list[["patch"]] <- results_patch
    }
    
    # Save interaction results
    interaction_file <- file.path(output_dir, "celltype_interactions.rds")
    saveRDS(results_list, interaction_file)
    logger$log_info("Saved interaction testing results to: %s", interaction_file)
  }
  
  # Save the SpatialExperiment object with spatial interaction annotations
  output_file <- file.path(output_dir, "spe_spatial_interactions.rds")
  saveRDS(spe, output_file)
  logger$log_info("Saved SpatialExperiment with spatial interaction annotations to: %s", output_file)
  
  if (memory_limit > 0) {
    gc_limit <- memory_limit * 1024 * 1024  # Convert to bytes
    message(sprintf("Setting memory limit to %d MB", memory_limit))
    utils::mem.limit(gc_limit)
  }
  
  invisible(spe)
}

if (interactive() || identical(environment(), globalenv())) {
  spe <- runSpatialInteractionAnalysis()
}
