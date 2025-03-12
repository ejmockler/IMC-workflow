# Load core modules
source("src/core/ConfigurationManager.R")
# source("src/core/Logger.R") # Comment out problematic logger
source("src/core/DependencyManager.R")
source("src/core/ResultsManager.R")
source("src/core/Reporter.R")
source("src/core/Dataloader.R")
source("src/core/MetadataHarmonizer.R")

# Analysis modules - only load top-level managers
source("src/analysis/spatial/SpatialAnalyzer.R")  # Will load its own components
source("src/analysis/markers/MarkerAnalyzer.R")
source("src/analysis/phenotyping/CellPhenotyper.R")
source("src/analysis/preprocessing/PreprocessingManager.R")  # Use the new manager
source("src/analysis/validation/ComparativeAnalysis.R")

# Visualization modules
source("src/visualization/core_visualization.R")
source("src/visualization/visualization_wrappers.R") 
source("src/visualization/marker_visualization.R")
source("src/visualization/phenotype_visualization.R")
source("src/visualization/spatial_visualization.R")
source("src/visualization/gated_cell_visualization.R")

# Load common utilities
`%||%` <- function(x, y) if (is.null(x)) y else x

# Simple logger functions for debugging
simple_logger <- function() {
  log_path <- NULL
  
  # Create a list to store the functions
  logger <- list()
  
  # Set the log file
  logger$set_log_file <- function(path) {
    log_path <<- path
    if (!is.null(log_path)) {
      dir.create(dirname(log_path), recursive = TRUE, showWarnings = FALSE)
      cat(paste("[", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "] Log initialized\n"), 
          file = log_path)
    }
  }
  
  # Log functions
  logger$info <- function(message) {
    log_entry <- paste("[", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "] INFO:", message)
    cat(paste(log_entry, "\n"))
    if (!is.null(log_path)) {
      cat(paste(log_entry, "\n"), file = log_path, append = TRUE)
    }
  }
  
  logger$warning <- function(message) {
    log_entry <- paste("[", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "] WARNING:", message)
    cat(paste(log_entry, "\n"))
    if (!is.null(log_path)) {
      cat(paste(log_entry, "\n"), file = log_path, append = TRUE)
    }
  }
  
  logger$error <- function(message) {
    log_entry <- paste("[", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "] ERROR:", message)
    cat(paste(log_entry, "\n"))
    if (!is.null(log_path)) {
      cat(paste(log_entry, "\n"), file = log_path, append = TRUE)
    }
  }
  
  return(logger)
}

#' Run the IMC workflow with specified options
#'
#' @param config_file Path to configuration file (optional)
#' @param workflow_type Type of workflow to run (unsupervised, gated, or both)
#' @param steps Analysis steps to run
#' @param output_dir Output directory (optional, overrides config)
#' @return List of SpatialExperiment objects
runIMCWorkflow <- function(
  config_file = NULL,
  workflow_type = c("unsupervised", "gated", "both"),
  steps = c("load", "batch_correction", "phenotyping", "spatial", "visualize", "report"),
  output_dir = NULL
) {
  # Initialize
  workflow_type <- match.arg(workflow_type)
  config <- ConfigurationManager$new()
  if (!is.null(config_file)) config$merge_with_defaults(config_file)
  
  # Override output directory if provided
  if (!is.null(output_dir)) config$config$paths$output_dir <- output_dir
  
  # Create logger
  logger <- simple_logger()
  log_file <- file.path(config$config$paths$output_dir, "logs", "workflow.log")
  logger$set_log_file(log_file)
  logger$info("Starting IMC Workflow")
  
  # Initialize all required modules
  data_loader <- DataLoader$new(config$config, logger)
  preprocessing_manager <- PreprocessingManager$new(config$config, logger)
  spatial_analyzer <- SpatialAnalyzer$new(config$config, logger)
  marker_analyzer <- MarkerAnalyzer$new(
    output_dir = config$config$paths$output_dir,
    logger = logger
  )
  cell_phenotyper <- CellPhenotyper$new(config$config, logger)
  reporter <- Reporter$new(config$config, logger)
  image_processor <- ImageProcessor$new(config$config, logger)
  metadata_harmonizer <- MetadataHarmonizer$new(config$config, logger)

  # Create a visualization config from the main config
  viz_config <- get_default_viz_config()
  viz_config$dimensions$width <- config$config$visualization$plot_dimensions$width %||% 10
  viz_config$dimensions$height <- config$config$visualization$plot_dimensions$height %||% 8
  viz_config$dimensions$dpi <- config$config$visualization$plot_dimensions$dpi %||% 300
  viz_config$point_size <- config$config$visualization$point_size %||% 1.5
  
  # Run selected workflow
  spe_objects <- list()
  
  # Data loading
  if ("load" %in% steps) {
    if (workflow_type %in% c("unsupervised", "both")) {
      logger$info("Running unsupervised cell phenotyping workflow")
      spe_objects$unsupervised <- data_loader$loadUnsupervisedData()
      spe_objects$unsupervised <- metadata_harmonizer$harmonize(spe_objects$unsupervised)
      
      # Use preprocessing manager for quality control
      spe_objects$unsupervised <- preprocessing_manager$preprocess(
        spe_objects$unsupervised, 
        is_gated = FALSE,
        steps = "quality_control"
      )
    }
    
    if (workflow_type %in% c("gated", "both")) {
      logger$info("Running gated cell workflow")
      spe_objects$gated <- data_loader$loadGatedCellData()
      spe_objects$gated <- metadata_harmonizer$harmonize(spe_objects$gated)
      
      # Use preprocessing manager for quality control
      spe_objects$gated <- preprocessing_manager$preprocess(
        spe_objects$gated, 
        is_gated = TRUE,
        steps = "quality_control"
      )
    }
  }
  
  # Batch correction
  if ("batch_correction" %in% steps && "unsupervised" %in% names(spe_objects)) {
    logger$info("Running batch correction on unsupervised data")
    
    # Use preprocessing manager for batch correction
    spe_objects$unsupervised <- preprocessing_manager$preprocess(
      spe_objects$unsupervised,
      is_gated = FALSE,
      steps = "batch_correction"
    )
  }
  
  # Phenotyping
  if ("phenotyping" %in% steps) {
    phenotyping_mode <- config$config$phenotyping$mode %||% "segmentation"
    
    if (phenotyping_mode == "segmentation" && "unsupervised" %in% names(spe_objects)) {
      logger$info("Running segmentation-based cell phenotyping on unsupervised data")
      
      # Use the CellPhenotyper class for segmentation-based phenotyping
      spe_objects$unsupervised <- cell_phenotyper$phenotypeCells(
        spe = spe_objects$unsupervised,
        use_corrected_embedding = config$config$phenotyping$segmentation$use_corrected_embedding,
        k = config$config$phenotyping$segmentation$k_nearest_neighbors,
        run_both_embeddings = config$config$phenotyping$segmentation$run_both_embeddings,
        seed = config$config$system$seed,
        n_cores = config$config$system$n_cores
      )
      
      # Generate visualizations if configured
      if (config$config$visualization$save_plots) {
        # Use visualization manager for phenotype visualizations
        plot <- create_visualization(
          spe = spe_objects$unsupervised,
          plot_type = "phenotype_composition",
          output_file = file.path(config$config$paths$output_dir, "phenotype_composition.pdf"),
          config = list(
            max_cells = config$config$visualization$max_points
          ),
          phenotype_column = "phenograph_corrected",
          reduction = "UMAP"
        )
      }
    }
    
    if (phenotyping_mode == "segmentation_free") {
      logger$info("Running segmentation-free phenotyping")
      
      # Load images if not already loaded
      images_list <- data_loader$loadImages()
      
      # Use MarkerAnalyzer for segmentation-free analysis
      noseg_results <- marker_analyzer$runImageAwareMarkerAnalysis(
        images_list,
        n_cores = config$config$system$n_cores,
        transform_method = config$config$phenotyping$segmentation_free$transform_method,
        threshold_method = config$config$phenotyping$segmentation_free$threshold_method,
        k_clusters = config$config$phenotyping$segmentation_free$k_clusters
      )
    }
  }
  
  # Spatial analysis
  if ("spatial" %in% steps) {
    logger$info("Running spatial analysis")
    
    if (!is.null(spe_objects$unsupervised)) {
      logger$info("Analyzing unsupervised data")
      spe_objects$unsupervised <- spatial_analyzer$analyze(
        spe_objects$unsupervised,
        steps = c("graphs", "communities", "interactions"),
        is_gated = FALSE
      )
    }
    
    if (!is.null(spe_objects$gated)) {
      logger$info("Analyzing gated cell data")
      spe_objects$gated <- spatial_analyzer$analyze(
        spe_objects$gated,
        steps = c("graphs", "communities", "interactions"),
        is_gated = TRUE
      )
    }
  }
  
  # Visualization
  if ("visualize" %in% steps) {
    logger$info("Generating visualizations")
    
    for (workflow in names(spe_objects)) {
      # Use visualization manager for all visualizations
      create_visualizations(
        spe = spe_objects[[workflow]],
        workflow_type = workflow,
        plot_types = "all",
        output_dir = config$config$paths$output_dir,
        config = get_default_viz_config()
      )
    }
  }
  
  # Reporting
  if ("report" %in% steps) {
    logger$info("Generating analysis report")
    report_path <- reporter$generateReport(spe_objects)
    logger$info(paste("Report generated at:", report_path))
  }
  
  # Process images if needed
  if ("process_images" %in% steps) {
    logger$info("Processing images")
    images_list <- data_loader$loadImages()
    masks_list <- data_loader$loadMasks()
    
    # Use the ImageProcessor class for image processing
    processed_data <- image_processor$processImages(
      images_list = images_list,
      masks_list = masks_list,
      spe = spe_objects$unsupervised
    )
    
    # Assess mask quality if configured
    if (config$config$processing$assess_mask_quality %||% TRUE) {
      mask_quality <- image_processor$assessMaskQuality(masks_list)
      quality_path <- file.path(config$config$paths$output_dir, "mask_quality.csv")
      write.csv(mask_quality, quality_path, row.names = FALSE)
      logger$info(paste("Mask quality assessment saved to:", quality_path))
    }
  }
  
  return(spe_objects)
}

# ====================================
# INTERACTIVE ENTRY POINT
# ====================================

#' Run the IMC workflow in debug mode
#' 
#' This function provides a simple way to run the workflow for debugging/development
#' Call this function to test the workflow after sourcing this file
#' 
#' @param workflow_type Type of workflow to run
#' @param steps Analysis steps to run
#' @return List of SpatialExperiment objects (invisibly)
debug_workflow <- function(
  workflow_type = "unsupervised",
  steps = c("load", "visualize")
) {
  # Simple header to show we're debugging
  cat("============================================\n")
  cat("          RUNNING IN DEBUG MODE             \n")
  cat("============================================\n\n")
  
  # Create debug output directory
  debug_output_dir <- file.path(getwd(), "debug_output")
  if (!dir.exists(debug_output_dir)) {
    dir.create(debug_output_dir, recursive = TRUE)
    cat(paste("Created debug output directory:", debug_output_dir, "\n"))
  }
  
  # Create logs subdirectory
  logs_dir <- file.path(debug_output_dir, "logs")
  if (!dir.exists(logs_dir)) {
    dir.create(logs_dir, recursive = TRUE)
  }
  
  # Run the workflow
  cat(paste("Workflow type:", workflow_type, "\n"))
  cat(paste("Steps:", paste(steps, collapse = ", "), "\n"))
  
  # Store results in global environment for inspection
  tryCatch({
    .GlobalEnv$debug_results <- runIMCWorkflow(
      config_file = NULL,
      workflow_type = workflow_type,
      steps = steps,
      output_dir = debug_output_dir
    )
    
    cat("\n============================================\n")
    cat("          DEBUG RUN COMPLETE                \n")
    cat("============================================\n")
    cat("Results stored in 'debug_results' variable\n")
    
    invisible(.GlobalEnv$debug_results)
  }, error = function(e) {
    cat("\n============================================\n")
    cat("          ERROR IN DEBUG RUN                \n")
    cat("============================================\n")
    cat(paste("Error:", e$message, "\n"))
  })
}

# Execute debug workflow when sourcing the file
debug_workflow()