# Load core modules
source("src/core/ConfigurationManager.R")
# source("src/core/Logger.R") # Comment out problematic logger
source("src/core/DependencyManager.R")
source("src/core/ResultsManager.R")
source("src/core/Reporter.R")
source("src/core/Dataloader.R")
source("src/core/MetadataHarmonizer.R")
source("src/core/ImageProcessor.R")

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
source("src/visualization/channel_overlay_visualization.R")  # Add the new channel overlay module

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

#' Run the IMC workflow
#'
#' Main entry point for the IMC workflow
#'
#' @param config_file Path to configuration file
#' @param workflow_type Type of workflow to run
#' @param steps Analysis steps to run
#' @param output_dir Output directory
#' @return List of SpatialExperiment objects
runIMCWorkflow <- function(
  config_file = NULL,
  workflow_type = c("unsupervised", "gated", "both"),
  steps = c("load", "batch_correction", "phenotyping", "spatial", "visualize", "report", "process_images", "channel_overlay"),
  output_dir = NULL
) {
  # Match arguments
  workflow_type <- match.arg(workflow_type)
  
  # Initialize logger
  logger <- simple_logger()
  logger$info("Starting IMC Workflow")
  
  # Debug statements to track execution
  cat("DEBUG: Step 1 - Initializing modules\n")
  
  # Initialize configuration
  config <- ConfigurationManager$new()
  
  # Set output directory if provided
  if (!is.null(output_dir)) {
    config$config$paths$output_dir <- output_dir
  }
  
  # Load config file if provided
  if (!is.null(config_file)) {
    if (file.exists(config_file)) {
      user_config <- yaml::read_yaml(config_file)
      config$merge_with_defaults(user_config)
    } else {
      logger$warning(paste("Config file not found:", config_file))
    }
  }
  
  # Initialize preprocessing components
  logger$info("Initializing preprocessing components")
  preprocessing_manager <- PreprocessingManager$new(config = config, logger = logger)
  
  # Initialize spatial analysis components
  logger$info("Loading spatial analysis components")
  spatial_analyzer <- SpatialAnalyzer$new(config = config, logger = logger)
  
  # Debug statements
  cat("DEBUG: Step 2 - Creating MarkerAnalyzer\n")
  
  # Only initialize MarkerAnalyzer if we use it, and we'll initialize it
  # when we have images for segmentation-free analysis
  marker_analyzer <- NULL  # Will be initialized when needed
  
  # Debug statements
  cat("DEBUG: Step 3 - Creating CellPhenotyper\n")
  
  # Initialize cell phenotyper
  cell_phenotyper <- CellPhenotyper$new(config = config, logger = logger)
  logger$info("CellPhenotyper initialized")
  
  # Debug statements
  cat("DEBUG: Step 4 - Creating Reporter\n")
  
  # Initialize reporter
  reporter <- Reporter$new(config = config, logger = logger)
  
  # Debug statements
  cat("DEBUG: Step 5 - Creating ImageProcessor\n")
  
  # Initialize image processor
  image_processor <- ImageProcessor$new(config = config, logger = logger)
  
  # Debug statements
  cat("DEBUG: Step 6 - Creating MetadataHarmonizer\n")
  
  # Initialize metadata harmonizer
  metadata_harmonizer <- MetadataHarmonizer$new(config = config, logger = logger)
  
  # Debug statements
  cat("DEBUG: Step 7 - Creating visualization config\n")
  
  # Initialize visualization configuration
  viz_config <- list(
    dimensions = list(
      width = config$config$visualization$width,
      height = config$config$visualization$height,
      dpi = config$config$visualization$dpi
    ),
    max_points = config$config$visualization$max_points,
    color_scheme = config$config$visualization$color_scheme
  )
  
  # Debug statements
  cat("DEBUG: Step 8 - Starting data loading\n")
  
  # Debug the configuration paths
  if (!is.null(config) && !is.null(config$config) && !is.null(config$config$paths)) {
    cat("DEBUG: config$config$paths exists\n")
    cat("DEBUG: data_dir =", config$config$paths$data_dir, "\n")
    cat("DEBUG: panels$default =", config$config$paths$panels$default, "\n")
  } else {
    cat("DEBUG: config$config$paths is NULL or incomplete\n")
  }
  
  # Initialize data loader
  # Need to create a wrapper around config to match what DataLoader expects
  data_loader_config <- list(
    paths = config$config$paths
  )
  data_loader <- DataLoader$new(config = data_loader_config, logger = logger)
  
  # Store SpatialExperiment objects
  spe_objects <- list()
  
  # Debug statements
  cat("DEBUG: Step 9 - Running unsupervised workflow\n")
  
  # Run unsupervised workflow
  if (workflow_type %in% c("unsupervised", "both")) {
    logger$info("Running unsupervised cell phenotyping workflow")
    
    # Debug statements
    cat("DEBUG: Step 10 - Calling loadUnsupervisedData\n")
    
    # Load unsupervised data within a tryCatch block to catch errors
    tryCatch({
      spe_objects$unsupervised <- data_loader$loadUnsupervisedData()
      
      # Debug statements
      cat("DEBUG: Step 11 - Harmonizing metadata\n")
      
      # Harmonize metadata
      spe_objects$unsupervised <- metadata_harmonizer$harmonize(spe_objects$unsupervised)
      
      # Debug statements
      cat("DEBUG: Step 12 - Preprocessing quality control\n")
      
      # Use preprocessing manager for quality control
      spe_objects$unsupervised <- preprocessing_manager$preprocess(
        spe_objects$unsupervised, 
        is_gated = FALSE,
        steps = "quality_control"
      )
      
      # Debug statements
      cat("DEBUG: Step 13 - Checking if batch_correction is in steps\n")
    }, error = function(e) {
      cat("DEBUG: Error in data loading or preprocessing:", e$message, "\n")
      logger$error(paste("Failed to load or preprocess data:", e$message))
      # Create an empty SPE to avoid downstream errors
      spe_objects$unsupervised <- NULL
    })
  }
  
  # Batch correction
  if ("batch_correction" %in% steps && "unsupervised" %in% names(spe_objects)) {
    logger$info("Running batch correction on unsupervised data")
    
    # Debug statements
    cat("DEBUG: Step 14 - Running batch correction\n")
    
    # Use preprocessing manager for batch correction
    spe_objects$unsupervised <- preprocessing_manager$preprocess(
      spe_objects$unsupervised,
      is_gated = FALSE,
      steps = "batch_correction"
    )
  }
  
  # Phenotyping
  if ("phenotyping" %in% steps) {
    # Debug statements
    cat("DEBUG: Step 15 - Starting phenotyping\n")
    
    phenotyping_mode <- config$config$phenotyping$mode %||% "segmentation"
    
    if (phenotyping_mode == "segmentation" && "unsupervised" %in% names(spe_objects)) {
      logger$info("Running segmentation-based cell phenotyping on unsupervised data")
      
      # Debug statements
      cat("DEBUG: Step 16 - Running segmentation-based phenotyping\n")
      
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
        # Debug statements
        cat("DEBUG: Step 17 - Creating phenotype composition plot\n")
        
        # Use visualization manager for phenotype visualizations
        tryCatch({
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
        }, error = function(e) {
          cat("DEBUG: Error in phenotype composition plot:", e$message, "\n")
        })
      }
    }
    
    if (phenotyping_mode == "segmentation_free") {
      logger$info("Running segmentation-free phenotyping")
      
      # Debug statements
      cat("DEBUG: Step 18 - Running segmentation-free phenotyping\n")
      
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
    # Debug statements
    cat("DEBUG: Step 19 - Starting spatial analysis\n")
    
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
    # Debug statements
    cat("DEBUG: Step 20 - Starting visualization\n")
    
    logger$info("Generating visualizations")
    
    # Ensure output directory exists
    visualization_dir <- file.path(config$config$paths$output_dir, "visualizations")
    
    # Debug statements
    cat("DEBUG: Visualization directory path:", visualization_dir, "\n")
    cat("DEBUG: Output directory exists:", dir.exists(config$config$paths$output_dir), "\n")
    
    tryCatch({
      if (!dir.exists(visualization_dir)) {
        cat("DEBUG: Creating visualization directory\n")
        dir.create(visualization_dir, recursive = TRUE)
        logger$info(paste("Created visualization directory:", visualization_dir))
      }
      
      # Debug statement to show the output directory
      cat("DEBUG: Visualization output directory:", visualization_dir, "\n")
      
      for (workflow in names(spe_objects)) {
        # Skip if the SPE object is NULL
        if (is.null(spe_objects[[workflow]])) {
          cat("DEBUG: Skipping visualization for workflow", workflow, "- data is NULL\n")
          logger$warning(paste("Skipping visualization for", workflow, "workflow - no data available"))
          next
        }
        
        # Debug the workflow being processed
        cat("DEBUG: Creating visualizations for workflow:", workflow, "\n")
        
        # Determine plot types to generate
        plot_types <- c("markers", "phenotypes", "spatial")
        
        # Add channel overlay if requested
        if ("channel_overlay" %in% steps) {
          plot_types <- c(plot_types, "channel_overlays")
        }
        
        # Use visualization manager for all visualizations within try-catch for better error reporting
        tryCatch({
          create_visualizations(
            spe = spe_objects[[workflow]],
            workflow_type = workflow,
            plot_types = plot_types,
            output_dir = visualization_dir,
            config = viz_config
          )
          logger$info(paste("Created visualizations for", workflow, "workflow"))
        }, error = function(e) {
          cat("DEBUG: Error in create_visualizations:", e$message, "\n")
          logger$error(paste("Failed to create visualizations for", workflow, "workflow:", e$message))
        })
      }
    }, error = function(e) {
      cat("DEBUG: Error in visualization step:", e$message, "\n")
    })
  }
  
  # Reporting
  if ("report" %in% steps) {
    # Debug statements
    cat("DEBUG: Step 21 - Starting reporting\n")
    
    logger$info("Generating analysis report")
    report_path <- reporter$generateReport(spe_objects)
    logger$info(paste("Report generated at:", report_path))
  }
  
  # Process images if needed
  if ("process_images" %in% steps) {
    # Debug statements
    cat("DEBUG: Step 22 - Starting image processing\n")
    
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
      # Debug statements
      cat("DEBUG: Step 23 - Assessing mask quality\n")
      
      mask_quality <- image_processor$assessMaskQuality(masks_list)
      quality_path <- file.path(config$config$paths$output_dir, "mask_quality.csv")
      
      # Debug statements
      cat("DEBUG: Quality path:", quality_path, "\n")
      cat("DEBUG: Output directory exists:", dir.exists(config$config$paths$output_dir), "\n")
      
      tryCatch({
        write.csv(mask_quality, quality_path, row.names = FALSE)
        logger$info(paste("Mask quality assessment saved to:", quality_path))
      }, error = function(e) {
        cat("DEBUG: Error writing mask quality CSV:", e$message, "\n")
      })
    }
  }
  
  # Debug statements
  cat("DEBUG: Step 24 - Workflow complete\n")
  
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
  # Wrap entire execution in tryCatch to get better error information
  tryCatch({
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
    cat("Trace:\n")
    traceback()
  })
}

# Execute debug workflow when sourcing the file
debug_workflow()

# ====================================
# CHANNEL OVERLAY DEBUGGING
# ====================================

#' Run channel overlay visualization in debug mode
#' 
#' This function provides a simple way to test the channel overlay visualization
#' It uses the saved images and masks from a previous run
#' 
#' @param output_dir Directory to save visualizations to
#' @return Invisibly returns path to output directory
debug_channel_overlay <- function(
  output_dir = "debug_output/visualizations/channel_overlays"
) {
  # Wrap entire execution in tryCatch to get better error information
  tryCatch({
    # Simple header to show we're debugging
    cat("============================================\n")
    cat("     DEBUGGING CHANNEL OVERLAY VISUALIZATION\n")
    cat("============================================\n\n")
    
    # Create debug output directory
    if (!dir.exists(output_dir)) {
      dir.create(output_dir, recursive = TRUE)
      cat(paste("Created output directory:", output_dir, "\n"))
    }
    
    # Check for images and masks
    images_path <- "debug_output/images.rds"
    masks_path <- "debug_output/masks.rds"
    
    if (!file.exists(images_path)) {
      stop("Images file not found at ", images_path)
    }
    
    if (!file.exists(masks_path)) {
      warning("Masks file not found at ", masks_path, " - single cell preview will be skipped")
      masks <- NULL
    } else {
      masks <- readRDS(masks_path)
      cat("Loaded masks from ", masks_path, "\n")
    }
    
    # Load images
    images <- readRDS(images_path)
    cat("Loaded images from ", images_path, "\n")
    
    # Source required files
    source("src/visualization/core_visualization.R")
    source("src/visualization/channel_overlay_visualization.R")
    
    # Create channel overlays
    result_paths <- create_channel_overlay_visualization(
      images = images,
      masks = masks,
      spe = NULL,
      output_dir = output_dir,
      channels_to_highlight = NULL,
      max_cells = 25,
      color_scheme = "viridis",
      width = 12,
      height = 10,
      dpi = 300
    )
    
    cat("\n============================================\n")
    cat("          CHANNEL OVERLAY COMPLETE          \n")
    cat("============================================\n")
    cat(paste("Results saved to:", output_dir, "\n"))
    
    invisible(output_dir)
  }, error = function(e) {
    cat("\n============================================\n")
    cat("          ERROR IN OVERLAY DEBUG            \n")
    cat("============================================\n")
    cat(paste("Error:", e$message, "\n"))
    cat("Trace:\n")
    traceback()
  })
}

# uncomment to run just the channel overlay debug
# debug_channel_overlay()