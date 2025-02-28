#' Configuration management for spatial analysis pipeline
#' @description Handles configuration validation, defaults, and merging of user settings.
#' This class provides a centralized configuration system for the entire analysis pipeline.
#'
#' @section Configuration Structure:
#' The configuration is organized into logical sections including paths,
#' analysis parameters, visualization settings, and module-specific parameters.
#'
#' @examples
#' # Basic usage
#' configManager <- ConfigurationManager$new()
#' 
#' # Custom configuration
#' user_config <- list(
#'   paths = list(steinbock_data = "my/custom/path"),
#'   batch_correction = list(batch_variable = "patient_id")
#' )
#' configManager$merge_with_defaults(user_config)
ConfigurationManager <- R6::R6Class("ConfigurationManager",
  public = list(
    #' @field config Current configuration
    config = NULL,
    
    #' Initialize with optional base configuration
    #' @description Creates a new ConfigurationManager with default settings
    initialize = function() {
      self$config <- self$get_default_config()
    },
    
    #' Get default configuration settings
    #' @description Returns the default configuration for the analysis pipeline
    #' @return A list with nested configuration parameters
    get_default_config = function() {
      list(
        # Data path configurations
        paths = list(
          # Directory with the steinbock processed single-cell data
          steinbock_data   = "data/", 
          
          # Directory with multi-channel images
          steinbock_images = "data/img/",
          
          # Directory with segmentation masks
          steinbock_masks  = "data/masks/",
          
          # The panel CSV file containing channel metadata
          panel            = "data/panel.csv",
          
          # External metadata file for sample-level annotation
          metadata_annotation = "data/Data_annotations_Karen/Metadata-Table 1.csv"
        ),
        
        # General analysis parameters used across multiple modules
        analysis_params = list(
          # Number of neighbors for spatial analyses (e.g., cell neighborhoods)
          k_neighbors         = 6,
          
          # Maximum distance (in pixels) to consider cells as neighbors
          distance_threshold  = 50,
          
          # Maximum points to use for visualizations to prevent memory issues
          max_points          = 5000
        ),
        
        # Visualization settings for plots and figures
        visualization_params = list(
          # Default width for saved plots (in inches)
          width  = 10,
          
          # Default height for saved plots (in inches)
          height = 8,
          
          # Resolution for raster outputs (PNG, TIFF)
          dpi    = 300
        ),
        
        # Output settings for saving analysis results
        output = list(
          # Directory where all results will be saved
          dir         = "output",
          
          # Whether to save plots generated during analysis
          save_plots  = TRUE,
          
          # Whether to save intermediate data objects
          save_data   = TRUE
        ),
        
        # Batch correction settings for sample/patient integration
        batch_correction = list(
          # Column in SPE metadata used to identify batches
          batch_variable = "sample_id",
          
          # Random seed for reproducible results
          seed = 220228,
          
          # Number of principal components to use for batch correction
          num_pcs = 50
        ),
        
        # Cell phenotyping parameters for clustering
        phenotyping = list(
          # Connectivity parameter for Rphenoannoy/Rphenograph clustering
          # Controls the granularity of clustering (higher = more clusters)
          k_nearest_neighbors = 45,
          
          # Random seed for reproducible clustering
          seed = 220619,
          
          # Whether to use batch-corrected embedding for clustering
          use_corrected_embedding = TRUE,
          
          # Whether to use approximate nearest neighbors (faster but less precise)
          use_approximate_nn = TRUE,
          
          # Number of CPU cores to use for parallel processing (1 = no parallelization)
          n_cores = 1
        ),
        
        # Parameters for marker analysis without segmentation
        marker_analysis = list(
          # Input file path for image data (NULL = use default path)
          input_file = NULL,
          
          # Number of pixels to sample for analysis (controls memory usage)
          n_pixels = 50000000,
          
          # Whether to transform data (e.g., log, arcsinh)
          transformation = TRUE,
          
          # Whether to save visualization plots
          save_plots = TRUE,
          
          # Memory limit in MB (0 = no limit)
          memory_limit = 0,
          
          # Number of CPU cores for parallel processing
          n_cores = 1
        )
      )
    },
    
    #' Merge user config with defaults
    #' @description Combines user-provided configuration with default settings
    #' @param user_config User provided configuration list
    #' @return The ConfigurationManager instance (for method chaining)
    merge_with_defaults = function(user_config) {
      if (is.null(user_config)) {
        # When no user configuration is provided, keep the default configuration.
        return(invisible(self))
      }
      
      # Recursive merge of user config with defaults.
      merged <- modifyList(self$config, user_config, keep.null = FALSE)
      self$validate_config(merged)
      self$config <- merged
      invisible(self)
    },
    
    #' Validate configuration
    #' @description Ensures configuration parameters are valid and consistent
    #' @param config The configuration list to validate
    #' @return The validated configuration (invisibly)
    validate_config = function(config) {
      # Ensure required input paths are present.
      required_paths <- c("steinbock_data", "steinbock_images", "steinbock_masks", "panel")
      missing_paths <- required_paths[!required_paths %in% names(config$paths)]
      if (length(missing_paths) > 0) {
        stop("Missing required paths in configuration: ", paste(missing_paths, collapse = ", "))
      }
      
      # Validate that input directories exist.
      if (!dir.exists(config$paths$steinbock_data)) {
        stop("The specified steinbock_data directory does not exist: ", config$paths$steinbock_data)
      }
      
      if (!dir.exists(config$paths$steinbock_images)) {
        stop("The specified steinbock_images directory does not exist: ", config$paths$steinbock_images)
      }
      
      if (!dir.exists(config$paths$steinbock_masks)) {
        stop("The specified steinbock_masks directory does not exist: ", config$paths$steinbock_masks)
      }
      
      # Validate that the panel file exists.
      if (!file.exists(config$paths$panel)) {
        stop("Panel file not found: ", config$paths$panel)
      }
      
      # Validate general analysis parameters
      if (!is.null(config$analysis_params$k_neighbors)) {
        if (config$analysis_params$k_neighbors < 1) {
          stop("analysis_params.k_neighbors must be positive")
        }
      }
      
      if (!is.null(config$analysis_params$distance_threshold)) {
        if (config$analysis_params$distance_threshold <= 0) {
          stop("analysis_params.distance_threshold must be positive")
        }
      }
      
      # Validate output configuration: create output directory if it does not exist.
      if (!is.null(config$output$dir)) {
        if (!dir.exists(config$output$dir)) {
          dir.create(config$output$dir, recursive = TRUE)
        }
      }
      
      # Validate phenotyping parameters
      if (!is.null(config$phenotyping$k_nearest_neighbors)) {
        if (config$phenotyping$k_nearest_neighbors < 1) {
          stop("phenotyping.k_nearest_neighbors must be positive")
        }
      }
      
      if (!is.null(config$phenotyping$n_cores)) {
        if (config$phenotyping$n_cores < 1 || !is.numeric(config$phenotyping$n_cores)) {
          stop("phenotyping.n_cores must be a positive integer")
        }
      }
      
      # Validate batch correction parameters
      if (!is.null(config$batch_correction$num_pcs)) {
        if (config$batch_correction$num_pcs < 1) {
          stop("batch_correction.num_pcs must be positive")
        }
      }
      
      # Validate marker analysis parameters
      if (!is.null(config$marker_analysis$n_pixels)) {
        if (config$marker_analysis$n_pixels < 1000) {
          stop("marker_analysis.n_pixels must be at least 1000")
        }
      }
      
      if (!is.null(config$marker_analysis$n_cores)) {
        if (config$marker_analysis$n_cores < 1 || !is.numeric(config$marker_analysis$n_cores)) {
          stop("marker_analysis.n_cores must be a positive integer")
        }
      }
      
      if (!is.null(config$marker_analysis$memory_limit)) {
        if (config$marker_analysis$memory_limit < 0) {
          stop("marker_analysis.memory_limit must be non-negative")
        }
      }
      
      invisible(config)
    }
  )
)
