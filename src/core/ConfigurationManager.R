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
        # Consolidated paths section
        paths = list(
          data_dir = "data/",
          image_dir = "data/img/",
          masks_dir = "data/masks/",
          gated_cells_dir = "data/gated_cells/",
          output_dir = "output/",
          panels = list(
            default = "data/panel.csv",
            alternative = "data/panel2.csv"
          ),
          metadata = "data/Data_annotations_Karen/Metadata-Table 1.csv"
        ),
        
        # Consolidated cell analysis parameters
        cell_analysis = list(
          # General parameters
          transformation = TRUE,
          qc_filter = TRUE,
          min_cell_area = 10,
          
          # Combined phenotyping parameters (for both unsupervised and gated)
          use_gated_cells = FALSE,  # Switch between workflows
          gated_cell_files = list(
            "immune" = "Immune_cells.rds",
            "endothelial" = "Endothelial_cells.rds",
            "fibroblasts" = "Fibroblasts.rds",
            "m2_macrophages" = "M2_Macrophages.rds",
            "non_m2_macrophages" = "Non_M2_Macrophages.rds"
          ),
          
          # Unsupervised parameters
          k_nearest_neighbors = 45,
          batch_correction = TRUE,
          batch_variable = "sample_id",
          n_pcs = 50
        ),
        
        # Consolidated spatial analysis parameters
        spatial_analysis = list(
          build_graphs = TRUE,
          graph_types = c("knn", "expansion", "delaunay"),
          distance_threshold = 50,
          k_neighbors = 20,
          
          # Communities
          community_methods = c("graph_based", "celltype_aggregation"),
          size_threshold = 10,
          
          # Interactions
          test_interactions = TRUE,
          interaction_methods = c("classic", "patch")
        ),
        
        # Visualization and reporting
        visualization = list(
          save_plots = TRUE,
          max_points = 5000,
          plot_dimensions = c(width = 10, height = 8, dpi = 300),
          cell_size = 1,
          color_scheme = "viridis"
        ),
        
        # Simplified system parameters
        system = list(
          n_cores = 1,
          memory_limit = 0,
          seed = 220619,
          save_intermediate = TRUE,
          verbose = TRUE
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
      required_paths <- c("data_dir", "image_dir", "masks_dir", "panels$default")
      missing_paths <- required_paths[!required_paths %in% names(config$paths)]
      if (length(missing_paths) > 0) {
        stop("Missing required paths in configuration: ", paste(missing_paths, collapse = ", "))
      }
      
      # Validate that input directories exist.
      if (!dir.exists(config$paths$data_dir)) {
        stop("The specified data_dir directory does not exist: ", config$paths$data_dir)
      }
      
      if (!dir.exists(config$paths$image_dir)) {
        stop("The specified image_dir directory does not exist: ", config$paths$image_dir)
      }
      
      if (!dir.exists(config$paths$masks_dir)) {
        stop("The specified masks_dir directory does not exist: ", config$paths$masks_dir)
      }
      
      # Validate that the panel file exists.
      if (!file.exists(config$paths$panels$default)) {
        stop("Panel file not found: ", config$paths$panels$default)
      }
      
      # Validate cell analysis parameters
      if (!is.null(config$cell_analysis$k_nearest_neighbors)) {
        if (config$cell_analysis$k_nearest_neighbors < 1) {
          stop("cell_analysis.k_nearest_neighbors must be positive")
        }
      }
      
      if (!is.null(config$cell_analysis$n_pcs)) {
        if (config$cell_analysis$n_pcs < 1) {
          stop("cell_analysis.n_pcs must be positive")
        }
      }
      
      # Validate spatial analysis parameters
      if (!is.null(config$spatial_analysis$k_neighbors)) {
        if (config$spatial_analysis$k_neighbors < 1) {
          stop("spatial_analysis.k_neighbors must be positive")
        }
      }
      
      if (!is.null(config$spatial_analysis$distance_threshold)) {
        if (config$spatial_analysis$distance_threshold <= 0) {
          stop("spatial_analysis.distance_threshold must be positive")
        }
      }
      
      # Validate visualization parameters
      if (!is.null(config$visualization$max_points)) {
        if (config$visualization$max_points < 1000) {
          stop("visualization.max_points must be at least 1000")
        }
      }
      
      invisible(config)
    }
  )
)
