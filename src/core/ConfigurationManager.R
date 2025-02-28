#' Configuration management for spatial analysis pipeline
#' @description Handles configuration validation, defaults, and merging
ConfigurationManager <- R6::R6Class("ConfigurationManager",
  public = list(
    #' @field config Current configuration
    config = NULL,
    
    #' Initialize with optional base configuration
    initialize = function() {
      self$config <- self$get_default_config()
    },
    
    #' Get default configuration settings
    get_default_config = function() {
      list(
        paths = list(
          # Directory with the steinbock processed single-cell data.
          steinbock_data   = "data/",
          # Directory with multi-channel images.
          steinbock_images = "data/img/",
          # Directory with segmentation masks.
          steinbock_masks  = "data/masks/",
          # The panel CSV file containing channel metadata.
          panel            = "data/panel.csv",
          # External metadata (for sample-level annotation)
          metadata_annotation = "data/Data_annotations_Karen/Metadata-Table 1.csv"
        ),
        analysis_params = list(
          k_neighbors         = 6,
          distance_threshold  = 50,
          max_points          = 5000
        ),
        visualization_params = list(
          width  = 10,
          height = 8,
          dpi    = 300
        ),
        output = list(
          dir         = "output",
          save_plots  = TRUE,
          save_data   = TRUE
        )
      )
    },
    
    #' Merge user config with defaults
    #' @param user_config User provided configuration list
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
    #' @param config The configuration list to validate.
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
      
      # Validate analysis parameters.
      if (!is.null(config$analysis_params$k_neighbors)) {
        if (config$analysis_params$k_neighbors < 1) {
          stop("k_neighbors must be positive")
        }
      }
      
      # Validate output configuration: create output directory if it does not exist.
      if (!is.null(config$output$dir)) {
        if (!dir.exists(config$output$dir)) {
          dir.create(config$output$dir, recursive = TRUE)
        }
      }
      
      invisible(config)
    }
  )
) 