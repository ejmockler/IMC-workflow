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
          spe = "/Users/noot/Documents/IMC/data/spe.rds",
          images = "data/images.rds",
          masks = "data/masks.rds",
          panel = "data/panel.csv",
          imc = "data/imc"
        ),
        analysis_params = list(
          k_neighbors = 6,
          distance_threshold = 50,
          max_points = 5000
        ),
        visualization_params = list(
          width = 10,
          height = 8,
          dpi = 300
        ),
        output = list(
          dir = "output",
          save_plots = TRUE,
          save_data = TRUE
        )
      )
    },
    
    #' Merge user config with defaults
    #' @param user_config User provided configuration
    merge_with_defaults = function(user_config) {
      if (is.null(user_config)) {
        # When no user configuration is provided, don't return the config list directly.
        # Just leave the default config in place and return the ConfigurationManager object.
        return(invisible(self))
      }
      
      # Recursive merge of user config with defaults
      merged <- modifyList(self$config, user_config, keep.null = FALSE)
      self$validate_config(merged)
      self$config <- merged
      invisible(self)
    },
    
    #' Validate configuration
    validate_config = function(config) {
      # Validate paths
      required_paths <- c("spe", "images", "panel")
      missing_paths <- required_paths[!required_paths %in% names(config$paths)]
      if (length(missing_paths) > 0) {
        stop("Missing required paths: ", paste(missing_paths, collapse = ", "))
      }
      
      # Validate analysis parameters
      if (!is.null(config$analysis_params$k_neighbors)) {
        if (config$analysis_params$k_neighbors < 1) {
          stop("k_neighbors must be positive")
        }
      }
      
      # Validate output configuration
      if (!is.null(config$output$dir)) {
        if (!dir.exists(config$output$dir)) {
          dir.create(config$output$dir, recursive = TRUE)
        }
      }
      
      invisible(config)
    }
  )
) 