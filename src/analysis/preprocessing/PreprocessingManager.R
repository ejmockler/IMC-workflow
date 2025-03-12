# PreprocessingManager.R
# High-level manager for preprocessing components

# Load required preprocessing components
source("src/analysis/preprocessing/BatchCorrection.R")
source("src/analysis/preprocessing/QualityControl.R")

#' @import R6
#' @import SpatialExperiment

# Make sure we have the null-coalescing operator available
if (!exists("%||%")) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
}

#' PreprocessingManager class
#' @description Manages preprocessing steps including quality control and batch correction
PreprocessingManager <- R6::R6Class("PreprocessingManager",
  public = list(
    #' @field config Configuration parameters
    config = NULL,
    
    #' @field logger Logger object
    logger = NULL,
    
    #' @field quality_control QualityControl object
    quality_control = NULL,
    
    #' @field batch_corrector BatchCorrection object
    batch_corrector = NULL,
    
    #' @description
    #' Initialize a new PreprocessingManager object
    #' @param config Configuration parameters
    #' @param logger Logger object
    initialize = function(config = NULL, logger = NULL) {
      self$config <- config
      self$logger <- logger %||% Logger$new("PreprocessingManager")
      
      # Initialize components
      self$logger$info("Initializing preprocessing components")
      self$quality_control <- QualityControl$new(config, self$logger)
      self$batch_corrector <- BatchCorrection$new(config, self$logger)
    },
    
    #' @description
    #' Preprocess a SpatialExperiment object
    #' @param spe SpatialExperiment object
    #' @param is_gated Whether the data is from gated cells
    #' @param steps Vector of preprocessing steps to run
    #' @return Preprocessed SpatialExperiment object
    preprocess = function(spe, 
                          is_gated = FALSE,
                          steps = c("quality_control", "batch_correction")) {
      self$logger$info(paste("Preprocessing", ifelse(is_gated, "gated", "unsupervised"), "data"))
      
      # Run quality control if needed
      if ("quality_control" %in% steps) {
        self$logger$info("Running quality control")
        spe <- self$quality_control$processData(spe, is_gated)
        
        # Save intermediate result if configured
        if (!is.null(self$config$paths$output_dir) && self$config$system$save_intermediate %||% TRUE) {
          output_file <- file.path(
            self$config$paths$output_dir, 
            paste0("spe_qc", ifelse(is_gated, "_gated", ""), ".rds")
          )
          saveRDS(spe, output_file)
          self$logger$info(paste("Saved quality-controlled data to", output_file))
        }
      }
      
      # Run batch correction if needed and not gated
      if ("batch_correction" %in% steps && !is_gated) {
        # Check if batch correction is enabled in config
        if (self$config$cell_analysis$batch_correction %||% TRUE) {
          self$logger$info("Running batch correction")
          spe <- self$batch_corrector$runBatchCorrection(spe)
          
          # Save intermediate result if configured
          if (!is.null(self$config$paths$output_dir) && self$config$system$save_intermediate %||% TRUE) {
            output_file <- file.path(
              self$config$paths$output_dir, 
              "spe_batch_corrected.rds"
            )
            saveRDS(spe, output_file)
            self$logger$info(paste("Saved batch-corrected data to", output_file))
          }
        } else {
          self$logger$info("Batch correction disabled in configuration, skipping")
        }
      } else if ("batch_correction" %in% steps && is_gated) {
        self$logger$info("Batch correction not applicable for gated data, skipping")
      }
      
      return(spe)
    },
    
    #' @description
    #' Preprocess multiple SpatialExperiment objects
    #' @param spe_list List of SpatialExperiment objects
    #' @param is_gated Whether the data is from gated cells
    #' @param steps Vector of preprocessing steps to run
    #' @return List of preprocessed SpatialExperiment objects
    batchPreprocess = function(spe_list, 
                              is_gated = FALSE,
                              steps = c("quality_control", "batch_correction")) {
      self$logger$info(paste("Batch preprocessing", length(spe_list), "samples"))
      
      # Process each sample
      result_list <- lapply(names(spe_list), function(sample_name) {
        self$logger$info(paste("Processing sample:", sample_name))
        spe_list[[sample_name]] <- self$preprocess(spe_list[[sample_name]], is_gated, steps)
        return(spe_list[[sample_name]])
      })
      
      names(result_list) <- names(spe_list)
      return(result_list)
    }
  ),
  
  private = list(
    # Helper methods can be added here
  )
) 