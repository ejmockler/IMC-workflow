# Data transformation class for marker analysis
# Handles normalization and transformation of pixel data

#' DataTransformer class
#' 
#' Applies various transformations to prepare marker data for analysis
DataTransformer <- R6::R6Class("DataTransformer",
  private = list(
    .logger = NULL,
    
    # Helper method to track performance
    .trackPerformance = function(start_time, stage_name) {
      current_time <- Sys.time()
      elapsed <- as.numeric(difftime(current_time, start_time, units = "secs"))
      mem_used <- utils::memory.size()
      message(sprintf("[PERF] %s completed in %.2f seconds, memory usage: %.2f MB", 
                    stage_name, elapsed, mem_used))
      return(current_time)
    }
  ),
  
  public = list(
    #' Initialize a new DataTransformer
    #' 
    #' @param logger Optional logger object
    initialize = function(logger = NULL) {
      private$.logger <- logger
      return(invisible(self))
    },
    
    #' Apply log transformation to pixel data
    #' 
    #' @param pixel_data Matrix of pixel data to transform
    #' @return Transformed pixel data
    logTransform = function(pixel_data) {
      start_time <- Sys.time()
      message("Applying log transformation to pixel data...")
      
      # Apply log transformation (with small offset to handle zeros)
      transformed_data <- log1p(pixel_data)
      
      private$.trackPerformance(start_time, "Log transformation")
      return(transformed_data)
    },
    
    #' Scale each marker to [0,1] range
    #' 
    #' @param pixel_data Matrix of pixel data to transform
    #' @return Scaled pixel data
    scaleToUnitRange = function(pixel_data) {
      start_time <- Sys.time()
      message("Scaling pixel data to [0,1] range...")
      
      # Make a copy of the input data
      scaled_data <- pixel_data
      
      # Scale each marker to [0,1] range
      for (i in 1:ncol(scaled_data)) {
        col_min <- min(scaled_data[, i], na.rm = TRUE)
        col_max <- max(scaled_data[, i], na.rm = TRUE)
        
        if (col_max > col_min) {
          scaled_data[, i] <- (scaled_data[, i] - col_min) / (col_max - col_min)
        }
      }
      
      private$.trackPerformance(start_time, "Unit range scaling")
      return(scaled_data)
    },
    
    #' Apply Z-score normalization to pixel data
    #' 
    #' @param pixel_data Matrix of pixel data to transform
    #' @return Z-score normalized pixel data
    zScoreNormalize = function(pixel_data) {
      start_time <- Sys.time()
      message("Applying Z-score normalization to pixel data...")
      
      # Make a copy of the input data
      normalized_data <- pixel_data
      
      # Apply Z-score normalization to each column
      for (i in 1:ncol(normalized_data)) {
        col_mean <- mean(normalized_data[, i], na.rm = TRUE)
        col_sd <- sd(normalized_data[, i], na.rm = TRUE)
        
        if (col_sd > 0) {
          normalized_data[, i] <- (normalized_data[, i] - col_mean) / col_sd
        }
      }
      
      private$.trackPerformance(start_time, "Z-score normalization")
      return(normalized_data)
    },
    
    #' Apply quantile normalization to pixel data
    #' 
    #' @param pixel_data Matrix of pixel data to transform
    #' @return Quantile normalized pixel data
    quantileNormalize = function(pixel_data) {
      start_time <- Sys.time()
      message("Applying quantile normalization to pixel data...")
      
      if (!requireNamespace("preprocessCore", quietly = TRUE)) {
        message("Installing preprocessCore package for quantile normalization...")
        if (!requireNamespace("BiocManager", quietly = TRUE)) {
          install.packages("BiocManager")
        }
        BiocManager::install("preprocessCore", update = FALSE, ask = FALSE)
      }
      
      require(preprocessCore)
      
      # Apply quantile normalization
      normalized_data <- preprocessCore::normalize.quantiles(as.matrix(pixel_data))
      colnames(normalized_data) <- colnames(pixel_data)
      
      private$.trackPerformance(start_time, "Quantile normalization")
      return(normalized_data)
    },
    
    #' Apply arcsinh transformation
    #' 
    #' @param pixel_data Matrix of pixel data to transform
    #' @param cofactor Cofactor for arcsinh transformation (default: 5)
    #' @return Transformed pixel data
    arcsinhTransform = function(pixel_data, cofactor = 5) {
      start_time <- Sys.time()
      message(sprintf("Applying arcsinh transformation with cofactor %f...", cofactor))
      
      # Apply arcsinh transformation
      transformed_data <- asinh(pixel_data / cofactor)
      
      private$.trackPerformance(start_time, "Arcsinh transformation")
      return(transformed_data)
    },
    
    #' Standard transformation pipeline for marker analysis
    #' 
    #' Applies log transformation followed by unit range scaling
    #' 
    #' @param pixel_data Matrix of pixel data to transform
    #' @return Transformed pixel data
    standardTransform = function(pixel_data) {
      start_time <- Sys.time()
      message("Applying standard transformation pipeline...")
      
      # Apply log transformation
      transformed_data <- self$logTransform(pixel_data)
      
      # Scale to unit range
      transformed_data <- self$scaleToUnitRange(transformed_data)
      
      message("Standard transformation complete")
      private$.trackPerformance(start_time, "Standard transformation pipeline")
      
      return(transformed_data)
    },
    
    #' Full transformation pipeline with multiple methods
    #' 
    #' @param pixel_data Matrix of pixel data to transform
    #' @param method Transformation method: "standard", "arcsinh", "zscore", "quantile"
    #' @param cofactor Cofactor for arcsinh transformation (if applicable)
    #' @return Transformed pixel data
    transform = function(pixel_data, method = "standard", cofactor = 5) {
      start_time <- Sys.time()
      message(sprintf("Transforming pixel data using %s method...", method))
      
      transformed_data <- switch(method,
        "standard" = self$standardTransform(pixel_data),
        "arcsinh" = self$arcsinhTransform(pixel_data, cofactor),
        "zscore" = self$zScoreNormalize(pixel_data),
        "quantile" = self$quantileNormalize(pixel_data),
        stop(sprintf("Unknown transformation method: %s", method))
      )
      
      private$.trackPerformance(start_time, paste("Full transformation -", method))
      return(transformed_data)
    }
  )
) 