# Correlation analyzer for marker analysis
# Calculates correlations between markers across images

library(parallel)
library(foreach)
library(doParallel)

#' CorrelationAnalyzer class
#' 
#' Analyzes correlations between markers across images
CorrelationAnalyzer <- R6::R6Class("CorrelationAnalyzer",
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
    #' Initialize a new CorrelationAnalyzer
    #' 
    #' @param logger Optional logger object
    initialize = function(logger = NULL) {
      private$.logger <- logger
      return(invisible(self))
    },
    
    #' Calculate correlation matrices for each image
    #' 
    #' @param pixel_data Matrix of pixel data
    #' @param pixel_image_index Vector of image indices for each pixel
    #' @param n_cores Number of cores to use for parallel processing
    #' @param min_pixels_per_image Minimum number of pixels required per image
    #' @return List of correlation matrices by image
    calculateImageCorrelations = function(pixel_data, pixel_image_index, marker_names, 
                                         n_cores = NULL, min_pixels_per_image = 50) {
      start_time <- Sys.time()
      
      # Determine number of cores to use
      if (is.null(n_cores)) {
        # Use a conservative 30% of available cores to avoid memory issues
        n_cores <- max(1, floor(parallel::detectCores() * 0.3))
      }
      
      # Get unique image indices
      unique_image_indices <- sort(unique(pixel_image_index))
      n_markers <- ncol(pixel_data)
      
      message(sprintf("Calculating correlations for each of %d images using %d cores...", 
                      length(unique_image_indices), n_cores))
      
      # Create and register cluster for parallel processing
      cl <- parallel::makeCluster(n_cores, type = "PSOCK")
      on.exit(parallel::stopCluster(cl), add = TRUE)  # Ensure cluster is stopped on exit
      
      doParallel::registerDoParallel(cl)
      
      # Export required variables to cluster
      parallel::clusterExport(cl, 
                             varlist = c("pixel_data", "pixel_image_index", 
                                        "marker_names", "n_markers", "min_pixels_per_image"), 
                             envir = environment())
      
      # Calculate correlations in parallel
      image_correlations <- parallel::parLapply(cl, unique_image_indices, function(img_idx) {
        # Get pixels from this image
        img_pixels <- which(pixel_image_index == img_idx)
        
        # Skip if too few pixels
        if (length(img_pixels) < min_pixels_per_image) {
          return(NULL)
        }
        
        # Get image data
        img_data <- pixel_data[img_pixels, ]
        
        # Handle potential issues
        if (any(is.na(img_data))) {
          img_data[is.na(img_data)] <- 0
        }
        
        # Calculate correlation matrix for this image
        tryCatch({
          result <- cor(img_data, use = "pairwise.complete.obs")
          # Validate result
          if (any(is.na(result)) || !all(is.finite(result))) {
            return(NULL)
          }
          colnames(result) <- marker_names
          rownames(result) <- marker_names
          return(result)
        }, error = function(e) {
          return(NULL)
        })
      })
      
      # Remove NULL results
      image_correlations <- image_correlations[!sapply(image_correlations, is.null)]
      
      private$.trackPerformance(start_time, "Image correlations")
      return(image_correlations)
    },
    
    #' Calculate average correlation across all images
    #' 
    #' @param image_correlations List of correlation matrices by image
    #' @param marker_names Names of markers
    #' @return Average correlation matrix
    calculateAverageCorrelation = function(image_correlations, marker_names) {
      n_markers <- length(marker_names)
      
      # Initialize matrix for average correlation
      combined_corr <- matrix(0, nrow = n_markers, ncol = n_markers)
      colnames(combined_corr) <- marker_names
      rownames(combined_corr) <- marker_names
      
      # Better validation before combining correlations
      valid_correlations <- 0
      for (i in 1:length(image_correlations)) {
        corr_matrix <- image_correlations[[i]]
        # Check if the result is a proper matrix with correct dimensions
        if (!is.null(corr_matrix) && is.matrix(corr_matrix) && 
            nrow(corr_matrix) == n_markers && ncol(corr_matrix) == n_markers) {
          combined_corr <- combined_corr + corr_matrix
          valid_correlations <- valid_correlations + 1
        }
      }
      
      # Only divide if we have valid correlations
      if (valid_correlations > 0) {
        avg_corr <- combined_corr / valid_correlations
      } else {
        avg_corr <- matrix(NA, nrow = n_markers, ncol = n_markers)
        rownames(avg_corr) <- marker_names
        colnames(avg_corr) <- marker_names
      }
      
      return(avg_corr)
    },
    
    #' Calculate correlation variance across images
    #' 
    #' @param image_correlations List of correlation matrices by image
    #' @param marker_names Names of markers
    #' @return Matrix of correlation variances
    calculateCorrelationVariance = function(image_correlations, marker_names) {
      n_markers <- length(marker_names)
      
      # Create matrix for variances
      corr_var <- matrix(0, nrow = n_markers, ncol = n_markers)
      colnames(corr_var) <- marker_names
      rownames(corr_var) <- marker_names
      
      # Calculate variance for each cell in the correlation matrix
      for (i in 1:n_markers) {
        for (j in 1:n_markers) {
          # Extract correlation values for this marker pair across images
          corr_values <- sapply(image_correlations, function(x) {
            if (is.null(x)) return(NA)
            if (is.matrix(x) && nrow(x) == n_markers && ncol(x) == n_markers) {
              return(x[i, j])
            } else {
              return(NA)
            }
          })
          
          # Calculate variance (excluding NAs)
          corr_var[i, j] <- var(corr_values, na.rm = TRUE)
        }
      }
      
      return(corr_var)
    },
    
    #' Analyze marker correlations
    #' 
    #' @param pixel_data Matrix of pixel data
    #' @param pixel_image_index Vector of image indices for each pixel
    #' @param marker_names Names of markers
    #' @param n_cores Number of cores to use for parallel processing
    #' @return List with correlation results
    analyze = function(pixel_data, pixel_image_index, marker_names, n_cores = NULL) {
      start_time <- Sys.time()
      message("Starting correlation analysis...")
      
      # Calculate correlation for each image
      image_correlations <- self$calculateImageCorrelations(
        pixel_data, 
        pixel_image_index, 
        marker_names,
        n_cores = n_cores
      )
      
      # Calculate overall correlation
      if (length(image_correlations) > 0) {
        message("Calculating overall correlation...")
        overall_corr <- self$calculateAverageCorrelation(image_correlations, marker_names)
        
        message("Calculating correlation variance across images...")
        corr_var <- self$calculateCorrelationVariance(image_correlations, marker_names)
        
        correlation_results <- list(
          by_image = image_correlations,
          overall = overall_corr,
          variance = corr_var
        )
      } else {
        message("Warning: No valid correlations calculated. Check image data.")
        n_markers <- length(marker_names)
        correlation_results <- list(
          by_image = list(),
          overall = matrix(NA, nrow = n_markers, ncol = n_markers, 
                         dimnames = list(marker_names, marker_names)),
          variance = matrix(NA, nrow = n_markers, ncol = n_markers, 
                          dimnames = list(marker_names, marker_names))
        )
      }
      
      # Run garbage collection to free memory
      gc(verbose = FALSE)
      
      private$.trackPerformance(start_time, "Correlation analysis")
      return(correlation_results)
    }
  )
) 