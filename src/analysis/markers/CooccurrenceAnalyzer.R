# Co-occurrence analyzer for marker analysis
# Calculates co-occurrence patterns between markers across images

library(parallel)
library(foreach)
library(doParallel)

#' CooccurrenceAnalyzer class
#' 
#' Analyzes co-occurrence patterns between markers across images
CooccurrenceAnalyzer <- R6::R6Class("CooccurrenceAnalyzer",
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
    #' Initialize a new CooccurrenceAnalyzer
    #' 
    #' @param logger Optional logger object
    initialize = function(logger = NULL) {
      private$.logger <- logger
      return(invisible(self))
    },
    
    #' Calculate thresholds for marker presence/absence
    #' 
    #' @param pixel_data Matrix of pixel data
    #' @param method Method for calculating thresholds: "median", "percentile", "otsu"
    #' @param percentile Percentile to use if method is "percentile"
    #' @return Vector of thresholds for each marker
    calculateThresholds = function(pixel_data, method = "median", percentile = 0.75) {
      start_time <- Sys.time()
      message(sprintf("Calculating marker expression thresholds for co-occurrence using %s method...", method))
      
      if (method == "median") {
        # Calculate median for each marker as threshold
        thresholds <- apply(pixel_data, 2, median, na.rm = TRUE)
      } else if (method == "percentile") {
        # Calculate specified percentile for each marker
        thresholds <- apply(pixel_data, 2, function(x) quantile(x, probs = percentile, na.rm = TRUE))
      } else if (method == "otsu") {
        # Implement Otsu's method for thresholding
        if (!requireNamespace("EBImage", quietly = TRUE)) {
          message("Installing EBImage package for Otsu thresholding...")
          if (!requireNamespace("BiocManager", quietly = TRUE)) {
            install.packages("BiocManager")
          }
          BiocManager::install("EBImage", update = FALSE, ask = FALSE)
        }
        
        require(EBImage)
        
        # Calculate Otsu threshold for each marker
        thresholds <- apply(pixel_data, 2, function(x) {
          # Normalize to 0-1 range for EBImage
          x_norm <- (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
          # Calculate Otsu threshold
          otsu_thresh <- otsu(x_norm)
          # Convert back to original scale
          thresh <- otsu_thresh * (max(x, na.rm = TRUE) - min(x, na.rm = TRUE)) + min(x, na.rm = TRUE)
          return(thresh)
        })
      } else {
        stop(sprintf("Unknown threshold method: %s", method))
      }
      
      # Report the thresholds
      for (i in 1:length(thresholds)) {
        message(sprintf("  Marker %s threshold: %.4f", names(thresholds)[i], thresholds[i]))
      }
      
      private$.trackPerformance(start_time, "Threshold calculation")
      return(thresholds)
    },
    
    #' Calculate co-occurrence matrices for each image
    #' 
    #' @param pixel_data Matrix of pixel data
    #' @param pixel_image_index Vector of image indices for each pixel
    #' @param marker_names Names of markers
    #' @param marker_thresholds Threshold values for each marker
    #' @param n_cores Number of cores to use for parallel processing
    #' @param min_pixels_per_image Minimum number of pixels required per image
    #' @return List of co-occurrence matrices by image
    calculateImageCooccurrences = function(pixel_data, pixel_image_index, marker_names, 
                                         marker_thresholds, n_cores = NULL, 
                                         min_pixels_per_image = 50) {
      start_time <- Sys.time()
      
      # Determine number of cores to use
      if (is.null(n_cores)) {
        # Use a conservative 30% of available cores to avoid memory issues
        n_cores <- max(1, floor(parallel::detectCores() * 0.3))
      }
      
      # Get unique image indices
      unique_image_indices <- sort(unique(pixel_image_index))
      n_markers <- length(marker_names)
      
      message(sprintf("Calculating co-occurrences for each of %d images using %d cores...", 
                     length(unique_image_indices), n_cores))
      
      # Create and register cluster for parallel processing
      co_cl <- parallel::makeCluster(n_cores, type = "PSOCK")
      on.exit(parallel::stopCluster(co_cl), add = TRUE)  # Ensure cluster is stopped on exit
      
      doParallel::registerDoParallel(co_cl)
      
      # Export required variables to cluster
      parallel::clusterExport(co_cl, 
                            varlist = c("pixel_data", "pixel_image_index", "marker_names", 
                                      "marker_thresholds", "n_markers", "min_pixels_per_image"), 
                            envir = environment())
      
      # Calculate co-occurrences in parallel
      image_cooccurrences <- parallel::parLapply(co_cl, unique_image_indices, function(img_idx) {
        # Get pixels from this image
        img_pixels <- which(pixel_image_index == img_idx)
        
        # Skip if too few pixels
        if (length(img_pixels) < min_pixels_per_image) {
          return(NULL)
        }
        
        # Get image data
        img_data <- pixel_data[img_pixels, ]
        
        # Handle NAs
        if (any(is.na(img_data))) {
          img_data[is.na(img_data)] <- 0
        }
        
        # Calculate binary presence
        binary_presence <- matrix(FALSE, nrow = nrow(img_data), ncol = ncol(img_data))
        for (m in 1:ncol(img_data)) {
          binary_presence[, m] <- img_data[, m] > marker_thresholds[m]
        }
        
        # Calculate co-occurrence matrix
        cooc_matrix <- matrix(0, nrow = n_markers, ncol = n_markers)
        
        # Loop through each marker pair
        for (i in 1:n_markers) {
          for (j in 1:n_markers) {
            # Count co-occurrence (both markers present)
            cooc_matrix[i, j] <- sum(binary_presence[, i] & binary_presence[, j]) / 
                                nrow(binary_presence)
          }
        }
        
        # Set dimension names
        colnames(cooc_matrix) <- marker_names
        rownames(cooc_matrix) <- marker_names
        
        return(cooc_matrix)
      })
      
      # Remove NULL entries
      image_cooccurrences <- image_cooccurrences[!sapply(image_cooccurrences, is.null)]
      
      private$.trackPerformance(start_time, "Image co-occurrences")
      return(image_cooccurrences)
    },
    
    #' Calculate average co-occurrence across all images
    #' 
    #' @param image_cooccurrences List of co-occurrence matrices by image
    #' @param marker_names Names of markers
    #' @return Average co-occurrence matrix
    calculateAverageCooccurrence = function(image_cooccurrences, marker_names) {
      n_markers <- length(marker_names)
      
      # Initialize matrix for average co-occurrence
      combined_cooc <- matrix(0, nrow = n_markers, ncol = n_markers)
      colnames(combined_cooc) <- marker_names
      rownames(combined_cooc) <- marker_names
      
      # Sum co-occurrences across images
      for (i in 1:length(image_cooccurrences)) {
        combined_cooc <- combined_cooc + image_cooccurrences[[i]]
      }
      
      # Calculate average
      if (length(image_cooccurrences) > 0) {
        avg_cooc <- combined_cooc / length(image_cooccurrences)
      } else {
        avg_cooc <- matrix(NA, nrow = n_markers, ncol = n_markers)
        rownames(avg_cooc) <- marker_names
        colnames(avg_cooc) <- marker_names
      }
      
      return(avg_cooc)
    },
    
    #' Calculate co-occurrence variance across images
    #' 
    #' @param image_cooccurrences List of co-occurrence matrices by image
    #' @param marker_names Names of markers
    #' @return Matrix of co-occurrence variances
    calculateCooccurrenceVariance = function(image_cooccurrences, marker_names) {
      n_markers <- length(marker_names)
      
      # Create matrix for variances
      cooc_var <- matrix(0, nrow = n_markers, ncol = n_markers)
      colnames(cooc_var) <- marker_names
      rownames(cooc_var) <- marker_names
      
      # Calculate variance for each cell in the co-occurrence matrix
      for (i in 1:n_markers) {
        for (j in 1:n_markers) {
          # Extract co-occurrence values for this marker pair across images
          cooc_values <- sapply(image_cooccurrences, function(x) x[i, j])
          
          # Calculate variance
          cooc_var[i, j] <- var(cooc_values, na.rm = TRUE)
        }
      }
      
      return(cooc_var)
    },
    
    #' Analyze marker co-occurrences
    #' 
    #' @param pixel_data Matrix of pixel data
    #' @param pixel_image_index Vector of image indices for each pixel
    #' @param marker_names Names of markers
    #' @param threshold_method Method for calculating thresholds
    #' @param n_cores Number of cores to use for parallel processing
    #' @return List with co-occurrence results
    analyze = function(pixel_data, pixel_image_index, marker_names, 
                      threshold_method = "median", n_cores = NULL) {
      start_time <- Sys.time()
      message("Starting co-occurrence analysis...")
      
      # Calculate thresholds for markers
      marker_thresholds <- self$calculateThresholds(pixel_data, method = threshold_method)
      
      # Calculate co-occurrence for each image
      image_cooccurrences <- self$calculateImageCooccurrences(
        pixel_data, 
        pixel_image_index, 
        marker_names,
        marker_thresholds,
        n_cores = n_cores
      )
      
      # Calculate overall co-occurrence
      if (length(image_cooccurrences) > 0) {
        message("Calculating overall co-occurrence...")
        overall_cooc <- self$calculateAverageCooccurrence(image_cooccurrences, marker_names)
        
        message("Calculating co-occurrence variance across images...")
        cooc_var <- self$calculateCooccurrenceVariance(image_cooccurrences, marker_names)
        
        cooccurrence_results <- list(
          by_image = image_cooccurrences,
          overall = overall_cooc,
          variance = cooc_var,
          thresholds = marker_thresholds
        )
      } else {
        message("Warning: No valid co-occurrences calculated. Check image data.")
        n_markers <- length(marker_names)
        cooccurrence_results <- list(
          by_image = list(),
          overall = matrix(NA, nrow = n_markers, ncol = n_markers, 
                         dimnames = list(marker_names, marker_names)),
          variance = matrix(NA, nrow = n_markers, ncol = n_markers, 
                          dimnames = list(marker_names, marker_names)),
          thresholds = marker_thresholds
        )
      }
      
      # Run garbage collection to free memory
      gc(verbose = FALSE)
      
      private$.trackPerformance(start_time, "Co-occurrence analysis")
      return(cooccurrence_results)
    }
  )
) 