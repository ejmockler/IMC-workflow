#' Image Processing for IMC Analysis
#' @description Handles image loading, processing, and quality control for IMC data.
#' Provides functionality for background subtraction, normalization, and mask quality assessment.
#'
#' @import R6
#'
ImageProcessor <- R6::R6Class("ImageProcessor",
  public = list(
    #' @field config Configuration parameters
    config = NULL,
    
    #' @field logger Logger object
    logger = NULL,
    
    #' @description
    #' Initialize a new ImageProcessor object
    #' @param config Configuration parameters
    #' @param logger Logger object
    initialize = function(config = NULL, logger = NULL) {
      cat("DEBUG: ImageProcessor initialize - Starting\n")
      self$config <- config
      self$logger <- logger %||% Logger$new("ImageProcessor")
      
      cat("DEBUG: ImageProcessor initialize - Checking required packages\n")
      # Ensure required packages
      if (!requireNamespace("cytomapper", quietly = TRUE)) {
        self$logger$warn("cytomapper package not available, some functionality may be limited")
        cat("DEBUG: ImageProcessor initialize - cytomapper package not available\n")
      } else {
        cat("DEBUG: ImageProcessor initialize - cytomapper package loaded\n")
      }
      
      if (!requireNamespace("S4Vectors", quietly = TRUE)) {
        self$logger$warn("S4Vectors package not available, some functionality may be limited")
        cat("DEBUG: ImageProcessor initialize - S4Vectors package not available\n")
      } else {
        cat("DEBUG: ImageProcessor initialize - S4Vectors package loaded\n")
      }
      
      cat("DEBUG: ImageProcessor initialize - Complete\n")
    },
    
    #' @description
    #' Process images and masks for analysis
    #' @param images_list List of images
    #' @param masks_list List of masks
    #' @param spe SpatialExperiment object (optional, for channel names)
    #' @return List of processed images and masks
    processImages = function(images_list, masks_list, spe = NULL) {
      self$logger$info("Processing image data")
      
      # Check inputs
      if (is.null(images_list) || length(images_list) == 0) {
        self$logger$warn("No images to process")
        return(NULL)
      }
      
      # Verify image and mask filename consistency
      if (!identical(sort(names(images_list)), sort(names(masks_list)))) {
        missing_masks <- setdiff(names(images_list), names(masks_list))
        missing_images <- setdiff(names(masks_list), names(images_list))
        if (length(missing_masks) > 0) {
          self$logger$warn(sprintf("Images missing masks: %s", paste(missing_masks, collapse = ", ")))
        }
        if (length(missing_images) > 0) {
          self$logger$warn(sprintf("Masks missing images: %s", paste(missing_images, collapse = ", ")))
        }
        self$logger$error("Mismatch between image and mask filenames")
        return(NULL)
      }
      
      # Set channel names from SPE if available
      if (!is.null(spe)) {
        expected_channels <- rownames(spe)
        
        # Set channel names
        tryCatch({
          require(cytomapper)
          channelNames(images_list) <- expected_channels
          self$logger$info("Channel names set from SPE")
        }, error = function(e) {
          self$logger$error(sprintf("Failed to set channel names: %s", e$message))
        })
      } else if (file.exists(file.path(self$config$paths$output_dir, "spe_processed.rds"))) {
        # Try to load from file if SPE not provided
        spe <- readRDS(file.path(self$config$paths$output_dir, "spe_processed.rds"))
        expected_channels <- rownames(spe)
        
        # Set channel names
        tryCatch({
          require(cytomapper)
          channelNames(images_list) <- expected_channels
          self$logger$info("Channel names set from SPE")
        }, error = function(e) {
          self$logger$error(sprintf("Failed to set channel names: %s", e$message))
        })
      }
      
      # Apply metadata
      meta_df <- S4Vectors::DataFrame(sample_id = names(images_list))
      mcols(images_list) <- mcols(masks_list) <- meta_df
      
      # Extract configuration parameters
      use_background_subtraction <- self$config$processing$background_subtraction
      if (is.null(use_background_subtraction)) use_background_subtraction <- TRUE
      
      # Process each image
      for (img_name in names(images_list)) {
        self$logger$info(sprintf("Processing image %s", img_name))
        
        # Get dimensions
        dims <- dim(images_list[[img_name]])
        
        if (length(dims) == 3) {
          # For each channel
          for (ch in 1:dims[3]) {
            channel_data <- images_list[[img_name]][,,ch]
            
            # Background subtraction if enabled
            if (use_background_subtraction) {
              bg_value <- quantile(channel_data, 0.05, na.rm = TRUE)
              channel_data <- pmax(channel_data - bg_value, 0)
            }
            
            # Normalize using min-max scaling
            rng <- range(channel_data, na.rm = TRUE)
            if (rng[2] > rng[1]) {
              images_list[[img_name]][,,ch] <- (channel_data - rng[1]) / (rng[2] - rng[1])
            }
          }
        }
        
        # Attach corresponding mask
        attr(images_list[[img_name]], "mask") <- masks_list[[img_name]]
        
        # Compute and log basic segmentation quality metrics
        mask_data <- masks_list[[img_name]]
        if (!is.null(mask_data)) {
          cell_pixel_counts <- table(as.vector(mask_data))
          mean_cell_area <- mean(as.numeric(cell_pixel_counts))
          self$logger$info(sprintf("Image %s segmentation: mean cell area = %f", img_name, mean_cell_area))
        }
      }
      
      # Add processing metadata
      attr(images_list, "processing") <- list(
        date = Sys.time(),
        parameters = list(
          background_subtraction = use_background_subtraction
        )
      )
      
      # Save processed images
      output_path <- file.path(self$config$paths$output_dir, "images_processed.rds")
      saveRDS(images_list, file = output_path)
      self$logger$info(sprintf("Processed images saved to %s", output_path))
      
      return(list(processed_images = images_list, processed_masks = masks_list))
    },
    
    #' @description
    #' Assess quality of segmentation masks
    #' @param masks_list List of masks
    #' @return Data frame with quality metrics
    assessMaskQuality = function(masks_list) {
      self$logger$info("Assessing mask quality")
      
      if (is.null(masks_list) || length(masks_list) == 0) {
        self$logger$warn("No masks to assess")
        return(NULL)
      }
      
      # Process each mask
      quality_metrics <- lapply(names(masks_list), function(mask_name) {
        mask_data <- masks_list[[mask_name]]
        
        # Skip if NULL
        if (is.null(mask_data)) {
          return(NULL)
        }
        
        # Calculate metrics
        cell_ids <- setdiff(unique(as.vector(mask_data)), 0)  # 0 is background
        n_cells <- length(cell_ids)
        
        # Cell sizes
        cell_pixel_counts <- table(as.vector(mask_data))
        cell_pixel_counts <- cell_pixel_counts[names(cell_pixel_counts) != "0"]
        
        # Calculate metrics
        mean_area <- mean(as.numeric(cell_pixel_counts))
        median_area <- median(as.numeric(cell_pixel_counts))
        min_area <- min(as.numeric(cell_pixel_counts))
        max_area <- max(as.numeric(cell_pixel_counts))
        
        # Return as data frame row
        data.frame(
          mask_name = mask_name,
          n_cells = n_cells,
          mean_area = mean_area,
          median_area = median_area,
          min_area = min_area,
          max_area = max_area,
          stringsAsFactors = FALSE
        )
      })
      
      # Combine into a single data frame
      quality_metrics <- do.call(rbind, quality_metrics)
      
      # Log summary
      self$logger$info(sprintf("Assessed %d masks, average cells per mask: %.1f", 
                               nrow(quality_metrics), 
                               mean(quality_metrics$n_cells)))
      
      return(quality_metrics)
    }
  ),
  
  private = list(
    # Helper methods can be added here
  )
) 