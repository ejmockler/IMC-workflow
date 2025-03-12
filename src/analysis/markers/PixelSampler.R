# Pixel sampling class for marker analysis
# Extracts pixel data from CytoImageList objects

#' PixelSampler class
#' 
#' Handles sampling pixels from CytoImageList objects for marker analysis
#' Maintains tracking between pixels and their source images
PixelSampler <- R6::R6Class("PixelSampler",
  private = list(
    .images = NULL,
    .pixel_data = NULL,
    .pixel_data_transformed = NULL,
    .marker_names = NULL,
    .n_markers = NULL,
    .pixel_image_index = NULL,
    .pixel_image_names = NULL,
    
    # Track performance metrics during analysis
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
    #' Initialize a PixelSampler with images
    #' 
    #' @param images A CytoImageList object from cytomapper containing images
    initialize = function(images) {
      if (is.null(images)) {
        stop("Images must be provided")
      }
      if (!inherits(images, "CytoImageList")) {
        stop("Images must be a CytoImageList object")
      }
      
      # Initialize private fields
      private$.images <- images
      
      # Extract marker names from the images (CytoImageList has channelNames)
      private$.marker_names <- channelNames(images)
      private$.n_markers <- length(private$.marker_names)
      
      # Log basic image information
      message(sprintf("Initialized PixelSampler with %d images and %d markers",
                     length(images), private$.n_markers))
      for (i in seq_along(images)) {
        message(sprintf("  Image %d: %s, dimensions: %s", 
                       i, names(images)[i], paste(dim(images[[i]]), collapse="x")))
      }
      
      return(invisible(self))
    },
    
    #' Sample pixels from images, keeping track of source image
    #' 
    #' @param n_pixels Number of pixels to sample
    #' @return Matrix of pixel data
    samplePixels = function(n_pixels) {
      start_time <- Sys.time()
      
      # First calculate total available pixels across all images
      total_available_pixels <- 0
      image_dimensions <- list()
      
      for (i in seq_along(private$.images)) {
        img <- private$.images[[i]]
        img_dims <- dim(img)
        image_dimensions[[i]] <- img_dims
        total_available_pixels <- total_available_pixels + prod(img_dims[1:2])
      }
      
      message(sprintf("Total available pixels across all images: %d", total_available_pixels))
      
      # Determine how many pixels we can actually use
      n_pixels_to_use <- min(n_pixels, total_available_pixels)
      
      # Provide feedback if requested pixels exceeds available
      if (n_pixels > total_available_pixels) {
        message(sprintf("Warning: Requested %d pixels, but only %d are available. Using all available pixels.", 
                       n_pixels, total_available_pixels))
      }
      
      # Pre-allocate result matrices - significant speedup
      pixel_data <- matrix(0, ncol = private$.n_markers, nrow = n_pixels_to_use)
      colnames(pixel_data) <- private$.marker_names
      pixel_image_index <- integer(n_pixels_to_use)
      pixel_image_names <- character(n_pixels_to_use)
      
      # Counter for filled rows
      current_row <- 1
      
      # Loop through images and sample pixels - with vectorized operations
      for (i in seq_along(private$.images)) {
        # Track progress through images
        message(sprintf("Processing image %d of %d...", i, length(private$.images)))
        
        # Get current image and dimensions
        img <- private$.images[[i]]
        img_dims <- image_dimensions[[i]]
        total_img_pixels <- prod(img_dims[1:2])
        
        # Calculate pixels to sample from this image proportionally
        img_fraction <- total_img_pixels / total_available_pixels
        n_to_sample <- min(ceiling(n_pixels_to_use * img_fraction), total_img_pixels)
        
        # If we're sampling all pixels, provide feedback
        if (n_to_sample == total_img_pixels) {
          message(sprintf("Using all %d pixels from image %d", n_to_sample, i))
        } else {
          message(sprintf("Sampling %d pixels from image %d (%.1f%% of image)", 
                         n_to_sample, i, 100 * n_to_sample / total_img_pixels))
        }
        
        # Calculate end row for this batch
        end_row <- current_row + n_to_sample - 1
        if (end_row > n_pixels_to_use) {
          n_to_sample <- n_pixels_to_use - current_row + 1
          end_row <- n_pixels_to_use
        }
        
        # Sample random pixel positions without replacement
        sampled_indices <- sample(total_img_pixels, n_to_sample)
        
        # Convert to row, col coordinates (vectorized)
        img_width <- img_dims[2]
        row_indices <- ((sampled_indices - 1) %/% img_width) + 1
        col_indices <- ((sampled_indices - 1) %% img_width) + 1
        
        # VECTORIZED APPROACH: Process in chunks of 10,000 pixels at a time
        # This avoids memory issues with very large images
        chunk_size <- 10000
        for (chunk_start in seq(1, n_to_sample, by=chunk_size)) {
          chunk_end <- min(chunk_start + chunk_size - 1, n_to_sample)
          chunk_rows <- chunk_start:chunk_end
          
          # Track progress for large images
          if (n_to_sample > 100000 && chunk_start %% 100000 == 1) {
            message(sprintf("  Processing pixels %d-%d of %d...", 
                           chunk_start, chunk_end, n_to_sample))
          }
          
          # Get row indices for this chunk
          chunk_row_idx <- row_indices[chunk_rows]
          chunk_col_idx <- col_indices[chunk_rows]
          
          # Process each marker channel for this chunk of pixels
          for (k in 1:private$.n_markers) {
            # Extract all pixels for this marker in one operation
            pixel_data[current_row:(current_row + length(chunk_rows) - 1), k] <- 
              img[cbind(chunk_row_idx, chunk_col_idx, rep(k, length(chunk_rows)))]
          }
          
          # Update image indices for this chunk
          pixel_image_index[current_row:(current_row + length(chunk_rows) - 1)] <- i
          pixel_image_names[current_row:(current_row + length(chunk_rows) - 1)] <- names(private$.images)[i]
          
          # Update row counter
          current_row <- current_row + length(chunk_rows)
        }
        
        # Force garbage collection every few images to prevent memory buildup
        if (i %% 5 == 0) {
          gc(verbose = FALSE)
        }
      }
      
      # Store the sampled pixel data and image indices
      private$.pixel_data <- pixel_data
      private$.pixel_image_index <- pixel_image_index
      private$.pixel_image_names <- pixel_image_names
      
      message(sprintf("Sampled %d pixels. Data matrix dimensions: %d rows x %d columns",
                     nrow(pixel_data), nrow(pixel_data), ncol(pixel_data)))
      
      # Track performance
      private$.trackPerformance(start_time, "Pixel sampling")
      
      return(invisible(pixel_data))
    },
    
    #' Get the sampled pixel data
    #' @param transformed Whether to return transformed data (if available)
    #' @return Matrix of pixel data
    getPixelData = function(transformed = FALSE) {
      if (transformed && !is.null(private$.pixel_data_transformed)) {
        return(private$.pixel_data_transformed)
      } else {
        return(private$.pixel_data)
      }
    },
    
    #' Set transformed pixel data
    #' @param data Transformed pixel data
    setTransformedData = function(data) {
      private$.pixel_data_transformed <- data
      return(invisible(self))
    },
    
    #' Get the marker names
    #' @return Character vector of marker names
    getMarkerNames = function() {
      return(private$.marker_names)
    },
    
    #' Get number of markers
    #' @return Integer count of markers
    getMarkerCount = function() {
      return(private$.n_markers)
    },
    
    #' Get the image indices for sampled pixels
    #' @return Integer vector of image indices
    getPixelImageIndices = function() {
      return(private$.pixel_image_index)
    },
    
    #' Get the image names for sampled pixels
    #' @return Character vector of image names
    getPixelImageNames = function() {
      return(private$.pixel_image_names)
    },
    
    #' Get the original images
    #' @return CytoImageList of images
    getImages = function() {
      return(private$.images)
    }
  )
) 