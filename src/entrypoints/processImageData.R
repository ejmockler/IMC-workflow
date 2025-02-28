#' Process IMC Image Data: Loading and Preprocessing Images and Segmentation Masks
#'
#' This entrypoint loads multi‑channel images and segmentation masks using cytomapper,
#' sets channel names from the SpatialExperiment, attaches masks to images,
#' and performs quality control checks.
#'
#' @return A CytoImageList containing fully processed images with masks attached.
#'
#' @example
#'   images <- runProcessImageData()

source("src/core/ConfigurationManager.R")
source("src/core/Logger.R")

runProcessImageData <- function() {
  # Initialize configuration and logger
  configManager <- ConfigurationManager$new()
  logger <- Logger$new(log_file = "logs/processImageData.log", log_level = "INFO")
  
  # Load required libraries
  library(cytomapper)
  library(imcRtools)
  
  # Load images and masks using configured paths
  logger$log_info("Loading images and masks...")
  images <- loadImages(configManager$config$paths$steinbock_images)
  masks <- loadImages(configManager$config$paths$steinbock_masks, as.is = TRUE)
  logger$log_info("Loaded %d images and %d masks.", length(images), length(masks))
  
  # Verify image and mask filename consistency
  if (!identical(sort(names(images)), sort(names(masks)))) {
    missing_masks <- setdiff(names(images), names(masks))
    missing_images <- setdiff(names(masks), names(images))
    if (length(missing_masks) > 0) {
      logger$log_warning("Images missing masks: %s", paste(missing_masks, collapse = ", "))
    }
    if (length(missing_images) > 0) {
      logger$log_warning("Masks missing images: %s", paste(missing_images, collapse = ", "))
    }
    stop("Mismatch between image and mask filenames")
  }
  
  # Read the processed SpatialExperiment to get channel names
  spe <- readRDS(file.path(configManager$config$output$dir, "spe_processed.rds"))
  expected_channels <- rownames(spe)
  logger$log_info("Setting %d channel names from SPE...", length(expected_channels))
  
  # Set channel names (with validation)
  tryCatch({
    channelNames(images) <- expected_channels
    logger$log_info("Channel names set successfully")
  }, error = function(e) {
    logger$log_error("Failed to set channel names: %s", e$message)
    stop(e)
  })
  
  # We simply record the sample_id now.
  meta_df <- DataFrame(
    sample_id = names(images)
  )

  # Apply the same minimal metadata to both images and masks.
  mcols(images) <- mcols(masks) <- meta_df
  
  # Process each image: normalize intensities and attach masks
  for (img_name in names(images)) {
    # Normalize intensities using min-max scaling
    dims <- dim(images[[img_name]])
    if (length(dims) == 3) {
      for(ch in seq_len(dims[3])) {
        channel_data <- images[[img_name]][,,ch]
        rng <- range(channel_data, na.rm = TRUE)
        images[[img_name]][,,ch] <- (channel_data - rng[1]) / (rng[2] - rng[1] + 1e-8)
      }
    }
    
    # Attach corresponding mask
    attr(images[[img_name]], "mask") <- masks[[img_name]]
    
    # Compute and log basic segmentation quality metrics
    mask_data <- masks[[img_name]]
    if (!is.null(mask_data)) {
      cell_pixel_counts <- table(as.vector(mask_data))
      mean_cell_area <- mean(as.numeric(cell_pixel_counts))
      logger$log_info("Image %s segmentation: mean cell area = %f", img_name, mean_cell_area)
    }
  }
  
  # Save processed images
  output_path <- file.path(configManager$config$output$dir, "images_processed.rds")
  saveRDS(images, file = output_path)
  logger$log_info("Processed images saved to %s", output_path)
  
  invisible(images)
}

if (interactive() || identical(environment(), globalenv())) {
  runProcessImageData()
} 