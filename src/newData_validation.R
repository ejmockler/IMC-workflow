# Load required packages
library(tiff)              # For reading TIFF files
library(magrittr)          # For the pipe (%>%), if desired
library(cytomapper)        # For CytoImageList handling
# Assume other dependencies are loaded as needed
source("src/visualization/VisualizationFunctions.R")

#############################
## Configuration Module
#############################
get_new_data_config <- function() {
  # Centralized configuration for raw image and mask data.
  list(
    paths = list(
      images_rds = "data/images.rds",    # Path to existing images RData
      masks_dir = "data/masks",          # Directory with mask TIFF files
      panel     = "data/panel.csv",      # Panel configuration file
      imc_dir   = "data/241218_IMC_Alun" # (Optional) Additional directory for raw IMC text files
    )
  )
}

#############################
## Paths Validation for Raw Data
#############################
validate_new_paths <- function(config) {
  paths <- config$paths
  
  if (!file.exists(paths$images_rds)) {
    stop(sprintf("Images RData file missing: %s", paths$images_rds))
  } else {
    message(sprintf("Found images RData file: %s", paths$images_rds))
  }
  
  if (!dir.exists(paths$masks_dir)) {
    stop(sprintf("Mask directory missing: %s", paths$masks_dir))
  } else {
    message(sprintf("Found mask directory: %s", paths$masks_dir))
  }
  
  if (!file.exists(paths$panel)) {
    stop(sprintf("Panel file missing: %s", paths$panel))
  } else {
    message(sprintf("Found panel file: %s", paths$panel))
  }
}

#############################
## Validate Mask Values
#############################
validate_mask_values <- function(masks) {
  message("Validating mask values...")
  
  for (i in seq_along(masks)) {
    mask <- masks[[i]]
    mask_name <- names(masks)[i]
    
    if (!is.null(mask)) {
      # Get unique values
      unique_vals <- unique(as.vector(mask))
      min_val <- min(unique_vals)
      max_val <- max(unique_vals)
      
      # Check if values are binary (0 and 1)
      if (!all(unique_vals %in% c(0, 1))) {
        warning(sprintf("Mask '%s' contains non-binary values. Range: [%f, %f], Unique values: %s", 
                       mask_name, min_val, max_val, paste(unique_vals, collapse = ", ")))
      } else if (length(unique_vals) == 1) {
        # Check if mask has any variability
        warning(sprintf("Mask '%s' has no variability - contains only %d values", 
                       mask_name, unique_vals[1]))
      } else {
        message(sprintf("Mask '%s' contains valid binary values with variability.", mask_name))
      }
    }
  }
}

#############################
## Load Raw Data: Images, Masks and Panel CSV
#############################
load_raw_data <- function(paths) {
  message("Loading images and masks...")
  
  # Load the images RData object
  images <- readRDS(paths$images_rds)
  message(sprintf("Loaded images object with %d images", length(images)))
  
  # List TIFF files for masks (accept .tif or .tiff extensions)
  mask_files <- list.files(
    paths$masks_dir, pattern = "\\.(tif|tiff)$", 
    full.names = TRUE, ignore.case = TRUE
  )
  
  if (length(mask_files) == 0) {
    warning("No TIFF files found in the mask directory.")
  } else {
    message(sprintf("Found %d mask files.", length(mask_files)))
  }
  
  # Read in all masks
  masks <- lapply(mask_files, function(f) {
    message(sprintf("Reading mask: %s", f))
    tryCatch({
      tiff::readTIFF(f, native = FALSE)
    }, error = function(e) {
      warning(sprintf("Error reading mask %s: %s", f, e$message))
      NULL
    })
  })
  names(masks) <- basename(mask_files)
  
  # Load panel CSV file
  panel <- read.csv(paths$panel, stringsAsFactors = FALSE)
  
  # Validate mask values
  validate_mask_values(masks)
  
  return(list(
    images = images,
    masks  = masks,
    panel  = panel
  ))
}

#############################
## Map IMC Files from Filenames
#############################
map_imc_files <- function(file_names) {
  # Define patterns based on the naming convention (adapted for TIFF extension)
  patterns <- list(
    roi_pattern  = "^IMC_241218_Alun_ROI_D([0-9]+)_M([1-2])_[0-9]+_([0-9]+)\\.(tif|tiff)$",
    sham_pattern = "^IMC_241218_Alun_ROI_Sam[0-9]+_[0-9]+_([0-9]+)\\.(tif|tiff)$",
    test_pattern = "^IMC_241218_Alun_ROI_Test[0-9]+_([0-9]+)\\.(tif|tiff)$"
  )
  
  valid_files <- character(0)
  imc_info_list <- list()
  
  # Process ROI files
  roi_files <- grep(patterns$roi_pattern, file_names, value = TRUE)
  valid_files <- c(valid_files, roi_files)
  if (length(roi_files) > 0) {
    roi_info <- data.frame(
      filename   = roi_files,
      type       = "ROI",
      day        = as.numeric(gsub(patterns$roi_pattern, "\\1", roi_files)),
      mouse      = as.numeric(gsub(patterns$roi_pattern, "\\2", roi_files)),
      sample_num = as.numeric(gsub(patterns$roi_pattern, "\\3", roi_files)),
      stringsAsFactors = FALSE
    )
    imc_info_list[[length(imc_info_list) + 1]] <- roi_info
  }
  
  # Process Sham files
  sham_files <- grep(patterns$sham_pattern, file_names, value = TRUE)
  valid_files <- c(valid_files, sham_files)
  if (length(sham_files) > 0) {
    sham_info <- data.frame(
      filename   = sham_files,
      type       = "Sham",
      day        = NA,  # No day information for Sham regions
      mouse      = NA,  # No mouse information for Sham regions
      sample_num = as.numeric(gsub(patterns$sham_pattern, "\\1", sham_files)),
      stringsAsFactors = FALSE
    )
    imc_info_list[[length(imc_info_list) + 1]] <- sham_info
  }
  
  # Process Test files
  test_files <- grep(patterns$test_pattern, file_names, value = TRUE)
  valid_files <- c(valid_files, test_files)
  if (length(test_files) > 0) {
    test_info <- data.frame(
      filename   = test_files,
      type       = "Test",
      day        = NA,
      mouse      = NA,
      sample_num = as.numeric(gsub(patterns$test_pattern, "\\1", test_files)),
      stringsAsFactors = FALSE
    )
    imc_info_list[[length(imc_info_list) + 1]] <- test_info
  }
  
  if (length(valid_files) == 0) {
    warning("No image files match the expected IMC naming patterns.")
    return(NULL)
  }
  
  imc_info <- do.call(rbind, imc_info_list)
  message(sprintf("Mapped %d IMC image file(s) with contextual metadata.", nrow(imc_info)))
  return(imc_info)
}

#############################
## Channel Validation Function
#############################
validate_channels <- function(images, panel) {
  # Get the kept channels from panel.csv (assumes panel has a "keep" column)
  kept_panel <- panel[panel$keep == 1, ]
  expected_channels <- nrow(kept_panel)
  message(sprintf("Panel indicates %d kept channel(s).", expected_channels))
  
  for (i in seq_along(images)) {
    img <- images[[i]]
    file_name <- names(images)[i]
    
    # Check if the image is multi-channel (3D array) or grayscale (2D)
    if (length(dim(img)) < 3) {
      if (expected_channels > 1) {
        warning(sprintf("Image '%s' appears grayscale but panel.csv indicates multiple kept channels.", file_name))
      } else {
        message(sprintf("Image '%s' channel alignment validated as grayscale.", file_name))
      }
    } else {
      raw_channels <- dim(img)[3]
      if (raw_channels != expected_channels) {
        warning(sprintf("Image '%s' has %d channels but panel.csv indicates %d kept channels.", file_name, raw_channels, expected_channels))
      } else {
        message(sprintf("Image '%s' channel alignment validated.", file_name))
      }
    }
  }
}

#############################
## Main Validation, Mapping and Visualization Pipeline
#############################
validate_new_imc_data <- function() {
  message("Starting IMC raw data validation and mapping...\n")
  
  # Retrieve configuration for raw data
  config <- get_new_data_config()
  
  # Validate that required directories and files exist
  validate_new_paths(config)
  
  # Load raw images, masks, and panel data
  data <- load_raw_data(config$paths)
  
  # Validate channel alignment using panel.csv information.
  # (The Steinbock pipeline should have saved and ordered channels according to panel.csv.)
  validate_channels(data$images, data$panel)
  
  # Map file metadata from image filenames (using the naming convention)
  imc_info <- map_imc_files(names(data$images))
  if (is.null(imc_info)) {
    warning("Mapping information could not be generated from image filenames.")
  } else {
    message("IMC file mapping based on filenames:")
    print(imc_info)
  }
  
  # Visualize the raw IMC data using the visualization function.
  # (The visualization function is expected to work directly with raw images and masks plus mapping info.)
  message("Calling visualization function with raw image and mask data...")
  # visualize_raw_imc_data(8, data$images, data$masks, data$panel, imc_info)

  # Specific channels:
  visualize_raw_imc_data(8, data$images, data$masks, data$panel, imc_info,
                        channels = c("CD31", "CD34", "CD44", "DNA1", "DNA2"))
  
  message("\nRaw data validation, mapping and visualization complete.")
  return(list(
    data     = data,
    imc_info = imc_info,
    config   = config
  ))
}

#############################
## Execute the Pipeline
#############################
validated_data <- validate_new_imc_data()