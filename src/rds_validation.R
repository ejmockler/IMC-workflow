# Load required packages for dependency management
library(R6)
source("src/core/DependencyManager.R")
# Source the visualization functions (newly added)

# Validate the environment and install any missing packages before proceeding
dep_manager <- DependencyManager$new()
dep_manager$validate_environment()

# Explicitly load libraries that are needed for the validation process
# (These packages have been ensured to be installed by DependencyManager)
library(SpatialExperiment)
library(cytomapper)

#----------------------------------
# Comprehensive IMC Data Validation
# Using modular design (Single Responsibility, Dependency Injection,
# and Functional Composition) for ease of maintenance and clarity.
#----------------------------------

#############################
## Configuration Module
#############################
get_validation_config <- function() {
  # Centralized configuration object
  list(
    paths = list(
      spe = "/Users/noot/Documents/IMC/data/spe.rds",
      images = "/Users/noot/Documents/IMC/data/images.rds",
      masks = "/Users/noot/Documents/IMC/data/masks.rds",
      panel = "data/panel.csv",
      imc_dir = "data/241218_IMC_Alun"  # IMC txt file directory
    ),
    expected_samples = 1:26  # Expected sample numbers in the IMC study
  )
}

#############################
## Module 1: Paths Validation
#############################
validate_paths <- function(config) {
  paths <- config$paths
  for (name in names(paths)) {
    if (!file.exists(paths[[name]])) {
      stop(sprintf("Required path missing: %s at %s", name, paths[[name]]))
    }
    message(sprintf("Found %s: %s", name, paths[[name]]))
  }
  
  # Validate IMC raw files separately (only files starting with "IMC_")
  imc_files <- list.files(paths$imc_dir, pattern = "^IMC_.*(\\.txt)?$")
  message(sprintf("\nFound %d IMC raw files", length(imc_files)))
  
  return(list(paths = paths, imc_files = imc_files))
}

#############################
## Module 2: Data Loading and Basic Validation
#############################
load_and_validate_data <- function(paths) {
  message("\nLoading data files...")
  spe <- readRDS(paths$spe)
  images <- readRDS(paths$images)
  masks <- readRDS(paths$masks)
  panel <- read.csv(paths$panel, stringsAsFactors = FALSE)

  message("First image dimensions:")
  print(dim(images[[1]]))
  
  # Fix marker names in SPE object
  message("\nFixing marker names in SPE object...")
  message("Current marker names:")
  print(rownames(spe))
  
  # Fix CD11b name in SPE object
  rownames(spe)[rownames(spe) == "CD11b 1"] <- "CD11b"
  
  message("\nUpdated marker names:")
  print(rownames(spe))
  print(spe)
  
  # Also fix in assay names if present
  for (assay_name in names(assays(spe))) {
    rownames(assay(spe, assay_name))[rownames(assay(spe, assay_name)) == "CD11b 1"] <- "CD11b"
  }
  
  # Save the updated SPE object
  saveRDS(spe, paths$spe)
  message("\nSPE object updated and saved with corrected marker names.")
  
  # ---- New Fallback for Intensities ----
  if (!("intensities" %in% assayNames(spe))) {
    if ("exprs" %in% assayNames(spe)) {
      message("No 'intensities' assay found in SPE. Creating 'intensities' assay from 'exprs'.")
      assay(spe, "intensities") <- assay(spe, "exprs")
    } else {
      warning("Neither 'intensities' nor 'exprs' assay is available in SPE.")
    }
  }
  # ---- End Fallback ----
  
  # Print images object details
  message("images object:")
  print(images)

  # Fix channel names for the entire images object
  if (!is.null(images) && length(images) > 0) {
    # Get all channel names from the images list at once
    current_channels <- channelNames(images)
    if ("CD11b 1" %in% current_channels) {
      new_channels <- current_channels
      new_channels[new_channels == "CD11b 1"] <- "CD11b"
      channelNames(images) <- new_channels
      message("Images object channel names corrected.")
    }
    # Save the updated images object
    saveRDS(images, paths$images)
    message("\nImages object updated and saved with corrected channel names.")
  }
  
  # Print masks object details
  message("masks object:")
  print(masks)
  
  # Fix channel names for the entire masks object
  if (!is.null(masks) && length(masks) > 0) {
    current_channels <- channelNames(masks)
    if ("CD11b 1" %in% current_channels) {
      new_channels <- current_channels
      new_channels[new_channels == "CD11b 1"] <- "CD11b"
      channelNames(masks) <- new_channels
      message("Masks object channel names corrected.")
    }
    # Save the updated masks object
    saveRDS(masks, paths$masks)
    message("\nMasks object updated and saved with corrected channel names.")
  }
  
  message("\nBasic data dimensions:")
  message("SPE dimensions: ", paste(dim(spe), collapse = " x "))
  message("Number of images: ", length(images))
  message("Number of masks: ", length(masks))
  message("Panel dimensions: ", paste(dim(panel), collapse = " x "))
  
  if (length(images) != length(masks)) {
    warning("Mismatch between number of images and masks")
  }
  
  return(list(spe = spe, images = images, masks = masks, panel = panel))
}

#############################
## Module 3: Sample and IMC File Mapping Validation
#############################
validate_sample_mapping <- function(spe, imc_files, config) {
  message("\n=== Sample and IMC File Validation ===")
  
  # Extract sample numbers from SPE sample_id: "IMC_241218_Alun_###"
  spe_samples <- sort(unique(as.numeric(gsub(".*_", "", colData(spe)$sample_id))))
  message("\nSPE samples (", length(spe_samples), "):")
  message(paste(sprintf("%03d", spe_samples), collapse = ", "))
  
  # Define patterns for valid IMC files for ROI, Sham, and Test regions.
  # ROI example: "IMC_241218_Alun_ROI_D7_M1_01_21.txt"
  # Sham example: "IMC_241218_Alun_ROI_Sam1_03_4.txt"
  # Test example: "IMC_241218_Alun_ROI_Test01_1.txt"
  patterns <- list(
    roi_pattern  = "^IMC_241218_Alun_ROI_D([0-9]+)_M([1-2])_[0-9]+_([0-9]+)\\.txt$",
    sham_pattern = "^IMC_241218_Alun_ROI_Sam[0-9]+_[0-9]+_([0-9]+)\\.txt$",
    test_pattern = "^IMC_241218_Alun_ROI_Test[0-9]+_([0-9]+)\\.txt$"
  )
  
  valid_files <- character(0)
  imc_info_list <- list()
  
  # Process ROI files
  roi_files <- grep(patterns$roi_pattern, imc_files, value = TRUE)
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
  sham_files <- grep(patterns$sham_pattern, imc_files, value = TRUE)
  valid_files <- c(valid_files, sham_files)
  if (length(sham_files) > 0) {
    sham_info <- data.frame(
      filename   = sham_files,
      type       = "Sham",
      day        = NA,  # No day information for sham regions
      mouse      = NA,  # No mouse information for sham regions
      sample_num = as.numeric(gsub(patterns$sham_pattern, "\\1", sham_files)),
      stringsAsFactors = FALSE
    )
    imc_info_list[[length(imc_info_list) + 1]] <- sham_info
  }
  
  # Process Test files
  test_files <- grep(patterns$test_pattern, imc_files, value = TRUE)
  valid_files <- c(valid_files, test_files)
  if (length(test_files) > 0) {
    test_info <- data.frame(
      filename   = test_files,
      type       = "Test",
      day        = NA,  # No day information for test regions
      mouse      = NA,  # No mouse information for test regions
      sample_num = as.numeric(gsub(patterns$test_pattern, "\\1", test_files)),
      stringsAsFactors = FALSE
    )
    imc_info_list[[length(imc_info_list) + 1]] <- test_info
  }
  
  if (length(valid_files) == 0) {
    warning("No IMC files found matching any expected pattern (ROI, Sham, or Test).")
    return(NULL)
  }
  
  imc_info <- do.call(rbind, imc_info_list)
  
  message("\nValid IMC files found: ", nrow(imc_info))
  message("\nIMC file structure:")
  message("Types: ", paste(sort(unique(imc_info$type)), collapse = ", "))
  
  if ("ROI" %in% imc_info$type) {
    message("ROI files - Days: ", paste(sort(unique(na.omit(imc_info$day))), collapse = ", "))
    message("ROI files - Mice: ", paste(sort(unique(na.omit(imc_info$mouse))), collapse = ", "))
  }
  
  if ("Sham" %in% imc_info$type) {
    sham_subset <- imc_info[imc_info$type == "Sham", ]
    message("Sham files - Count: ", nrow(sham_subset))
    message("Sham files - Sample numbers: ", paste(sort(unique(sham_subset$sample_num)), collapse = ", "))
  } else {
    message("No Sham files detected.")
  }
  
  if ("Test" %in% imc_info$type) {
    test_subset <- imc_info[imc_info$type == "Test", ]
    message("Test files - Count: ", nrow(test_subset))
    message("Test files - Sample numbers: ", paste(sort(unique(test_subset$sample_num)), collapse = ", "))
  } else {
    message("No Test files detected.")
  }
  
  message("Combined Sample numbers from all file types: ", paste(sort(unique(imc_info$sample_num)), collapse = ", "))
  
  # Cross-reference with SPE samples
  missing_in_spe <- setdiff(imc_info$sample_num, spe_samples)
  missing_in_imc <- setdiff(spe_samples, imc_info$sample_num)
  
  if (length(missing_in_spe) > 0) {
    warning("IMC files found with sample numbers not in SPE: ", paste(missing_in_spe, collapse = ", "))
  }
  if (length(missing_in_imc) > 0) {
    warning("SPE samples without corresponding IMC files: ", paste(missing_in_imc, collapse = ", "))
  }
  
  return(imc_info)
}

#############################
## Module 4: Marker and Channel Validation
#############################
validate_markers <- function(spe, images, panel) {
  message("\n=== Marker and Channel Validation ===")
  
  # Define expected markers (cell-type signatures)
  cell_markers <- list(
    macrophages = c("CD45", "CD11b", "CD206"),
    neutrophils  = c("CD45", "Ly6G"),
    endothelial  = c("CD31"),
    fibroblasts  = c("CD140b")
  )
  
  # Use the panel file to determine which markers are retained
  kept_markers <- panel$name[panel$keep == 1]
  message("\nKept markers in panel:")
  print(kept_markers)
  
  # Validate markers for each cell type
  message("\nCell type marker validation:")
  for (cell_type in names(cell_markers)) {
    required <- cell_markers[[cell_type]]
    available <- required[required %in% kept_markers]
    missing <- setdiff(required, kept_markers)
    
    message(sprintf("\n%s:", cell_type))
    message("- Required: ", paste(required, collapse = ", "))
    message("- Available: ", paste(available, collapse = ", "))
    if (length(missing) > 0) {
      warning(sprintf("Missing markers for %s: %s", 
                      cell_type, paste(missing, collapse = ", ")))
    }
  }
  
  # Compare SPE features (rownames) to kept markers in panel
  features <- rownames(spe)
  message("\nFeature alignment:")
  message("- Features in SPE: ", length(features))
  message("- Kept markers in panel: ", length(kept_markers))
  
  missing_in_spe <- setdiff(kept_markers, features)
  extra_in_spe <- setdiff(features, kept_markers)
  if (length(missing_in_spe) > 0 || length(extra_in_spe) > 0) {
    warning("Mismatched features: markers in panel not in SPE or vice-versa")
  }
}

#############################
## Module 5: Spatial Coordinate Validation
#############################
validate_spatial <- function(spe) {
  message("\n=== Spatial Coordinate Validation ===")
  coords <- spatialCoords(spe)
  
  # Detailed analysis per dimension (X and Y)
  for (dim in c("X", "Y")) {
    dim_idx <- if (dim == "X") 1 else 2
    dim_vals <- coords[, dim_idx]
    
    message(sprintf("\n%s coordinate:", dim))
    message("- Range: ", paste(range(dim_vals), collapse = " to "))
    message("- Unique values: ", length(unique(dim_vals)))
    message("- Decimal places: ", max(nchar(sub(".*\\.", "", as.character(dim_vals)))))
    
    sorted_unique <- sort(unique(dim_vals))
    intervals <- diff(sorted_unique)
    message("- Min interval: ", min(intervals, na.rm = TRUE))
    message("- Max interval: ", max(intervals, na.rm = TRUE))
  }
  
  message("\nSpatial summary:")
  message("Total points: ", nrow(coords))
  message("Points at integer positions: ",
          sum(coords[, 1] %% 1 == 0), " (X) vs ", 
          sum(coords[, 2] %% 1 == 0), " (Y)")
}

#############################
## Main Validation Pipeline
#############################
validate_imc_data <- function() {
  message("Starting comprehensive IMC data validation...\n")
  
  # Configuration injection
  config <- get_validation_config()
  
  # 1. Validate file paths and IMC txt files
  path_info <- validate_paths(config)
  
  # 2. Load data from RDS and CSV files
  data <- load_and_validate_data(path_info$paths)
  
  # 3. Validate sample mapping using the IMC txt files directory
  imc_info <- validate_sample_mapping(data$spe, path_info$imc_files, config)
  
  # 4. Validate marker and channel information
  validate_markers(data$spe, data$images, data$panel)
  
  # 5. Validate spatial coordinates
  validate_spatial(data$spe)
  
  message("\nValidation complete.")
  return(list(
    data = data,
    imc_info = imc_info,
    config = config
  ))
}

#############################
## Execute the Pipeline
#############################
validated_data <- validate_imc_data() 