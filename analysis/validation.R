# Load required packages for dependency management
library(R6)
source("analysis/core/DependencyManager.R")

# Validate the environment and install any missing packages before proceeding
dep_manager <- DependencyManager$new()
dep_manager$validate_environment()

# Explicitly load libraries that are needed for the validation process
# (These packages have been ensured to be installed by DependencyManager)
library(SpatialExperiment)

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
  
  # Fix marker names in SPE object
  message("\nFixing marker names in SPE object...")
  message("Current marker names:")
  print(rownames(spe))
  
  # Fix CD11b name
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
  
  # Fix channel names in images object
  if (!is.null(images) && "CD11b 1" %in% channelNames(images)) {
    channelNames(images)[channelNames(images) == "CD11b 1"] <- "CD11b"
    saveRDS(images, paths$images)
    message("\nImages object updated and saved with corrected channel names.")
  }
  
  # Fix channel names in masks object
  if (!is.null(masks) && "CD11b 1" %in% channelNames(masks)) {
    channelNames(masks)[channelNames(masks) == "CD11b 1"] <- "CD11b"
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
  
  # From the IMC filenames, extract Day, Mouse, ROI using regular expressions,
  # assuming the naming pattern is like: "D1_M1_ROI1.txt"
  imc_info <- data.frame(
    filename = imc_files,
    day = gsub("D([1-7])_.*", "\\1", imc_files),
    mouse = gsub(".*_M([1-2])_.*", "\\1", imc_files),
    roi = gsub(".*_ROI([0-9]+).*", "\\1", imc_files),
    stringsAsFactors = FALSE
  )
  
  message("\nIMC file structure:")
  message("Days: ", paste(unique(imc_info$day), collapse = ", "))
  message("Mice: ", paste(unique(imc_info$mouse), collapse = ", "))
  message("ROIs per condition: ", paste(table(imc_info$day, imc_info$mouse), collapse = ", "))
  
  # Report any missing samples based on our expected sample numbers
  missing_numbers <- setdiff(config$expected_samples, spe_samples)
  if (length(missing_numbers) > 0) {
    message("\nMissing sample numbers: ",
            paste(sprintf("%03d", missing_numbers), collapse = ", "))
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