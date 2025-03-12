# QualityControl.R
# Responsible for data quality control, filtering, and transformation

#' @import R6
#' @import SpatialExperiment

QualityControl <- R6::R6Class(
  "QualityControl",
  
  public = list(
    #' @field config Configuration parameters
    config = NULL,
    
    #' @field logger Logger object
    logger = NULL,
    
    #' @description
    #' Initialize a new QualityControl object
    #' @param config Configuration parameters
    #' @param logger Logger object
    initialize = function(config = NULL, logger = NULL) {
      self$config <- config
      self$logger <- logger %||% Logger$new("QualityControl")
    },
    
    #' @description
    #' Process data with quality control and transformations
    #' @param spe SpatialExperiment object
    #' @param is_gated Whether the data is from gated cells
    #' @return Processed SpatialExperiment object
    processData = function(spe, is_gated = FALSE) {
      self$logger$info(paste("Processing", ifelse(is_gated, "gated", "unsupervised"), "data"))
      
      # Apply transformations to count data
      spe <- self$transformCounts(spe)
      
      # Quality control filtering
      spe <- self$performQC(spe)
      
      # Flag channels for downstream analysis
      spe <- self$flagChannels(spe)
      
      # Run dimensionality reduction if not gated
      if (!is_gated) {
        spe <- self$runDimensionalityReduction(spe)
      }
      
      return(spe)
    },
    
    #' @description
    #' Transform count data using asinh transformation
    #' @param spe SpatialExperiment object
    #' @return SpatialExperiment object with transformed counts
    transformCounts = function(spe) {
      self$logger$info("Transforming count data")
      
      # Get cofactor from config
      cofactor <- self$config$processing$asinh_cofactor
      if (is.null(cofactor)) cofactor <- 5
      
      # Transform counts if they exist
      if ("counts" %in% assayNames(spe)) {
        counts <- assay(spe, "counts")
        self$logger$info(paste("Applying asinh transformation with cofactor", cofactor))
        assay(spe, "exprs") <- asinh(counts / cofactor)
      } else {
        self$logger$warn("No 'counts' assay found, skipping transformation")
      }
      
      return(spe)
    },
    
    #' @description
    #' Perform quality control filtering
    #' @param spe SpatialExperiment object
    #' @return Filtered SpatialExperiment object
    performQC = function(spe) {
      self$logger$info("Performing quality control filtering")
      
      # Filter cells based on area, etc.
      min_area <- self$config$processing$min_cell_area
      if (is.null(min_area)) min_area <- 10
      
      if ("area" %in% colnames(colData(spe))) {
        n_before <- ncol(spe)
        keep_cells <- spe$area >= min_area
        spe <- spe[, keep_cells]
        n_after <- ncol(spe)
        n_filtered <- n_before - n_after
        
        self$logger$info(sprintf(
          "Filtered %d cells (%.1f%%) with area < %d", 
          n_filtered, 
          100 * n_filtered / n_before,
          min_area
        ))
      } else {
        self$logger$warn("No 'area' column found, skipping area-based filtering")
      }
      
      # Additional QC metrics
      if ("counts" %in% assayNames(spe)) {
        # Calculate total counts per cell
        spe$total_counts <- colSums(assay(spe, "counts"))
        
        # Calculate number of detected markers (above zero)
        spe$n_detected <- colSums(assay(spe, "counts") > 0)
        
        # Optionally filter by total counts
        min_counts <- self$config$processing$min_total_counts
        if (!is.null(min_counts)) {
          n_before <- ncol(spe)
          keep_cells <- spe$total_counts >= min_counts
          spe <- spe[, keep_cells]
          n_after <- ncol(spe)
          n_filtered <- n_before - n_after
          
          self$logger$info(sprintf(
            "Filtered %d cells (%.1f%%) with total counts < %d", 
            n_filtered, 
            100 * n_filtered / n_before,
            min_counts
          ))
        }
      }
      
      return(spe)
    },
    
    #' @description
    #' Flag channels for downstream analysis
    #' @param spe SpatialExperiment object
    #' @return SpatialExperiment object with flagged channels
    flagChannels = function(spe) {
      self$logger$info("Flagging channels for analysis")
      
      # Get panel file path
      panel_path <- self$config$paths$panel
      
      if (!is.null(panel_path) && file.exists(panel_path)) {
        panel_data <- read.csv(panel_path)
        
        if ("keep" %in% colnames(panel_data)) {
          # Extract channel names to keep
          keep_markers <- panel_data$name[panel_data$keep == 1]
          metadata(spe)$keep_markers <- keep_markers
          self$logger$info(paste("Flagged", length(keep_markers), "markers for analysis"))
          
          # Create new assay with only the kept markers if needed
          if ("counts" %in% assayNames(spe)) {
            keep_indices <- match(keep_markers, rownames(spe))
            keep_indices <- keep_indices[!is.na(keep_indices)]
            
            if (length(keep_indices) > 0) {
              assay(spe, "filtered_counts") <- assay(spe, "counts")[keep_indices, ]
              
              if ("exprs" %in% assayNames(spe)) {
                assay(spe, "filtered_exprs") <- assay(spe, "exprs")[keep_indices, ]
              }
              
              self$logger$info(paste("Created filtered assays with", length(keep_indices), "markers"))
            }
          }
        } else {
          self$logger$warn("Panel file does not contain 'keep' column")
        }
      } else {
        self$logger$warn("Panel file not found, skipping channel flagging")
      }
      
      return(spe)
    },
    
    #' @description
    #' Run dimensionality reduction methods
    #' @param spe SpatialExperiment object
    #' @return SpatialExperiment object with reduced dimensions
    runDimensionalityReduction = function(spe) {
      self$logger$info("Running dimensionality reduction")
      
      # Determine which assay to use
      assay_name <- "exprs"
      if ("filtered_exprs" %in% assayNames(spe) && 
          self$config$processing$use_filtered_markers_for_pca) {
        assay_name <- "filtered_exprs"
        self$logger$info("Using filtered markers for dimensionality reduction")
      }
      
      # Run PCA if scater is available
      if (requireNamespace("scater", quietly = TRUE)) {
        # PCA
        self$logger$info("Running PCA")
        spe <- scater::runPCA(spe, exprs_values = assay_name)
        
        # UMAP
        if (requireNamespace("uwot", quietly = TRUE)) {
          self$logger$info("Running UMAP")
          spe <- scater::runUMAP(spe)
        } else {
          self$logger$warn("Package 'uwot' not available, skipping UMAP")
        }
        
        # t-SNE
        if (requireNamespace("Rtsne", quietly = TRUE)) {
          self$logger$info("Running t-SNE")
          spe <- scater::runTSNE(spe)
        } else {
          self$logger$warn("Package 'Rtsne' not available, skipping t-SNE")
        }
      } else {
        self$logger$warn("Package 'scater' not available, skipping dimensionality reduction")
      }
      
      return(spe)
    }
  ),
  
  private = list(
    # Helper methods can be added here
  )
) 