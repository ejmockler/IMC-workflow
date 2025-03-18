DataLoader <- R6::R6Class("DataLoader",
  public = list(
    config = NULL,
    logger = NULL,
    
    initialize = function(config, logger) {
      self$config <- config
      self$logger <- logger
    },
    
    # Unified method to load either unsupervised or gated data
    loadData = function(use_gated = NULL) {
      # Determine whether to use gated cells from config if not explicitly provided
      if (is.null(use_gated)) {
        use_gated <- self$config$cell_analysis$use_gated_cells
      }
      
      if (use_gated) {
        return(self$loadGatedCellData())
      } else {
        return(self$loadUnsupervisedData())
      }
    },
    
    # Load all unsupervised data
    loadUnsupervisedData = function() {
      self$logger$info("Loading Steinbock data")
      spe <- self$importSteinbockData()
      
      self$logger$info("Annotating data")
      spe <- self$annotateSteinbockData(spe)
      
      self$logger$info("Processing data")
      spe <- self$processSteinbockData(spe)
      
      if (self$config$cell_analysis$batch_correction) {
        self$logger$info("Performing batch correction")
        spe <- self$performBatchCorrection(spe)
      }
      
      self$logger$info("Performing unsupervised phenotyping")
      spe <- self$performPhenotyping(spe)
      
      # Save processed data if configured
      if (self$config$system$save_intermediate) {
        output_path <- file.path(self$config$paths$output_dir, "spe_phenotyped.rds")
        saveRDS(spe, output_path)
        self$logger$info(paste("Saved phenotyped SPE to:", output_path))
      }
      
      return(spe)
    },
    
    # Load gated cell data
    loadGatedCellData = function() {
      self$logger$info("Loading gated cell data")
      
      # Construct paths to gated cell files
      gated_files <- lapply(self$config$cell_analysis$gated_cell_files, function(filename) {
        file.path(self$config$paths$gated_cells_dir, filename)
      })
      
      # Load each file
      gated_spes <- list()
      for (cell_type in names(gated_files)) {
        file_path <- gated_files[[cell_type]]
        if (file.exists(file_path)) {
          self$logger$info(paste("Loading", cell_type, "cells from", file_path))
          gated_spes[[cell_type]] <- readRDS(file_path)
          
          # Add cell type annotation
          gated_spes[[cell_type]]$gated_celltype <- cell_type
        } else {
          self$logger$warn(paste("File not found:", file_path))
        }
      }
      
      # Merge SPEs if we have multiple
      if (length(gated_spes) > 1) {
        self$logger$info("Merging gated cell datasets")
        merged_spe <- self$mergeSpatialExperiments(gated_spes)
      } else if (length(gated_spes) == 1) {
        merged_spe <- gated_spes[[1]]
      } else {
        self$logger$error("No gated cell files could be loaded")
        stop("Failed to load any gated cell files")
      }
      
      # Save merged data if configured
      if (self$config$system$save_intermediate) {
        output_path <- file.path(self$config$paths$output_dir, "spe_gated_cells.rds")
        saveRDS(merged_spe, output_path)
        self$logger$info(paste("Saved gated cell SPE to:", output_path))
      }
      
      return(merged_spe)
    },
    
    # Import Steinbock data
    importSteinbockData = function() {
      # Ensure required packages are loaded
      requireNamespace("imcRtools", quietly = TRUE)
      requireNamespace("cytomapper", quietly = TRUE)
      
      cat("DEBUG: importSteinbockData - Starting\n")
      
      # Get paths from configuration
      steinbock_data_path <- self$config$paths$data_dir
      panel_file <- self$config$paths$panels$default
      
      cat(sprintf("DEBUG: importSteinbockData - Using data_dir: %s\n", steinbock_data_path))
      cat(sprintf("DEBUG: importSteinbockData - Using panel_file: %s\n", panel_file))
      
      # Get the panel filename only (strip the path)
      panel_filename <- basename(panel_file)
      
      # Check if paths exist
      cat(sprintf("DEBUG: importSteinbockData - Checking paths exist:\n"))
      cat(sprintf("DEBUG: data_dir exists: %s\n", dir.exists(steinbock_data_path)))
      cat(sprintf("DEBUG: intensities folder exists: %s\n", dir.exists(file.path(steinbock_data_path, "intensities"))))
      cat(sprintf("DEBUG: regionprops folder exists: %s\n", dir.exists(file.path(steinbock_data_path, "regionprops"))))
      cat(sprintf("DEBUG: neighbors folder exists: %s\n", dir.exists(file.path(steinbock_data_path, "neighbors"))))
      cat(sprintf("DEBUG: panel file exists: %s\n", file.exists(panel_file)))
      
      # Load SpatialExperiment object with explicit settings
      self$logger$info(paste("Loading data from:", steinbock_data_path))
      
      cat("DEBUG: importSteinbockData - About to call imcRtools::read_steinbock\n")
      tryCatch({
        spe <- imcRtools::read_steinbock(
          path = steinbock_data_path,
          intensities_folder = "intensities",  # Use default or customize
          regionprops_folder = "regionprops",  # Use default or customize  
          graphs_folder = "neighbors",         # Use default or customize
          panel_file = panel_file              # Use our configured panel path
        )
        cat("DEBUG: importSteinbockData - Successfully loaded data with read_steinbock\n")
      }, error = function(e) {
        cat(sprintf("DEBUG: importSteinbockData - Error in read_steinbock: %s\n", e$message))
        stop(e)
      })
      
      cat("DEBUG: importSteinbockData - After read_steinbock\n")
      
      # Optionally load images and masks
      if (dir.exists(self$config$paths$image_dir) && dir.exists(self$config$paths$masks_dir)) {
        self$logger$info("Loading images and masks")
        cat("DEBUG: importSteinbockData - Loading images and masks\n")
        
        tryCatch({
          images <- cytomapper::loadImages(self$config$paths$image_dir)
          cat("DEBUG: importSteinbockData - Images loaded successfully\n")
        }, error = function(e) {
          cat(sprintf("DEBUG: importSteinbockData - Error loading images: %s\n", e$message))
        })
        
        tryCatch({
          masks <- cytomapper::loadImages(self$config$paths$masks_dir, as.is = TRUE)
          cat("DEBUG: importSteinbockData - Masks loaded successfully\n")
        }, error = function(e) {
          cat(sprintf("DEBUG: importSteinbockData - Error loading masks: %s\n", e$message))
        })
        
        # Set channel names to match SPE
        cytomapper::channelNames(images) <- rownames(spe)
        
        # Save images and masks if configured
        if (self$config$system$save_intermediate) {
          output_dir <- self$config$paths$output_dir
          if (!dir.exists(output_dir)) {
            dir.create(output_dir, recursive = TRUE)
          }
          saveRDS(images, file = file.path(output_dir, "images.rds"))
          saveRDS(masks, file = file.path(output_dir, "masks.rds"))
        }
      }
      
      # Save SPE if configured
      if (self$config$system$save_intermediate) {
        output_dir <- self$config$paths$output_dir
        if (!dir.exists(output_dir)) {
          dir.create(output_dir, recursive = TRUE)
        }
        saveRDS(spe, file = file.path(output_dir, "spe.rds"))
      }
      
      return(spe)
    },
    
    # Annotate Steinbock data with metadata
    annotateSteinbockData = function(spe) {
      # Assign unique cell identifiers
      colnames(spe) <- paste0(spe$sample_id, "_", spe$ObjectNumber)
      self$logger$info("Unique cell identifiers assigned to SPE")
      
      # Check if we have external metadata to integrate
      metadata_path <- self$config$paths$metadata
      if (!is.null(metadata_path) && file.exists(metadata_path)) {
        self$logger$info(paste("Merging external metadata from:", metadata_path))
        
        # Read metadata file with options to preserve original column names
        metadata <- read.csv(metadata_path, stringsAsFactors = FALSE, check.names = FALSE, strip.white = TRUE)
        
        # DEBUG: Print column names to see what's actually in the metadata file
        cat("DEBUG: Metadata file column names:\n")
        cat(paste(colnames(metadata), collapse = ", "), "\n")
        
        # Determine join keys (can customize based on your files)
        join_key_spe <- "sample_id" 
        join_key_csv <- "File Name"  # Adjust based on actual CSV header
        
        # Check if columns exist in respective data frames
        if (!(join_key_spe %in% colnames(colData(spe)))) {
          self$logger$warn(paste("Join key", join_key_spe, "not found in SPE"))
          return(spe)
        }
        
        # Clean up column names for more robust matching - trim whitespace
        col_names_clean <- trimws(colnames(metadata))
        join_key_csv_clean <- trimws(join_key_csv)
        
        # Check if the column exists (exactly or with trimmed whitespace)
        if (!(join_key_csv %in% colnames(metadata)) && !(join_key_csv_clean %in% col_names_clean)) {
          # Print more details about the search
          cat("DEBUG: Looking for join key:", join_key_csv, "\n")
          cat("DEBUG: It wasn't found in the metadata columns shown above.\n")
          
          # Try to find similar columns that might be the right one
          similar_cols <- grep(gsub(" ", ".*", join_key_csv, ignore.case = TRUE), colnames(metadata), ignore.case = TRUE, value = TRUE)
          cat("DEBUG: Similar column names found:", paste(similar_cols, collapse = ", "), "\n")
          
          # If we found similar columns, use the first one
          if (length(similar_cols) > 0) {
            join_key_csv <- similar_cols[1]
            cat("DEBUG: Using alternative column:", join_key_csv, "\n")
          } else {
            self$logger$warn(paste("Join key", join_key_csv, "not found in metadata file"))
            return(spe)
          }
        } else if (!(join_key_csv %in% colnames(metadata)) && join_key_csv_clean %in% col_names_clean) {
          # If we found a match with trimmed whitespace, use that column
          join_key_csv <- colnames(metadata)[which(col_names_clean == join_key_csv_clean)][1]
          cat("DEBUG: Using column with trimmed whitespace:", join_key_csv, "\n")
        }
        
        # Create mapping table
        spe_ids <- unique(spe[[join_key_spe]])
        matching_rows <- metadata[metadata[[join_key_csv]] %in% spe_ids, ]
        
        if (nrow(matching_rows) == 0) {
          self$logger$warn("No matching rows found in metadata")
          return(spe)
        }
        
        # Join metadata
        for (row_idx in 1:nrow(matching_rows)) {
          sample_id <- matching_rows[row_idx, join_key_csv]
          
          # Add all columns from metadata to matching cells
          for (col_name in setdiff(colnames(matching_rows), join_key_csv)) {
            spe[[col_name]][spe[[join_key_spe]] == sample_id] <- matching_rows[row_idx, col_name]
          }
        }
        
        self$logger$info("External metadata merged into SPE colData")
      }
      
      # Save annotated SPE if configured
      if (self$config$system$save_intermediate) {
        output_path <- file.path(self$config$paths$output_dir, "spe_annotated.rds")
        saveRDS(spe, output_path)
        self$logger$info(paste("Saved annotated SPE to:", output_path))
      }
      
      return(spe)
    },
    
    # Process Steinbock data: transformation, QC, dimensionality reduction
    processSteinbockData = function(spe) {
      # Apply asinh transformation to counts
      counts <- SummarizedExperiment::assay(spe, "counts")
      SummarizedExperiment::assay(spe, "exprs") <- asinh(counts)
      self$logger$info("Counts transformed using asinh")
      
      # Quality control based on cell area
      if (self$config$cell_analysis$qc_filter) {
        # Identify the correct column for cell area
        area_column <- NULL
        available_cols <- colnames(SummarizedExperiment::colData(spe))
        
        if ("cell_area" %in% available_cols) {
          area_column <- "cell_area"
        } else if ("area" %in% available_cols) {
          area_column <- "area"
        } else {
          self$logger$warn("Could not identify cell area column, skipping QC filtering")
          area_column <- NULL
        }
        
        if (!is.null(area_column)) {
          # Calculate bounds using median/MAD approach
          median_area <- stats::median(spe[[area_column]], na.rm = TRUE)
          mad_area <- stats::mad(spe[[area_column]], na.rm = TRUE)
          
          multiplier_mad <- 5  # Can be adjusted in config if needed
          lower_bound <- max(0, median_area - multiplier_mad * mad_area)
          upper_bound <- median_area + multiplier_mad * mad_area
          
          # Apply filtering
          n_cells_original <- ncol(spe)
          cells_to_keep <- which(spe[[area_column]] >= lower_bound & spe[[area_column]] <= upper_bound)
          
          if (length(cells_to_keep) > 0) {
            spe <- spe[, cells_to_keep]
            self$logger$info(sprintf("QC filtering: kept %d of %d cells (%.1f%%)", 
                             ncol(spe), n_cells_original, 100*ncol(spe)/n_cells_original))
          } else {
            self$logger$warn("QC filtering would remove all cells, skipping")
          }
        }
      }
      
      # Flag channels for analysis (exclude DNA/Histone markers)
      SummarizedExperiment::rowData(spe)$use_channel <- !grepl("DNA|Histone", rownames(spe))
      self$logger$info("Flagged channels for downstream analysis")
      
      # Perform dimensionality reduction
      if (requireNamespace("scater", quietly = TRUE) && 
          requireNamespace("BiocSingular", quietly = TRUE)) {
        self$logger$info("Performing dimensionality reduction")
        
        # Set seed for reproducibility
        set.seed(self$config$system$seed)
        
        # Run PCA
        spe <- scater::runPCA(spe, 
                     subset_row = SummarizedExperiment::rowData(spe)$use_channel, 
                     exprs_values = "exprs",
                     BSPARAM = BiocSingular::ExactParam())
        
        # Determine dimensions for UMAP
        pca_dims <- ncol(SingleCellExperiment::reducedDim(spe, "PCA"))
        n_dims <- min(50, pca_dims)
        
        # Run UMAP
        spe <- scater::runUMAP(spe, 
                      dimred = "PCA", 
                      n_dimred = n_dims)
        
        # Run t-SNE
        spe <- scater::runTSNE(spe, 
                      subset_row = SummarizedExperiment::rowData(spe)$use_channel, 
                      exprs_values = "exprs")
        
        self$logger$info("Dimensionality reduction completed (PCA, UMAP, t-SNE)")
      } else {
        self$logger$warn("Required packages for dimensionality reduction not available")
      }
      
      # Save processed SPE if configured
      if (self$config$system$save_intermediate) {
        output_path <- file.path(self$config$paths$output_dir, "spe_processed.rds")
        saveRDS(spe, output_path)
        self$logger$info(paste("Saved processed SPE to:", output_path))
      }
      
      return(spe)
    },
    
    # Placeholder for batch correction method
    performBatchCorrection = function(spe) {
      self$logger$info("Batch correction not yet implemented")
      return(spe)
    },
    
    # Placeholder for phenotyping method
    performPhenotyping = function(spe) {
      self$logger$info("Phenotyping not yet implemented")
      return(spe)
    },
    
    # Merge multiple SPE objects
    mergeSpatialExperiments = function(spe_list) {
      if (length(spe_list) == 0) {
        stop("Empty list provided to mergeSpatialExperiments")
      }
      
      # Use the first SPE as a base
      result <- spe_list[[1]]
      
      # If there are more SPEs, merge them
      if (length(spe_list) > 1) {
        for (i in 2:length(spe_list)) {
          # Ensure column names don't conflict
          if (any(colnames(result) %in% colnames(spe_list[[i]]))) {
            self$logger$warn("Duplicate cell names detected during merge, making them unique")
            colnames(spe_list[[i]]) <- paste0(names(spe_list)[i], "_", colnames(spe_list[[i]]))
          }
          
          # Merge the SPEs
          result <- cbind(result, spe_list[[i]])
        }
      }
      
      self$logger$info(sprintf("Successfully merged %d datasets with %d total cells", 
                      length(spe_list), ncol(result)))
      
      return(result)
    },
    
    # Load images from the configured image directory
    loadImages = function() {
      # Ensure required packages are loaded
      requireNamespace("cytomapper", quietly = TRUE)
      
      # Get image directory from configuration
      image_dir <- self$config$paths$image_dir
      
      if (!dir.exists(image_dir)) {
        self$logger$error(paste("Image directory does not exist:", image_dir))
        stop("Image directory not found")
      }
      
      self$logger$info(paste("Loading images from:", image_dir))
      
      # Load images using cytomapper
      images <- cytomapper::loadImages(image_dir)
      
      return(images)
    },
    
    # Load masks from the configured masks directory
    loadMasks = function() {
      # Ensure required packages are loaded
      requireNamespace("cytomapper", quietly = TRUE)
      
      # Get masks directory from configuration
      masks_dir <- self$config$paths$masks_dir
      
      if (!dir.exists(masks_dir)) {
        self$logger$error(paste("Masks directory does not exist:", masks_dir))
        stop("Masks directory not found")
      }
      
      self$logger$info(paste("Loading masks from:", masks_dir))
      
      # Load masks using cytomapper (with as.is=TRUE for proper handling of mask values)
      masks <- cytomapper::loadImages(masks_dir, as.is = TRUE)
      
      return(masks)
    }
  )
)
