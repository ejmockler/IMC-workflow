#' Perform Batch Correction on Processed Single-Cell Data
#'
#' This entrypoint loads the processed SpatialExperiment object, performs batch correction
#' using the fastMNN method from the batchelor package, and saves the batch-corrected
#' data with integrated low-dimensional embeddings for downstream analysis.
#'
#' @return The batch-corrected SpatialExperiment object.
#'
#' @example
#'   spe_corrected <- runBatchCorrection()

source("src/core/ConfigurationManager.R")
source("src/core/Logger.R")

runBatchCorrection <- function() {
  # Initialize configuration and logger
  configManager <- ConfigurationManager$new()
  logger <- Logger$new(log_file = "logs/batchCorrection.log", log_level = "INFO")
  
  # Load the processed SPE object
  spe <- readRDS(file.path(configManager$config$output$dir, "spe_processed.rds"))
  logger$log_info("Loaded processed SPE with %d cells.", ncol(spe))
  
  # Perform batch correction using fastMNN
  library(batchelor)
  library(scater)
  
  # Get the batch variable from config (default to patient_id or sample_id)
  batch_var <- configManager$config$batch_correction$batch_variable
  if (is.null(batch_var)) {
    if ("patient_id" %in% colnames(SummarizedExperiment::colData(spe))) {
      batch_var <- "patient_id"
    } else if ("sample_id" %in% colnames(SummarizedExperiment::colData(spe))) {
      batch_var <- "sample_id"
    } else {
      logger$log_error("No suitable batch variable found in SPE colData")
      stop("Batch variable must be specified in config or available in SPE")
    }
  }
  logger$log_info("Using '%s' as batch variable for correction", batch_var)
  
  # Run fastMNN with auto-merge option
  seed <- configManager$config$batch_correction$seed
  num_pcs <- configManager$config$batch_correction$num_pcs
  logger$log_info("Running fastMNN batch correction...")
  
  # Save original column names to restore later
  original_colnames <- colnames(spe)
  
  # Calculate appropriate number of PCs based on data dimensions
  # This ensures we don't request more singular values than available
  n_features <- sum(rowData(spe)$use_channel)
  n_cells <- ncol(spe)
  max_pcs <- min(50, min(n_features, n_cells) - 1)
  
  # Get requested PCs from config or use calculated max
  num_pcs <- min(num_pcs %||% 50, max_pcs)
  logger$log_info("Using %d principal components for batch correction", num_pcs)
  
  # Try to run fastMNN with adjusted parameters
  tryCatch({
    out <- fastMNN(spe, 
                   batch = spe[[batch_var]],
                   auto.merge = TRUE,
                   subset.row = rowData(spe)$use_channel,
                   assay.type = "exprs",
                   d = num_pcs)
    
    # Transfer correction results to main SPE object
    reducedDim(spe, "fastMNN") <- reducedDim(out, "corrected")
    logger$log_info("Batch correction completed, low-dimensional embedding saved as 'fastMNN'")
    
    # Compute UMAP on corrected data
    set.seed(seed %||% 220228)
    spe <- runUMAP(spe, dimred = "fastMNN", name = "UMAP_corrected")
    logger$log_info("UMAP computed on batch-corrected data")
    
    # Log batch correction diagnostics
    merge_info <- metadata(out)$merge.info
    logger$log_info("Batch correction merge diagnostics:")
    for (i in seq_len(nrow(merge_info))) {
      # Use as.character() or convert to string representation that works
      left_label <- if(is(merge_info$left[i], "S4")) "Batch group" else as.character(merge_info$left[i])
      right_label <- if(is(merge_info$right[i], "S4")) "Batch group" else as.character(merge_info$right[i])
      
      logger$log_info("  Merge %d: %s + %s, batch size: %.3f, max lost var: %.3f", 
                     i, 
                     left_label,
                     right_label,
                     merge_info$batch.size[i], 
                     rowMax(merge_info$lost.var)[i])
    }
    
    # After batch correction, restore column names if they were lost
    if (!is.null(original_colnames) && length(original_colnames) > 0) {
      if (is.null(colnames(spe)) || length(colnames(spe)) == 0) {
        logger$log_info("Restoring original column names after batch correction")
        colnames(spe) <- original_colnames
      }
    }
    
  }, error = function(e) {
    logger$log_error("Error during batch correction: %s", e$message)
    logger$log_warning("Attempting fallback with standard SVD...")
    
    # Fallback to standard SVD
    out <- fastMNN(spe, 
                   batch = spe[[batch_var]],
                   auto.merge = TRUE,
                   subset.row = rowData(spe)$use_channel,
                   assay.type = "exprs",
                   d = num_pcs,
                   BSPARAM = BiocSingular::ExactParam())
    
    # Transfer correction results to main SPE object
    reducedDim(spe, "fastMNN") <- reducedDim(out, "corrected")
    logger$log_info("Batch correction completed with fallback method")
    
    # Compute UMAP on corrected data
    set.seed(seed %||% 220228)
    spe <- runUMAP(spe, dimred = "fastMNN", name = "UMAP_corrected")
    logger$log_info("UMAP computed on batch-corrected data")
  })
  
  # Save the corrected object
  saveRDS(spe, file = file.path(configManager$config$output$dir, "spe_batch_corrected.rds"))
  logger$log_info("Batch-corrected SPE saved to output directory: %s", configManager$config$output$dir)
  
  # Optional visualization if running interactively
  if (interactive()) {
    message("Generating batch correction visualization...")
    library(dittoSeq)
    library(cowplot)
    
    # Check if column names exist, if not, add them
    if (is.null(colnames(spe)) || length(colnames(spe)) == 0) {
      logger$log_warning("No column names found in SPE object. Adding generic cell names.")
      colnames(spe) <- paste0("cell", seq_len(ncol(spe)))
    }
    
    # Compare UMAP before and after correction
    p1 <- dittoDimPlot(spe, var = batch_var, 
                       reduction.use = "UMAP", size = 0.2) + 
      ggtitle(paste(batch_var, "on UMAP before correction"))
      
    p2 <- dittoDimPlot(spe, var = batch_var, 
                       reduction.use = "UMAP_corrected", size = 0.2) + 
      ggtitle(paste(batch_var, "on UMAP after correction"))
    
    plot_grid(p1, p2)
    
    # Save the visualization
    ggsave(file.path(configManager$config$output$dir, "batch_correction_comparison.png"), 
           plot = plot_grid(p1, p2), width = 10, height = 5)
  }
  
  invisible(spe)
}

if (interactive() || identical(environment(), globalenv())) {
  spe_corrected <- runBatchCorrection()
}