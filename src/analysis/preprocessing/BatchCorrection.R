# BatchCorrection.R
# Responsible for batch correction of IMC data

#' @import R6
#' @import SpatialExperiment

BatchCorrection <- R6::R6Class(
  "BatchCorrection",
  
  public = list(
    #' @field config Configuration parameters
    config = NULL,
    
    #' @field logger Logger object
    logger = NULL,
    
    #' @description
    #' Initialize a new BatchCorrection object
    #' @param config Configuration parameters
    #' @param logger Logger object
    initialize = function(config = NULL, logger = NULL) {
      self$config <- config
      self$logger <- logger %||% Logger$new("BatchCorrection")
      
      # Verify required packages
      if (!requireNamespace("batchelor", quietly = TRUE)) {
        stop("Required package not available: batchelor")
      }
    },
    
    #' @description
    #' Run batch correction on a SpatialExperiment object
    #' @param spe SpatialExperiment object
    #' @param batch_var Column name in colData containing batch information
    #' @param num_pcs Number of principal components to use
    #' @param assay_name Name of assay containing expression values
    #' @param seed Random seed for reproducibility
    #' @return SpatialExperiment object with batch correction
    runBatchCorrection = function(
      spe,
      batch_var = NULL,
      num_pcs = NULL,
      assay_name = "exprs",
      seed = NULL
    ) {
      self$logger$info("Running batch correction")
      
      # Extract configuration parameters
      if (is.null(batch_var)) {
        batch_var <- self$config$batch_correction$batch_variable
        if (is.null(batch_var)) {
          batch_var <- "sample_id"
          self$logger$warn(paste("No batch variable specified, using default:", batch_var))
        }
      }
      
      if (is.null(num_pcs)) {
        num_pcs <- self$config$batch_correction$num_pcs
        if (is.null(num_pcs)) {
          num_pcs <- 50
          self$logger$info(paste("Using default number of PCs:", num_pcs))
        }
      }
      
      if (is.null(seed)) {
        seed <- self$config$batch_correction$seed %||% 12345
      }
      
      # Check if batch variable exists
      if (!batch_var %in% colnames(colData(spe))) {
        stop(paste("Batch variable", batch_var, "not found in colData"))
      }
      
      # Check if there are multiple batches
      batches <- as.character(spe[[batch_var]])
      batch_levels <- unique(batches)
      
      if (length(batch_levels) < 2) {
        self$logger$warn(paste("Only one batch found:", batch_levels))
        self$logger$info("Skipping batch correction")
        return(spe)
      }
      
      self$logger$info(paste(
        "Performing batch correction using variable:", batch_var,
        "with", length(batch_levels), "batches"
      ))
      
      # Extract expression data
      if (!assay_name %in% assayNames(spe)) {
        stop(paste("Assay", assay_name, "not found"))
      }
      
      expr_data <- assay(spe, assay_name)
      
      # Set seed for reproducibility
      set.seed(seed)
      
      # Run fastMNN
      self$logger$info(paste("Running fastMNN with", num_pcs, "PCs"))
      mnn_out <- batchelor::fastMNN(expr_data, batch = batches, d = num_pcs)
      
      # Store corrected PCs in reducedDim
      reducedDim(spe, "MNN") <- reducedDim(mnn_out, "corrected")
      
      # Add metadata about batch correction
      metadata(spe)$batch_correction <- list(
        method = "fastMNN",
        batch_variable = batch_var,
        num_pcs = num_pcs,
        date = Sys.time(),
        batches = batch_levels
      )
      
      # Run UMAP on batch-corrected dimensions
      if (requireNamespace("scater", quietly = TRUE)) {
        self$logger$info("Running UMAP on batch-corrected dimensions")
        spe <- scater::runUMAP(spe, dimred = "MNN", name = "UMAP_MNN")
      }
      
      # Generate diagnostic plots
      if (requireNamespace("ggplot2", quietly = TRUE) && 
          !is.null(self$config$paths$output_dir)) {
        self$generateDiagnosticPlots(spe, batch_var)
      }
      
      self$logger$info("Batch correction complete")
      return(spe)
    },
    
    #' @description
    #' Generate diagnostic plots for batch correction
    #' @param spe SpatialExperiment object with batch correction
    #' @param batch_var Column name in colData containing batch information
    #' @return NULL
    generateDiagnosticPlots = function(spe, batch_var) {
      self$logger$info("Generating batch correction diagnostic plots")
      
      # Create output directory if it doesn't exist
      plot_dir <- file.path(self$config$paths$output_dir, "batch_correction_plots")
      if (!dir.exists(plot_dir)) {
        dir.create(plot_dir, recursive = TRUE)
      }
      
      # 1. Before correction: PCA colored by batch
      if ("PCA" %in% reducedDimNames(spe)) {
        p1 <- scater::plotReducedDim(spe, "PCA", 
                                    colour_by = batch_var, 
                                    point_size = 0.5)
        ggsave(file.path(plot_dir, "batch_pca_before.png"), p1, width = 8, height = 6)
      }
      
      # 2. Before correction: UMAP colored by batch
      if ("UMAP" %in% reducedDimNames(spe)) {
        p2 <- scater::plotReducedDim(spe, "UMAP", 
                                    colour_by = batch_var, 
                                    point_size = 0.5)
        ggsave(file.path(plot_dir, "batch_umap_before.png"), p2, width = 8, height = 6)
      }
      
      # 3. After correction: UMAP colored by batch
      if ("UMAP_MNN" %in% reducedDimNames(spe)) {
        p3 <- scater::plotReducedDim(spe, "UMAP_MNN", 
                                    colour_by = batch_var, 
                                    point_size = 0.5)
        ggsave(file.path(plot_dir, "batch_umap_after.png"), p3, width = 8, height = 6)
      }
      
      # 4. MNN dimensions colored by batch
      if ("MNN" %in% reducedDimNames(spe)) {
        mnn_df <- as.data.frame(reducedDim(spe, "MNN")[, 1:2])
        mnn_df$batch <- spe[[batch_var]]
        
        p4 <- ggplot2::ggplot(mnn_df, ggplot2::aes(x = V1, y = V2, color = batch)) +
          ggplot2::geom_point(size = 0.5, alpha = 0.6) +
          ggplot2::labs(title = "MNN Dimensions",
               x = "MNN Dim 1", y = "MNN Dim 2") +
          ggplot2::theme_minimal()
        
        ggsave(file.path(plot_dir, "batch_mnn_dims.png"), p4, width = 8, height = 6)
      }
      
      self$logger$info(paste("Saved diagnostic plots to", plot_dir))
    }
  ),
  
  private = list(
    # Helper methods can be added here
  )
) 