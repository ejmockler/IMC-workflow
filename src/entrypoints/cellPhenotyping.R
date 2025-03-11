#' Perform Cell Phenotyping via Clustering
#'
#' This entrypoint loads the batch-corrected SpatialExperiment object, applies
#' clustering algorithms (Rphenoannoy) to identify cell phenotypes, and saves
#' the annotated cells with cluster assignments for downstream analysis.
#'
#' @return The SpatialExperiment object with cell phenotype annotations.
#'
#' @example
#'   spe_phenotyped <- runCellPhenotyping()

source("src/core/ConfigurationManager.R")
source("src/core/Logger.R")

runCellPhenotyping <- function() {
  # Initialize configuration and logger
  configManager <- ConfigurationManager$new()
  logger <- Logger$new(log_file = "logs/cellPhenotyping.log", log_level = "INFO")
  
  # Load the batch-corrected SPE object
  spe <- readRDS(file.path(configManager$config$output$dir, "spe_batch_corrected.rds"))
  logger$log_info("Loaded batch-corrected SPE with %d cells.", ncol(spe))
  
  # Load required libraries
  library(igraph)
  library(dittoSeq)
  library(viridis)
  
  # Check which phenograph implementation is available
  use_rphenoannoy <- requireNamespace("Rphenoannoy", quietly = TRUE)
  use_rphenograph <- requireNamespace("Rphenograph", quietly = TRUE)
  
  # Report which package is available
  if (use_rphenoannoy) {
    logger$log_info("Using Rphenoannoy package for clustering")
    library(Rphenoannoy)
  } else if (use_rphenograph) {
    logger$log_info("Using Rphenograph package for clustering")
    library(Rphenograph)
  } else {
    # Try to install one of them
    logger$log_warning("No phenograph implementation found. Attempting to install...")
    tryCatch({
      logger$log_info("Attempting to install Rphenoannoy...")
      devtools::install_github("stuchly/Rphenoannoy")
      library(Rphenoannoy)
      use_rphenoannoy <- TRUE
    }, error = function(e) {
      logger$log_warning("Failed to install Rphenoannoy: %s", e$message)
      logger$log_info("Attempting to install Rphenograph...")
      devtools::install_github("JinmiaoChenLab/Rphenograph")
      library(Rphenograph)
      use_rphenograph <- TRUE
    })
    
    if (!use_rphenoannoy && !use_rphenograph) {
      stop("Neither Rphenoannoy nor Rphenograph could be installed. Please install manually.")
    }
  }
  
  # Get phenotyping parameters from configuration
  k_param <- configManager$config$phenotyping$k_nearest_neighbors
  use_corrected <- configManager$config$phenotyping$use_corrected_embedding
  seed_value <- configManager$config$phenotyping$seed
  n_cores <- configManager$config$phenotyping$n_cores
  
  # Prepare data for clustering
  logger$log_info("Preparing data for clustering (k=%d)...", k_param)
  
  if (use_corrected) {
    # Use the batch-corrected low-dimensional embedding
    logger$log_info("Using batch-corrected embedding for clustering")
    mat <- reducedDim(spe, "fastMNN")
  } else {
    # Use the transformed expression values
    logger$log_info("Using expression values for clustering")
    mat <- t(assay(spe, "exprs")[rowData(spe)$use_channel,])
  }
  
  # Run clustering
  set.seed(seed_value %||% 220619)
  logger$log_info("Running clustering...")
  
  if (use_rphenoannoy) {
    # Check what arguments the installed version of Rphenoannoy accepts
    if (length(formals(Rphenoannoy)) > 2) {
      # Newer version with additional parameters
      use_approx <- configManager$config$phenotyping$use_approximate_nn %||% TRUE
      
      logger$log_info("Using extended Rphenoannoy with additional parameters")
      
      # Try to call with all parameters, with a fallback
      tryCatch({
        out <- Rphenoannoy(mat, k = k_param, approx = use_approx, parallel = n_cores)
      }, error = function(e) {
        logger$log_warning("Error with extended parameters: %s", e$message)
        logger$log_info("Falling back to basic Rphenoannoy call")
        out <<- Rphenoannoy(mat, k = k_param)
      })
    } else {
      # Basic version with only data and k parameters
      logger$log_info("Using basic Rphenoannoy")
      out <- Rphenoannoy(mat, k = k_param)
    }
  } else {
    # Rphenograph doesn't have approx or parallel parameters
    logger$log_info("Using Rphenograph")
    out <- Rphenograph(mat, k = k_param)
  }
  
  clusters <- factor(membership(out[[2]]))
  
  # Store cluster assignments in SPE
  column_name <- if (use_corrected) "phenograph_corrected" else "phenograph_raw"
  spe[[column_name]] <- clusters
  
  logger$log_info("Identified %d clusters across %d cells", 
                 length(unique(clusters)), length(clusters))
  
  # Optional: Run additional clustering on the other embedding
  if (configManager$config$phenotyping$run_both_embeddings %||% FALSE) {
    logger$log_info("Running clustering on alternative embedding...")
    
    if (use_corrected) {
      alt_mat <- t(assay(spe, "exprs")[rowData(spe)$use_channel,])
      alt_col <- "phenograph_raw"
    } else {
      alt_mat <- reducedDim(spe, "fastMNN")
      alt_col <- "phenograph_corrected"
    }
    
    # Use the same clustering approach as before
    if (use_rphenoannoy) {
      if (length(formals(Rphenoannoy)) > 2) {
        tryCatch({
          alt_out <- Rphenoannoy(alt_mat, k = k_param, approx = use_approx, parallel = n_cores)
        }, error = function(e) {
          alt_out <<- Rphenoannoy(alt_mat, k = k_param)
        })
      } else {
        alt_out <- Rphenoannoy(alt_mat, k = k_param)
      }
    } else {
      alt_out <- Rphenograph(alt_mat, k = k_param)
    }
    
    alt_clusters <- factor(membership(alt_out[[2]]))
    spe[[alt_col]] <- alt_clusters
    
    logger$log_info("Alternative clustering identified %d clusters", 
                   length(unique(alt_clusters)))
  }
  
  # Save the phenotyped SPE object
  saveRDS(spe, file = file.path(configManager$config$output$dir, "spe_phenotyped.rds"))
  logger$log_info("Phenotyped SPE saved to output directory: %s", configManager$config$output$dir)
  
  # Optional visualization if running interactively
  if (interactive()) {
    message("Generating phenotyping visualizations...")
    
    # Check if column names exist, if not, add them
    if (is.null(colnames(spe)) || length(colnames(spe)) == 0) {
      logger$log_warning("No column names found in SPE object. Adding generic cell names.")
      colnames(spe) <- paste0("cell", seq_len(ncol(spe)))
    }
    
    # First check if there are any patient_id or sample_id columns for annotations
    annot_columns <- c(column_name)
    if ("patient_id" %in% colnames(SummarizedExperiment::colData(spe))) {
      annot_columns <- c(annot_columns, "patient_id")
    } else if ("sample_id" %in% colnames(SummarizedExperiment::colData(spe))) {
      annot_columns <- c(annot_columns, "sample_id")
    }
    
    # Visualize clusters on UMAP
    reduction <- if (use_corrected) "UMAP_corrected" else "UMAP"
    
    p1 <- dittoDimPlot(spe, var = column_name, 
                      reduction.use = reduction, size = 0.2,
                      do.label = TRUE) +
      ggtitle("Phenograph clusters on UMAP")
    
    print(p1)
    
    # Sample a subset of cells for heatmap visualization
    set.seed(220619)
    cur_cells <- sample(seq_len(ncol(spe)), min(2000, ncol(spe)))
    
    # Create heatmap of marker expression by cluster
    tryCatch({
      h1 <- dittoHeatmap(spe[,cur_cells], 
                        genes = rownames(spe)[rowData(spe)$use_channel],
                        assay = "exprs", scale = "none",
                        heatmap.colors = viridis(100), 
                        annot.by = annot_columns)
      
      print(h1)
    }, error = function(e) {
      logger$log_warning("Error creating heatmap: %s", e$message)
      logger$log_info("Trying simplified heatmap...")
      
      # Try a simplified version with just the clusters
      h1 <- dittoHeatmap(spe[,cur_cells], 
                        genes = rownames(spe)[rowData(spe)$use_channel],
                        assay = "exprs", scale = "none",
                        heatmap.colors = viridis(100),
                        annot.by = column_name)
      print(h1)
    })
    
    # Save visualizations
    ggsave(file.path(configManager$config$output$dir, "phenograph_clusters.png"), 
           plot = p1, width = 8, height = 6)
  }
  
  # Set up batch information if available
  batch_var <- configManager$config$phenotyping$batch_variable
  if (is.null(batch_var)) {
    if ("patient_id" %in% colnames(SummarizedExperiment::colData(spe))) {
      batch_var <- "patient_id"
    } else if ("sample_id" %in% colnames(SummarizedExperiment::colData(spe))) {
      batch_var <- "sample_id"
    }
  }
  
  invisible(spe)
}

if (interactive() || identical(environment(), globalenv())) {
  spe_phenotyped <- runCellPhenotyping()
}