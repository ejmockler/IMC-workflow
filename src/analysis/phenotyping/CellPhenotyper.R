#' CellPhenotyper class for segmentation-based cell phenotyping
#' 
#' Implements clustering algorithms for identifying cell phenotypes from segmented cells
#' in SpatialExperiment objects. This class supports both batch-corrected and raw expression
#' based phenotyping using Rphenoannoy/Rphenograph algorithms.

library(R6)
library(SpatialExperiment)
library(igraph)

#' CellPhenotyper class
#'
#' @description
#' R6 class for performing cell phenotyping via clustering on SpatialExperiment objects
#' containing segmented cells.
CellPhenotyper <- R6::R6Class("CellPhenotyper",
  private = list(
    .config = NULL,
    .logger = NULL,
    
    # Track performance
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
    #' Initialize a CellPhenotyper object
    #' 
    #' @param config Configuration object or list
    #' @param logger Optional logger object
    initialize = function(config = NULL, logger = NULL) {
      private$.config <- config
      private$.logger <- logger
      
      if (is.null(private$.logger)) {
        private$.logger <- Logger$new("CellPhenotyper")
      }
      
      private$.logger$info("CellPhenotyper initialized")
    },
    
    #' Perform clustering on a SpatialExperiment object
    #' 
    #' @param spe SpatialExperiment object with expression data
    #' @param use_corrected_embedding Whether to use batch-corrected embedding for clustering
    #' @param k Number of nearest neighbors for clustering (k parameter)
    #' @param run_both_embeddings Whether to run clustering on both corrected and raw embeddings
    #' @param seed Random seed for reproducibility  
    #' @param n_cores Number of cores for parallel processing
    #' @return SpatialExperiment object with added phenotype cluster assignments
    phenotypeCells = function(
      spe, 
      use_corrected_embedding = TRUE, 
      k = 45,
      run_both_embeddings = FALSE,
      seed = 220619,
      n_cores = 1
    ) {
      start_time <- Sys.time()
      
      private$.logger$info("Performing cell phenotyping via clustering")
      
      # Check required packages
      if (!requireNamespace("igraph", quietly = TRUE)) {
        stop("Package 'igraph' is required for cell phenotyping")
      }
      
      # Check which phenograph implementation is available
      use_rphenoannoy <- requireNamespace("Rphenoannoy", quietly = TRUE)
      use_rphenograph <- requireNamespace("Rphenograph", quietly = TRUE)
      
      # Report which package is available
      if (use_rphenoannoy) {
        private$.logger$info("Using Rphenoannoy package for clustering")
        library(Rphenoannoy)
      } else if (use_rphenograph) {
        private$.logger$info("Using Rphenograph package for clustering")
        library(Rphenograph)
      } else {
        # Error if no appropriate package
        stop("Either Rphenoannoy or Rphenograph is required for phenotyping")
      }
      
      # Prepare data for clustering
      private$.logger$info(sprintf("Preparing data for clustering (k=%d)...", k))
      
      # Process the primary embedding
      if (use_corrected_embedding) {
        # Use the batch-corrected low-dimensional embedding
        private$.logger$info("Using batch-corrected embedding for clustering")
        if ("fastMNN" %in% names(reducedDims(spe))) {
          mat <- reducedDim(spe, "fastMNN")
        } else if ("MNN" %in% names(reducedDims(spe))) {
          mat <- reducedDim(spe, "MNN")
        } else {
          private$.logger$warn("No batch-corrected embedding found. Falling back to expression values.")
          use_corrected_embedding <- FALSE
          use_channels <- rowData(spe)$use_channel %||% rep(TRUE, nrow(spe))
          mat <- t(assay(spe, "exprs")[use_channels,])
        }
      } 
      
      if (!use_corrected_embedding) {
        # Use the transformed expression values
        private$.logger$info("Using expression values for clustering")
        use_channels <- rowData(spe)$use_channel %||% rep(TRUE, nrow(spe))
        mat <- t(assay(spe, "exprs")[use_channels,])
      }
      
      # Run clustering
      set.seed(seed)
      private$.logger$info("Running clustering...")
      
      if (use_rphenoannoy) {
        # Check what arguments the installed version of Rphenoannoy accepts
        if (length(formals(Rphenoannoy)) > 2) {
          # Newer version with additional parameters
          private$.logger$info("Using extended Rphenoannoy with additional parameters")
          
          # Try to call with all parameters, with a fallback
          tryCatch({
            out <- Rphenoannoy(mat, k = k, approx = TRUE, parallel = n_cores)
          }, error = function(e) {
            private$.logger$warn(sprintf("Error with extended parameters: %s", e$message))
            private$.logger$info("Falling back to basic Rphenoannoy call")
            out <<- Rphenoannoy(mat, k = k)
          })
        } else {
          # Basic version with only data and k parameters
          private$.logger$info("Using basic Rphenoannoy")
          out <- Rphenoannoy(mat, k = k)
        }
      } else {
        # Rphenograph doesn't have approx or parallel parameters
        private$.logger$info("Using Rphenograph")
        out <- Rphenograph(mat, k = k)
      }
      
      clusters <- factor(membership(out[[2]]))
      
      # Store cluster assignments in SPE
      column_name <- if (use_corrected_embedding) "phenograph_corrected" else "phenograph_raw"
      spe[[column_name]] <- clusters
      
      private$.logger$info(sprintf("Identified %d clusters across %d cells", 
                     length(unique(clusters)), length(clusters)))
      
      # Optional: Run additional clustering on the other embedding
      if (run_both_embeddings) {
        private$.logger$info("Running clustering on alternative embedding...")
        
        if (use_corrected_embedding) {
          alt_mat <- t(assay(spe, "exprs")[rowData(spe)$use_channel,])
          alt_col <- "phenograph_raw"
        } else {
          if ("fastMNN" %in% names(reducedDims(spe))) {
            alt_mat <- reducedDim(spe, "fastMNN")
          } else if ("MNN" %in% names(reducedDims(spe))) {
            alt_mat <- reducedDim(spe, "MNN")
          } else {
            private$.logger$warn("No batch-corrected embedding found for alternative clustering.")
            alt_mat <- NULL
          }
          alt_col <- "phenograph_corrected"
        }
        
        if (!is.null(alt_mat)) {
          # Use the same clustering approach as before
          if (use_rphenoannoy) {
            if (length(formals(Rphenoannoy)) > 2) {
              tryCatch({
                alt_out <- Rphenoannoy(alt_mat, k = k, approx = TRUE, parallel = n_cores)
              }, error = function(e) {
                alt_out <<- Rphenoannoy(alt_mat, k = k)
              })
            } else {
              alt_out <- Rphenoannoy(alt_mat, k = k)
            }
          } else {
            alt_out <- Rphenograph(alt_mat, k = k)
          }
          
          alt_clusters <- factor(membership(alt_out[[2]]))
          spe[[alt_col]] <- alt_clusters
          
          private$.logger$info("Alternative clustering identified %d clusters", 
                       length(unique(alt_clusters)))
        }
      }
      
      # Track performance
      private$.trackPerformance(start_time, "Cell phenotyping")
      
      return(spe)
    },
    
    #' Generate visualizations of phenotyping results
    #' 
    #' @param spe SpatialExperiment object with phenotype assignments
    #' @param output_dir Directory to save visualizations
    #' @param max_cells Maximum number of cells to include in heatmap
    #' @param column_name Column name with cluster assignments
    #' @param reduction Dimensionality reduction to use for plotting
    #' @return List of plot objects
    visualizePhenotypes = function(
      spe,
      output_dir = "output",
      max_cells = 2000,
      column_name = "phenograph_corrected",
      reduction = "UMAP"
    ) {
      if (!requireNamespace("dittoSeq", quietly = TRUE)) {
        private$.logger$warn("dittoSeq package required for visualization")
        return(NULL)
      }
      
      if (!requireNamespace("ggplot2", quietly = TRUE)) {
        private$.logger$warn("ggplot2 package required for visualization")
        return(NULL)
      }
      
      if (!dir.exists(output_dir)) {
        dir.create(output_dir, recursive = TRUE)
      }
      
      plots <- list()
      
      # Check if column exists
      if (!(column_name %in% colnames(colData(spe)))) {
        private$.logger$warn(sprintf("Column %s not found in SPE object", column_name))
        return(NULL)
      }
      
      # First check if there are any patient_id or sample_id columns for annotations
      annot_columns <- c(column_name)
      if ("patient_id" %in% colnames(SummarizedExperiment::colData(spe))) {
        annot_columns <- c(annot_columns, "patient_id")
      } else if ("sample_id" %in% colnames(SummarizedExperiment::colData(spe))) {
        annot_columns <- c(annot_columns, "sample_id")
      }
      
      # Visualize clusters on dimensionality reduction
      private$.logger$info("Creating dimensionality reduction plot")
      p1 <- dittoSeq::dittoDimPlot(spe, var = column_name, 
                    reduction.use = reduction, size = 0.2,
                    do.label = TRUE) +
        ggplot2::ggtitle(sprintf("Phenograph clusters on %s", reduction))
      
      plots$dimplot <- p1
      
      # Save the plot
      plot_file <- file.path(output_dir, sprintf("phenograph_clusters_%s.png", reduction))
      ggplot2::ggsave(plot_file, plot = p1, width = 8, height = 6)
      private$.logger$info(sprintf("Saved cluster plot to %s", plot_file))
      
      # Sample a subset of cells for heatmap visualization
      set.seed(220619)
      n_cells <- min(max_cells, ncol(spe))
      cur_cells <- sample(seq_len(ncol(spe)), n_cells)
      
      # Create heatmap of marker expression by cluster
      if (requireNamespace("ComplexHeatmap", quietly = TRUE)) {
        private$.logger$info("Creating heatmap of marker expression by cluster")
        
        tryCatch({
          use_channels <- rowData(spe)$use_channel %||% rep(TRUE, nrow(spe))
          h1 <- dittoSeq::dittoHeatmap(spe[,cur_cells], 
                          genes = rownames(spe)[use_channels],
                          assay = "exprs", scale = "none",
                          heatmap.colors = viridis::viridis(100), 
                          annot.by = annot_columns)
          
          plots$heatmap <- h1
          
          # Save the heatmap
          png(file.path(output_dir, "phenograph_heatmap.png"), 
              width = 1200, height = 1000, res = 150)
          ComplexHeatmap::draw(h1)
          dev.off()
          private$.logger$info("Saved heatmap to phenograph_heatmap.png")
          
        }, error = function(e) {
          private$.logger$warn(sprintf("Error creating heatmap: %s", e$message))
        })
      }
      
      return(plots)
    }
  )
) 