#' Cell type-based neighborhood analysis
#' 
#' @description Analyzes spatial neighborhoods based on cell type composition
#' rather than graph topology. This approach identifies spatial patterns by aggregating
#' cell type information in local neighborhoods.
#'
#' @details Extends the CommunityBase class with methods for cell type-based
#' neighborhood analysis using the imcRtools package.

library(R6)
library(SpatialExperiment)

source("src/analysis/spatial/community/CommunityBase.R")

CellTypeNeighborhood <- R6::R6Class("CellTypeNeighborhood",
  inherit = CommunityBase,
  
  public = list(
    #' @description Create a new CellTypeNeighborhood object
    #' @param spe SpatialExperiment object with spatial graphs and cell types
    #' @param logger Logger object
    #' @param n_cores Number of cores for parallel processing
    #' @param dependency_manager DependencyManager object
    initialize = function(spe, logger = NULL, n_cores = 1, dependency_manager = NULL) {
      super$initialize(spe, logger, n_cores, dependency_manager)
      
      # Load specialized dependencies
      private$loadCellTypeAnalysisDependencies()
      
      invisible(self)
    },
    
    #' @description Aggregate neighbors based on cell types
    #' @param colPairName Name of the spatial graph to use
    #' @param celltype_column Column containing cell type information
    #' @param result_column Name of the column to store neighborhood assignments
    #' @param auto_build_graph Whether to automatically build the graph if missing
    #' @return Updated SpatialExperiment object
    aggregateCellTypeNeighbors = function(
      colPairName = "knn_interaction_graph",
      celltype_column = "celltype",
      result_column = "cn_celltypes",
      auto_build_graph = TRUE
    ) {
      if (!is.null(self$logger)) self$logger$log_info("Aggregating neighbors by cell type")
      
      # Check if required columns exist
      if (!celltype_column %in% colnames(SummarizedExperiment::colData(self$spe))) {
        if (!is.null(self$logger)) {
          self$logger$log_warning("Cell type column '%s' not found", celltype_column)
        }
        return(self$spe)
      }
      
      # Check if the required graph exists
      if (!self$checkSpatialGraphs(colPairName, auto_build_graph)) {
        if (!is.null(self$logger)) self$logger$log_error("Required graph '%s' not found", colPairName)
        return(self$spe)
      }
      
      # Make sure imcRtools is available
      if (!requireNamespace("imcRtools", quietly = TRUE)) {
        if (!is.null(self$logger)) {
          self$logger$log_error("imcRtools package is required for cell type neighborhood analysis")
        }
        return(self$spe)
      }
      
      # Aggregate neighbor information based on cell type
      tryCatch({
        self$spe <- imcRtools::aggregateNeighbors(
          self$spe, 
          colPairName = colPairName, 
          aggregate_by = "metadata", 
          count_by = celltype_column
        )
        
        # Rename the result column if it's not the default
        if (result_column != "cn_celltypes" && "aggregatedNeighbors" %in% colnames(SummarizedExperiment::colData(self$spe))) {
          self$spe[[result_column]] <- self$spe$aggregatedNeighbors
          self$spe$aggregatedNeighbors <- NULL
        }
        
        if (!is.null(self$logger)) {
          self$logger$log_info("Cell type-based neighborhoods created")
        }
      }, error = function(e) {
        if (!is.null(self$logger)) {
          self$logger$log_error("Error in cell type aggregation: %s", e$message)
        }
      })
      
      return(self$spe)
    },
    
    #' @description Cluster cells based on their neighborhood composition
    #' @param neighbor_column Column containing aggregated neighborhood information
    #' @param n_clusters Number of clusters for k-means
    #' @param result_column Name of the column to store cluster assignments
    #' @param seed Random seed for reproducibility
    #' @return Updated SpatialExperiment object
    clusterNeighborhoods = function(
      neighbor_column = "aggregatedNeighbors",
      n_clusters = 6,
      result_column = "neighborhood_cluster",
      seed = 220705
    ) {
      if (!is.null(self$logger)) self$logger$log_info("Clustering neighborhoods (k=%d)", n_clusters)
      
      # Check if the neighborhood column exists
      if (!neighbor_column %in% colnames(SummarizedExperiment::colData(self$spe))) {
        if (!is.null(self$logger)) {
          self$logger$log_warning("Neighborhood column '%s' not found", neighbor_column)
        }
        return(self$spe)
      }
      
      # Get the neighborhood data
      neighborhood_data <- self$spe[[neighbor_column]]
      
      # Check if it's a valid data format
      if (!is.matrix(neighborhood_data) && !is.data.frame(neighborhood_data)) {
        if (!is.null(self$logger)) {
          self$logger$log_error("Neighborhood data is not in the expected format")
        }
        return(self$spe)
      }
      
      # Set seed for reproducibility
      set.seed(seed)
      
      # Run k-means clustering
      km <- stats::kmeans(neighborhood_data, centers = n_clusters)
      
      # Add cluster assignments to SPE object
      self$spe[[result_column]] <- factor(km$cluster)
      
      if (!is.null(self$logger)) {
        self$logger$log_info("Neighborhoods clustered into %d groups", n_clusters)
      }
      
      return(self$spe)
    },
    
    #' @description Calculate enrichment of neighborhoods in each cell type
    #' @param cluster_column Column containing neighborhood cluster assignments
    #' @param celltype_column Column containing cell type information
    #' @param img_id Column containing image identifiers
    #' @param result_column Name of the column to store enrichment results
    #' @return Updated SpatialExperiment object
    calculateTypeEnrichment = function(
      cluster_column = "neighborhood_cluster",
      celltype_column = "celltype",
      img_id = "sample_id",
      result_column = "neighborhood_enrichment"
    ) {
      if (!is.null(self$logger)) self$logger$log_info("Calculating cell type enrichment in neighborhoods")
      
      # Check if required columns exist
      required_cols <- c(cluster_column, celltype_column, img_id)
      missing_cols <- required_cols[!required_cols %in% colnames(SummarizedExperiment::colData(self$spe))]
      
      if (length(missing_cols) > 0) {
        if (!is.null(self$logger)) {
          self$logger$log_warning("Required columns missing: %s", paste(missing_cols, collapse = ", "))
        }
        return(self$spe)
      }
      
      # Get unique cell types and clusters
      cell_types <- unique(self$spe[[celltype_column]])
      clusters <- levels(self$spe[[cluster_column]])
      
      # Calculate enrichment for each image
      images <- unique(self$spe[[img_id]])
      enrichment_results <- list()
      
      for (img in images) {
        # Subset to current image
        img_cells <- self$spe[[img_id]] == img
        if (sum(img_cells) == 0) next
        
        # Get cell types and clusters for this image
        img_types <- self$spe[[celltype_column]][img_cells]
        img_clusters <- self$spe[[cluster_column]][img_cells]
        
        # Calculate the total number of cells in each cluster
        cluster_totals <- table(img_clusters)
        
        # Calculate the enrichment of each cell type in each cluster
        enrichment_matrix <- matrix(0, nrow = length(cell_types), ncol = length(clusters))
        rownames(enrichment_matrix) <- cell_types
        colnames(enrichment_matrix) <- clusters
        
        for (cell_type in cell_types) {
          type_cells <- img_types == cell_type
          if (sum(type_cells) == 0) next
          
          # Count cells of this type in each cluster
          type_cluster_counts <- table(img_clusters[type_cells])
          
          # Calculate enrichment (observed/expected)
          for (clust in names(type_cluster_counts)) {
            observed <- type_cluster_counts[clust]
            expected <- sum(type_cells) * cluster_totals[clust] / sum(img_cells)
            enrichment_matrix[cell_type, clust] <- observed / expected
          }
        }
        
        enrichment_results[[img]] <- enrichment_matrix
      }
      
      # Store the enrichment results in the SPE object
      if (length(enrichment_results) > 0) {
        S4Vectors::metadata(self$spe)[[result_column]] <- enrichment_results
        
        if (!is.null(self$logger)) {
          self$logger$log_info("Enrichment calculated for %d images", length(enrichment_results))
        }
      }
      
      return(self$spe)
    },
    
    #' @description Run complete neighborhood analysis pipeline
    #' @param celltype_column Column containing cell type information
    #' @param colPairName Name of the spatial graph to use
    #' @param n_clusters Number of clusters for neighborhood clustering
    #' @param auto_build_graph Whether to automatically build the graph if missing
    #' @param seed Random seed for reproducibility
    #' @return Updated SpatialExperiment object
    runNeighborhoodAnalysis = function(
      celltype_column = "celltype",
      colPairName = "knn_interaction_graph",
      n_clusters = 6,
      auto_build_graph = TRUE,
      seed = 220705
    ) {
      if (!is.null(self$logger)) self$logger$log_info("Running complete neighborhood analysis")
      
      # Step 1: Aggregate neighbors by cell type
      self$spe <- self$aggregateCellTypeNeighbors(
        colPairName = colPairName,
        celltype_column = celltype_column,
        auto_build_graph = auto_build_graph
      )
      
      # Step 2: Cluster neighborhoods
      self$spe <- self$clusterNeighborhoods(
        n_clusters = n_clusters,
        seed = seed
      )
      
      # Step 3: Calculate enrichment
      self$spe <- self$calculateTypeEnrichment(
        celltype_column = celltype_column
      )
      
      if (!is.null(self$logger)) self$logger$log_info("Neighborhood analysis completed")
      
      return(self$spe)
    }
  ),
  
  private = list(
    #' @description Load cell type analysis dependencies
    loadCellTypeAnalysisDependencies = function() {
      required_packages <- c("imcRtools")
      
      if (!is.null(self$dependency_manager)) {
        for (pkg in required_packages) {
          self$dependency_manager$ensure_package(pkg)
        }
      } else {
        for (pkg in required_packages) {
          if (!requireNamespace(pkg, quietly = TRUE)) {
            warning(sprintf("Package '%s' is required but not installed", pkg))
          }
        }
      }
    }
  )
) 