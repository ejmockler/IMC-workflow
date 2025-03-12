#' Base class for spatial community detection
#' 
#' @description Provides core functionality and shared methods for all community detection approaches
#'
#' @details This class serves as the foundation for more specialized community detection classes.
#' It handles common dependencies, shared utility methods, and ensures consistent interfaces.

library(R6)
library(SpatialExperiment)

CommunityBase <- R6::R6Class("CommunityBase",
  public = list(
    #' @field spe SpatialExperiment object
    spe = NULL,
    
    #' @field logger Logger object
    logger = NULL,
    
    #' @field n_cores Number of cores for parallel processing
    n_cores = NULL,
    
    #' @field dependency_manager DependencyManager object
    dependency_manager = NULL,
    
    #' @description Create a new CommunityBase object
    #' @param spe SpatialExperiment object with spatial coordinates and cell types
    #' @param logger Logger object for status updates
    #' @param n_cores Number of cores for parallel processing
    #' @param dependency_manager DependencyManager object for package management
    initialize = function(spe, logger = NULL, n_cores = 1, dependency_manager = NULL) {
      self$spe <- spe
      self$logger <- logger
      self$n_cores <- n_cores
      
      # If no dependency manager is provided, create one
      if (is.null(dependency_manager)) {
        # Simply try to create the DependencyManager - if it fails, we'll get NULL
        tryCatch({
          self$dependency_manager <- DependencyManager$new(logger = logger)
        }, error = function(e) {
          if (!is.null(logger)) logger$log_warning("Could not create DependencyManager: %s", e$message)
        })
      } else {
        self$dependency_manager <- dependency_manager
      }
      
      # Load required packages
      private$loadDependencies()
      
      invisible(self)
    },
    
    #' @description Verify required spatial graphs exist
    #' @param required_graphs List of required graph names
    #' @param auto_build Whether to automatically build missing graphs
    #' @return Logical indicating if all required graphs exist
    checkSpatialGraphs = function(required_graphs, auto_build = FALSE) {
      # Get all colPairs in the SPE object
      all_pairs <- S4Vectors::metadata(self$spe)$spatialCoords$colPairNames
      
      if (is.null(all_pairs)) {
        if (!is.null(self$logger)) self$logger$log_warning("No spatial graphs found in SPE object")
        if (auto_build) {
          if (!is.null(self$logger)) self$logger$log_info("Attempting to build required spatial graphs")
          # Import and use SpatialGraph class to build graphs
          spatial_graph <- SpatialGraph$new(self$spe, self$logger, self$n_cores, self$dependency_manager)
          self$spe <- spatial_graph$buildGraph("knn")
          self$spe <- spatial_graph$buildGraph("expansion")
          
          all_pairs <- S4Vectors::metadata(self$spe)$spatialCoords$colPairNames
          if (is.null(all_pairs)) {
            if (!is.null(self$logger)) self$logger$log_error("Failed to build spatial graphs")
            return(FALSE)
          }
        } else {
          return(FALSE)
        }
      }
      
      # Check if all required graphs exist
      missing_graphs <- required_graphs[!required_graphs %in% all_pairs]
      if (length(missing_graphs) > 0) {
        if (!is.null(self$logger)) {
          self$logger$log_warning("Missing required spatial graphs: %s", 
                                 paste(missing_graphs, collapse = ", "))
        }
        if (auto_build) {
          if (!is.null(self$logger)) self$logger$log_info("Attempting to build missing spatial graphs")
          spatial_graph <- SpatialGraph$new(self$spe, self$logger, self$n_cores, self$dependency_manager)
          for (graph_name in missing_graphs) {
            # Extract graph type from name (assume naming convention like "knn_interaction_graph")
            graph_type <- strsplit(graph_name, "_")[[1]][1]
            self$spe <- spatial_graph$buildGraph(graph_type)
          }
          
          # Check again if all required graphs exist
          all_pairs <- S4Vectors::metadata(self$spe)$spatialCoords$colPairNames
          missing_graphs <- required_graphs[!required_graphs %in% all_pairs]
          if (length(missing_graphs) > 0) {
            if (!is.null(self$logger)) {
              self$logger$log_error("Failed to build required spatial graphs: %s", 
                                   paste(missing_graphs, collapse = ", "))
            }
            return(FALSE)
          }
        } else {
          return(FALSE)
        }
      }
      
      return(TRUE)
    },
    
    #' @description Get all cell types from SPE object
    #' @param col_name Column containing cell type information
    #' @return Vector of cell types
    getCellTypes = function(col_name = "celltype") {
      if (!col_name %in% colnames(SummarizedExperiment::colData(self$spe))) {
        if (!is.null(self$logger)) self$logger$log_warning("Cell type column '%s' not found", col_name)
        return(NULL)
      }
      return(unique(self$spe[[col_name]]))
    },
    
    #' @description Get all images from SPE object
    #' @param img_id Column containing image identifiers
    #' @return Vector of image identifiers
    getImages = function(img_id = "sample_id") {
      if (!img_id %in% colnames(SummarizedExperiment::colData(self$spe))) {
        if (!is.null(self$logger)) self$logger$log_warning("Image ID column '%s' not found", img_id)
        return(NULL)
      }
      return(unique(self$spe[[img_id]]))
    }
  ),
  
  private = list(
    #' @description Load required packages
    loadDependencies = function() {
      required_packages <- c("SpatialExperiment", "igraph", "imcRtools")
      
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