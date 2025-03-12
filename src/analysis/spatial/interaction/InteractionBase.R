#' Base class for spatial interaction analysis
#' 
#' @description Provides core functionality and shared methods for all spatial interaction 
#' analysis approaches.
#'
#' @details This class serves as the foundation for more specialized interaction analysis classes.
#' It handles common dependencies, utility methods, and ensures consistent interfaces.

library(R6)
library(SpatialExperiment)

InteractionBase <- R6::R6Class("InteractionBase",
  public = list(
    #' @field spe SpatialExperiment object
    spe = NULL,
    
    #' @field logger Logger object
    logger = NULL,
    
    #' @field n_cores Number of cores for parallel processing
    n_cores = NULL,
    
    #' @field dependency_manager DependencyManager object
    dependency_manager = NULL,
    
    #' @description Create a new InteractionBase object
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
    
    #' @description Check if required phenotype information is available
    #' @param phenotype_column Column containing phenotype information
    #' @param min_categories Minimum number of phenotype categories required
    #' @return Logical indicating if phenotype information is valid
    checkPhenotypeData = function(phenotype_column = "celltype", min_categories = 2) {
      # Check if column exists
      if (!phenotype_column %in% colnames(SummarizedExperiment::colData(self$spe))) {
        if (!is.null(self$logger)) {
          self$logger$log_warning("Phenotype column '%s' not found", phenotype_column)
        }
        return(FALSE)
      }
      
      # Check if there are enough categories
      n_categories <- length(unique(self$spe[[phenotype_column]]))
      if (n_categories < min_categories) {
        if (!is.null(self$logger)) {
          self$logger$log_warning(
            "Not enough phenotype categories (found %d, need at least %d)",
            n_categories, min_categories
          )
        }
        return(FALSE)
      }
      
      return(TRUE)
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