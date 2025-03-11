#' SpatialGraph class for building and managing spatial interaction graphs
#' 
#' @description Handles construction of various spatial interaction graphs from
#' a SpatialExperiment object, supporting kNN, expansion, and Delaunay methods.
#'
#' @details This class implements the graph construction functionality shown in the
#' "Spatial interaction graphs" section of the workshop, using the imcRtools package.

SpatialGraph <- R6::R6Class("SpatialGraph",
  public = list(
    #' @field spe SpatialExperiment object
    spe = NULL,
    
    #' @field img_id Column name for sample/image ID
    img_id = NULL,
    
    #' @field logger Logger object
    logger = NULL,
    
    #' @field n_cores Number of cores for parallel processing
    n_cores = NULL,
    
    #' @field dependency_manager DependencyManager object
    dependency_manager = NULL,
    
    #' @description Create a new SpatialGraph object
    #' @param spe SpatialExperiment object
    #' @param img_id Column name for sample/image ID
    #' @param logger Logger object
    #' @param n_cores Number of cores for parallel processing
    #' @param dependency_manager DependencyManager object
    initialize = function(spe, img_id = "sample_id", logger = NULL, n_cores = 1, dependency_manager = NULL) {
      self$spe <- spe
      self$img_id <- img_id
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
    
    #' @description Build a k-nearest neighbor graph
    #' @param k Number of neighbors to use
    #' @param max_dist Maximum distance to consider (optional)
    #' @param name Name for the graph (optional)
    #' @return Updated SpatialExperiment object
    buildKnnGraph = function(k = 20, max_dist = NULL, name = "knn_interaction_graph") {
      if (!is.null(self$logger)) self$logger$log_info("Building kNN graph with k = %d", k)
      
      args <- list(
        self$spe,
        img_id = self$img_id,
        type = "knn",
        k = k,
        name = name
      )
      
      if (!is.null(max_dist)) {
        args$max_dist <- max_dist
      }
      
      self$spe <- do.call(imcRtools::buildSpatialGraph, args)
      return(self$spe)
    },
    
    #' @description Build an expansion graph
    #' @param threshold Distance threshold
    #' @param name Name for the graph (optional)
    #' @return Updated SpatialExperiment object
    buildExpansionGraph = function(threshold = 20, name = "expansion_interaction_graph") {
      if (!is.null(self$logger)) self$logger$log_info("Building expansion graph with threshold = %d", threshold)
      
      self$spe <- imcRtools::buildSpatialGraph(
        self$spe,
        img_id = self$img_id,
        type = "expansion",
        threshold = threshold,
        name = name
      )
      
      return(self$spe)
    },
    
    #' @description Build a Delaunay triangulation graph
    #' @param max_dist Maximum distance to consider
    #' @param name Name for the graph (optional)
    #' @return Updated SpatialExperiment object
    buildDelaunayGraph = function(max_dist = 50, name = "delaunay_interaction_graph") {
      if (!is.null(self$logger)) self$logger$log_info("Building Delaunay graph with max_dist = %d", max_dist)
      
      self$spe <- imcRtools::buildSpatialGraph(
        self$spe,
        img_id = self$img_id,
        type = "delaunay",
        max_dist = max_dist,
        name = name
      )
      
      return(self$spe)
    },
    
    #' @description Get a spatial graph by name
    #' @param name Name of the graph
    #' @return SelfHits object representing the graph
    getGraph = function(name) {
      if (!name %in% SingleCellExperiment::colPairNames(self$spe)) {
        if (!is.null(self$logger)) self$logger$log_warning("Graph '%s' not found in SPE", name)
        return(NULL)
      }
      return(SingleCellExperiment::colPair(self$spe, name))
    },
    
    #' @description Plot a spatial graph
    #' @param node_color_by Column to color nodes by
    #' @param sample_id Sample ID to plot (if NULL, plots all samples)
    #' @param graph_name Name of the graph to plot
    #' @param ... Additional parameters to pass to plotSpatial
    #' @return ggplot object
    plotGraph = function(node_color_by = "celltype", sample_id = NULL, graph_name = "knn_interaction_graph", ...) {
      # Use dependency manager to ensure required packages
      private$ensurePackage("ggplot2")
      private$ensurePackage("viridis")
      
      plot_spe <- self$spe
      if (!is.null(sample_id)) {
        plot_spe <- plot_spe[, plot_spe[[self$img_id]] == sample_id]
      }
      
      p <- imcRtools::plotSpatial(
        plot_spe,
        node_color_by = node_color_by,
        img_id = self$img_id,
        draw_edges = TRUE,
        colPairName = graph_name,
        nodes_first = FALSE,
        edge_color_fix = "grey",
        ...
      )
      
      if ("celltype" %in% colnames(SummarizedExperiment::colData(self$spe)) && 
          node_color_by == "celltype" && 
          "color_vectors" %in% names(SpatialExperiment::metadata(self$spe))) {
        p <- p + ggplot2::scale_color_manual(values = SpatialExperiment::metadata(self$spe)$color_vectors$celltype)
      }
      
      return(p)
    }
  ),
  
  private = list(
    #' @description Load required dependencies
    loadDependencies = function() {
      # If dependency manager is available, use it
      if (!is.null(self$dependency_manager)) {
        self$dependency_manager$install_missing_packages()
        return()
      }
    },
    
    #' @description Ensure a package is installed
    ensurePackage = function(pkg_name) {
      # If dependency manager is available, use it
      if (!is.null(self$dependency_manager)) {
        return(self$dependency_manager$ensure_package(pkg_name))
      }
      
      # Otherwise, use direct installation
      if (!requireNamespace(pkg_name, quietly = TRUE)) {
        if (!is.null(self$logger)) self$logger$log_info("Installing package: %s", pkg_name)
        install.packages(pkg_name)
      }
      return(requireNamespace(pkg_name, quietly = TRUE))
    }
  )
)