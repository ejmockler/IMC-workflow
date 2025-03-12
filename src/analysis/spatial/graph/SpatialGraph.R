# SpatialGraph.R
# Responsible for building and managing spatial interaction graphs

#' @import R6
#' @import SpatialExperiment
#' @import scater

SpatialGraph <- R6::R6Class(
  "SpatialGraph",
  
  public = list(
    #' @field config Configuration parameters
    config = NULL,
    
    #' @field logger Logger object
    logger = NULL,
    
    #' @field output_dir Output directory for results
    output_dir = NULL,
    
    #' @description
    #' Initialize a new SpatialGraph object
    #' @param config Configuration parameters
    #' @param logger Logger object
    initialize = function(config = NULL, logger = NULL) {
      if (!is.null(config)) {
        self$config <- config
        if (!is.null(config$paths$output_dir)) {
          self$output_dir <- config$paths$output_dir
        }
      }
      
      self$logger <- logger %||% Logger$new("SpatialGraph")
      
      # Create output directory if it doesn't exist
      if (!is.null(self$output_dir) && !dir.exists(self$output_dir)) {
        dir.create(self$output_dir, recursive = TRUE)
      }
      
      # Verify required packages
      if (!requireNamespace("RANN", quietly = TRUE) || 
          !requireNamespace("igraph", quietly = TRUE)) {
        stop("Required packages not available: RANN, igraph")
      }
    },
    
    #' @description
    #' Build spatial graphs using specified methods
    #' @param spe SpatialExperiment object
    #' @param methods Vector of graph building methods
    #' @param params List of method-specific parameters
    #' @return Updated SpatialExperiment object with graphs
    buildGraphs = function(spe, 
                           methods = c("knn", "expansion", "delaunay"),
                           params = list()) {
      self$logger$info("Building spatial graphs")
      
      # Get spatial coordinates
      if (is.null(spatialCoords(spe))) {
        stop("SpatialExperiment object does not contain spatial coordinates")
      }
      
      coords <- spatialCoords(spe)
      
      for (method in methods) {
        self$logger$info(paste("Building", method, "graph"))
        
        graph <- switch(
          method,
          knn = self$buildKNNGraph(
            coords, 
            k = params$knn_k %||% 10
          ),
          expansion = self$buildExpansionGraph(
            coords, 
            radius = params$expansion_radius %||% 30
          ),
          delaunay = self$buildDelaunayGraph(coords),
          stop(paste("Unknown graph method:", method))
        )
        
        # Store graph in SpatialExperiment metadata
        if (is.null(metadata(spe)$spatial_graphs)) {
          metadata(spe)$spatial_graphs <- list()
        }
        
        metadata(spe)$spatial_graphs[[method]] <- graph
      }
      
      # After building graphs, if visualization is enabled
      if (!is.null(self$config) && 
          !is.null(self$config$visualization$save_plots) && 
          self$config$visualization$save_plots) {
        
        # Create visualization factory
        viz_factory <- VisualizationFactory$new()
        
        # For each graph type, create visualization
        for (method in methods) {
          if (method %in% names(metadata(spe)$spatial_graphs)) {
            # Create graph visualization
            viz <- viz_factory$create_visualization(
              "spatial_graph", 
              spe, 
              results = NULL,
              graph_name = method,
              node_color_by = "celltype"  # Use appropriate column
            )
            
            # Save to appropriate output directory
            output_dir <- file.path(self$config$paths$output_dir, "visualizations", "spatial_graphs")
            dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
            viz$save(file.path(output_dir, paste0(method, "_graph.pdf")))
            
            self$logger$info(paste("Saved", method, "graph visualization"))
          }
        }
      }
      
      return(spe)
    },
    
    #' @description
    #' Build k-nearest neighbors graph
    #' @param coords Matrix of spatial coordinates
    #' @param k Number of nearest neighbors
    #' @return igraph object
    buildKNNGraph = function(coords, k = 10) {
      self$logger$info(paste("Building k-NN graph with k =", k))
      
      # Find k nearest neighbors
      nn <- RANN::nn2(coords, k = k + 1)
      
      # Create edge list
      edges <- lapply(1:nrow(nn$idx), function(i) {
        # Skip the first index (which is the point itself)
        cbind(i, nn$idx[i, 2:(k+1)])
      })
      
      edges <- do.call(rbind, edges)
      
      # Create igraph object
      g <- igraph::graph_from_edgelist(edges, directed = FALSE)
      
      # Add edge weights (distances)
      igraph::E(g)$weight <- nn$dist[edges]
      
      # Check for disconnected components
      comps <- igraph::components(g)
      if (comps$no > 1) {
        self$logger$warn(paste(
          "KNN graph has", comps$no, "disconnected components.",
          "Largest component has", max(comps$csize), "nodes"
        ))
      }
      
      return(g)
    },
    
    #' @description
    #' Build expansion radius graph
    #' @param coords Matrix of spatial coordinates
    #' @param radius Expansion radius
    #' @return igraph object
    buildExpansionGraph = function(coords, radius = 30) {
      self$logger$info(paste("Building expansion graph with radius =", radius))
      
      # Build distance matrix
      dists <- as.matrix(dist(coords))
      
      # Create adjacency matrix
      adj <- dists <= radius
      diag(adj) <- FALSE
      
      # Create igraph object
      g <- igraph::graph_from_adjacency_matrix(adj, mode = "undirected", weighted = TRUE)
      
      # Add edge weights (distances)
      igraph::E(g)$weight <- dists[adj]
      
      # Check for disconnected components
      comps <- igraph::components(g)
      if (comps$no > 1) {
        self$logger$warn(paste(
          "Expansion graph has", comps$no, "disconnected components.",
          "Largest component has", max(comps$csize), "nodes"
        ))
      }
      
      return(g)
    },
    
    #' @description
    #' Build Delaunay triangulation graph
    #' @param coords Matrix of spatial coordinates
    #' @return igraph object
    buildDelaunayGraph = function(coords) {
      self$logger$info("Building Delaunay triangulation graph")
      
      if (!requireNamespace("deldir", quietly = TRUE)) {
        stop("Package 'deldir' is required for Delaunay triangulation")
      }
      
      # Apply Delaunay triangulation
      tri <- deldir::deldir(coords[,1], coords[,2])
      
      # Get edges from triangulation
      edges <- cbind(tri$delsgs$ind1, tri$delsgs$ind2)
      
      # Create igraph object
      g <- igraph::graph_from_edgelist(edges, directed = FALSE)
      
      # Calculate edge weights (distances)
      weights <- sapply(1:nrow(edges), function(i) {
        p1 <- coords[edges[i, 1], ]
        p2 <- coords[edges[i, 2], ]
        sqrt(sum((p1 - p2)^2))
      })
      
      # Add edge weights
      igraph::E(g)$weight <- weights
      
      # Check for disconnected components
      comps <- igraph::components(g)
      if (comps$no > 1) {
        self$logger$warn(paste(
          "Delaunay graph has", comps$no, "disconnected components.",
          "Largest component has", max(comps$csize), "nodes"
        ))
      }
      
      return(g)
    }
  ),
  
  private = list(
    # Helper methods can be added here
  )
) 