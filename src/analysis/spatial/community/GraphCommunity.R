#' Graph-based community detection for spatial analysis
#' 
#' @description Specializes in detecting communities from spatial graphs using
#' graph theory algorithms such as Louvain, Leiden, and label propagation.
#'
#' @details Extends the CommunityBase class with methods specific to graph-based
#' community detection methods. This class handles detection of cellular communities
#' using established graph algorithms.

library(R6)
library(SpatialExperiment)

source("src/analysis/spatial/community/CommunityBase.R")

GraphCommunity <- R6::R6Class("GraphCommunity",
  inherit = CommunityBase,
  
  public = list(
    #' @description Create a new GraphCommunity object
    #' @param spe SpatialExperiment object with spatial graphs
    #' @param logger Logger object
    #' @param n_cores Number of cores for parallel processing
    #' @param dependency_manager DependencyManager object
    initialize = function(spe, logger = NULL, n_cores = 1, dependency_manager = NULL) {
      super$initialize(spe, logger, n_cores, dependency_manager)
      
      # Load graph-specific dependencies
      private$loadGraphDependencies()
      
      invisible(self)
    },
    
    #' @description Detect communities using the specified algorithm
    #' @param method Community detection algorithm: "louvain" (default), "leiden", "walktrap", or "label_prop"
    #' @param colPairName Name of the graph to use for community detection
    #' @param size_threshold Minimum size for communities to keep
    #' @param result_column Name of the column to store community assignments
    #' @param auto_build_graph Whether to automatically build the graph if missing
    #' @return Updated SpatialExperiment object
    detectCommunities = function(
      method = "louvain",
      colPairName = "knn_interaction_graph",
      size_threshold = 10,
      result_column = "community",
      auto_build_graph = TRUE
    ) {
      if (!is.null(self$logger)) self$logger$log_info("Detecting communities using %s algorithm", method)
      
      # Check if the required graph exists
      if (!self$checkSpatialGraphs(colPairName, auto_build_graph)) {
        if (!is.null(self$logger)) self$logger$log_error("Required graph '%s' not found", colPairName)
        return(self$spe)
      }
      
      # Get the interaction graph
      cells_interaction_graph <- igraph::graph_from_adjacency_matrix(
        SpatialExperiment::adjacencyMatrix(self$spe, colPairName),
        mode = "undirected",
        weighted = TRUE
      )
      
      # Run the selected community detection algorithm
      communities <- switch(method,
        "louvain" = {
          if (!is.null(self$logger)) self$logger$log_info("Using Louvain community detection")
          igraph::cluster_louvain(cells_interaction_graph)
        },
        "leiden" = {
          if (!is.null(self$logger)) self$logger$log_info("Using Leiden community detection")
          if (!requireNamespace("leiden", quietly = TRUE)) {
            if (!is.null(self$logger)) self$logger$log_warning("leiden package not available, falling back to Louvain")
            igraph::cluster_louvain(cells_interaction_graph)
          } else {
            leiden::leiden(cells_interaction_graph)
          }
        },
        "walktrap" = {
          if (!is.null(self$logger)) self$logger$log_info("Using walktrap community detection")
          igraph::cluster_walktrap(cells_interaction_graph)
        },
        "label_prop" = {
          if (!is.null(self$logger)) self$logger$log_info("Using label propagation community detection")
          igraph::cluster_label_prop(cells_interaction_graph)
        },
        {
          if (!is.null(self$logger)) self$logger$log_warning("Unknown method '%s', using default (Louvain)", method)
          igraph::cluster_louvain(cells_interaction_graph)
        }
      )
      
      # Convert communities to a factor and add to the SPE object
      community_assignments <- as.factor(igraph::membership(communities))
      
      # Apply size threshold to filter out small communities
      if (size_threshold > 0) {
        if (!is.null(self$logger)) {
          self$logger$log_info("Filtering out communities smaller than %d cells", size_threshold)
        }
        
        # Count cells per community
        community_counts <- table(community_assignments)
        small_communities <- names(community_counts[community_counts < size_threshold])
        
        # Set small communities to NA
        if (length(small_communities) > 0) {
          community_assignments[community_assignments %in% small_communities] <- NA
          if (!is.null(self$logger)) {
            self$logger$log_info("Filtered out %d small communities", length(small_communities))
          }
        }
      }
      
      # Store the community assignments in the SPE object
      self$spe[[result_column]] <- community_assignments
      
      if (!is.null(self$logger)) {
        self$logger$log_info("Detected %d communities with %d cells",
                           length(unique(community_assignments[!is.na(community_assignments)])),
                           sum(!is.na(community_assignments)))
      }
      
      return(self$spe)
    },
    
    #' @description Add community metadata to SPE object
    #' @param community_column Name of the column containing community assignments
    #' @param phenotype_column Name of the column containing cell phenotypes
    #' @param img_id Column containing image identifiers
    #' @param result_prefix Prefix for result columns
    #' @return Updated SpatialExperiment object
    addCommunityMetadata = function(
      community_column = "community",
      phenotype_column = "celltype",
      img_id = "sample_id",
      result_prefix = "comm"
    ) {
      if (!is.null(self$logger)) self$logger$log_info("Adding community metadata")
      
      # Check if required columns exist
      required_cols <- c(community_column, phenotype_column, img_id)
      missing_cols <- required_cols[!required_cols %in% colnames(SummarizedExperiment::colData(self$spe))]
      
      if (length(missing_cols) > 0) {
        if (!is.null(self$logger)) {
          self$logger$log_warning("Required columns missing: %s", paste(missing_cols, collapse = ", "))
        }
        return(self$spe)
      }
      
      # Get the communities
      communities <- self$spe[[community_column]]
      
      # Remove NA communities
      valid_cells <- !is.na(communities)
      if (sum(valid_cells) == 0) {
        if (!is.null(self$logger)) self$logger$log_warning("No valid communities found")
        return(self$spe)
      }
      
      # Calculate the size of each community
      community_sizes <- table(communities[valid_cells])
      
      # Calculate the composition of each community by phenotype
      # For each image and community, calculate the number and percentage of cells of each phenotype
      images <- unique(self$spe[[img_id]])
      community_ids <- sort(as.numeric(names(community_sizes)))
      
      # Create a data frame to store the results
      community_data <- data.frame()
      
      # Loop through each image
      for (img in images) {
        img_cells <- self$spe[[img_id]] == img & valid_cells
        if (sum(img_cells) == 0) {
          if (!is.null(self$logger)) {
            self$logger$log_warning("No valid communities found in image '%s'", img)
          }
          next
        }
        
        # Loop through each community in this image
        for (comm in community_ids) {
          comm_cells <- communities == comm & img_cells
          if (sum(comm_cells) == 0) {
            next
          }
          
          # Calculate phenotype composition
          phenotypes <- self$spe[[phenotype_column]][comm_cells]
          phenotype_counts <- table(phenotypes)
          
          # Calculate percentages
          phenotype_percent <- 100 * phenotype_counts / sum(phenotype_counts)
          
          # Dominant phenotype
          dominant_phenotype <- names(phenotype_counts)[which.max(phenotype_counts)]
          
          # Add to the data frame
          community_data <- rbind(community_data, data.frame(
            image = img,
            community = comm,
            size = sum(comm_cells),
            dominant_phenotype = dominant_phenotype,
            entropy = private$calculateEntropy(phenotype_counts),
            stringsAsFactors = FALSE
          ))
        }
      }
      
      # Add to SPE object
      if (nrow(community_data) > 0) {
        community_metadata_column <- paste0(result_prefix, "_metadata")
        S4Vectors::metadata(self$spe)[[community_metadata_column]] <- community_data
        
        if (!is.null(self$logger)) {
          self$logger$log_info("Added metadata for %d communities", nrow(community_data))
        }
      }
      
      return(self$spe)
    },
    
    #' @description Calculate spatial metrics for communities
    #' @param community_column Name of the column containing community assignments
    #' @param img_id Column containing image identifiers
    #' @param spatial_cols Column names for spatial coordinates
    #' @param result_column Name of the column to store spatial metrics
    #' @return Updated SpatialExperiment object
    calculateSpatialMetrics = function(
      community_column = "community",
      img_id = "sample_id",
      spatial_cols = c("X_position", "Y_position"),
      result_column = "comm_spatial_metrics"
    ) {
      if (!is.null(self$logger)) self$logger$log_info("Calculating spatial metrics for communities")
      
      # Check if required columns exist
      required_cols <- c(community_column, img_id, spatial_cols)
      missing_cols <- required_cols[!required_cols %in% colnames(SummarizedExperiment::colData(self$spe))]
      
      if (length(missing_cols) > 0) {
        if (!is.null(self$logger)) {
          self$logger$log_warning("Required columns missing: %s", paste(missing_cols, collapse = ", "))
        }
        return(self$spe)
      }
      
      # Get the communities
      communities <- self$spe[[community_column]]
      
      # Remove NA communities
      valid_cells <- !is.na(communities)
      if (sum(valid_cells) == 0) {
        if (!is.null(self$logger)) self$logger$log_warning("No valid communities found")
        return(self$spe)
      }
      
      # Calculate spatial metrics
      images <- unique(self$spe[[img_id]])
      community_ids <- unique(communities[valid_cells])
      
      # Create a data frame to store the results
      spatial_metrics <- data.frame()
      
      for (img in images) {
        img_cells <- self$spe[[img_id]] == img & valid_cells
        if (sum(img_cells) == 0) continue
        
        for (comm in community_ids) {
          comm_cells <- communities == comm & img_cells
          if (sum(comm_cells) < 3) {
            # Skip communities with too few cells for meaningful metrics
            next
          }
          
          # Extract spatial coordinates
          x_coords <- self$spe[[spatial_cols[1]]][comm_cells]
          y_coords <- self$spe[[spatial_cols[2]]][comm_cells]
          
          # Calculate centroid
          centroid_x <- mean(x_coords)
          centroid_y <- mean(y_coords)
          
          # Calculate dispersion
          dispersion <- mean(sqrt((x_coords - centroid_x)^2 + (y_coords - centroid_y)^2))
          
          # Calculate area approximation (convex hull)
          if (requireNamespace("geometry", quietly = TRUE) && length(x_coords) >= 3) {
            tryCatch({
              points <- cbind(x_coords, y_coords)
              hull <- geometry::convhulln(points)
              hull_points <- points[unique(c(hull)), ]
              area <- geometry::polyarea(hull_points[,1], hull_points[,2])
            }, error = function(e) {
              area <- NA
            })
          } else {
            # Approximate area using bounding box
            area <- (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
          }
          
          # Add to the data frame
          spatial_metrics <- rbind(spatial_metrics, data.frame(
            image = img,
            community = comm,
            centroid_x = centroid_x,
            centroid_y = centroid_y,
            dispersion = dispersion,
            area = area,
            stringsAsFactors = FALSE
          ))
        }
      }
      
      # Add to SPE object
      if (nrow(spatial_metrics) > 0) {
        S4Vectors::metadata(self$spe)[[result_column]] <- spatial_metrics
        
        if (!is.null(self$logger)) {
          self$logger$log_info("Added spatial metrics for %d communities", nrow(spatial_metrics))
        }
      }
      
      return(self$spe)
    }
  ),
  
  private = list(
    #' @description Load graph-specific dependencies
    loadGraphDependencies = function() {
      required_packages <- c("igraph", "leiden")
      
      if (!is.null(self$dependency_manager)) {
        for (pkg in required_packages) {
          self$dependency_manager$ensure_package(pkg)
        }
      } else {
        for (pkg in required_packages) {
          if (!requireNamespace(pkg, quietly = TRUE)) {
            warning(sprintf("Package '%s' is recommended but not installed", pkg))
          }
        }
      }
    },
    
    #' @description Calculate the entropy of a distribution
    #' @param counts Vector of counts
    #' @return Entropy value
    calculateEntropy = function(counts) {
      p <- counts / sum(counts)
      -sum(p * log2(p), na.rm = TRUE)
    }
  )
) 