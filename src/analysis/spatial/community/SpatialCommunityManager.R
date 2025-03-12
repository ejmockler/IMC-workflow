#' Spatial Community Manager
#' 
#' @description Orchestrates various spatial community detection approaches
#' and serves as the main entry point for community detection functionality.
#'
#' @details This class manages the different community detection methods and provides
#' a unified interface for the analysis workflow. It delegates to specialized classes
#' for specific detection algorithms.

library(R6)
library(SpatialExperiment)

source("src/analysis/spatial/community/CommunityBase.R")
source("src/analysis/spatial/community/GraphCommunity.R")
source("src/analysis/spatial/community/CellTypeNeighborhood.R")
source("src/analysis/spatial/community/LisaClustering.R")

SpatialCommunityManager <- R6::R6Class("SpatialCommunityManager",
  public = list(
    #' @field spe SpatialExperiment object
    spe = NULL,
    
    #' @field logger Logger object
    logger = NULL,
    
    #' @field n_cores Number of cores for parallel processing
    n_cores = NULL,
    
    #' @field dependency_manager DependencyManager object
    dependency_manager = NULL,
    
    #' @field graph_community GraphCommunity instance
    graph_community = NULL,
    
    #' @field cell_neighborhood CellTypeNeighborhood instance
    cell_neighborhood = NULL,
    
    #' @field lisa_clustering LisaClustering instance
    lisa_clustering = NULL,
    
    #' @description Create a new SpatialCommunityManager object
    #' @param spe SpatialExperiment object
    #' @param logger Logger object
    #' @param n_cores Number of cores for parallel processing
    #' @param dependency_manager DependencyManager object
    initialize = function(spe, logger = NULL, n_cores = 1, dependency_manager = NULL) {
      self$spe <- spe
      self$logger <- logger
      self$n_cores <- n_cores
      self$dependency_manager <- dependency_manager
      
      # Initialize component instances as needed
      
      invisible(self)
    },
    
    #' @description Run community analysis using the specified methods
    #' @param methods Vector of methods to run ("graph_based", "celltype_aggregation", "expression_aggregation", "lisa")
    #' @param graph_method Graph community detection algorithm (for graph_based method)
    #' @param colPairName Name of the graph to use
    #' @param celltype_column Column containing cell type information
    #' @param size_threshold Minimum size for communities to keep
    #' @param n_clusters Number of clusters for neighborhood clustering
    #' @param auto_build_graph Whether to automatically build the graph if missing
    #' @return Updated SpatialExperiment object
    runCommunityAnalysis = function(
      methods = c("graph_based", "celltype_aggregation"),
      graph_method = "louvain",
      colPairName = "knn_interaction_graph",
      celltype_column = "celltype",
      size_threshold = 10,
      n_clusters = 6,
      auto_build_graph = TRUE
    ) {
      if (!is.null(self$logger)) {
        self$logger$log_info("Running spatial community analysis with methods: %s", 
                           paste(methods, collapse = ", "))
      }
      
      # Run graph-based community detection
      if ("graph_based" %in% methods) {
        if (!is.null(self$logger)) self$logger$log_info("Running graph-based community detection")
        
        # Initialize GraphCommunity if not already done
        if (is.null(self$graph_community)) {
          self$graph_community <- GraphCommunity$new(
            self$spe, self$logger, self$n_cores, self$dependency_manager
          )
        } else {
          # Update SPE in the existing instance
          self$graph_community$spe <- self$spe
        }
        
        # Run graph-based community detection
        self$spe <- self$graph_community$detectCommunities(
          method = graph_method,
          colPairName = colPairName,
          size_threshold = size_threshold,
          auto_build_graph = auto_build_graph
        )
        
        # Add community metadata
        self$spe <- self$graph_community$addCommunityMetadata(
          community_column = "community",
          phenotype_column = celltype_column
        )
        
        # Calculate spatial metrics
        self$spe <- self$graph_community$calculateSpatialMetrics()
      }
      
      # Run cell type aggregation
      if ("celltype_aggregation" %in% methods) {
        if (!is.null(self$logger)) self$logger$log_info("Running cell type neighborhood analysis")
        
        # Initialize CellTypeNeighborhood if not already done
        if (is.null(self$cell_neighborhood)) {
          self$cell_neighborhood <- CellTypeNeighborhood$new(
            self$spe, self$logger, self$n_cores, self$dependency_manager
          )
        } else {
          # Update SPE in the existing instance
          self$cell_neighborhood$spe <- self$spe
        }
        
        # Run neighborhood analysis
        self$spe <- self$cell_neighborhood$runNeighborhoodAnalysis(
          celltype_column = celltype_column,
          colPairName = colPairName,
          n_clusters = n_clusters,
          auto_build_graph = auto_build_graph
        )
      }
      
      # Add expression aggregation functionality if needed
      if ("expression_aggregation" %in% methods) {
        if (!is.null(self$logger)) {
          self$logger$log_info("Expression aggregation not yet implemented")
        }
      }
      
      # Add LISA functionality if needed
      if ("lisa" %in% methods) {
        if (!is.null(self$logger)) self$logger$log_info("Running LISA-based spatial clustering")
        
        # Initialize LisaClustering if not already done
        if (is.null(self$lisa_clustering)) {
          self$lisa_clustering <- LisaClustering$new(
            self$spe, self$logger, self$n_cores, self$dependency_manager
          )
        } else {
          # Update SPE in the existing instance
          self$lisa_clustering$spe <- self$spe
        }
        
        # Run LISA clustering with appropriate parameters
        self$spe <- self$lisa_clustering$performLisaClustering(
          radii = lisa_radii %||% c(10, 20, 50),
          n_clusters = n_clusters,
          img_id = img_id_column,
          celltype_column = celltype_column,
          result_column = "lisa_clusters",
          save_visualizations = save_visualizations
        )
        
        # Create spatial visualization if requested
        if (save_visualizations) {
          self$lisa_clustering$visualizeSpatialLisaClusters(
            save_path = file.path(visualization_dir, "lisa_clusters_spatial.png")
          )
        }
      }
      
      if (!is.null(self$logger)) {
        self$logger$log_info("Spatial community analysis completed")
      }
      
      return(self$spe)
    },
    
    #' @description Run a complete community analysis workflow
    #' @param img_id Column containing image identifiers
    #' @param celltype_column Column containing cell type information
    #' @param all_methods Whether to run all available methods
    #' @param colPairName Name of the graph to use
    #' @param auto_build_graph Whether to automatically build the graph if missing
    #' @return Updated SpatialExperiment object
    runCompleteWorkflow = function(
      img_id = "sample_id",
      celltype_column = "celltype",
      all_methods = FALSE,
      colPairName = "knn_interaction_graph",
      auto_build_graph = TRUE
    ) {
      if (!is.null(self$logger)) {
        self$logger$log_info("Running complete community analysis workflow")
      }
      
      # Determine which methods to run
      methods <- c("graph_based", "celltype_aggregation")
      if (all_methods) {
        methods <- c(methods, "expression_aggregation", "lisa")
      }
      
      # Check requirements
      if (!celltype_column %in% colnames(SummarizedExperiment::colData(self$spe))) {
        if (!is.null(self$logger)) {
          self$logger$log_warning("Cell type column '%s' not found, some methods may fail", 
                                celltype_column)
        }
      }
      
      # Run community analysis
      self$spe <- self$runCommunityAnalysis(
        methods = methods,
        celltype_column = celltype_column,
        colPairName = colPairName,
        auto_build_graph = auto_build_graph
      )
      
      return(self$spe)
    },
    
    #' @description Get community statistics
    #' @param community_column Column containing community assignments
    #' @param celltype_column Column containing cell type information
    #' @param img_id Column containing image identifiers
    #' @return Data frame with community statistics
    getCommunityStats = function(
      community_column = "community",
      celltype_column = "celltype",
      img_id = "sample_id"
    ) {
      if (!is.null(self$logger)) {
        self$logger$log_info("Gathering community statistics")
      }
      
      # Check if required columns exist
      required_cols <- c(community_column, celltype_column, img_id)
      missing_cols <- required_cols[!required_cols %in% colnames(SummarizedExperiment::colData(self$spe))]
      
      if (length(missing_cols) > 0) {
        if (!is.null(self$logger)) {
          self$logger$log_warning("Required columns missing: %s", paste(missing_cols, collapse = ", "))
        }
        return(NULL)
      }
      
      # Get the communities
      communities <- self$spe[[community_column]]
      
      # Remove NA communities
      valid_cells <- !is.na(communities)
      if (sum(valid_cells) == 0) {
        if (!is.null(self$logger)) self$logger$log_warning("No valid communities found")
        return(NULL)
      }
      
      # Get metadata
      comm_metadata_col <- paste0(sub("_.*$", "", community_column), "_metadata")
      comm_metadata <- S4Vectors::metadata(self$spe)[[comm_metadata_col]]
      
      # Get spatial metrics
      spatial_metrics_col <- paste0(sub("_.*$", "", community_column), "_spatial_metrics")
      spatial_metrics <- S4Vectors::metadata(self$spe)[[spatial_metrics_col]]
      
      # Combine and return
      if (!is.null(comm_metadata) && !is.null(spatial_metrics)) {
        stats <- merge(comm_metadata, spatial_metrics, by = c("image", "community"))
        return(stats)
      } else if (!is.null(comm_metadata)) {
        return(comm_metadata)
      } else if (!is.null(spatial_metrics)) {
        return(spatial_metrics)
      } else {
        # No precomputed stats, calculate basic ones
        stats <- data.frame()
        
        # Get unique images and communities
        images <- unique(self$spe[[img_id]])
        community_ids <- sort(unique(communities[valid_cells]))
        
        for (img in images) {
          img_cells <- self$spe[[img_id]] == img & valid_cells
          
          for (comm in community_ids) {
            comm_cells <- communities == comm & img_cells
            if (sum(comm_cells) == 0) next
            
            # Calculate basic stats
            cell_types <- table(self$spe[[celltype_column]][comm_cells])
            dominant_type <- names(cell_types)[which.max(cell_types)]
            
            stats <- rbind(stats, data.frame(
              image = img,
              community = comm,
              size = sum(comm_cells),
              dominant_type = dominant_type,
              stringsAsFactors = FALSE
            ))
          }
        }
        
        return(stats)
      }
    }
  )
) 