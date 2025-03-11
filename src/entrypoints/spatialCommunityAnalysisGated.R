#' Perform Spatial Community and Neighborhood Analysis with Gated Cells
#'
#' This entrypoint analyzes the spatial organization of gated cells by detecting communities
#' and neighborhoods based on spatial graphs. It is a modified version of spatialCommunityAnalysis.R
#' that works with gated cell data instead of phenograph clusters.
#'
#' @return A SpatialExperiment object with community and neighborhood annotations.

source("src/core/DependencyManager.R")
source("src/core/ConfigurationManager.R")
source("src/core/Logger.R")
source("src/core/ColumnUtilities.R")
source("src/analysis/SpatialCommunity.R")
source("src/analysis/SpatialAnalysisWorkflows.R")

# Explicitly load required packages
library(SpatialExperiment)
library(SummarizedExperiment)
library(spicyR)
library(imcRtools)
library(BiocParallel)

#' @title Run Spatial Community Analysis with Gated Cells
#' @description Main entry point for spatial community analysis using gated cells
#' @param input_file Path to input file containing SpatialExperiment object with gated cells
#' @param output_dir Directory to save outputs
#' @param community_detection Vector of community detection methods to use
#' @param size_threshold Minimum community size
#' @param cn_method Method for cellular neighborhood analysis
#' @param cn_k k parameter for kNN in cellular neighborhood analysis
#' @param lisa_radii Vector of radii for LISA clustering
#' @param n_clusters Number of clusters for aggregation methods
#' @param memory_limit Optional memory limit in MB
#' @param n_cores Number of cores to use for processing
#' @return SpatialExperiment object with community annotations
runSpatialCommunityAnalysisGated <- function(
  input_file = NULL,
  output_dir = NULL,
  community_detection = NULL,
  size_threshold = NULL,
  cn_method = NULL,
  cn_k = NULL,
  lisa_radii = NULL,
  n_clusters = NULL,
  memory_limit = 0,
  n_cores = NULL
) {
  # Initialize configuration and logger
  configManager <- ConfigurationManager$new()
  logger <- Logger$new(log_file = "logs/spatialCommunityAnalysisGated.log", log_level = "INFO")
  
  # Initialize dependencies
  dependencyManager <- DependencyManager$new(logger = logger)
  dependencyManager$validate_environment()
  
  # Get parameters from config or use provided values
  input_file <- input_file %||% configManager$config$gated_cell_analysis$output_file %||% "output/spe_gated_cells.rds"
  output_dir <- output_dir %||% configManager$config$output$dir %||% "output"
  community_detection <- community_detection %||% configManager$config$community_analysis$methods %||% c("graph_based", "celltype_aggregation", "expression_aggregation", "lisa")
  size_threshold <- size_threshold %||% configManager$config$community_analysis$size_threshold %||% 10
  cn_method <- cn_method %||% configManager$config$community_analysis$cn_method %||% "knn"
  cn_k <- cn_k %||% configManager$config$community_analysis$cn_k %||% 20
  lisa_radii <- lisa_radii %||% configManager$config$community_analysis$lisa_radii %||% c(10, 20, 50)
  n_clusters <- n_clusters %||% configManager$config$community_analysis$n_clusters %||% 6
  
  # Initialize cores if not specified
  if (is.null(n_cores)) {
    n_cores <- configManager$config$community_analysis$n_cores %||% max(1, floor(parallel::detectCores() * 0.7))
  }
  
  # Load the SpatialExperiment object with gated cells
  logger$log_info("Loading SpatialExperiment with gated cells from: %s", input_file)
  spe <- readRDS(input_file)
  
  # Check if the gated_celltype column exists
  if (!"gated_celltype" %in% colnames(colData(spe))) {
    stop("The 'gated_celltype' column is missing from the SpatialExperiment object. Please run loadGatedCells.R first.")
  }
  
  # Check if spatial graphs exist, if not, build them
  if (!any(grepl("interaction_graph", colPairNames(spe)))) {
    logger$log_info("No spatial graphs found. Building spatial graphs...")
    
    # Source the buildSpatialGraphs.R script
    source("src/entrypoints/buildSpatialGraphs.R")
    
    # Build spatial graphs
    spe <- buildSpatialGraphs(
      input_file = input_file,
      output_dir = output_dir,
      graph_types = c("knn", "expansion", "delaunay"),
      knn_k = 20,
      expansion_threshold = 20,
      delaunay_max_dist = 50,
      img_id = "sample_id"
    )
  }
  
  # Perform spatial community detection
  if ("graph_based" %in% community_detection) {
    logger$log_info("Performing graph-based community detection...")
    
    # Use gated_celltype for grouping
    spe <- detectCommunity(
      spe,
      colPairName = "neighborhood",
      size_threshold = size_threshold,
      group_by = "gated_celltype",
      BPPARAM = SerialParam(RNGseed = 220819)
    )
    
    logger$log_info("Graph-based community detection completed.")
  }
  
  # Perform cellular neighborhood analysis based on cell types
  if ("celltype_aggregation" %in% community_detection) {
    logger$log_info("Performing cellular neighborhood analysis based on cell types...")
    
    # Aggregate neighbors by cell type
    spe <- aggregateNeighbors(
      spe,
      colPairName = "knn_interaction_graph",
      aggregate_by = "metadata",
      count_by = "gated_celltype"
    )
    
    # Cluster cells based on neighborhood composition
    set.seed(220705)
    cn_clusters <- kmeans(spe$aggregatedNeighbors, centers = n_clusters)
    spe$cn_gated_celltypes <- as.factor(cn_clusters$cluster)
    
    logger$log_info("Cellular neighborhood analysis based on cell types completed.")
  }
  
  # Perform cellular neighborhood analysis based on expression
  if ("expression_aggregation" %in% community_detection) {
    logger$log_info("Performing cellular neighborhood analysis based on expression...")
    
    # Check if use_channel column exists
    if ("use_channel" %in% colnames(rowData(spe))) {
      subset_rows <- rowData(spe)$use_channel
    } else {
      # Use all channels if use_channel is not defined
      subset_rows <- rep(TRUE, nrow(spe))
    }
    
    # Aggregate neighbors by expression
    spe <- aggregateNeighbors(
      spe,
      colPairName = "knn_interaction_graph",
      aggregate_by = "expression",
      assay_type = "exprs",
      subset_row = subset_rows
    )
    
    # Cluster cells based on mean expression in neighborhood
    set.seed(220705)
    cn_clusters <- kmeans(spe$mean_aggregatedExpression, centers = n_clusters)
    spe$cn_expression_gated <- as.factor(cn_clusters$cluster)
    
    logger$log_info("Cellular neighborhood analysis based on expression completed.")
  }
  
  # Perform LISA clustering
  if ("lisa" %in% community_detection) {
    logger$log_info("Performing LISA clustering...")
    
    # Create a SegmentedCells object for LISA clustering
    cells <- data.frame(row.names = colnames(spe))
    cells$ObjectNumber <- spe$ObjectNumber
    cells$ImageNumber <- spe$sample_id
    cells$AreaShape_Center_X <- spatialCoords(spe)[,"Pos_X"]
    cells$AreaShape_Center_Y <- spatialCoords(spe)[,"Pos_Y"]
    cells$cellType <- spe$gated_celltype
    
    lisa_sc <- SegmentedCells(cells, cellProfiler = TRUE)
    
    # Calculate LISA curves
    lisaCurves <- lisa(lisa_sc, Rs = lisa_radii)
    
    # Set NA to 0
    lisaCurves[is.na(lisaCurves)] <- 0
    
    # Cluster cells based on LISA curves
    set.seed(220705)
    lisa_clusters <- kmeans(lisaCurves, centers = n_clusters)$cluster
    
    # Add LISA clusters to SPE
    spe$lisa_clusters_gated <- as.factor(lisa_clusters)
    
    logger$log_info("LISA clustering completed.")
  }
  
  # Save the SpatialExperiment object with community annotations
  output_file <- file.path(output_dir, "spe_communities_gated.rds")
  saveRDS(spe, output_file)
  logger$log_info("Saved SpatialExperiment with community annotations to: %s", output_file)
  
  # Set memory limit if specified
  if (memory_limit > 0) {
    gc_limit <- memory_limit * 1024 * 1024  # Convert to bytes
    message(sprintf("Setting memory limit to %d MB", memory_limit))
    utils::mem.limit(gc_limit)
  }
  
  # Return the SpatialExperiment object
  invisible(spe)
}

# Run the function if this script is executed directly
if (interactive() || identical(environment(), globalenv())) {
  spe_communities <- runSpatialCommunityAnalysisGated()
} 