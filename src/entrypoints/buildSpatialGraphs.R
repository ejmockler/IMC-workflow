#' Build Spatial Graphs for IMC Data
#'
#' This entrypoint constructs multiple spatial graphs (kNN, expansion, Delaunay)
#' from processed IMC data. These graphs form the foundation for all downstream
#' spatial analyses.
#'
#' @return A SpatialExperiment object with multiple spatial graphs attached.

source("src/core/DependencyManager.R")
source("src/core/ConfigurationManager.R")
source("src/core/Logger.R")
source("src/analysis/SpatialGraph.R")  # New module to create

buildSpatialGraphs <- function(
  input_file = NULL,
  output_dir = NULL,
  graph_types = NULL,
  knn_k = NULL,
  expansion_threshold = NULL,
  delaunay_max_dist = NULL,
  img_id = NULL,
  memory_limit = 0,
  n_cores = NULL
) {
  # Initialize configuration and logger
  configManager <- ConfigurationManager$new()
  logger <- Logger$new(log_file = "logs/buildSpatialGraphs.log", log_level = "INFO")
  
  # Initialize dependencies
  dependencyManager <- DependencyManager$new(logger = logger)
  dependencyManager$validate_environment()
  
  # Get parameters from config or use provided values
  input_file <- input_file %||% configManager$config$spatial_analysis$input_file %||% "output/spe_phenotyped.rds"
  output_dir <- output_dir %||% configManager$config$output$dir %||% "output"
  graph_types <- graph_types %||% configManager$config$spatial_analysis$graph_types %||% c("knn", "expansion", "delaunay")
  knn_k <- knn_k %||% configManager$config$spatial_analysis$knn_k %||% 100
  expansion_threshold <- expansion_threshold %||% configManager$config$spatial_analysis$expansion_threshold %||% 20
  delaunay_max_dist <- delaunay_max_dist %||% configManager$config$spatial_analysis$delaunay_max_dist %||% 50
  img_id <- img_id %||% configManager$config$spatial_analysis$img_id %||% "sample_id"
  
  # Initialize cores if not specified
  if (is.null(n_cores)) {
    n_cores <- configManager$config$spatial_analysis$n_cores %||% max(1, floor(parallel::detectCores() * 0.7))
  }
  
  # Load the SpatialExperiment object
  logger$log_info("Loading SpatialExperiment from: %s", input_file)
  spe <- readRDS(input_file)
  
  # Build spatial graphs
  logger$log_info("Building spatial graphs: %s", paste(graph_types, collapse = ", "))
  
  # Create the SpatialGraph object
  spatialGraph <- SpatialGraph$new(
    spe = spe,
    img_id = img_id,
    logger = logger
  )
  
  # Build the spatial graphs
  for (graph_type in graph_types) {
    if (graph_type == "knn") {
      logger$log_info("Building kNN graph with k = %d", knn_k)
      spe <- spatialGraph$buildKnnGraph(k = knn_k)
    } else if (graph_type == "expansion") {
      logger$log_info("Building expansion graph with threshold = %d", expansion_threshold)
      spe <- spatialGraph$buildExpansionGraph(threshold = expansion_threshold)
    } else if (graph_type == "delaunay") {
      logger$log_info("Building Delaunay graph with max_dist = %d", delaunay_max_dist)
      spe <- spatialGraph$buildDelaunayGraph(max_dist = delaunay_max_dist)
    }
  }
  
  # Save the SpatialExperiment object with spatial graphs
  output_file <- if (!is.null(configManager$config$spatial_analysis$output_file)) {
    configManager$config$spatial_analysis$output_file
  } else {
    file.path(output_dir, "spe_spatial_graphs.rds")
  }
  
  logger$log_info("Saved SpatialExperiment with spatial graphs to: %s", output_file)
  saveRDS(spe, output_file)
  
  if (memory_limit > 0) {
    gc_limit <- memory_limit * 1024 * 1024  # Convert to bytes
    message(sprintf("Setting memory limit to %d MB", memory_limit))
    utils::mem.limit(gc_limit)
  }
  
  invisible(spe)
}

if (interactive() || identical(environment(), globalenv())) {
  spe <- buildSpatialGraphs()
}
