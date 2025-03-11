#' Perform Spatial Community and Neighborhood Analysis
#'
#' This entrypoint analyzes the spatial organization of cells by detecting communities
#' and neighborhoods based on spatial graphs. It supports multiple methods including
#' graph-based community detection, cellular neighborhood aggregation, and LISA clustering.
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

#' @title Run Spatial Community Analysis
#' @description Main entry point for spatial community analysis
#' @param input_file Path to input file containing SpatialExperiment object with spatial graphs
#' @param output_dir Directory to save outputs
#' @param community_detection Vector of community detection methods to use
#' @param compartment_column Column to use for compartment-based community detection
#' @param size_threshold Minimum community size
#' @param cn_method Method for cellular neighborhood analysis
#' @param cn_k k parameter for kNN in cellular neighborhood analysis
#' @param lisa_radii Vector of radii for LISA clustering
#' @param n_clusters Number of clusters for aggregation methods
#' @param phenotyping_column Column to use for cell phenotyping
#' @param memory_limit Optional memory limit in MB
#' @param n_cores Number of cores to use for processing
#' @return SpatialExperiment object with community annotations
runSpatialCommunityAnalysis <- function(
  input_file = NULL,
  output_dir = NULL,
  community_detection = NULL,
  compartment_column = NULL,
  size_threshold = NULL,
  cn_method = NULL,
  cn_k = NULL,
  lisa_radii = NULL,
  n_clusters = NULL,
  phenotyping_column = NULL,  # New parameter to specify which phenotyping result to use
  memory_limit = 0,
  n_cores = NULL
) {
  # ---- Initialize configuration, logging, and dependencies ----
  configManager <- ConfigurationManager$new()
  logger <- Logger$new(log_file = "logs/spatialCommunityAnalysis.log", log_level = "INFO")
  
  logger$log_info("Starting Spatial Community Analysis")
  
  dependencyManager <- DependencyManager$new(logger = logger)
  dependencyManager$validate_environment()
  
  # ---- Get parameters from config or use provided values ----
  input_file <- input_file %||% configManager$config$community_analysis$input_file %||% "output/spe_spatial_graphs.rds"
  output_dir <- output_dir %||% configManager$config$output$dir %||% "output"
  community_detection <- community_detection %||% configManager$config$community_analysis$methods %||% 
    c("graph_based", "celltype_aggregation", "expression_aggregation", "lisa")
  compartment_column <- compartment_column %||% configManager$config$community_analysis$compartment_column %||% "celltype"
  size_threshold <- size_threshold %||% configManager$config$community_analysis$size_threshold %||% 10
  cn_method <- cn_method %||% configManager$config$community_analysis$cn_method %||% "knn"
  cn_k <- cn_k %||% configManager$config$community_analysis$cn_k %||% 20
  lisa_radii <- lisa_radii %||% configManager$config$community_analysis$lisa_radii %||% c(10, 20, 50)
  n_clusters <- n_clusters %||% configManager$config$community_analysis$n_clusters %||% 6
  phenotyping_column <- phenotyping_column %||% configManager$config$community_analysis$phenotyping_column %||% "phenograph_corrected"
  direct_celltype_analysis <- configManager$config$community_analysis$direct_celltype_analysis %||% TRUE
  require_spatial_graphs <- configManager$config$community_analysis$require_spatial_graphs %||% TRUE
  
  # ---- Set up parallel processing ----
  if (is.null(n_cores)) {
    n_cores <- configManager$config$community_analysis$n_cores %||% max(1, floor(parallel::detectCores() * 0.7))
  }
  logger$log_info("Using %d cores for processing", n_cores)
  
  # ---- Ensure output directory exists ----
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
    logger$log_info("Created output directory: %s", output_dir)
  }
  
  # ---- Step 1: Check for spatial graphs and build if needed ----
  if (require_spatial_graphs && !file.exists(input_file)) {
    logger$log_warning("Spatial graphs file not found: %s", input_file)
    logger$log_info("Building spatial graphs before proceeding with community analysis")
    
    # Use the spatial analysis input file to build graphs
    spatial_input <- configManager$config$spatial_analysis$input_file %||% "output/spe_phenotyped.rds"
    
    if (!file.exists(spatial_input)) {
      logger$log_error("Phenotyped SPE file not found: %s. Please run cellPhenotyping.R first.", spatial_input)
      stop("Input file not found. Run cellPhenotyping.R first to generate the phenotyped SpatialExperiment object.")
    }
    
    logger$log_info("Loading phenotyped SpatialExperiment from: %s", spatial_input)
    spe <- readRDS(spatial_input)
    
    # Build spatial graphs
    logger$log_info("Building spatial graphs")
    
    # Create SpatialGraph object
    source("src/analysis/SpatialGraph.R")
    spatialGraph <- SpatialGraph$new(
      spe = spe,
      logger = logger,
      n_cores = n_cores
    )
    
    # Build each graph type
    graph_types <- configManager$config$spatial_analysis$graph_types %||% c("knn", "expansion", "delaunay")
    
    if ("knn" %in% graph_types) {
      k_param <- configManager$config$spatial_analysis$knn_k %||% 20
      logger$log_info("Building KNN graph with k=%d", k_param)
      spe <- spatialGraph$buildKNNGraph(k = k_param)
    }
    
    if ("expansion" %in% graph_types) {
      threshold <- configManager$config$spatial_analysis$expansion_threshold %||% 20
      logger$log_info("Building expansion graph with threshold=%d", threshold)
      spe <- spatialGraph$buildExpansionGraph(threshold = threshold)
    }
    
    if ("delaunay" %in% graph_types) {
      max_dist <- configManager$config$spatial_analysis$delaunay_max_dist %||% 50
      logger$log_info("Building Delaunay graph with max_dist=%d", max_dist)
      spe <- spatialGraph$buildDelaunayGraph(max_dist = max_dist)
    }
    
    # Save the SPE with spatial graphs
    saveRDS(spe, input_file)
    logger$log_info("Saved SpatialExperiment with spatial graphs to: %s", input_file)
  }
  
  # ---- Step 2: Load SpatialExperiment object ----
  logger$log_info("Loading SpatialExperiment from: %s", input_file)
  spe <- readRDS(input_file)
  
  # ---- Step 3: Prepare SpatialExperiment object for analysis ----
  # Ensure the SpatialExperiment has necessary graphs for our analysis
  spe <- ensureSpatialGraphs(spe, c("neighborhood", "knn_interaction_graph"), logger)
  
  # Create SpatialCommunity object before cell type preparation
  spatialCommunity <- SpatialCommunity$new(
    spe = spe,
    logger = logger,
    n_cores = n_cores
  )
  
  # ---- Step 4: Prepare cell types and other metadata ----
  spe <- prepareForSpatialAnalysis(spe, phenotyping_column, logger, spatialCommunity)
  
  # Update spatialCommunity's SPE object with the prepared data
  spatialCommunity$spe <- spe
  
  # ---- Step 5: Create immune cell classification if needed ----
  if (!("is_immune" %in% colnames(SummarizedExperiment::colData(spe)))) {
    logger$log_info("Classifying immune cells based on cell type classification")
    
    # First approach: Use celltype_classified if available
    if ("celltype_classified" %in% colnames(SummarizedExperiment::colData(spe))) {
      # Identify cells classified as immune type
      cell_types <- SummarizedExperiment::colData(spe)$celltype_classified
      is_immune <- grepl("Immune|CD45|CD3|CD4|CD8|CD11b|CD11c|CD20|Ly6G|CD206|CD68|MHCII", cell_types, ignore.case = TRUE)
      
      SummarizedExperiment::colData(spe)$is_immune <- is_immune
      logger$log_info("Identified %d immune cells out of %d total cells using cell type classification", 
                     sum(is_immune), length(is_immune))
    }
    # Fall back to marker-based approach if needed
    else if ("counts" %in% assayNames(spe)) {
      # Get marker definitions from configuration
      immune_markers <- configManager$config$community_analysis$immune_markers
      if (is.null(immune_markers) || length(immune_markers) == 0) {
        # Default immune markers if not specified in config
        immune_markers <- c("CD45", "Ly6G", "CD11b")
        logger$log_info("Using default immune markers: %s", paste(immune_markers, collapse=", "))
      }
      
      # Function to check if a marker exists in the dataset
      marker_exists <- function(marker) {
        return(marker %in% rownames(assay(spe, "counts")))
      }
      
      # Filter to only markers that exist in the dataset
      available_markers <- immune_markers[sapply(immune_markers, marker_exists)]
      
      if (length(available_markers) > 0) {
        logger$log_info("Using immune markers: %s", paste(available_markers, collapse=", "))
        
        # Get marker expression
        marker_expr <- assay(spe, "counts")[available_markers, , drop = FALSE]
        
        # Calculate median expression per marker
        marker_medians <- apply(marker_expr, 1, median, na.rm = TRUE)
        
        # A cell is immune if it has high expression of any immune marker
        # Using 75th percentile as a more sensitive threshold (changed from 50th)
        is_immune <- apply(marker_expr, 2, function(cell_values) {
          any(cell_values > quantile(marker_expr, 0.75, na.rm = TRUE))
        })
        
        SummarizedExperiment::colData(spe)$is_immune <- is_immune
        logger$log_info("Identified %d immune cells out of %d total cells using marker expression", 
                       sum(is_immune), length(is_immune))
      } else {
        logger$log_warn("No immune markers found in dataset. Setting all cells as non-immune.")
        SummarizedExperiment::colData(spe)$is_immune <- FALSE
      }
    } else {
      logger$log_warn("No assay data found for marker-based immune cell detection")
      SummarizedExperiment::colData(spe)$is_immune <- FALSE
    }
    
    # Final fallback - use celltype column if it contains "Immune"
    if (sum(SummarizedExperiment::colData(spe)$is_immune) == 0 && 
        "celltype" %in% colnames(SummarizedExperiment::colData(spe))) {
      is_immune <- grepl("Immune|CD45|CD3|CD4|CD8|CD11b|CD11c|CD20|B cell|T cell|Macrophage", 
                         SummarizedExperiment::colData(spe)$celltype, ignore.case = TRUE)
      SummarizedExperiment::colData(spe)$is_immune <- is_immune
      logger$log_info("Identified %d immune cells out of %d total cells using celltype column", 
                     sum(is_immune), length(is_immune))
    }
  }
  
  # ---- Step 6: Create SpatialCommunity object and perform analysis ----
  # SpatialCommunity object already created in Step 3
  
  # If phenograph_corrected exists, classify cells by marker expression
  if ("phenograph_corrected" %in% colnames(SummarizedExperiment::colData(spe))) {
    logger$log_info("Classifying cells by marker expression")
    
    # Create output directory for cell type visualizations
    cell_type_dir <- file.path(output_dir, "cell_type_analysis")
    ensureDirectory(cell_type_dir, TRUE, logger)
    
    # Classify cells based on marker expression
    spe <- spatialCommunity$classifyCellTypesByMarkers(
      output_column = "celltype_classified",
      use_config = TRUE
    )
    
    # Visualize the results
    spatialCommunity$visualizeCellTypesByCluster(
      phenograph_column = "phenograph_corrected",
      celltype_column = "celltype_classified",
      save_path = file.path(cell_type_dir, "celltype_by_cluster.png")
    )
    
    # Also visualize marker expression by cluster
    spatialCommunity$visualizeClusterMarkerExpression(
      phenograph_column = "phenograph_corrected",
      save_path = file.path(cell_type_dir, "marker_expression_by_cluster.png")
    )
    
    # Update celltype for further analysis if classification worked
    if ("celltype_classified_cluster" %in% colnames(SummarizedExperiment::colData(spe))) {
      logger$log_info("Using cluster-based cell type classification for analysis")
      spe$celltype <- spe$celltype_classified_cluster
      spatialCommunity$spe <- spe
    }
  }
  
  # ---- Step 7: Perform community detection ----
  # Build parameter list for community detection
  community_params <- list(
    size_threshold = size_threshold,
    n_clusters = n_clusters,
    compartment_column = compartment_column,
    direct_celltype_analysis = direct_celltype_analysis,
    lisa_radii = lisa_radii
  )
  
  # Run community detection methods
  spe <- performCommunityDetection(
    spatialCommunity = spatialCommunity, 
    methods = community_detection,
    parameters = community_params,
    logger = logger
  )
  
  # ---- Step 8: Save results ----
  output_file <- file.path(output_dir, "spe_communities.rds")
  if (!is.null(configManager$config$community_analysis$output_file)) {
    output_file <- configManager$config$community_analysis$output_file
  }
  saveRDS(spe, output_file)
  logger$log_info("Saved SpatialExperiment with community annotations to: %s", output_file)
  
  # ---- Step 9: Run immune infiltration analysis if possible ----
  if ("is_immune" %in% colnames(SummarizedExperiment::colData(spe))) {
    logger$log_info("Running immune infiltration analysis")
    
    # Determine region and condition columns if they exist
    region_col <- findMatchingColumn(spe, c("Details", "ROI", "Region"), logger, "region")
    condition_col <- findMatchingColumn(spe, c("condition", "Condition", "treatment", "Treatment"), logger, "condition")
    
    # Create visualization output directory
    immune_dir <- file.path(output_dir, "immune_infiltration")
    ensureDirectory(immune_dir, TRUE, logger)
    
    # Run the analysis
    infiltration_results <- spatialCommunity$analyzeImmuneInfiltration(
      immune_column = "is_immune",
      region_column = region_col,
      condition_column = condition_col,
      save_dir = immune_dir
    )
    
    # Save the results
    if (!is.null(infiltration_results)) {
      saveRDS(infiltration_results, file.path(output_dir, "immune_infiltration_results.rds"))
      logger$log_info("Saved immune infiltration analysis results")
    }
    
    # Create spatial visualization of immune infiltration across ROIs
    logger$log_info("Creating spatial visualizations of immune infiltration")
    
    # Identify the appropriate image ID column
    img_id_column <- findImageIdColumn(spe, logger)
    
    # Run spatial visualization if we have an image ID column
    if (!is.null(img_id_column)) {
      spatial_vis_results <- spatialCommunity$visualizeImmuneInfiltrationSpatial(
        immune_column = "is_immune",
        img_id_column = img_id_column,
        save_dir = file.path(immune_dir, "spatial")
      )
      
      # Save spatial visualization results
      if (!is.null(spatial_vis_results)) {
        saveRDS(spatial_vis_results, file.path(output_dir, "immune_infiltration_spatial_results.rds"))
        logger$log_info("Saved spatial immune infiltration visualization results")
      }
    } else {
      logger$log_warning("Could not create spatial immune infiltration visualizations: no image ID column found")
    }
  } else {
    logger$log_info("Skipping immune infiltration analysis (no 'is_immune' column found)")
  }
  
  # ---- Step 10: Analyze marker expression profiles for cell types ----
  if ("exprs" %in% assayNames(spe)) {
    logger$log_info("Analyzing marker expression profiles for each cell type")
    
    # Get marker data
    marker_data <- assay(spe, "exprs")
    
    # Calculate mean expression per cell type
    cell_types <- unique(spe$celltype)
    type_marker_means <- sapply(cell_types, function(ct) {
      cells_in_type <- which(spe$celltype == ct)
      rowMeans(marker_data[, cells_in_type, drop = FALSE])
    })
    colnames(type_marker_means) <- cell_types
    
    # Create marker analysis directory
    marker_dir <- file.path(output_dir, "marker_analysis")
    ensureDirectory(marker_dir, TRUE, logger)
    
    # Save marker expression matrix
    saveRDS(type_marker_means, file.path(marker_dir, "celltype_marker_expression.rds"))
    
    # Identify top markers for each cell type
    for (ct in cell_types) {
      # Calculate relative expression vs other types
      rel_expr <- type_marker_means[, ct] / 
        rowMeans(type_marker_means[, colnames(type_marker_means) != ct, drop = FALSE])
      
      # Get top markers
      top_idx <- order(rel_expr, decreasing = TRUE)[1:5]
      logger$log_info("Top markers for cell type %s:", ct)
      for (i in top_idx) {
        marker <- rownames(marker_data)[i]
        logger$log_info("  %s: %.2f-fold higher than other types", 
                       marker, rel_expr[i])
      }
    }
  }
  
  # ---- Step 11: Create cluster to cell type mapping visualization ----
  if ("phenograph_corrected" %in% colnames(SummarizedExperiment::colData(spe)) && 
      "celltype" %in% colnames(SummarizedExperiment::colData(spe))) {
    
    logger$log_info("Creating cluster to cell type mapping visualization")
    
    # Create mapping table
    mapping_data <- data.frame(
      Cluster = SummarizedExperiment::colData(spe)$phenograph_corrected,
      CellType = SummarizedExperiment::colData(spe)$celltype
    )
    
    # Tabulate the mapping
    mapping_table <- table(mapping_data$Cluster, mapping_data$CellType)
    
    # Log the mapping relationships
    logger$log_info("Cluster to cell type mapping:")
    for (i in 1:nrow(mapping_table)) {
      cluster <- rownames(mapping_table)[i]
      types <- colnames(mapping_table)[mapping_table[i,] > 0]
      counts <- mapping_table[i, mapping_table[i,] > 0]
      logger$log_info("  Cluster %s → %s (%s cells)", 
                     cluster,
                     paste(types, collapse="/"),
                     paste(counts, collapse="/"))
    }
    
    # Create visualization directory
    mapping_dir <- file.path(output_dir, "cluster_mapping")
    ensureDirectory(mapping_dir, TRUE, logger)
    
    # Save mapping as CSV for reference
    mapping_csv <- file.path(mapping_dir, "cluster_celltype_mapping.csv")
    write.csv(mapping_table, mapping_csv)
    logger$log_info("Saved cluster to cell type mapping to %s", mapping_csv)
  }
  
  # Create LISA clusters to cell type mapping if LISA clusters exist
  if ("lisa_clusters" %in% colnames(SummarizedExperiment::colData(spe)) && 
      "celltype" %in% colnames(SummarizedExperiment::colData(spe))) {
    
    logger$log_info("Creating LISA cluster to cell type mapping visualization")
    
    # Create mapping table for LISA clusters
    lisa_mapping_data <- data.frame(
      Cluster = SummarizedExperiment::colData(spe)$lisa_clusters,
      CellType = SummarizedExperiment::colData(spe)$celltype
    )
    
    # Tabulate the mapping
    lisa_mapping_table <- table(lisa_mapping_data$Cluster, lisa_mapping_data$CellType)
    
    # Log the mapping relationships
    logger$log_info("LISA cluster to cell type mapping:")
    for (i in 1:nrow(lisa_mapping_table)) {
      cluster <- rownames(lisa_mapping_table)[i]
      types <- colnames(lisa_mapping_table)[lisa_mapping_table[i,] > 0]
      counts <- lisa_mapping_table[i, lisa_mapping_table[i,] > 0]
      logger$log_info("  LISA Cluster %s → %s (%s cells)", 
                     cluster,
                     paste(types, collapse="/"),
                     paste(counts, collapse="/"))
    }
    
    # Create visualization directory if it doesn't exist
    mapping_dir <- file.path(output_dir, "cluster_mapping")
    ensureDirectory(mapping_dir, TRUE, logger)
    
    # Save mapping as CSV for reference
    lisa_mapping_csv <- file.path(mapping_dir, "lisa_cluster_celltype_mapping.csv")
    write.csv(lisa_mapping_table, lisa_mapping_csv)
    logger$log_info("Saved LISA cluster to cell type mapping to %s", lisa_mapping_csv)
    
    # Visualize LISA clusters by cell type using the SpatialCommunity class
    if ("celltype_classified" %in% colnames(SummarizedExperiment::colData(spe))) {
      lisa_viz_dir <- file.path(output_dir, "lisa_clusters")
      ensureDirectory(lisa_viz_dir, TRUE, logger)
      
      # Generate and save the visualization
      spatialCommunity$visualizeCellTypesByLisaCluster(
        lisa_column = "lisa_clusters",
        celltype_column = "celltype_classified",
        save_path = file.path(lisa_viz_dir, "celltype_by_lisa_cluster.png")
      )
    }
  }
  
  # ---- Step 12: Create comprehensive visualizations ----
  createSpatialVisualizations(
    spatialCommunity = spatialCommunity,
    output_dir = output_dir,
    visualization_types = c("celltype", "community", "interaction", "marker", "umap"),
    logger = logger
  )
  
  logger$log_info("Spatial community analysis completed successfully")
  
  invisible(spe)
}

# Run the analysis if script is called directly
if (interactive() || identical(environment(), globalenv())) {
  spe <- runSpatialCommunityAnalysis()
}

