#' Spatial Analysis Workflow Functions
#'
#' This file contains modular workflow functions for spatial analysis operations.
#' These functions are designed to be called from entrypoint scripts.

source("src/core/ColumnUtilities.R")

#' Prepare SpatialExperiment object for analysis
#'
#' @param spe SpatialExperiment object
#' @param phenotyping_column Column containing phenotyping assignments
#' @param logger Logger object
#' @param spatialCommunity Optional SpatialCommunity object for additional classification
#'
#' @return Prepared SpatialExperiment object
prepareForSpatialAnalysis <- function(spe, phenotyping_column = "phenograph_corrected", logger = NULL, spatialCommunity = NULL) {
  # Ensure SpatialExperiment object has unique cell IDs as column names
  if (is.null(colnames(spe)) || anyDuplicated(colnames(spe)) > 0 || any(colnames(spe) == "")) {
    if (!is.null(logger)) logger$log_warning("SpatialExperiment object has missing or non-unique column names. Creating unique cell IDs.")
    
    # Generate unique cell IDs if needed
    if (!"cell_id" %in% colnames(SummarizedExperiment::colData(spe))) {
      # Try to use any existing identifier
      id_candidates <- c("CellID", "cell_id", "CellNumber", "ObjectNumber")
      id_col <- findMatchingColumn(spe, id_candidates, logger, "cell identifier")
      
      if (!is.null(id_col)) {
        if (!is.null(logger)) logger$log_info("Using '%s' column to create unique cell IDs", id_col)
        
        # If we have ImageNumber or similar, use it to make IDs more unique
        img_col <- findImageIdColumn(spe, logger)
        if (!is.null(img_col)) {
          if (!is.null(logger)) logger$log_info("Combining with '%s' for image-level uniqueness", img_col)
          cell_ids <- paste0("cell_", spe[[img_col]], "_", spe[[id_col]])
        } else {
          cell_ids <- paste0("cell_", spe[[id_col]])
        }
      } else {
        # Create numeric IDs if no existing ID column
        if (!is.null(logger)) logger$log_info("No ID column found. Creating sequential cell IDs.")
        cell_ids <- paste0("cell_", seq_len(ncol(spe)))
      }
      
      # Make sure they're unique
      if (anyDuplicated(cell_ids) > 0) {
        if (!is.null(logger)) logger$log_warning("Generated cell IDs still contain duplicates. Adding unique suffix.")
        cell_ids <- make.unique(cell_ids, sep = "_")
      }
      
      # Store the original colnames if they exist
      if (!is.null(colnames(spe)) && all(colnames(spe) != "")) {
        spe$original_colnames <- colnames(spe)
      }
      
      # Set the new colnames
      colnames(spe) <- cell_ids
      if (!is.null(logger)) logger$log_info("Created and assigned unique cell IDs as column names")
    } else {
      # Use existing cell_id column
      if (!is.null(logger)) logger$log_info("Using existing 'cell_id' column as column names")
      colnames(spe) <- spe$cell_id
    }
  }
  
  # Set up celltype column based on phenotyping results
  if (!"celltype" %in% colnames(SummarizedExperiment::colData(spe))) {
    if (!is.null(logger)) logger$log_info("Setting up celltype column from available phenotyping data")
    
    # First, if we have a SpatialCommunity object, use marker-based classification
    if (!is.null(spatialCommunity) && "exprs" %in% SummarizedExperiment::assayNames(spe)) {
      if (!is.null(logger)) logger$log_info("Classifying cells by marker expression using SpatialCommunity")
      
      # Classify cells based on marker expression
      spe <- spatialCommunity$classifyCellTypesByMarkers(
        output_column = "celltype_classified",
        use_config = TRUE
      )
      
      # Update spatialCommunity's SPE object
      spatialCommunity$spe <- spe
      
      # If successful, use cluster-based cell type classification as the celltype column
      if ("celltype_classified_cluster" %in% colnames(SummarizedExperiment::colData(spe))) {
        if (!is.null(logger)) logger$log_info("Using cluster-based cell type classification for celltype column")
        SummarizedExperiment::colData(spe)$celltype <- SummarizedExperiment::colData(spe)$celltype_classified_cluster
        
        # Add detailed cell type reporting
        if (!is.null(logger)) {
          celltype_counts <- table(SummarizedExperiment::colData(spe)$celltype)
          logger$log_info("Cell type distribution:")
          for (ct in names(celltype_counts)) {
            logger$log_info("  Type %s: %d cells (%.2f%%)", 
                          ct, 
                          celltype_counts[ct], 
                          (celltype_counts[ct]/ncol(spe))*100)
          }
        }
        return(spe)
      }
    }
    
    # If marker-based classification failed or wasn't available, continue with phenotyping column
    # Try to use the specified phenotyping column first
    if (phenotyping_column %in% colnames(SummarizedExperiment::colData(spe))) {
      if (!is.null(logger)) logger$log_info("Using %s column for cell types", phenotyping_column)
      SummarizedExperiment::colData(spe)$celltype <- SummarizedExperiment::colData(spe)[[phenotyping_column]]
      
      # Add detailed cell type reporting
      if (!is.null(logger)) {
        celltype_counts <- table(SummarizedExperiment::colData(spe)$celltype)
        logger$log_info("Cell type distribution:")
        for (ct in names(celltype_counts)) {
          logger$log_info("  Type %s: %d cells (%.2f%%)", 
                         ct, 
                         celltype_counts[ct], 
                         (celltype_counts[ct]/ncol(spe))*100)
        }
      }
    } else {
      # Try to find any available phenotyping column
      phenotyping_options <- c(
        "phenograph_corrected", "phenograph_raw",  # From cellPhenotyping.R
        "cluster", "phenotype"                     # Possible other columns
      )
      
      for (col in phenotyping_options) {
        if (col %in% colnames(SummarizedExperiment::colData(spe))) {
          if (!is.null(logger)) logger$log_info("Using %s column for cell types", col)
          SummarizedExperiment::colData(spe)$celltype <- SummarizedExperiment::colData(spe)[[col]]
          break
        }
      }
      
      # If no celltype column exists yet, try to use clustering results
      if (!"celltype" %in% colnames(SummarizedExperiment::colData(spe))) {
        # This is a simplified version - the full cell type determination logic would be extracted
        # from the original file but is too lengthy to include here 
        cluster_col <- findPhenographColumn(spe, logger)
        
        if (!is.null(cluster_col)) {
          if (!is.null(logger)) logger$log_info("Using clustering column '%s' as basis for cell types", cluster_col)
          SummarizedExperiment::colData(spe)$celltype <- paste0("Cluster_", SummarizedExperiment::colData(spe)[[cluster_col]])
        } else {
          if (!is.null(logger)) logger$log_warning("No suitable phenotyping or clustering column found. Creating a dummy celltype column.")
          SummarizedExperiment::colData(spe)$celltype <- paste0("Cluster_", as.integer(runif(ncol(spe), 1, 5)))
        }
      }
    }
    
    # Convert to factor and assign meaningful labels if numeric
    if (is.numeric(SummarizedExperiment::colData(spe)$celltype)) {
      SummarizedExperiment::colData(spe)$celltype <- factor(SummarizedExperiment::colData(spe)$celltype, 
                                                         levels = sort(unique(SummarizedExperiment::colData(spe)$celltype)))
      levels(SummarizedExperiment::colData(spe)$celltype) <- paste0("Cluster_", levels(SummarizedExperiment::colData(spe)$celltype))
    } else {
      SummarizedExperiment::colData(spe)$celltype <- as.factor(SummarizedExperiment::colData(spe)$celltype)
    }
    
    if (!is.null(logger)) logger$log_info("Created celltype column with %d unique cell types", 
                                       length(levels(SummarizedExperiment::colData(spe)$celltype)))
  } else {
    if (!is.null(logger)) logger$log_info("Using existing celltype column with %d unique cell types", 
                                       length(unique(SummarizedExperiment::colData(spe)$celltype)))
  }
  
  return(spe)
}

#' Ensure required spatial graphs exist in the SpatialExperiment object
#'
#' @param spe SpatialExperiment object
#' @param required_graphs List of required graph names
#' @param logger Logger object
#'
#' @return SpatialExperiment object with required graphs
ensureSpatialGraphs <- function(spe, required_graphs = c("neighborhood", "knn_interaction_graph"), logger = NULL) {
  if (!is.null(logger)) logger$log_info("Available spatial graphs: %s", 
                                     paste(SingleCellExperiment::colPairNames(spe), collapse=", "))
  
  # Check if we have any spatial graphs
  if (length(SingleCellExperiment::colPairNames(spe)) == 0) {
    if (!is.null(logger)) logger$log_error("No spatial graphs found in the SpatialExperiment object")
    stop("No spatial graphs found. Make sure spatial graphs are built (knn, expansion, or delaunay).")
  }
  
  # Ensure required graphs exist by copying from available graphs if needed
  for (required_graph in required_graphs) {
    if (!(required_graph %in% SingleCellExperiment::colPairNames(spe))) {
      # Find a suitable graph to copy from
      source_graph <- SingleCellExperiment::colPairNames(spe)[1]
      if ("knn" %in% SingleCellExperiment::colPairNames(spe) && grepl("knn", required_graph)) {
        source_graph <- "knn"
      }
      
      if (!is.null(logger)) logger$log_warning("'%s' graph not found, using %s instead", required_graph, source_graph)
      spe <- imcRtools::copyColPair(spe, from = source_graph, to = required_graph)
      if (!is.null(logger)) logger$log_info("Created '%s' graph from %s", required_graph, source_graph)
    }
  }
  
  return(spe)
}

#' Perform community detection based on spatial graphs
#'
#' @param spatialCommunity SpatialCommunity object
#' @param methods Vector of community detection methods to use
#' @param parameters List of parameters for different methods
#' @param logger Logger object
#'
#' @return Updated SpatialExperiment object with community annotations
performCommunityDetection <- function(spatialCommunity, methods = c("graph_based"), 
                                     parameters = list(), logger = NULL) {
  # Get the SPE object
  spe <- spatialCommunity$spe
  
  # IMPORTANT: Ensure column names are unique and set
  if (is.null(colnames(spe)) || anyDuplicated(colnames(spe)) > 0 || any(colnames(spe) == "")) {
    if (!is.null(logger)) logger$log_warning("Missing or non-unique column names detected before community detection. Fixing...")
    
    # Generate unique cell IDs if needed
    if (!"cell_id" %in% colnames(SummarizedExperiment::colData(spe))) {
      cell_ids <- paste0("cell_", seq_len(ncol(spe)))
      if (!is.null(logger)) logger$log_info("Creating sequential cell IDs for %d cells", length(cell_ids))
    } else {
      cell_ids <- spe$cell_id
      if (!is.null(logger)) logger$log_info("Using existing cell_id column for column names")
    }
    
    # Make sure they're unique
    if (anyDuplicated(cell_ids) > 0) {
      if (!is.null(logger)) logger$log_warning("Generated cell IDs contain duplicates. Adding unique suffix.")
      cell_ids <- make.unique(cell_ids, sep = "_")
    }
    
    # Store the original colnames if they exist
    if (!is.null(colnames(spe)) && all(colnames(spe) != "")) {
      spe$original_colnames <- colnames(spe)
    }
    
    # Set the new colnames
    colnames(spe) <- cell_ids
    
    # Update the spatialCommunity object
    spatialCommunity$spe <- spe
    if (!is.null(logger)) logger$log_info("Fixed column names for community detection")
  }
  
  # Store is_immune column to restore it if needed
  has_is_immune <- "is_immune" %in% colnames(SummarizedExperiment::colData(spe))
  is_immune_values <- NULL
  if (has_is_immune) {
    is_immune_values <- SummarizedExperiment::colData(spe)$is_immune
  }
  
  # Extract parameters with defaults
  size_threshold <- parameters$size_threshold %||% 10
  n_clusters <- parameters$n_clusters %||% 6
  compartment_column <- parameters$compartment_column %||% "celltype"
  direct_celltype_analysis <- parameters$direct_celltype_analysis %||% TRUE
  
  # Perform the requested community detection methods
  if ("graph_based" %in% methods) {
    if (!is.null(logger)) logger$log_info("Performing graph-based community detection")
    
    # If direct_celltype_analysis is enabled, use phenotyping directly
    if (direct_celltype_analysis) {
      if (!is.null(logger)) logger$log_info("Using direct cell type analysis (no artificial compartments)")
      
      # Directly use celltype as the group_by parameter for detectCommunity
      spe <- spatialCommunity$detectGraphCommunities(
        colPairName = "neighborhood", 
        group_by = "celltype",
        size_threshold = size_threshold
      )
    } else {
      # If not using direct celltype analysis, use compartment approach
      if (!(compartment_column %in% colnames(SummarizedExperiment::colData(spe)))) {
        # This is a simplified version of the compartment creation
        if (!is.null(logger)) logger$log_info("Creating %s column using cell abundance approach", compartment_column)
        
        # Default to most abundant vs others
        type_counts <- table(SummarizedExperiment::colData(spe)$celltype)
        most_abundant <- names(type_counts)[which.max(type_counts)]
        if (!is.null(logger)) logger$log_info("Using most abundant cell type as primary compartment: %s", most_abundant)
        SummarizedExperiment::colData(spe)$compartment <- ifelse(
          SummarizedExperiment::colData(spe)$celltype == most_abundant, 
          most_abundant, 
          "Other"
        )
        compartment_column <- "compartment"
      }
      
      spe <- spatialCommunity$detectGraphCommunities(
        colPairName = "neighborhood", 
        group_by = compartment_column,
        size_threshold = size_threshold
      )
    }
  }
  
  if ("celltype_aggregation" %in% methods) {
    if (!is.null(logger)) logger$log_info("Performing celltype-based neighborhood aggregation")
    spe <- spatialCommunity$aggregateCelltypeNeighbors(
      colPairName = "knn_interaction_graph",
      n_clusters = n_clusters
    )
  }
  
  if ("expression_aggregation" %in% methods) {
    if (!is.null(logger)) logger$log_info("Performing expression-based neighborhood aggregation")
    spe <- spatialCommunity$aggregateExpressionNeighbors(
      colPairName = "knn_interaction_graph",
      n_clusters = n_clusters
    )
  }
  
  if ("lisa" %in% methods) {
    lisa_radii <- parameters$lisa_radii %||% c(10, 20, 50)
    if (!is.null(logger)) logger$log_info("Performing LISA-based spatial clustering")
    spe <- spatialCommunity$performLisaClustering(
      radii = lisa_radii,
      n_clusters = n_clusters
    )
  }
  
  # Restore is_immune column if it was lost
  if (has_is_immune && !("is_immune" %in% colnames(SummarizedExperiment::colData(spe)))) {
    if (!is.null(logger)) logger$log_info("Restoring is_immune column that was lost during processing")
    SummarizedExperiment::colData(spe)$is_immune <- is_immune_values
  }
  
  # Update the SPE in the spatialCommunity object 
  spatialCommunity$spe <- spe
  
  return(spe)
}

#' Create comprehensive visualizations for spatial analysis
#'
#' @param spatialCommunity SpatialCommunity object
#' @param output_dir Base output directory
#' @param visualization_types Types of visualizations to create (default: all)
#' @param logger Logger for logging information
#' @return TRUE if successful
createSpatialVisualizations <- function(spatialCommunity, 
                                      output_dir, 
                                      visualization_types = c("celltype", "community", "interaction", "marker", "umap", "proximity"),
                                      logger = NULL) {
  # Create visualization directories
  vis_dir <- file.path(output_dir, "visualizations")
  celltype_dir <- file.path(vis_dir, "cell_types")
  interaction_dir <- file.path(vis_dir, "interactions")
  community_dir <- file.path(vis_dir, "communities")
  marker_dir <- file.path(vis_dir, "marker_expression")
  umap_dir <- file.path(vis_dir, "umap")
  proximity_dir <- file.path(vis_dir, "proximity")
  
  # Ensure directories exist
  for (dir in c(vis_dir, celltype_dir, interaction_dir, community_dir, marker_dir, umap_dir, proximity_dir)) {
    if (!dir.exists(dir)) {
      dir.create(dir, recursive = TRUE)
    }
  }
  
  # Create cell type spatial visualizations
  if ("celltype" %in% visualization_types) {
    if (!is.null(logger)) logger$log_info("Creating cell type spatial visualizations")
    celltype_col <- findCellTypeColumn(spatialCommunity$spe, logger)
    phenograph_col <- findPhenographColumn(spatialCommunity$spe, logger)
    
    if (!is.null(celltype_col) && !is.null(phenograph_col)) {
      celltype_viz <- spatialCommunity$visualizeCellTypes(
        celltype_column = celltype_col,
        phenograph_column = phenograph_col,
        density_plot = TRUE,
        save_dir = celltype_dir
      )
    } else {
      if (!is.null(logger)) logger$log_warning("Could not create cell type visualizations: missing required columns")
    }
  }
  
  # Create community visualizations if requested
  if ("community" %in% visualization_types && "community_id" %in% colnames(SummarizedExperiment::colData(spatialCommunity$spe))) {
    if (!is.null(logger)) logger$log_info("Creating community visualizations")
    spatialCommunity$visualizeCommunities(
      community_column = "community_id",
      save_dir = community_dir
    )
  }
  
  # Create marker expression visualizations if requested
  if ("marker" %in% visualization_types && 
      any(c("exprs", "counts") %in% SummarizedExperiment::assayNames(spatialCommunity$spe))) {
    if (!is.null(logger)) logger$log_info("Creating marker expression visualizations")
    
    # By cell type if available
    if ("celltype" %in% colnames(SummarizedExperiment::colData(spatialCommunity$spe))) {
      spatialCommunity$visualizeMarkerExpression(
        group_column = "celltype",
        save_dir = marker_dir
      )
    }
    
    # By community if available
    if ("community_id" %in% colnames(SummarizedExperiment::colData(spatialCommunity$spe))) {
      spatialCommunity$visualizeMarkerExpression(
        group_column = "community_id",
        save_dir = marker_dir
      )
    }
  }
  
  # Create UMAP visualization if requested
  if ("umap" %in% visualization_types && "exprs" %in% SummarizedExperiment::assayNames(spatialCommunity$spe)) {
    if (requireNamespace("umap", quietly = TRUE) && requireNamespace("ggplot2", quietly = TRUE)) {
      if (!is.null(logger)) logger$log_info("Creating UMAP visualization of cell types")
      
      # Get marker data
      marker_data <- t(SummarizedExperiment::assay(spatialCommunity$spe, "exprs"))
      
      # Run UMAP
      set.seed(42)
      umap_result <- tryCatch({
        umap::umap(marker_data)
      }, error = function(e) {
        if (!is.null(logger)) logger$log_warning("UMAP computation failed: %s", e$message)
        return(NULL)
      })
      
      if (!is.null(umap_result)) {
        # Create data frame for plotting
        celltype_col <- findCellTypeColumn(spatialCommunity$spe, logger) %||% "celltype"
        
        umap_df <- data.frame(
          UMAP1 = umap_result$layout[,1],
          UMAP2 = umap_result$layout[,2],
          CellType = spatialCommunity$spe[[celltype_col]]
        )
        
        # Save as RDS for later plotting
        saveRDS(umap_df, file.path(umap_dir, "celltype_umap.rds"))
        if (!is.null(logger)) logger$log_info("Saved UMAP coordinates to %s", file.path(umap_dir, "celltype_umap.rds"))
        
        # Create and save the plot
        umap_plot <- ggplot2::ggplot(umap_df, ggplot2::aes(x=UMAP1, y=UMAP2, color=CellType)) +
          ggplot2::geom_point(alpha=0.5, size=0.5) +
          ggplot2::theme_minimal() +
          ggplot2::ggtitle("UMAP visualization of cell types")
        
        ggplot2::ggsave(file.path(umap_dir, "celltype_umap.png"), umap_plot, width=10, height=8)
        if (!is.null(logger)) logger$log_info("Created UMAP visualization at %s", file.path(umap_dir, "celltype_umap.png"))
      }
    } else {
      if (!is.null(logger)) logger$log_warning("Could not create UMAP visualization: required packages not available")
    }
  }
  
  # Create proximity visualization
  if ("proximity" %in% visualization_types) {
    if (!is.null(logger)) logger$log_info("Creating cell type proximity visualization")
    
    # Use the appropriate cell type column
    celltype_column <- findCellTypeColumn(spatialCommunity$spe, logger) %||% "celltype"
    
    # Create cell type proximity visualization
    spatialCommunity$visualizeCellTypeProximity(
      celltype_column = celltype_column,
      save_dir = proximity_dir
    )
  }
  
  # Note: We're intentionally NOT including interaction analysis here as it's handled by spatialInteractionAnalysis.R
  if (!is.null(logger)) logger$log_info("Visualizations created successfully and saved to %s", vis_dir)
  
  return(invisible(TRUE))
} 