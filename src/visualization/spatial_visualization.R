# spatial_visualization.R
# Functions for spatial visualizations

# Load required libraries
library(ggplot2)
library(SpatialExperiment)

# Source core utilities
source("src/visualization/core_visualization.R")

#' Plot a spatial graph visualization
#'
#' @param spe SpatialExperiment object with spatial graphs
#' @param node_color_by Column to color nodes by (default: "celltype")
#' @param sample_id Sample ID to plot (if NULL, plots all samples)
#' @param graph_name Name of the graph to plot (default: "knn_graph")
#' @param img_id Column name for sample/image ID (default: "sample_id")
#' @param node_size Size of nodes in the graph (default: 1)
#' @param edge_alpha Transparency of edges (default: 0.2)
#' @param ... Additional parameters to pass to plotting functions
#' @return A ggplot object showing the spatial graph
plot_spatial_graph <- function(
  spe,
  node_color_by = "celltype",
  sample_id = NULL,
  graph_name = "knn_graph",
  img_id = "sample_id",
  node_size = 1,
  edge_alpha = 0.2,
  ...
) {
  # Check for required packages
  if (!requireNamespace("imcRtools", quietly = TRUE)) {
    stop("Package 'imcRtools' is required for plot_spatial_graph()")
  }
  
  # Filter to specific sample if requested
  plot_spe <- spe
  if (!is.null(sample_id) && img_id %in% colnames(colData(spe))) {
    plot_spe <- spe[, spe[[img_id]] == sample_id]
    if (ncol(plot_spe) == 0) {
      stop(paste("No cells found for sample_id:", sample_id))
    }
  }
  
  # Check if the graph exists
  if (!graph_name %in% colPairNames(plot_spe)) {
    stop(paste("Graph", graph_name, "not found in SpatialExperiment object"))
  }
  
  # Create plot
  p <- imcRtools::plotSpatial(
    plot_spe,
    node_color_by = node_color_by,
    img_id = img_id,
    draw_edges = TRUE,
    colPairName = graph_name,
    nodes_first = FALSE,
    node_size = node_size,
    edge_color_fix = "grey",
    edge_alpha = edge_alpha,
    ...
  )
  
  # Add custom color scheme if available
  if (node_color_by %in% colnames(colData(spe)) && 
      "color_vectors" %in% names(metadata(spe)) &&
      node_color_by %in% names(metadata(spe)$color_vectors)) {
    p <- p + ggplot2::scale_color_manual(values = metadata(spe)$color_vectors[[node_color_by]])
  }
  
  return(p)
}

#' Plot spatial communities
#'
#' @param spe SpatialExperiment object
#' @param community_column Column containing community assignments (default: "spatial_community")
#' @param sample_id Sample ID to plot (if NULL, plots all samples)
#' @param img_id Column name for sample/image ID (default: "sample_id")
#' @param point_size Size of points (default: 1)
#' @param title Plot title (default: "Spatial Communities")
#' @return A ggplot object
plot_spatial_community <- function(
  spe,
  community_column = "spatial_community",
  sample_id = NULL,
  img_id = "sample_id",
  point_size = 1,
  title = "Spatial Communities"
) {
  # Check if community column exists
  if (!community_column %in% colnames(colData(spe))) {
    stop(paste("Column", community_column, "not found in SpatialExperiment object"))
  }
  
  # Filter to specific sample if requested
  plot_spe <- spe
  if (!is.null(sample_id) && img_id %in% colnames(colData(spe))) {
    plot_spe <- spe[, spe[[img_id]] == sample_id]
    if (ncol(plot_spe) == 0) {
      stop(paste("No cells found for sample_id:", sample_id))
    }
  }
  
  # Create plot data
  plot_data <- data.frame(
    x = spatialCoords(plot_spe)[,1],
    y = spatialCoords(plot_spe)[,2],
    community = as.factor(plot_spe[[community_column]])
  )
  
  # If multiple samples, add sample information
  if (is.null(sample_id) && img_id %in% colnames(colData(plot_spe))) {
    plot_data$sample <- plot_spe[[img_id]]
    
    # Create plot with faceting
    p <- ggplot(plot_data, aes(x = x, y = y, color = community)) +
      geom_point(size = point_size) +
      facet_wrap(~ sample, scales = "free") +
      theme_minimal() +
      labs(title = title, color = "Community")
  } else {
    # Create plot without faceting
    p <- ggplot(plot_data, aes(x = x, y = y, color = community)) +
      geom_point(size = point_size) +
      theme_minimal() +
      labs(title = title, color = "Community")
  }
  
  return(p)
}

#' Plot cell-cell interaction heatmap
#'
#' @param spe SpatialExperiment object with cell-cell interactions
#' @param celltype_column Column with cell type information (default: "celltype")
#' @param interaction_name Name of the interaction measure (default: "neighborhood")
#' @param normalize Whether to normalize the interaction strengths (default: TRUE)
#' @param title Plot title (default: "Cell-Cell Interactions")
#' @return A ggplot object
plot_interaction_heatmap <- function(
  spe,
  celltype_column = "celltype",
  interaction_name = "neighborhood",
  normalize = TRUE,
  title = "Cell-Cell Interactions"
) {
  # Check if celltype column exists
  if (!celltype_column %in% colnames(colData(spe))) {
    stop(paste("Column", celltype_column, "not found in SpatialExperiment object"))
  }
  
  # Check if interaction data exists
  interaction_pattern <- paste0("^", interaction_name)
  if (!any(grepl(interaction_pattern, colPairNames(spe)))) {
    stop(paste("No interaction data found with name pattern:", interaction_pattern))
  }
  
  # Extract interaction data
  interaction_matrices <- colPairNames(spe)[grepl(interaction_pattern, colPairNames(spe))]
  
  # We'll use the first interaction matrix for now
  # In a more complex implementation, we might want to aggregate multiple matrices
  interaction_matrix <- interaction_matrices[1]
  
  # Get cell type information
  celltypes <- spe[[celltype_column]]
  unique_celltypes <- unique(celltypes)
  
  # Create interaction count matrix
  interaction_counts <- matrix(0, 
                              nrow = length(unique_celltypes), 
                              ncol = length(unique_celltypes),
                              dimnames = list(unique_celltypes, unique_celltypes))
  
  # Extract interaction pairs
  pairs <- as.matrix(colPair(spe, interaction_matrix))
  
  # Count interactions between cell types
  for (i in seq_len(nrow(pairs))) {
    from_cell <- pairs[i, 1]
    to_cell <- pairs[i, 2]
    from_type <- celltypes[from_cell]
    to_type <- celltypes[to_cell]
    interaction_counts[from_type, to_type] <- interaction_counts[from_type, to_type] + 1
  }
  
  # Normalize if requested
  if (normalize) {
    # Calculate expected frequencies
    cell_type_counts <- table(celltypes)
    expected <- outer(cell_type_counts, cell_type_counts) / sum(cell_type_counts)
    
    # Calculate interaction enrichment (observed / expected)
    interaction_counts <- interaction_counts / expected
    interaction_counts[is.na(interaction_counts)] <- 0
    interaction_counts[is.infinite(interaction_counts)] <- 0
  }
  
  # Convert to long format for ggplot
  plot_data <- reshape2::melt(interaction_counts, varnames = c("From", "To"), 
                            value.name = "Interaction")
  
  # Create heatmap
  p <- ggplot(plot_data, aes(x = From, y = To, fill = Interaction)) +
    geom_tile() +
    scale_fill_gradient2(
      low = "blue", mid = "white", high = "red",
      midpoint = if(normalize) 1 else median(plot_data$Interaction)
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(
      title = title,
      fill = if(normalize) "Enrichment" else "Count"
    )
  
  return(p)
}

#' Create spatial visualizations for a SpatialExperiment object
#'
#' @param spe SpatialExperiment object
#' @param output_dir Directory to save plots to
#' @param config Visualization configuration
#' @return Invisibly returns the output directory
create_spatial_visualizations <- function(spe, output_dir, config = NULL) {
  # Use default config if not provided
  if (is.null(config)) {
    config <- get_default_viz_config()
  }
  
  # Create spatial graph visualization if available
  graph_available <- FALSE
  for (graph_name in c("knn_graph", "delaunay_graph", "snn_graph")) {
    if (graph_name %in% colPairNames(spe)) {
      graph_available <- TRUE
      message(paste("Creating spatial graph visualization using", graph_name, "..."))
      
      # Determine color column
      color_column <- NULL
      for (col in c("celltype", "phenotype", "cluster", "phenograph_corrected")) {
        if (col %in% colnames(colData(spe))) {
          color_column <- col
          break
        }
      }
      
      if (is.null(color_column)) {
        message("No suitable column found for coloring nodes. Using first available categorical column.")
        cat_cols <- sapply(colData(spe), is.factor)
        if (any(cat_cols)) {
          color_column <- names(cat_cols)[which(cat_cols)[1]]
        } else {
          message("No categorical columns found. Skipping spatial graph visualization.")
          break
        }
      }
      
      # Create plot
      p <- plot_spatial_graph(
        spe,
        node_color_by = color_column,
        graph_name = graph_name,
        node_size = config$point_size
      )
      
      # Save plot
      save_plot(
        p,
        file.path(output_dir, paste0("spatial_", graph_name, ".pdf")),
        width = config$dimensions$width,
        height = config$dimensions$height,
        dpi = config$dimensions$dpi
      )
      
      break  # Just use the first available graph
    }
  }
  
  if (!graph_available) {
    message("No spatial graph found in SpatialExperiment object. Skipping spatial graph visualization.")
  }
  
  # Create spatial community visualization if available
  if ("spatial_community" %in% colnames(colData(spe))) {
    message("Creating spatial community visualization...")
    p <- plot_spatial_community(
      spe,
      community_column = "spatial_community",
      point_size = config$point_size
    )
    
    save_plot(
      p,
      file.path(output_dir, "spatial_communities.pdf"),
      width = config$dimensions$width,
      height = config$dimensions$height,
      dpi = config$dimensions$dpi
    )
  }
  
  # Create interaction heatmap if available
  if (any(grepl("neighborhood", colPairNames(spe)))) {
    message("Creating cell-cell interaction heatmap...")
    
    # Determine cell type column
    celltype_column <- NULL
    for (col in c("celltype", "phenotype", "cluster", "phenograph_corrected")) {
      if (col %in% colnames(colData(spe))) {
        celltype_column <- col
        break
      }
    }
    
    if (!is.null(celltype_column)) {
      p <- plot_interaction_heatmap(
        spe,
        celltype_column = celltype_column
      )
      
      save_plot(
        p,
        file.path(output_dir, "cell_interactions.pdf"),
        width = config$dimensions$width,
        height = config$dimensions$height,
        dpi = config$dimensions$dpi
      )
    } else {
      message("No suitable column found for cell types. Skipping interaction heatmap.")
    }
  }
  
  return(invisible(output_dir))
} 