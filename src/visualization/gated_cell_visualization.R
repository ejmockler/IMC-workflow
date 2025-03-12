# gated_cell_visualization.R
# Functions for visualizing gated cell data

# Load required libraries
library(ggplot2)
library(SpatialExperiment)

# Source core utilities
source("src/visualization/core_visualization.R")

#' Create spatial plot of gated cells
#'
#' @param spe SpatialExperiment object with gated cells
#' @param color_by Column to color cells by (default: "gated_celltype")
#' @param facet_by Column to facet plot by (default: NULL)
#' @param point_size Size of points in plot (default: 1)
#' @param title Plot title (default: "Gated Cell Spatial Distribution")
#' @return ggplot object
plot_gated_cells_spatial <- function(
  spe,
  color_by = "gated_celltype",
  facet_by = NULL,
  point_size = 1,
  title = "Gated Cell Spatial Distribution"
) {
  # Extract spatial coordinates and metadata
  plot_data <- data.frame(
    x = spatialCoords(spe)[,1],
    y = spatialCoords(spe)[,2],
    color_var = safe_get_column(spe, color_by)
  )
  
  if (!is.null(facet_by) && facet_by %in% colnames(colData(spe))) {
    plot_data$facet_var <- spe[[facet_by]]
  }
  
  # Create the plot
  p <- ggplot(plot_data, aes(x = x, y = y, color = color_var)) +
    geom_point(size = point_size) +
    theme_minimal() +
    labs(title = title, color = color_by)
  
  # Add faceting if requested
  if (!is.null(facet_by) && "facet_var" %in% colnames(plot_data)) {
    p <- p + facet_wrap(~ facet_var)
  }
  
  return(p)
}

#' Create cell type proportion barplot
#'
#' @param spe SpatialExperiment object with gated cells
#' @param celltype_col Column with cell type information (default: "gated_celltype")
#' @param group_by Optional column to group proportions by
#' @param title Plot title (default: "Cell Type Proportions")
#' @return ggplot object
plot_cell_type_proportions <- function(
  spe,
  celltype_col = "gated_celltype",
  group_by = NULL,
  title = "Cell Type Proportions"
) {
  # Check if celltype column exists
  if (!celltype_col %in% colnames(colData(spe))) {
    stop(paste("Column", celltype_col, "not found in SpatialExperiment object"))
  }
  
  # Calculate proportions
  if (is.null(group_by)) {
    # Overall proportions
    counts <- table(spe[[celltype_col]])
    props <- prop.table(counts) * 100
    
    plot_data <- data.frame(
      CellType = names(props),
      Proportion = as.numeric(props)
    )
    
    # Create plot
    p <- ggplot(plot_data, aes(x = reorder(CellType, -Proportion), 
                               y = Proportion, 
                               fill = CellType)) +
      geom_bar(stat = "identity") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      labs(title = title, x = "Cell Type", y = "Proportion (%)")
    
  } else {
    # Check if group_by column exists
    if (!group_by %in% colnames(colData(spe))) {
      stop(paste("Column", group_by, "not found in SpatialExperiment object"))
    }
    
    # Group by another variable
    counts <- table(spe[[group_by]], spe[[celltype_col]])
    props <- prop.table(counts, margin = 1) * 100
    
    plot_data <- as.data.frame.table(props)
    names(plot_data) <- c("Group", "CellType", "Proportion")
    
    # Create plot
    p <- ggplot(plot_data, aes(x = Group, y = Proportion, fill = CellType)) +
      geom_bar(stat = "identity", position = "fill") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      labs(title = title, x = group_by, y = "Proportion (%)")
  }
  
  return(p)
}

#' Create heatmap of marker expression by cell type
#'
#' @param spe SpatialExperiment object with gated cells
#' @param markers Vector of marker names to include (default: all markers)
#' @param celltype_col Column with cell type information (default: "gated_celltype")
#' @param scale_method Method for scaling expression values (default: "zscore")
#' @param title Plot title (default: "Marker Expression by Cell Type")
#' @return ggplot object
plot_marker_expression_by_celltype <- function(
  spe,
  markers = NULL,
  celltype_col = "gated_celltype",
  scale_method = c("zscore", "robust", "minmax"),
  title = "Marker Expression by Cell Type"
) {
  scale_method <- match.arg(scale_method)
  
  # If markers not specified, use all assayed genes
  if (is.null(markers)) {
    markers <- rownames(spe)
  } else {
    # Check that all markers are in the SpatialExperiment object
    missing_markers <- setdiff(markers, rownames(spe))
    if (length(missing_markers) > 0) {
      warning(paste("Some markers not found:", 
                    paste(missing_markers, collapse = ", ")))
      markers <- intersect(markers, rownames(spe))
    }
  }
  
  # Check if celltype column exists
  if (!celltype_col %in% colnames(colData(spe))) {
    stop(paste("Column", celltype_col, "not found in SpatialExperiment object"))
  }
  
  # Calculate mean expression per cell type
  expression_data <- assay(spe)[markers, , drop = FALSE]
  celltypes <- spe[[celltype_col]]
  
  # Calculate mean expression per cell type
  mean_expr <- t(apply(expression_data, 1, function(x) {
    tapply(x, celltypes, mean)
  }))
  
  # Scale the data
  scaled_expr <- t(apply(mean_expr, 1, function(x) {
    scale_values(x, method = scale_method)
  }))
  
  # Convert to data frame for ggplot
  plot_data <- reshape2::melt(scaled_expr, varnames = c("Marker", "CellType"), 
                            value.name = "Expression")
  
  # Create the heatmap
  p <- ggplot(plot_data, aes(x = CellType, y = Marker, fill = Expression)) +
    geom_tile() +
    scale_fill_gradient2(
      low = "blue", mid = "white", high = "red",
      midpoint = 0
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = title)
  
  return(p)
}

#' Create all gated cell visualizations
#'
#' Main function to create all gated cell visualizations
#'
#' @param spe SpatialExperiment object
#' @param output_dir Directory to save plots to
#' @param config Visualization configuration
#' @return Invisibly returns the output directory
create_gated_cell_visualizations <- function(spe, output_dir, config = NULL) {
  # Use default config if not provided
  if (is.null(config)) {
    config <- get_default_viz_config()
  }
  
  # Check if necessary columns exist
  celltype_col <- "gated_celltype"
  if (!celltype_col %in% colnames(colData(spe))) {
    warning("Column 'gated_celltype' not found. Skipping gated cell visualizations.")
    return(invisible(output_dir))
  }
  
  # Create spatial visualization
  message("Creating gated cell spatial distribution plot...")
  p <- plot_gated_cells_spatial(
    spe, 
    color_by = celltype_col,
    point_size = config$point_size,
    title = "Gated Cell Spatial Distribution"
  )
  save_plot(
    p,
    file.path(output_dir, "gated_spatial.pdf"),
    width = config$dimensions$width,
    height = config$dimensions$height,
    dpi = config$dimensions$dpi
  )
  
  # Create cell type proportions
  message("Creating cell type proportion plot...")
  p <- plot_cell_type_proportions(
    spe,
    celltype_col = celltype_col,
    title = "Gated Cell Type Proportions"
  )
  save_plot(
    p,
    file.path(output_dir, "gated_proportions.pdf"),
    width = config$dimensions$width,
    height = config$dimensions$height,
    dpi = config$dimensions$dpi
  )
  
  # Create marker expression heatmap
  message("Creating marker expression heatmap by cell type...")
  p <- plot_marker_expression_by_celltype(
    spe,
    celltype_col = celltype_col,
    title = "Marker Expression by Cell Type"
  )
  save_plot(
    p,
    file.path(output_dir, "gated_marker_expression.pdf"),
    width = config$dimensions$width,
    height = config$dimensions$height * 1.5, # Make taller for heatmap
    dpi = config$dimensions$dpi
  )
  
  return(invisible(output_dir))
} 