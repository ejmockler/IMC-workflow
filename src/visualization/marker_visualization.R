# marker_visualization.R
# Functions for marker expression visualizations

# Load required libraries
library(ggplot2)
library(reshape2)
library(SpatialExperiment)

# Source core utilities
source("src/visualization/core_visualization.R")

#' Plot marker expression heatmap
#'
#' @param spe SpatialExperiment object
#' @param markers Vector of marker names to include (default: all markers)
#' @param group_column Column to group cells by (default: "celltype")
#' @param scale_method Method for scaling expression values (default: "zscore")
#' @param title Plot title (default: "Marker Expression by Group")
#' @return A ggplot object
plot_marker_heatmap <- function(
  spe,
  markers = NULL,
  group_column = "celltype",
  scale_method = c("zscore", "robust", "minmax"),
  title = "Marker Expression by Group"
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
  
  # Check if group column exists
  if (!group_column %in% colnames(colData(spe))) {
    stop(paste("Column", group_column, "not found in SpatialExperiment object"))
  }
  
  # Calculate mean expression per group
  expression_data <- assay(spe)[markers, , drop = FALSE]
  groups <- spe[[group_column]]
  
  # Calculate mean expression per group
  mean_expr <- t(apply(expression_data, 1, function(x) {
    tapply(x, groups, mean, na.rm = TRUE)
  }))
  
  # Handle potential NA values from tapply
  mean_expr[is.na(mean_expr)] <- 0
  
  # Scale the data
  scaled_expr <- t(apply(mean_expr, 1, function(x) {
    scale_values(x, method = scale_method)
  }))
  
  # Convert to data frame for ggplot
  plot_data <- reshape2::melt(scaled_expr, varnames = c("Marker", "Group"), 
                            value.name = "Expression")
  
  # Create the heatmap
  p <- ggplot(plot_data, aes(x = Group, y = Marker, fill = Expression)) +
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

#' Plot marker expression on spatial coordinates
#'
#' @param spe SpatialExperiment object
#' @param marker Name of the marker to visualize
#' @param sample_id Sample ID to plot (if NULL, plots all samples)
#' @param img_id Column name for sample/image ID (default: "sample_id")
#' @param point_size Size of points (default: 1)
#' @param scale_method Method for scaling expression values (default: "robust")
#' @param color_scale Vector of colors for the gradient
#' @param title Plot title (default: marker name)
#' @return A ggplot object
plot_marker_spatial <- function(
  spe,
  marker,
  sample_id = NULL,
  img_id = "sample_id",
  point_size = 1,
  scale_method = c("robust", "minmax", "none"),
  color_scale = c("black", "blue", "cyan", "green", "yellow", "red"),
  title = NULL
) {
  scale_method <- match.arg(scale_method)
  
  # Check if marker exists
  if (!marker %in% rownames(spe)) {
    stop(paste("Marker", marker, "not found in SpatialExperiment object"))
  }
  
  # Filter to specific sample if requested
  plot_spe <- spe
  if (!is.null(sample_id) && img_id %in% colnames(colData(spe))) {
    plot_spe <- spe[, spe[[img_id]] == sample_id]
    if (ncol(plot_spe) == 0) {
      stop(paste("No cells found for sample_id:", sample_id))
    }
  }
  
  # Extract expression data for the marker
  expression_values <- assay(plot_spe)[marker, ]
  
  # Scale values if requested
  if (scale_method != "none") {
    expression_values <- scale_values(expression_values, method = scale_method)
  }
  
  # Create plot data
  plot_data <- data.frame(
    x = spatialCoords(plot_spe)[,1],
    y = spatialCoords(plot_spe)[,2],
    expression = expression_values
  )
  
  # If multiple samples, add sample information
  if (is.null(sample_id) && img_id %in% colnames(colData(plot_spe))) {
    plot_data$sample <- plot_spe[[img_id]]
    
    # Create plot with faceting
    p <- ggplot(plot_data, aes(x = x, y = y, color = expression)) +
      geom_point(size = point_size) +
      scale_color_gradientn(colors = color_scale) +
      facet_wrap(~ sample, scales = "free") +
      theme_minimal() +
      labs(
        title = title %||% marker,
        color = "Expression"
      )
  } else {
    # Create plot without faceting
    p <- ggplot(plot_data, aes(x = x, y = y, color = expression)) +
      geom_point(size = point_size) +
      scale_color_gradientn(colors = color_scale) +
      theme_minimal() +
      labs(
        title = title %||% marker,
        color = "Expression"
      )
  }
  
  return(p)
}

#' Plot a UMAP of cells colored by marker expression
#'
#' @param spe SpatialExperiment object with UMAP coordinates
#' @param marker Name of the marker to visualize
#' @param umap_cols Names of UMAP coordinate columns (default: c("UMAP1", "UMAP2"))
#' @param point_size Size of points (default: 1)
#' @param scale_method Method for scaling expression values (default: "robust")
#' @param color_scale Vector of colors for the gradient
#' @param title Plot title (default: marker name)
#' @return A ggplot object
plot_umap <- function(
  spe,
  marker = NULL,
  umap_cols = c("UMAP1", "UMAP2"),
  point_size = 1,
  scale_method = c("robust", "minmax", "none"),
  color_scale = c("black", "blue", "cyan", "green", "yellow", "red"),
  title = NULL
) {
  # Check if UMAP coordinates exist
  for (col in umap_cols) {
    if (!col %in% colnames(colData(spe))) {
      stop(paste("Column", col, "not found in SpatialExperiment object"))
    }
  }
  
  # Create base plot data
  plot_data <- data.frame(
    UMAP1 = spe[[umap_cols[1]]],
    UMAP2 = spe[[umap_cols[2]]]
  )
  
  # If marker is specified, color by marker expression
  if (!is.null(marker)) {
    scale_method <- match.arg(scale_method)
    
    # Check if marker exists
    if (!marker %in% rownames(spe)) {
      stop(paste("Marker", marker, "not found in SpatialExperiment object"))
    }
    
    # Extract expression data for the marker
    expression_values <- assay(spe)[marker, ]
    
    # Scale values if requested
    if (scale_method != "none") {
      expression_values <- scale_values(expression_values, method = scale_method)
    }
    
    plot_data$expression <- expression_values
    
    # Create plot colored by expression
    p <- ggplot(plot_data, aes(x = UMAP1, y = UMAP2, color = expression)) +
      geom_point(size = point_size) +
      scale_color_gradientn(colors = color_scale) +
      theme_minimal() +
      labs(
        title = title %||% paste("UMAP -", marker),
        color = "Expression"
      )
  } else {
    # Create plain UMAP plot
    p <- ggplot(plot_data, aes(x = UMAP1, y = UMAP2)) +
      geom_point(size = point_size) +
      theme_minimal() +
      labs(
        title = title %||% "UMAP"
      )
  }
  
  return(p)
}

#' Create all marker visualizations
#'
#' @param spe SpatialExperiment object
#' @param output_dir Directory to save plots to
#' @param config Visualization configuration
#' @return Invisibly returns the output directory
create_marker_visualizations <- function(spe, output_dir, config = NULL) {
  # Use default config if not provided
  if (is.null(config)) {
    config <- get_default_viz_config()
  }
  
  # Create marker heatmap
  message("Creating marker expression heatmap...")
  
  # Determine group column
  group_column <- NULL
  for (col in c("celltype", "phenotype", "cluster", "phenograph_corrected")) {
    if (col %in% colnames(colData(spe))) {
      group_column <- col
      break
    }
  }
  
  if (!is.null(group_column)) {
    p <- plot_marker_heatmap(
      spe,
      group_column = group_column,
      title = paste("Marker Expression by", group_column)
    )
    
    save_plot(
      p,
      file.path(output_dir, "marker_heatmap.pdf"),
      width = config$dimensions$width,
      height = config$dimensions$height * 1.5, # Make taller for heatmap
      dpi = config$dimensions$dpi
    )
  } else {
    message("No suitable column found for grouping. Skipping marker heatmap.")
  }
  
  # Create spatial visualizations for selected markers
  message("Creating spatial visualizations for markers...")
  
  # Select top markers based on variance
  n_markers <- min(10, nrow(spe))
  marker_vars <- apply(assay(spe), 1, var)
  top_markers <- names(sort(marker_vars, decreasing = TRUE))[1:n_markers]
  
  # Create a plot for each top marker
  marker_plots <- list()
  for (marker in top_markers) {
    message(paste("  Creating spatial plot for marker:", marker))
    p <- plot_marker_spatial(
      spe,
      marker = marker,
      point_size = config$point_size,
      color_scale = config$color_palettes$markers
    )
    
    # Save individual plot
    save_plot(
      p,
      file.path(output_dir, paste0("marker_spatial_", marker, ".pdf")),
      width = config$dimensions$width,
      height = config$dimensions$height,
      dpi = config$dimensions$dpi
    )
    
    marker_plots[[marker]] <- p
  }
  
  # Create combined plot of top markers
  message("Creating combined spatial visualization of top markers...")
  combined_plot <- combine_plots(
    marker_plots[1:min(6, length(marker_plots))],  # Take at most 6 for combined plot
    ncol = 2
  )
  
  save_plot(
    combined_plot,
    file.path(output_dir, "top_markers_spatial.pdf"),
    width = config$dimensions$width * 2,
    height = config$dimensions$height * 3,
    dpi = config$dimensions$dpi
  )
  
  # Create UMAP plot if coordinates available
  if (all(c("UMAP1", "UMAP2") %in% colnames(colData(spe)))) {
    message("Creating UMAP visualization...")
    
    # Basic UMAP plot
    p <- plot_umap(
      spe,
      point_size = config$point_size
    )
    
    save_plot(
      p,
      file.path(output_dir, "umap.pdf"),
      width = config$dimensions$width,
      height = config$dimensions$height,
      dpi = config$dimensions$dpi
    )
    
    # UMAP plots colored by top markers
    umap_plots <- list()
    for (marker in top_markers[1:min(6, length(top_markers))]) {
      message(paste("  Creating UMAP plot colored by marker:", marker))
      p <- plot_umap(
        spe,
        marker = marker,
        point_size = config$point_size,
        color_scale = config$color_palettes$markers
      )
      
      # Save individual plot
      save_plot(
        p,
        file.path(output_dir, paste0("umap_", marker, ".pdf")),
        width = config$dimensions$width,
        height = config$dimensions$height,
        dpi = config$dimensions$dpi
      )
      
      umap_plots[[marker]] <- p
    }
    
    # Create combined UMAP plot
    message("Creating combined UMAP visualization of top markers...")
    combined_umap <- combine_plots(
      umap_plots,
      ncol = 2
    )
    
    save_plot(
      combined_umap,
      file.path(output_dir, "umap_top_markers.pdf"),
      width = config$dimensions$width * 2,
      height = config$dimensions$height * 3,
      dpi = config$dimensions$dpi
    )
  }
  
  return(invisible(output_dir))
} 