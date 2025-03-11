# Script: gatedCellViz.R
# Description: Functions for visualizing gated cell data
# Author: Your Name
# Date: Current Date

library(ggplot2)
library(SpatialExperiment)
library(patchwork)

#' Create a spatial plot of gated cells
#'
#' @param spe SpatialExperiment object with gated cells
#' @param color_by Column to color cells by (default: "gated_celltype")
#' @param facet_by Column to facet plot by (default: NULL)
#' @param point_size Size of points in plot (default: 1)
#' @param title Plot title (default: "Gated Cell Spatial Distribution")
#' @return ggplot object
plotGatedCellsSpatial <- function(
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
    color_var = spe[[color_by]]
  )
  
  if (!is.null(facet_by)) {
    plot_data$facet_var <- spe[[facet_by]]
  }
  
  # Create the plot
  p <- ggplot(plot_data, aes(x = x, y = y, color = color_var)) +
    geom_point(size = point_size) +
    theme_minimal() +
    labs(title = title, color = color_by)
  
  # Add faceting if requested
  if (!is.null(facet_by)) {
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
plotCellTypeProportions <- function(
  spe,
  celltype_col = "gated_celltype",
  group_by = NULL,
  title = "Cell Type Proportions"
) {
  # Calculate proportions
  if (is.null(group_by)) {
    counts <- table(spe[[celltype_col]])
    props <- prop.table(counts) * 100
    
    plot_data <- data.frame(
      CellType = names(props),
      Proportion = as.numeric(props)
    )
    
    # Create plot
    p <- ggplot(plot_data, aes(x = reorder(CellType, -Proportion), y = Proportion, fill = CellType)) +
      geom_bar(stat = "identity") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      labs(title = title, x = "Cell Type", y = "Proportion (%)")
    
  } else {
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
