# visualization_wrappers.R
# High-level wrapper functions for visualization tasks

# Load required libraries
library(SpatialExperiment)
library(ggplot2)

# Source core utilities
source("src/visualization/core_visualization.R")

#' Create visualizations for a SpatialExperiment object
#'
#' Main entry point for creating visualizations. This function replaces the
#' VisualizationManager class in the previous implementation.
#'
#' @param spe SpatialExperiment object
#' @param workflow_type Type of workflow ("unsupervised" or "gated")
#' @param plot_types Vector of plot types to create
#' @param output_dir Directory to save plots to
#' @param config Optional configuration list
#' @return Invisibly returns path to output directory
create_visualizations <- function(
  spe, 
  workflow_type = c("unsupervised", "gated"),
  plot_types = c("markers", "phenotypes", "spatial", "all"),
  output_dir = "results/visualizations",
  config = NULL
) {
  # Match arguments
  workflow_type <- match.arg(workflow_type)
  plot_types <- match.arg(plot_types, several.ok = TRUE)
  
  # Expand "all" to include all plot types
  if ("all" %in% plot_types) {
    plot_types <- c("markers", "phenotypes", "spatial")
  }
  
  # Use default config if not provided
  if (is.null(config)) {
    config <- get_default_viz_config()
  }
  
  # Create output directory
  full_output_dir <- file.path(output_dir, workflow_type)
  dir.create(full_output_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Create marker visualizations
  if ("markers" %in% plot_types) {
    message("Creating marker visualizations...")
    create_marker_visualizations(spe, full_output_dir, config)
  }
  
  # Create phenotype visualizations
  if ("phenotypes" %in% plot_types) {
    message("Creating phenotype visualizations...")
    create_phenotype_visualizations(spe, full_output_dir, config)
  }
  
  # Create spatial visualizations
  if ("spatial" %in% plot_types) {
    message("Creating spatial visualizations...")
    create_spatial_visualizations(spe, full_output_dir, config)
  }
  
  # Create gated cell visualizations if applicable
  if (workflow_type == "gated") {
    message("Creating gated cell visualizations...")
    create_gated_cell_visualizations(spe, full_output_dir, config)
  }
  
  message(paste("Visualizations saved to", full_output_dir))
  return(invisible(full_output_dir))
}

#' Create a specific visualization
#'
#' Creates a single visualization of a specific type
#'
#' @param spe SpatialExperiment object
#' @param plot_type Type of plot to create
#' @param output_file Optional output file to save plot to
#' @param config Optional configuration list
#' @param ... Additional parameters passed to specific visualization functions
#' @return Plot object
create_visualization <- function(
  spe, 
  plot_type, 
  output_file = NULL,
  config = NULL,
  ...
) {
  # Use default config if not provided
  if (is.null(config)) {
    config <- get_default_viz_config()
  }
  
  # Create the requested visualization
  plot <- switch(
    plot_type,
    "umap" = plot_umap(spe, ...),
    "tsne" = plot_tsne(spe, ...),
    "marker_heatmap" = plot_marker_heatmap(spe, ...),
    "marker_spatial" = plot_marker_spatial(spe, ...),
    "phenotype_composition" = plot_phenotype_composition(spe, ...),
    "phenotype_spatial" = plot_phenotype_spatial(spe, ...),
    "spatial_graph" = plot_spatial_graph(spe, ...),
    "spatial_community" = plot_spatial_community(spe, ...),
    "interaction_heatmap" = plot_interaction_heatmap(spe, ...),
    "gated_spatial" = plot_gated_cells_spatial(spe, ...),
    "gated_proportions" = plot_cell_type_proportions(spe, ...),
    stop(paste("Unknown plot type:", plot_type))
  )
  
  # Save plot if output file is specified
  if (!is.null(output_file)) {
    save_plot(
      plot, 
      output_file, 
      width = config$dimensions$width,
      height = config$dimensions$height,
      dpi = config$dimensions$dpi
    )
  }
  
  return(plot)
}

# Placeholder functions that will be implemented in domain-specific files
create_marker_visualizations <- function(spe, output_dir, config) {
  message("Note: create_marker_visualizations() needs to be implemented in marker_visualization.R")
}

create_phenotype_visualizations <- function(spe, output_dir, config) {
  message("Note: create_phenotype_visualizations() needs to be implemented in phenotype_visualization.R")
}

create_spatial_visualizations <- function(spe, output_dir, config) {
  message("Note: create_spatial_visualizations() needs to be implemented in spatial_visualization.R")
}

create_gated_cell_visualizations <- function(spe, output_dir, config) {
  message("Note: create_gated_cell_visualizations() needs to be implemented in gated_cell_visualization.R")
} 