# visualization_wrappers.R
# High-level wrapper functions for visualization tasks

# Load required libraries
library(SpatialExperiment)
library(ggplot2)

# Source core utilities
source("src/visualization/core_visualization.R")
source("src/visualization/marker_visualization.R")
source("src/visualization/phenotype_visualization.R")
source("src/visualization/spatial_visualization.R")
source("src/visualization/gated_cell_visualization.R")
source("src/visualization/channel_overlay_visualization.R")

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
  plot_types = c("markers", "phenotypes", "spatial", "channel_overlays", "all"),
  output_dir = "results/visualizations",
  config = NULL
) {
  # Debug information
  cat("DEBUG: create_visualizations called\n")
  cat("DEBUG: workflow_type =", workflow_type, "\n")
  cat("DEBUG: plot_types =", paste(plot_types, collapse = ", "), "\n")
  cat("DEBUG: output_dir =", output_dir, "\n")
  
  # Validate inputs
  if (is.null(spe)) {
    stop("SpatialExperiment object is NULL")
  }
  
  if (is.null(output_dir) || !is.character(output_dir) || length(output_dir) != 1) {
    stop("output_dir must be a character string")
  }
  
  # Match arguments
  workflow_type <- match.arg(workflow_type)
  plot_types <- match.arg(plot_types, several.ok = TRUE)
  
  # Expand "all" to include all plot types
  if ("all" %in% plot_types) {
    plot_types <- c("markers", "phenotypes", "spatial", "channel_overlays")
  }
  
  # Use default config if not provided
  if (is.null(config)) {
    config <- get_default_viz_config()
  }
  
  # Create output directory
  full_output_dir <- file.path(output_dir, workflow_type)
  cat("DEBUG: Creating full output directory:", full_output_dir, "\n")
  
  tryCatch({
    dir.create(full_output_dir, recursive = TRUE, showWarnings = TRUE)
  }, error = function(e) {
    cat("DEBUG: Error creating output directory:", e$message, "\n")
    stop(paste("Failed to create output directory:", full_output_dir, "-", e$message))
  })
  
  # Verify directory was created
  if (!dir.exists(full_output_dir)) {
    stop(paste("Failed to create output directory:", full_output_dir))
  }
  
  # Create marker visualizations
  if ("markers" %in% plot_types) {
    cat("DEBUG: Creating marker visualizations...\n")
    tryCatch({
      create_marker_visualizations(spe, full_output_dir, config)
    }, error = function(e) {
      cat("DEBUG: Error in create_marker_visualizations:", e$message, "\n")
      warning(paste("Failed to create marker visualizations:", e$message))
    })
  }
  
  # Create phenotype visualizations
  if ("phenotypes" %in% plot_types) {
    cat("DEBUG: Creating phenotype visualizations...\n")
    tryCatch({
      create_phenotype_visualizations(spe, full_output_dir, config)
    }, error = function(e) {
      cat("DEBUG: Error in create_phenotype_visualizations:", e$message, "\n")
      warning(paste("Failed to create phenotype visualizations:", e$message))
    })
  }
  
  # Create spatial visualizations
  if ("spatial" %in% plot_types) {
    cat("DEBUG: Creating spatial visualizations...\n")
    tryCatch({
      create_spatial_visualizations(spe, full_output_dir, config)
    }, error = function(e) {
      cat("DEBUG: Error in create_spatial_visualizations:", e$message, "\n")
      warning(paste("Failed to create spatial visualizations:", e$message))
    })
  }
  
  # Create gated cell visualizations if applicable
  if (workflow_type == "gated") {
    cat("DEBUG: Creating gated cell visualizations...\n")
    tryCatch({
      create_gated_cell_visualizations(spe, full_output_dir, config)
    }, error = function(e) {
      cat("DEBUG: Error in create_gated_cell_visualizations:", e$message, "\n")
      warning(paste("Failed to create gated cell visualizations:", e$message))
    })
  }
  
  # Create channel overlay visualizations
  if ("channel_overlays" %in% plot_types) {
    cat("DEBUG: Creating channel overlay visualizations...\n")
    tryCatch({
      # Check if images and masks are available in the workspace
      images_path <- file.path(dirname(full_output_dir), "images.rds")
      masks_path <- file.path(dirname(full_output_dir), "masks.rds")
      
      # Look in common alternative locations if not found
      if (!file.exists(images_path)) {
        alt_paths <- c(
          file.path(config$output_dir, "images.rds"),
          file.path(config$paths$output_dir, "images.rds"),
          "output/images.rds",
          "debug_output/images.rds"
        )
        for (alt_path in alt_paths) {
          if (!is.null(alt_path) && file.exists(alt_path)) {
            cat("DEBUG: Found images at alternative path:", alt_path, "\n")
            images_path <- alt_path
            break
          }
        }
      }
      
      if (!file.exists(masks_path)) {
        alt_paths <- c(
          file.path(config$output_dir, "masks.rds"),
          file.path(config$paths$output_dir, "masks.rds"),
          "output/masks.rds",
          "debug_output/masks.rds"
        )
        for (alt_path in alt_paths) {
          if (!is.null(alt_path) && file.exists(alt_path)) {
            cat("DEBUG: Found masks at alternative path:", alt_path, "\n")
            masks_path <- alt_path
            break
          }
        }
      }
      
      # Load images and masks if they exist
      images <- NULL
      masks <- NULL
      
      if (file.exists(images_path)) {
        cat("DEBUG: Loading images from", images_path, "\n")
        tryCatch({
          images <- readRDS(images_path)
          cat("DEBUG: Successfully loaded images\n")
        }, error = function(e) {
          cat("DEBUG: Error loading images:", e$message, "\n")
          warning("Error loading images: ", e$message)
        })
      } else {
        cat("DEBUG: Images file not found at", images_path, "\n")
        warning("Images file not found. Channel overlay visualization will be skipped.")
      }
      
      if (file.exists(masks_path)) {
        cat("DEBUG: Loading masks from", masks_path, "\n")
        tryCatch({
          masks <- readRDS(masks_path)
          cat("DEBUG: Successfully loaded masks\n")
        }, error = function(e) {
          cat("DEBUG: Error loading masks:", e$message, "\n")
          warning("Error loading masks: ", e$message)
        })
      } else {
        cat("DEBUG: Masks file not found at", masks_path, "\n")
        warning("Masks file not found. Single-cell preview will be skipped.")
      }
      
      # Create channel overlay visualizations if images are available
      if (!is.null(images)) {
        overlay_dir <- file.path(full_output_dir, "channel_overlays")
        
        # Extract configuration values with proper NULL checks
        max_cells <- 25
        if (!is.null(config) && !is.null(config$visualization) && !is.null(config$visualization$max_cells)) {
          max_cells <- config$visualization$max_cells
        }
        
        color_scheme <- "viridis"
        if (!is.null(config) && !is.null(config$visualization) && !is.null(config$visualization$color_scheme)) {
          color_scheme <- config$visualization$color_scheme
        }
        
        width <- 12
        height <- 10
        dpi <- 300
        if (!is.null(config) && !is.null(config$dimensions)) {
          width <- config$dimensions$width %||% width
          height <- config$dimensions$height %||% height
          dpi <- config$dimensions$dpi %||% dpi
        }
        
        # Create overlay visualizations
        result_paths <- create_channel_overlay_visualization(
          images = images,
          masks = masks,
          spe = spe,
          output_dir = overlay_dir,
          channels_to_highlight = NULL,  # Use all channels
          max_cells = max_cells,
          color_scheme = color_scheme,
          width = width,
          height = height,
          dpi = dpi
        )
        
        cat("DEBUG: Channel overlay visualizations created at:", overlay_dir, "\n")
      }
    }, error = function(e) {
      cat("DEBUG: Error in create_channel_overlay_visualization:", e$message, "\n")
      warning(paste("Failed to create channel overlay visualizations:", e$message))
    })
  }
  
  cat("DEBUG: Visualizations saved to", full_output_dir, "\n")
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
  # Debug information
  cat("DEBUG: create_visualization called\n")
  cat("DEBUG: plot_type =", plot_type, "\n")
  cat("DEBUG: output_file =", output_file %||% "NULL", "\n")
  
  # Validate inputs
  if (is.null(spe)) {
    stop("SpatialExperiment object is NULL")
  }
  
  # Use default config if not provided
  if (is.null(config)) {
    config <- get_default_viz_config()
  }
  
  # Create the requested visualization
  plot <- tryCatch({
    switch(
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
  }, error = function(e) {
    cat("DEBUG: Error creating plot:", e$message, "\n")
    stop(paste("Failed to create", plot_type, "plot:", e$message))
  })
  
  # Save plot if output file is specified
  if (!is.null(output_file)) {
    cat("DEBUG: Saving plot to", output_file, "\n")
    tryCatch({
      save_plot(
        plot, 
        output_file, 
        width = config$dimensions$width,
        height = config$dimensions$height,
        dpi = config$dimensions$dpi
      )
    }, error = function(e) {
      cat("DEBUG: Error saving plot:", e$message, "\n")
      warning(paste("Failed to save plot to", output_file, ":", e$message))
    })
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