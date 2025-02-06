# This file contains all visualization-related functions

library(ggplot2)
library(reshape2)
library(gridExtra)

#' Visualization functions for spatial analysis pipeline
#'
#' These functions generate robust visualizations which are saved as PDF files
#' by the ResultsManager. They cover intensity metrics, channel distribution,
#' spatial hotspot overlays, and composite processed data visualizations.

#----------------------------------------------------------
# Function: plot_intensity_metrics_heatmap
#----------------------------------------------------------
#' @param intensity_metrics A numeric matrix with channels as rows and metrics as columns.
#' @return A ggplot heatmap object.
plot_intensity_metrics_heatmap <- function(intensity_metrics) {
  # Transform the intensity metrics matrix to a long format for ggplot.
  df <- melt(intensity_metrics, varnames = c("Channel", "Metric"), value.name = "Value")
  
  p <- ggplot(df, aes(x = Metric, y = Channel, fill = Value)) +
    geom_tile() +
    scale_fill_gradient2(
      low = "blue", mid = "white", high = "red",
      midpoint = median(df$Value, na.rm = TRUE)
    ) +
    labs(title = "Intensity Metrics Heatmap", x = "Metric", y = "Channel") +
    theme_minimal()
  
  return(p)
}

#----------------------------------------------------------
# Function: plot_intensity_distribution
#----------------------------------------------------------
#' @param intensity_values A numeric vector containing intensity values for a channel.
#' @param channel_name Name of the channel.
#' @param metadata Optional data frame with additional grouping info (e.g. ROI, Time, Replicate).
#' @return A ggplot object showing the intensity distribution.
plot_intensity_distribution <- function(intensity_values, channel_name, metadata = NULL) {
  threshold <- quantile(intensity_values, 0.9, na.rm = TRUE)
  
  # Combine intensity values with metadata if provided.
  if (!is.null(metadata)) {
    df <- data.frame(Intensity = intensity_values, metadata)
  } else {
    df <- data.frame(Intensity = intensity_values)
  }
  
  p <- ggplot(df, aes(x = Intensity)) +
    geom_histogram(aes(y = ..density..),
                   bins = 30, fill = "lightblue", color = "black", alpha = 0.7) +
    geom_density(color = "red", size = 1.2) +
    geom_vline(xintercept = threshold, linetype = "dashed", 
               color = "darkgreen", size = 1) +
    labs(
      title = paste("Intensity Distribution -", channel_name),
      subtitle = sprintf("90th percentile = %.2f", threshold),
      x = "Intensity", y = "Density"
    ) +
    theme_minimal()
  
  # Use faceting if grouping info is available.
  if (!is.null(metadata)) {
    # When both Time and Replicate are available, use a grid (this shows time vs. replicate).
    if (all(c("Time", "Replicate") %in% names(df))) {
      p <- p + facet_grid(Time ~ Replicate)
    } else if ("ROI" %in% names(df)) {
      p <- p + facet_wrap(~ ROI)
    }
  }
  
  return(p)
}

#----------------------------------------------------------
# Function: plot_spatial_hotspots
#----------------------------------------------------------
#' @param coords A matrix or data frame of spatial coordinates (with columns x and y).
#' @param intensity_values A numeric vector of intensities for a specific channel.
#' @param channel_name Name of the channel.
#' @return A ggplot object displaying cell locations colored by hotspot status.
plot_spatial_hotspots <- function(coords, intensity_values, channel_name) {
  threshold <- quantile(intensity_values, 0.9, na.rm = TRUE)
  hotspot_flag <- ifelse(intensity_values > threshold, "Hotspot", "Normal")
  df <- data.frame(
    x = coords[, 1],
    y = coords[, 2],
    Intensity = intensity_values,
    Status = factor(hotspot_flag, levels = c("Normal", "Hotspot"))
  )
  
  p <- ggplot(df, aes(x = x, y = y, color = Status)) +
    geom_point(alpha = 0.7, size = 2) +
    scale_color_manual(values = c("Normal" = "blue", "Hotspot" = "red")) +
    labs(
      title = paste("Spatial Hotspots -", channel_name),
      subtitle = sprintf("Cells above 90th percentile (%.2f)", threshold),
      x = "X Coordinate", y = "Y Coordinate"
    ) +
    theme_minimal()
  
  return(p)
}

#----------------------------------------------------------
# Function: visualize_processed_data
#----------------------------------------------------------
#' @param image_idx Index of the current image being visualized.
#' @param images The object containing image data.
#' @param panel_keep Data indicating which channels to keep.
#' @param spe The SpatialExperiment object.
#' @param masks Optional masks for visual overlays.
#' @return A composite plot (arranged using gridExtra) of processed data visualizations.
visualize_processed_data <- function(image_idx, images, panel_keep, spe, masks = NULL) {
  # For demonstration: 
  # 1. Create a density plot of intensities from the first channel.
  # 2. Create a placeholder plot for mask overlay.
  
  # Attempt to retrieve intensities (using the first channel as demonstration).
  intensities <- tryCatch(assay(spe, "intensities"), error = function(e) NULL)
  if (is.null(intensities)) {
    stop("No intensity data available in SPE.")
  }
  
  channel1 <- intensities[, 1]
  p1 <- ggplot(data.frame(Intensity = channel1), aes(x = Intensity)) +
    geom_density(fill = "lightblue", alpha = 0.7) +
    labs(
      title = paste("Image", image_idx, "Intensity Density (Channel 1)"),
      x = "Intensity", y = "Density"
    ) +
    theme_minimal()
  
  # Create a dummy plot for masks (if provided) or a placeholder.
  if (!is.null(masks)) {
    p2 <- ggplot() +
      geom_blank() +
      labs(title = paste("Image", image_idx, "Mask Overlay")) +
      theme_minimal()
  } else {
    p2 <- ggplot() +
      geom_blank() +
      labs(title = paste("Image", image_idx, "No Mask Provided")) +
      theme_minimal()
  }
  
  composite <- gridExtra::grid.arrange(p1, p2, ncol = 2)
  return(composite)
}

#----------------------------------------------------------
# Function: plot_intensity_hotspots
#----------------------------------------------------------
#' @param intensity_results The results list from an intensity analysis (should include metadata and the SPE object).
#' @param channel_name Name of the channel to plot hotspots for.
#' @param max_points Maximum number of points to display.
#' @param metadata Optional metadata; if NULL, attempts to use intensity_results$metadata.
#' @return A ggplot object showing spatial hotspots.
plot_intensity_hotspots <- function(intensity_results, channel_name, max_points, metadata = NULL) {
  # Extract the SPE object from intensity_results if available.
  if (is.null(intensity_results$spe)) {
    stop("SPE object not found in intensity_results")
  }
  spe <- intensity_results$spe
  intensities <- assay(spe, "intensities")
  coords <- spatialCoords(spe)
  
  # Find the column for the specified channel.
  channel_index <- match(channel_name, colnames(intensities))
  if (is.na(channel_index)) stop("Channel not found in intensities")
  
  x <- intensities[, channel_index]
  threshold <- quantile(x, 0.9, na.rm = TRUE)
  hotspot_flag <- x > threshold
  
  df <- data.frame(x = coords[,1], y = coords[,2], Intensity = x,
                   Status = ifelse(hotspot_flag, "Hotspot", "Normal"))
  df$Status <- factor(df$Status, levels = c("Normal", "Hotspot"))
  
  # Use provided metadata; if not provided, try using it from intensity_results.
  if (is.null(metadata) && !is.null(intensity_results$metadata)) {
    metadata <- intensity_results$metadata
  }
  if (!is.null(metadata)) {
    df <- cbind(df, metadata)
  }
  
  p <- ggplot(df, aes(x = x, y = y, color = Status)) +
    geom_point(alpha = 0.7, size = 2) +
    scale_color_manual(values = c("Normal" = "blue", "Hotspot" = "red")) +
    labs(
      title = paste("Spatial Hotspots -", channel_name),
      subtitle = sprintf("Cells above 90th percentile (%.2f)", threshold),
      x = "X Coordinate", y = "Y Coordinate"
    ) +
    theme_minimal()
  
  # Apply faceting if grouping metadata is available.
  if (!is.null(metadata)) {
    if (all(c("Time", "Replicate") %in% names(df))) {
      p <- p + facet_grid(Time ~ Replicate)
    } else if ("ROI" %in% names(df)) {
      p <- p + facet_wrap(~ ROI)
    }
  }
  
  return(p)
}

#' Factory for creating visualizations
VisualizationFactory <- R6::R6Class("VisualizationFactory",
  public = list(
    create_visualization = function(viz_type, data, results = NULL, ...) {
      viz_class <- switch(viz_type,
        "processed_data" = ProcessedDataVisualization,
        "neighborhood" = NeighborhoodVisualization,
        "temporal" = TemporalVisualization,
        "intensity_hotspots" = HotspotVisualization,
        stop("Unknown visualization type: ", viz_type)
      )
      
      viz_class$new(data = data, results = results, ...)
    }
  )
)

#' Base visualization class
Visualization <- R6::R6Class("Visualization",
  public = list(
    plot = NULL,
    data = NULL,
    results = NULL,
    
    initialize = function(data, results = NULL, ...) {
      self$data <- data
      self$results <- results
      self$plot <- self$create_plot(...)
    },
    
    create_plot = function(...) {
      stop("Abstract method: implement in subclass")
    },
    
    save = function(filename, width = 10, height = 8) {
      ggsave(filename, plot = self$plot, width = width, height = height)
    }
  )
)

#' Processed data visualization
ProcessedDataVisualization <- R6::R6Class("ProcessedDataVisualization",
  inherit = Visualization,
  public = list(
    create_plot = function(image_idx = 1, ...) {
      # Extract data
      images <- self$data$images
      panel_keep <- self$data$panel[self$data$panel$keep == 1, ]
      spe <- self$data$spe
      masks <- self$data$masks
      
      # Extract image data and dimensions
      img_data <- imageData(images)[[image_idx]]
      img_height <- dim(img_data)[1]
      img_width <- dim(img_data)[2]
      
      # Identify channels using panel_keep
      img_channels <- channelNames(images)
      channels_to_plot <- panel_keep$name[panel_keep$name %in% img_channels]
      channel_indices <- match(channels_to_plot, img_channels)
      
      # Create normalized plots for each channel
      plot_list <- lapply(seq_along(channels_to_plot), function(i) {
        channel_data <- img_data[,,channel_indices[i]]
        
        # Robust normalization using 1%-99% quantiles
        q1 <- quantile(channel_data, 0.01, na.rm = TRUE)
        q99 <- quantile(channel_data, 0.99, na.rm = TRUE)
        normalized_data <- (channel_data - q1) / (q99 - q1)
        normalized_data[normalized_data < 0] <- 0
        normalized_data[normalized_data > 1] <- 1
        
        # Create plot data
        df_plot <- data.frame(
          x = rep(1:img_width, each = img_height),
          y = rep(1:img_height, times = img_width),
          intensity = as.vector(normalized_data)
        )
        
        # Create individual channel plot
        ggplot(df_plot, aes(x = x, y = y, fill = intensity)) +
          geom_raster() +
          scale_fill_gradientn(colors = c("black", "yellow", "red")) +
          theme_minimal() +
          labs(title = channels_to_plot[i]) +
          theme(aspect.ratio = img_height/img_width)
      })
      
      # Arrange plots in a grid
      gridExtra::grid.arrange(grobs = plot_list, ncol = 3)
    }
  )
)

#' Neighborhood visualization
NeighborhoodVisualization <- R6::R6Class("NeighborhoodVisualization",
  inherit = Visualization,
  public = list(
    create_plot = function(...) {
      if (is.null(self$results$neighborhood)) {
        stop("No neighborhood analysis results available")
      }
      
      # Extract neighborhood results
      results <- self$results$neighborhood
      
      # Create network plot
      network_plot <- results$plots$network +
        labs(title = "Cell Neighborhood Network")
      
      # Create density plot
      density_plot <- results$plots$density +
        labs(title = "Cell Neighborhood Density")
      
      # Create distance distribution plot
      distance_plot <- results$plots$distance_dist +
        labs(title = "Neighbor Distance Distribution")
      
      # Arrange plots
      gridExtra::grid.arrange(
        network_plot, density_plot, distance_plot,
        ncol = 2
      )
    }
  )
)

#' Temporal visualization
TemporalVisualization <- R6::R6Class("TemporalVisualization",
  inherit = Visualization,
  public = list(
    create_plot = function(...) {
      if (is.null(self$results$temporal)) {
        stop("No temporal analysis results available")
      }
      
      # Extract temporal data
      temporal_data <- self$results$temporal
      
      # Create population changes plot
      pop_plot <- ggplot(reshape2::melt(temporal_data$cell_counts$immune)) +
        geom_bar(aes(x = Var1, y = value, fill = Var2), stat = "identity") +
        labs(title = "Immune Cell Population Changes",
             x = "Day", y = "Cell Count") +
        theme_minimal()
      
      # Create spatial distribution plot
      spatial_plot <- ggplot(temporal_data$spatial_df, 
                           aes(x = day, y = distance, color = cell_type)) +
        geom_boxplot() +
        labs(title = "Changes in Spatial Distribution",
             x = "Day", y = "Distance") +
        theme_minimal()
      
      # Arrange plots
      gridExtra::grid.arrange(pop_plot, spatial_plot, ncol = 2)
    }
  )
)

#' Hotspot visualization
HotspotVisualization <- R6::R6Class("HotspotVisualization",
  inherit = Visualization,
  public = list(
    create_plot = function(max_points = 5000, ...) {
      if (is.null(self$results$intensity)) {
        stop("No intensity analysis results available")
      }
      
      # Create correlation heatmap
      heatmap <- self$results$correlation_heatmap
      
      # Create hotspot plots for each channel
      channel_plots <- lapply(names(self$results$hotspot_analysis), function(channel) {
        plot_intensity_hotspots(self$results, channel, max_points)
      })
      
      # Arrange all plots
      plots <- c(list(heatmap), channel_plots)
      do.call(gridExtra::grid.arrange, c(plots, list(ncol = 2)))
    }
  )
) 