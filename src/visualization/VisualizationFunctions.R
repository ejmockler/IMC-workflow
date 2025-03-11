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
  # Attempt to retrieve "intensities" assay; if missing, fall back to "exprs".
  intensities <- tryCatch(assay(spe, "intensities"), error = function(e) NULL)
  if (is.null(intensities)) {
    message("'intensities' assay not found; attempting to use 'exprs' assay as fallback.")
    intensities <- tryCatch(assay(spe, "exprs"), error = function(e) NULL)
  }
  if (is.null(intensities)) {
    stop("No intensity or expression data available in SPE.")
  }
  
  # Left panel: Intensity density plot for Channel 1
  channel1 <- intensities[, 1]
  p1 <- ggplot(data.frame(Intensity = channel1), aes(x = Intensity)) +
    geom_density(fill = "lightblue", alpha = 0.7) +
    labs(
      title = paste("Image", image_idx, "Intensity Density (Channel 1)"),
      x = "Intensity", y = "Density"
    ) +
    theme_minimal()
  
  # Right panel: Real mask overlay plot using contour lines with fixed aspect ratio
  if (!is.null(masks) && length(masks) >= image_idx) {
    # Retrieve the image and corresponding mask
    img <- images[[image_idx]]
    mask <- masks[[image_idx]]
    
    # Check dimensions and warn if they do not match
    if (!all(dim(img)[1:2] == dim(mask))) {
      warning("Dimensions of image and mask do not match. Proceeding with original dimensions.")
    }
    
    # Convert the image to a raster for ggplot
    img_raster <- as.raster(img)
    
    # Create a data frame from the mask matrix for contour plotting
    # We assume that the mask is numeric (binary or integer-valued)
    mask_df <- data.frame(expand.grid(x = 1:ncol(mask), y = 1:nrow(mask)),
                          mask = as.vector(mask))
    
    # Create a composite overlay plot: image in the background and mask outlines over it.
    # The contour breaks = 0.5 is chosen for a binary mask (0/1 values).
    p2 <- ggplot() +
      annotation_raster(img_raster, xmin = 0, xmax = ncol(img), ymin = 0, ymax = nrow(img)) +
      geom_contour(data = mask_df, aes(x = x, y = y, z = mask),
                   breaks = 0.5, colour = "red", size = 0.8) +
      labs(title = paste("Image", image_idx, "Mask Overlay")) +
      theme_void() +
      coord_fixed(ratio = 1)  # Ensure the original aspect ratio is preserved
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
  coords <- SpatialExperiment::spatialCoords(spe)
  
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

#' Cell Type Proximity Visualization
#' Shows average distance relationships between different cell types
CellTypeProximityVisualization <- R6::R6Class("CellTypeProximityVisualization",
  inherit = Visualization,
  public = list(
    create_plot = function(celltype_column = "celltype", distance_metric = "euclidean", max_cells_per_type = 5000, ...) {
      # Extract data
      spe <- self$data
      
      # Find the best cell type column to use
      if (!celltype_column %in% colnames(colData(spe))) {
        possible_columns <- c("celltype_classified", "celltype_classified_cluster", "celltype")
        for (col in possible_columns) {
          if (col %in% colnames(colData(spe))) {
            celltype_column <- col
            message(paste("Using", celltype_column, "for cell type information"))
            break
          }
        }
      }
      
      # Extract required data
      cell_types <- as.character(spe[[celltype_column]])
      unique_types <- unique(cell_types)
      
      if (length(unique_types) <= 1) {
        warning("Only one cell type found, cannot create proximity visualization")
        return(ggplot() + 
                theme_void() + 
                annotate("text", x = 0, y = 0, label = "Only one cell type found"))
      }
      
      # Extract spatial coordinates
      if ("spatialCoords" %in% names(spe@int_metadata)) {
        coords <- spe@int_metadata$spatialCoords
      } else {
        # Find coordinate columns in colData
        coord_cols <- grep("^x$|^y$|Pos_X|Pos_Y|Location_Center_X|Location_Center_Y|Cell_X_Position|Cell_Y_Position", 
                         colnames(colData(spe)), value = TRUE, ignore.case = TRUE)
        
        if (length(coord_cols) >= 2) {
          # Select first matching X and Y columns
          x_col <- coord_cols[grep("x|X", coord_cols)[1]]
          y_col <- coord_cols[grep("y|Y", coord_cols)[1]]
          coords <- data.frame(x = spe[[x_col]], y = spe[[y_col]])
        } else {
          warning("No spatial coordinates found, cannot create proximity visualization")
          return(ggplot() + 
                  theme_void() + 
                  annotate("text", x = 0, y = 0, label = "No spatial coordinates found"))
        }
      }
      
      # Initialize distance matrix for average distances between cell types
      proximity_matrix <- matrix(NA, nrow = length(unique_types), ncol = length(unique_types))
      rownames(proximity_matrix) <- colnames(proximity_matrix) <- unique_types
      
      # For large datasets, subsample cells to improve performance
      cell_indices_by_type <- lapply(unique_types, function(type) {
        indices <- which(cell_types == type)
        if (length(indices) > max_cells_per_type) {
          set.seed(42)  # For reproducibility
          indices <- sample(indices, max_cells_per_type)
        }
        return(indices)
      })
      names(cell_indices_by_type) <- unique_types
      
      # Calculate median distance between each pair of cell types
      for (i in 1:length(unique_types)) {
        type_i <- unique_types[i]
        cells_i <- cell_indices_by_type[[type_i]]
        coords_i <- coords[cells_i, ]
        
        for (j in i:length(unique_types)) {
          type_j <- unique_types[j]
          cells_j <- cell_indices_by_type[[type_j]]
          coords_j <- coords[cells_j, ]
          
          if (i == j) {
            # For same cell type, calculate average distance to nearest neighbor of same type
            if (length(cells_i) > 1) {
              # Create distance matrix within the same type
              dist_matrix <- as.matrix(dist(coords_i, method = distance_metric))
              
              # Set diagonal to Inf to ignore self-distance
              diag(dist_matrix) <- Inf
              
              # Get minimum distance for each cell (nearest neighbor)
              min_distances <- apply(dist_matrix, 1, min)
              
              # Calculate median of minimum distances
              median_distance <- median(min_distances, na.rm = TRUE)
              proximity_matrix[type_i, type_j] <- median_distance
            } else {
              proximity_matrix[type_i, type_j] <- 0
            }
          } else {
            # For different cell types, calculate distances between all pairs
            # This is a simplified approach for large datasets
            # For each cell in type_i, find distance to closest cell in type_j
            closest_distances <- numeric(length(cells_i))
            
            for (k in 1:length(cells_i)) {
              dists <- sqrt((coords_i[k, "x"] - coords_j[, "x"])^2 + 
                           (coords_i[k, "y"] - coords_j[, "y"])^2)
              closest_distances[k] <- min(dists)
            }
            
            # Calculate median of minimum distances
            median_distance <- median(closest_distances, na.rm = TRUE)
            proximity_matrix[type_i, type_j] <- median_distance
            proximity_matrix[type_j, type_i] <- median_distance
          }
        }
      }
      
      # Convert to data frame for ggplot
      proximity_df <- reshape2::melt(proximity_matrix, 
                                    value.name = "distance", 
                                    varnames = c("CellType1", "CellType2"))
      
      # Create heatmap
      p <- ggplot(proximity_df, aes(x = CellType1, y = CellType2, fill = distance)) +
        geom_tile() +
        scale_fill_viridis_c(
          name = "Median Distance\n(spatial units)",
          option = "plasma", 
          direction = -1,  # Darker colors = closer proximity
          na.value = "grey80"
        ) +
        labs(
          title = "Cell Type Proximity Map",
          subtitle = "Darker colors indicate closer proximity between cell types",
          x = "Cell Type",
          y = "Cell Type"
        ) +
        theme_minimal() +
        theme(
          axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
          panel.grid = element_blank(),
          panel.background = element_rect(fill = "white"),
          plot.background = element_rect(fill = "white")
        )
      
      return(p)
    }
  )
)

#' Factory for creating visualizations
VisualizationFactory <- R6::R6Class("VisualizationFactory",
  public = list(
    create_visualization = function(viz_type, data, results = NULL, ...) {
      viz_class <- switch(viz_type,
        "processed_data" = ProcessedDataVisualization,
        "neighborhood" = NeighborhoodVisualization,
        "temporal" = TemporalVisualization,
        "intensity_hotspots" = HotspotVisualization,
        "community" = SpatialCommunityVisualization,
        "celltype_spatial" = CellTypeSpatialVisualization,
        "interaction" = InteractionVisualization,
        "immune_infiltration" = ImmuneInfiltrationVisualization,
        "marker_expression" = MarkerExpressionVisualization,
        "spatial_distance" = SpatialDistanceVisualization,
        "celltype_proximity" = CellTypeProximityVisualization,
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
      
      # Enhanced population changes plot
      pop_plot <- ggplot(reshape2::melt(temporal_data$cell_counts$immune), 
                         aes(x = factor(Var1), y = value, fill = Var2)) +
        geom_bar(stat = "identity", position = "dodge") +
        labs(title = "Immune Cell Population Changes Over Time",
             x = "Day",
             y = "Cell Count",
             fill = "Cell Type") +
        theme_minimal() +
        theme(text = element_text(size = 14))
      
      # Enhanced spatial distribution plot
      spatial_plot <- ggplot(temporal_data$spatial_df, 
                             aes(x = day, y = distance, color = cell_type)) +
        geom_boxplot() +
        labs(title = "Spatial Distribution of Cells Over Time",
             x = "Day",
             y = "Nearest Neighbor Distance",
             color = "Cell Type") +
        theme_minimal() +
        theme(text = element_text(size = 14))
      
      # Arrange both plots side-by-side
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

#' Spatial Community Visualization
SpatialCommunityVisualization <- R6::R6Class("SpatialCommunityVisualization",
  inherit = Visualization,
  public = list(
    create_plot = function(community_column = "community_id", img_id_column = "sample_id", 
                           overlay_markers = TRUE, ...) {
      # Extract data
      spe <- self$data
      
      # Check if required columns exist
      if (!(community_column %in% colnames(colData(spe)))) {
        stop(paste("Community column", community_column, "not found in SPE object"))
      }
      
      if (!(img_id_column %in% colnames(colData(spe)))) {
        # Try to find an alternative image ID column
        alternative_columns <- c("ImageNumber", "ImageID", "ROI", "Image")
        for (col in alternative_columns) {
          if (col %in% colnames(colData(spe))) {
            img_id_column <- col
            message(paste("Using", col, "as image ID column"))
            break
          }
        }
        if (!(img_id_column %in% colnames(colData(spe)))) {
          stop("No suitable image ID column found in SPE object")
        }
      }
      
      # Get unique ROIs
      rois <- unique(spe[[img_id_column]])
      
      # Create a multi-panel plot for communities across ROIs
      plot_list <- lapply(rois, function(roi) {
        # Subset data for this ROI
        roi_cells <- spe[[img_id_column]] == roi
        coords <- SpatialExperiment::spatialCoords(spe)[roi_cells, ]
        communities <- factor(spe[[community_column]][roi_cells])
        
        # Prepare data frame for plotting
        plot_df <- data.frame(
          x = coords[, 1],
          y = coords[, 2],
          Community = communities
        )
        
        # Add cell type information if available
        if ("celltype" %in% colnames(colData(spe))) {
          plot_df$CellType <- spe$celltype[roi_cells]
        }
        
        # Create community plot
        p <- ggplot(plot_df, aes(x = x, y = y, color = Community)) +
          geom_point(size = 1.5, alpha = 0.7) +
          scale_color_viridis_d(option = "turbo") +
          labs(title = paste("Communities in", roi)) +
          theme_minimal() +
          theme(
            panel.background = element_rect(fill = "white"),
            plot.background = element_rect(fill = "white", color = NA),
            aspect.ratio = 1
          )
        
        return(p)
      })
      
      # Arrange plots in a grid
      if (length(plot_list) > 1) {
        n_cols <- min(3, length(plot_list))
        arranged_plot <- gridExtra::grid.arrange(grobs = plot_list, ncol = n_cols)
        return(arranged_plot)
      } else if (length(plot_list) == 1) {
        return(plot_list[[1]])
      } else {
        stop("No plots created")
      }
    }
  )
)

#' Cell Type Spatial Distribution Visualization
CellTypeSpatialVisualization <- R6::R6Class("CellTypeSpatialVisualization",
  inherit = Visualization,
  public = list(
    create_plot = function(celltype_column = "celltype_classified", img_id_column = "sample_id", 
                          show_density = TRUE, cluster_column = "phenograph_corrected", ...) {
      # Extract data
      spe <- self$data
      
      # Check if required columns exist
      if (!(celltype_column %in% colnames(colData(spe)))) {
        # Try to find an alternative celltype column
        alternative_columns <- c("celltype", "celltype_classified_cluster", "celltype_cluster", "celltype_manual")
        for (col in alternative_columns) {
          if (col %in% colnames(colData(spe))) {
            celltype_column <- col
            message(paste("Using", col, "as cell type column"))
            break
          }
        }
        if (!(celltype_column %in% colnames(colData(spe)))) {
          stop("No suitable cell type column found in SPE object")
        }
      }
      
      if (!(img_id_column %in% colnames(colData(spe)))) {
        # Try to find an alternative image ID column
        alternative_columns <- c("ImageNumber", "ImageID", "ROI", "Image")
        for (col in alternative_columns) {
          if (col %in% colnames(colData(spe))) {
            img_id_column <- col
            message(paste("Using", col, "as image ID column"))
            break
          }
        }
        if (!(img_id_column %in% colnames(colData(spe)))) {
          stop("No suitable image ID column found in SPE object")
        }
      }
      
      # Get unique ROIs
      rois <- unique(spe[[img_id_column]])
      
      # Check if we need to map clusters to cell types
      use_cluster_mapping <- FALSE
      if (cluster_column %in% colnames(colData(spe)) && 
          any(grepl("^[0-9]+$", as.character(spe[[celltype_column]])))) {
        message("Cell type column appears to contain cluster numbers. Will attempt to map to proper cell types.")
        use_cluster_mapping <- TRUE
        
        # Create a mapping table from clusters to cell types
        if (cluster_column == celltype_column) {
          # If using the same column, rename with more descriptive labels
          cluster_levels <- levels(factor(spe[[cluster_column]]))
          cluster_names <- paste("Cell Type", cluster_levels)
          names(cluster_names) <- cluster_levels
        } else {
          # Try to infer a mapping from the data
          mapping_data <- data.frame(
            Cluster = spe[[cluster_column]],
            CellType = spe[[celltype_column]]
          )
          
          # For each cluster, find the most common cell type
          cluster_to_celltype <- aggregate(
            CellType ~ Cluster, 
            data = mapping_data, 
            FUN = function(x) {
              names(sort(table(x), decreasing = TRUE)[1])
            }
          )
          
          cluster_names <- cluster_to_celltype$CellType
          names(cluster_names) <- as.character(cluster_to_celltype$Cluster)
        }
      }
      
      # Create a multi-panel plot for cell types across ROIs
      spatial_plots <- lapply(rois, function(roi) {
        # Subset data for this ROI
        roi_cells <- spe[[img_id_column]] == roi
        coords <- SpatialExperiment::spatialCoords(spe)[roi_cells, ]
        
        # Get cell types, mapping if necessary
        if (use_cluster_mapping) {
          raw_celltypes <- spe[[celltype_column]][roi_cells]
          celltypes <- factor(cluster_names[as.character(raw_celltypes)])
        } else {
          celltypes <- factor(spe[[celltype_column]][roi_cells])
        }
        
        # Prepare data frame for plotting
        plot_df <- data.frame(
          x = coords[, 1],
          y = coords[, 2],
          CellType = celltypes
        )
        
        # Get a better color palette that works for many cell types
        n_types <- length(unique(celltypes))
        if (n_types <= 8) {
          colors <- RColorBrewer::brewer.pal(max(3, n_types), "Set2")
        } else if (n_types <= 12) {
          colors <- RColorBrewer::brewer.pal(n_types, "Paired")
        } else {
          colors <- viridis::viridis(n_types, option = "turbo")
        }
        
        # Clean up ROI name for title
        roi_name <- gsub("_", " ", basename(as.character(roi)))
        
        # Create cell type spatial plot
        p <- ggplot(plot_df, aes(x = x, y = y, color = CellType)) +
          geom_point(size = 0.8, alpha = 0.7) +
          scale_color_manual(values = colors) +
          labs(title = roi_name) +
          theme_minimal() +
          theme(
            panel.background = element_rect(fill = "white"),
            plot.background = element_rect(fill = "white", color = NA),
            aspect.ratio = 1,
            legend.position = "right",
            plot.title = element_text(size = 10, hjust = 0.5),
            axis.text = element_text(size = 7),
            axis.title = element_text(size = 8),
            legend.title = element_text(size = 8),
            legend.text = element_text(size = 7),
            legend.key.size = unit(0.5, "cm")
          )
        
        return(p)
      })
      
      # Create summary plots showing cell type proportions
      if (use_cluster_mapping) {
        # Substitute the mapped cell type names
        mapped_celltypes <- sapply(as.character(spe[[celltype_column]]), function(ct) {
          ifelse(ct %in% names(cluster_names), cluster_names[ct], ct)
        })
        celltype_counts <- as.data.frame(table(spe[[img_id_column]], mapped_celltypes))
      } else {
        celltype_counts <- as.data.frame(table(spe[[img_id_column]], spe[[celltype_column]]))
      }
      
      names(celltype_counts) <- c("ROI", "CellType", "Count")
      
      # Calculate percentages
      celltype_counts <- transform(celltype_counts, 
                                  Percentage = Count / ave(Count, ROI, FUN = sum) * 100)
      
      # Helper function to abbreviate long ROI names while keeping them distinct
      abbreviate_roi_names <- function(roi_names, max_length = 15) {
        if (all(nchar(roi_names) <= max_length)) {
          return(roi_names)  # No need to abbreviate
        }
        
        # Extract meaningful parts from ROI names
        abbreviated <- sapply(roi_names, function(name) {
          # Clean up name
          name <- gsub("_", " ", basename(as.character(name)))
          
          # Extract parts - common pattern: "IMC 241218 Alun ROI D1 M1 01 9"
          parts <- unlist(strsplit(name, " "))
          
          if (length(parts) >= 3) {
            # Extract key identifying information
            suffix <- tail(parts, 3)  # Last 3 parts often contain unique identifiers
            abbrev <- paste(suffix, collapse=" ")
            
            # If still too long, take just the last 2 parts
            if (nchar(abbrev) > max_length) {
              abbrev <- paste(tail(parts, 2), collapse=" ")
            }
            
            return(abbrev)
          } else {
            # Simple abbreviation if structure doesn't match expected
            return(substr(name, 1, max_length))
          }
        })
        
        # Ensure uniqueness
        if (length(unique(abbreviated)) < length(roi_names)) {
          # If duplicates, revert to simple abbreviation with unique identifiers
          abbreviated <- make.unique(substr(roi_names, 1, max_length))
        }
        
        return(abbreviated)
      }
      
      # Shorten ROI names for readability
      original_rois <- as.character(celltype_counts$ROI)
      celltype_counts$ROI_display <- abbreviate_roi_names(original_rois)
      
      # Keep original ROI names for reference
      celltype_counts$ROI_full <- original_rois
      
      # Create bar plot of cell type proportions with better handling of long names
      proportion_plot <- ggplot(celltype_counts, aes(x = ROI_display, y = Percentage, fill = CellType)) +
        geom_bar(stat = "identity", position = "stack") +
        labs(title = "Cell Type Proportions by ROI", 
             y = "Percentage", 
             x = "ROI") +
        theme_minimal() +
        theme(
          panel.background = element_rect(fill = "white"),
          plot.background = element_rect(fill = "white", color = NA),
          axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 8),
          plot.title = element_text(size = 12, hjust = 0.5),
          legend.position = "right"
        )
      
      # Create density ridges plot with better handling of long text
      if (show_density && requireNamespace("ggridges", quietly = TRUE)) {
        # Combine all ROIs and calculate distances
        all_distances <- data.frame()
        
        for (roi in rois) {
          roi_cells <- spe[[img_id_column]] == roi
          coords <- SpatialExperiment::spatialCoords(spe)[roi_cells, ]
          
          # Get cell types, mapping if necessary
          if (use_cluster_mapping) {
            raw_celltypes <- spe[[celltype_column]][roi_cells]
            celltypes <- cluster_names[as.character(raw_celltypes)]
          } else {
            celltypes <- spe[[celltype_column]][roi_cells]
          }
          
          # Calculate distances from each cell to center of ROI
          center_x <- mean(coords[, 1])
          center_y <- mean(coords[, 2])
          distances <- sqrt((coords[, 1] - center_x)^2 + (coords[, 2] - center_y)^2)
          
          roi_distances <- data.frame(
            ROI = roi,
            CellType = celltypes,
            Distance = distances
          )
          
          all_distances <- rbind(all_distances, roi_distances)
        }
        
        # Create density ridges plot with better layout
        density_plot <- ggplot(all_distances, aes(x = Distance, y = CellType, fill = CellType)) +
          ggridges::geom_density_ridges(alpha = 0.7, scale = 0.9) +
          labs(title = "Cell Type Spatial Distribution", 
               x = "Distance from ROI Center", 
               y = "Cell Type") +
          theme_minimal() +
          theme(
            panel.background = element_rect(fill = "white"),
            plot.background = element_rect(fill = "white", color = NA),
            legend.position = "none",
            plot.title = element_text(size = 12, hjust = 0.5)
          )
        
        # Combine all plots
        plot_list <- c(spatial_plots, list(proportion_plot, density_plot))
      } else {
        plot_list <- c(spatial_plots, list(proportion_plot))
      }
      
      # Arrange plots in a grid with better layout
      n_plots <- length(plot_list)
      
      # Determine optimal layout based on number of plots
      if (n_plots <= 3) {
        n_cols <- n_plots
      } else if (n_plots <= 6) {
        n_cols <- min(3, n_plots)
      } else if (n_plots <= 12) {
        n_cols <- 4  # Use 4 columns for many plots
      } else {
        n_cols <- 5  # Use 5 columns for very many plots
      }
      
      # Calculate rows
      n_rows <- ceiling(n_plots / n_cols)
      
      # Place summary plots at bottom with full width if possible
      if (n_plots > 4) {
        # Calculate how many plots on the last row
        last_row_count <- n_plots - (n_rows - 1) * n_cols
        
        # If last row isn't full, place summary plots there
        if (last_row_count < n_cols && last_row_count > 0) {
          # Just proportion plot at the end with full width
          if (length(plot_list) == n_rows * n_cols - n_cols + 1) {
            # One plot on last row, make it span all columns
            layout_matrix <- matrix(c(1:(n_plots-1), rep(n_plots, n_cols)), 
                                   nrow = n_rows, 
                                   ncol = n_cols, 
                                   byrow = TRUE)
            
            # Set relative heights with more space for the summary row
            heights <- c(rep(1, n_rows-1), 1.2)
            
            arranged_plot <- gridExtra::grid.arrange(
              grobs = plot_list,
              layout_matrix = layout_matrix,
              heights = heights
            )
            
            return(arranged_plot)
          }
        }
      }
      
      # Default arrangement
      arranged_plot <- gridExtra::grid.arrange(
        grobs = plot_list, 
        ncol = n_cols
      )
      
      return(arranged_plot)
    }
  )
)

#' Cell-Cell Interaction Visualization
InteractionVisualization <- R6::R6Class("InteractionVisualization",
  inherit = Visualization,
  public = list(
    create_plot = function(interaction_results = NULL, colPairName = "neighborhood", ...) {
      # Extract data
      spe <- self$data
      
      # Find the best cell type column to use
      celltype_column <- "celltype"
      possible_columns <- c("celltype_classified", "celltype_classified_cluster", "celltype")
      for (col in possible_columns) {
        if (col %in% colnames(colData(spe))) {
          celltype_column <- col
          message(paste("Using", celltype_column, "for cell type information"))
          break
        }
      }
      
      # If interaction results are provided, use them
      if (!is.null(interaction_results)) {
        # Create interaction heatmap
        if ("sigval" %in% colnames(interaction_results)) {
          # Summarize interaction results by cell type pair
          interaction_summary <- as.data.frame(interaction_results) %>%
            dplyr::group_by(from_label, to_label) %>%
            dplyr::summarize(sum_sigval = sum(sigval, na.rm = TRUE)) %>%
            dplyr::ungroup()
          
          # Create a matrix for heatmap
          cell_types <- unique(c(interaction_summary$from_label, interaction_summary$to_label))
          interaction_matrix <- matrix(0, nrow = length(cell_types), ncol = length(cell_types))
          rownames(interaction_matrix) <- colnames(interaction_matrix) <- cell_types
          
          for (i in 1:nrow(interaction_summary)) {
            from <- as.character(interaction_summary$from_label[i])
            to <- as.character(interaction_summary$to_label[i])
            interaction_matrix[from, to] <- interaction_summary$sum_sigval[i]
          }
          
          # Create heatmap
          interaction_df <- reshape2::melt(interaction_matrix, varnames = c("from_label", "to_label"), value.name = "sum_sigval")
          
          heatmap_plot <- ggplot(interaction_df, aes(x = from_label, y = to_label, fill = sum_sigval)) +
            geom_tile() +
            scale_fill_gradient2(low = scales::muted("blue"), mid = "white", high = scales::muted("red")) +
            labs(title = "Cell-Cell Interaction Summary", 
                 x = "From Cell Type", 
                 y = "To Cell Type",
                 fill = "Interaction\nScore") +
            theme_minimal() +
            theme(
              panel.background = element_rect(fill = "white"),
              plot.background = element_rect(fill = "white", color = NA),
              axis.text.x = element_text(angle = 45, hjust = 1)
            )
          
          return(heatmap_plot)
        } else {
          message("Interaction results don't contain expected 'sigval' column. Calculating new interactions.")
        }
      }
      
      # If we reach here, calculate interaction statistics
      if (!requireNamespace("imcRtools", quietly = TRUE)) {
        stop("imcRtools package required for interaction analysis")
      }
      
      # Check if we have the required graph and cell type information
      if (!(colPairName %in% SingleCellExperiment::colPairNames(spe))) {
        stop(paste("Spatial graph", colPairName, "not found in SPE object"))
      }
      
      if (!(celltype_column %in% colnames(colData(spe)))) {
        stop(paste("Cell type information", celltype_column, "not found in SPE object"))
      }
      
      # Identify the appropriate image ID column
      img_id_column <- NULL
      for (col_name in c("sample_id", "ImageNumber", "ImageID", "ROI", "Image")) {
        if (col_name %in% colnames(colData(spe))) {
          img_id_column <- col_name
          break
        }
      }
      
      if (is.null(img_id_column)) {
        stop("No suitable image ID column found in SPE object")
      }
      
      # Calculate interaction statistics using imcRtools
      set.seed(42)  # For reproducible permutation tests
      message("Calculating cell-cell interactions...")
      interaction_results <- imcRtools::testInteractions(
        spe,
        group_by = img_id_column,
        label = celltype_column,  # Use the identified column
        colPairName = colPairName
      )
      
      # Summarize interaction results by cell type pair
      interaction_summary <- as.data.frame(interaction_results) %>%
        dplyr::group_by(from_label, to_label) %>%
        dplyr::summarize(sum_sigval = sum(sigval, na.rm = TRUE)) %>%
        dplyr::ungroup()
      
      # Create a matrix for heatmap
      cell_types <- unique(c(interaction_summary$from_label, interaction_summary$to_label))
      interaction_matrix <- matrix(0, nrow = length(cell_types), ncol = length(cell_types))
      rownames(interaction_matrix) <- colnames(interaction_matrix) <- cell_types
      
      for (i in 1:nrow(interaction_summary)) {
        from <- as.character(interaction_summary$from_label[i])
        to <- as.character(interaction_summary$to_label[i])
        interaction_matrix[from, to] <- interaction_summary$sum_sigval[i]
      }
      
      # Create heatmap
      interaction_df <- reshape2::melt(interaction_matrix, varnames = c("from_label", "to_label"), value.name = "sum_sigval")
      
      heatmap_plot <- ggplot(interaction_df, aes(x = from_label, y = to_label, fill = sum_sigval)) +
        geom_tile() +
        scale_fill_gradient2(low = scales::muted("blue"), mid = "white", high = scales::muted("red")) +
        labs(title = "Cell-Cell Interaction Summary", 
             x = "From Cell Type", 
             y = "To Cell Type",
             fill = "Interaction\nScore") +
        theme_minimal() +
        theme(
          panel.background = element_rect(fill = "white"),
          plot.background = element_rect(fill = "white", color = NA),
          axis.text.x = element_text(angle = 45, hjust = 1)
        )
      
      return(heatmap_plot)
    }
  )
)

#' Immune Infiltration Spatial Visualization
ImmuneInfiltrationVisualization <- R6::R6Class("ImmuneInfiltrationVisualization",
  inherit = Visualization,
  public = list(
    create_plot = function(immune_column = "is_immune", img_id_column = "sample_id", 
                          include_metrics = TRUE, ...) {
      # Extract data
      spe <- self$data
      
      # Check if required columns exist
      if (!(immune_column %in% colnames(colData(spe)))) {
        stop(paste("Immune column", immune_column, "not found in SPE object"))
      }
      
      if (!(img_id_column %in% colnames(colData(spe)))) {
        # Try to find an alternative image ID column
        alternative_columns <- c("ImageNumber", "ImageID", "ROI", "Image")
        for (col in alternative_columns) {
          if (col %in% colnames(colData(spe))) {
            img_id_column <- col
            message(paste("Using", col, "as image ID column"))
            break
          }
        }
        if (!(img_id_column %in% colnames(colData(spe)))) {
          stop("No suitable image ID column found in SPE object")
        }
      }
      
      # Get unique ROIs
      rois <- unique(spe[[img_id_column]])
      
      # Create a multi-panel plot for immune infiltration across ROIs
      spatial_plots <- lapply(rois, function(roi) {
        # Subset data for this ROI
        roi_cells <- spe[[img_id_column]] == roi
        coords <- SpatialExperiment::spatialCoords(spe)[roi_cells, ]
        is_immune <- spe[[immune_column]][roi_cells]
        
        # Prepare data frame for plotting
        plot_df <- data.frame(
          x = coords[, 1],
          y = coords[, 2],
          IsImmune = factor(is_immune, levels = c(FALSE, TRUE), 
                            labels = c("Non-immune", "Immune"))
        )
        
        # Add cell type information if available
        if ("celltype" %in% colnames(colData(spe))) {
          plot_df$CellType <- spe$celltype[roi_cells]
        }
        
        # Create immune infiltration spatial plot
        p <- ggplot(plot_df, aes(x = x, y = y, color = IsImmune)) +
          geom_point(size = 1, alpha = 0.7) +
          scale_color_manual(values = c("Non-immune" = "lightgrey", "Immune" = "darkred")) +
          labs(title = paste("Immune Infiltration in", roi),
               color = "Cell Type") +
          theme_minimal() +
          theme(
            panel.background = element_rect(fill = "white"),
            plot.background = element_rect(fill = "white", color = NA),
            aspect.ratio = 1
          )
        
        return(p)
      })
      
      # Create summary metrics for immune infiltration
      if (include_metrics) {
        # Calculate immune percentage by ROI
        immune_summary <- data.frame()
        
        for (roi in rois) {
          roi_cells <- spe[[img_id_column]] == roi
          is_immune <- spe[[immune_column]][roi_cells]
          
          # Calculate metrics
          total_cells <- sum(roi_cells)
          immune_cells <- sum(is_immune[roi_cells])
          immune_percent <- (immune_cells / total_cells) * 100
          
          # Calculate nearest neighbor distances between immune cells
          coords <- SpatialExperiment::spatialCoords(spe)[roi_cells, ]
          immune_coords <- coords[is_immune[roi_cells], ]
          
          # Only calculate if there are at least 2 immune cells
          if (nrow(immune_coords) >= 2) {
            # Calculate pairwise distances
            dist_matrix <- as.matrix(dist(immune_coords))
            # Get nearest neighbor distances (excluding self)
            diag(dist_matrix) <- Inf
            nearest_dist <- apply(dist_matrix, 1, min)
            median_dist <- median(nearest_dist)
          } else {
            median_dist <- NA
          }
          
          roi_summary <- data.frame(
            ROI = roi,
            TotalCells = total_cells,
            ImmuneCells = immune_cells,
            ImmunePercent = immune_percent,
            MedianNNDist = median_dist
          )
          
          immune_summary <- rbind(immune_summary, roi_summary)
        }
        
        # Create bar plot of immune percentages
        percent_plot <- ggplot(immune_summary, aes(x = ROI, y = ImmunePercent)) +
          geom_bar(stat = "identity", fill = "darkred") +
          labs(title = "Immune Cell Percentage by ROI", 
               y = "Immune %", 
               x = "ROI") +
          theme_minimal() +
          theme(
            panel.background = element_rect(fill = "white"),
            plot.background = element_rect(fill = "white", color = NA),
            axis.text.x = element_text(angle = 45, hjust = 1)
          )
        
        # Create scatter plot of immune cell count vs total cells
        count_plot <- ggplot(immune_summary, aes(x = TotalCells, y = ImmuneCells)) +
          geom_point(size = 3) +
          geom_text(aes(label = ROI), vjust = -1) +
          geom_smooth(method = "lm", se = FALSE, color = "darkred") +
          labs(title = "Immune Cell Count vs Total Cells", 
               x = "Total Cells", 
               y = "Immune Cells") +
          theme_minimal() +
          theme(
            panel.background = element_rect(fill = "white"),
            plot.background = element_rect(fill = "white", color = NA)
          )
        
        # Create nearest neighbor distance plot
        dist_plot <- ggplot(immune_summary, aes(x = ROI, y = MedianNNDist)) +
          geom_bar(stat = "identity", fill = "steelblue") +
          labs(title = "Median Distance Between Immune Cells", 
               y = "Distance (μm)", 
               x = "ROI") +
          theme_minimal() +
          theme(
            panel.background = element_rect(fill = "white"),
            plot.background = element_rect(fill = "white", color = NA),
            axis.text.x = element_text(angle = 45, hjust = 1)
          )
        
        # Add summary plots to the list
        plot_list <- c(spatial_plots, list(percent_plot, count_plot, dist_plot))
      } else {
        plot_list <- spatial_plots
      }
      
      # Arrange plots in a grid
      n_cols <- min(3, length(plot_list))
      arranged_plot <- gridExtra::grid.arrange(grobs = plot_list, ncol = n_cols)
      return(arranged_plot)
    }
  )
)

#' Marker Expression Heatmap Visualization
MarkerExpressionVisualization <- R6::R6Class("MarkerExpressionVisualization",
  inherit = Visualization,
  public = list(
    create_plot = function(group_column = "celltype", assay_name = "exprs", 
                          top_markers = 25, show_enrichment = TRUE, ...) {
      # Extract data
      spe <- self$data
      
      # Check if required columns and assays exist
      if (!(group_column %in% colnames(colData(spe)))) {
        stop(paste("Group column", group_column, "not found in SPE object"))
      }
      
      if (!(assay_name %in% SummarizedExperiment::assayNames(spe))) {
        # Try to find an alternative assay
        alternative_assays <- c("counts", "logcounts", "normcounts")
        for (assay in alternative_assays) {
          if (assay %in% SummarizedExperiment::assayNames(spe)) {
            assay_name <- assay
            message(paste("Using", assay, "as expression assay"))
            break
          }
        }
        if (!(assay_name %in% SummarizedExperiment::assayNames(spe))) {
          stop("No suitable expression assay found in SPE object")
        }
      }
      
      # Get expression data
      expr_data <- SummarizedExperiment::assay(spe, assay_name)
      
      # Get unique groups
      groups <- unique(spe[[group_column]])
      
      # Calculate mean expression per group
      mean_expr <- matrix(0, nrow = nrow(expr_data), ncol = length(groups))
      rownames(mean_expr) <- rownames(expr_data)
      colnames(mean_expr) <- groups
      
      for (i in seq_along(groups)) {
        group_cells <- which(spe[[group_column]] == groups[i])
        if (length(group_cells) > 0) {
          mean_expr[, i] <- rowMeans(expr_data[, group_cells, drop = FALSE])
        }
      }
      
      # Calculate marker variance across groups to identify most informative markers
      marker_variance <- apply(mean_expr, 1, var)
      top_marker_indices <- order(marker_variance, decreasing = TRUE)[1:min(top_markers, length(marker_variance))]
      top_mean_expr <- mean_expr[top_marker_indices, ]
      
      # Scale the data for better visualization
      scaled_expr <- t(scale(t(top_mean_expr)))
      
      # Create heatmap using pheatmap
      if (requireNamespace("pheatmap", quietly = TRUE)) {
        # Set up colors
        heatmap_colors <- colorRampPalette(c("navy", "white", "firebrick3"))(100)
        
        # Create heatmap
        heatmap_plot <- pheatmap::pheatmap(
          scaled_expr,
          color = heatmap_colors,
          cluster_rows = TRUE,
          cluster_cols = TRUE,
          fontsize_row = 10,
          fontsize_col = 10,
          main = paste("Scaled Marker Expression by", group_column),
          silent = TRUE  # Return the object instead of plotting
        )
        
        # If requested, also create marker enrichment scores
        if (show_enrichment) {
          # Calculate marker enrichment score (log fold change vs mean of other groups)
          enrichment_scores <- matrix(0, nrow = nrow(top_mean_expr), ncol = ncol(top_mean_expr))
          rownames(enrichment_scores) <- rownames(top_mean_expr)
          colnames(enrichment_scores) <- colnames(top_mean_expr)
          
          for (i in 1:ncol(top_mean_expr)) {
            # For each group, calculate log fold change vs mean of other groups
            other_groups <- setdiff(1:ncol(top_mean_expr), i)
            if (length(other_groups) > 0) {
              mean_other <- rowMeans(top_mean_expr[, other_groups, drop = FALSE])
              # Add small pseudocount to avoid division by zero
              enrichment_scores[, i] <- log2((top_mean_expr[, i] + 0.1) / (mean_other + 0.1))
            }
          }
          
          # Create enrichment heatmap
          enrichment_heatmap <- pheatmap::pheatmap(
            enrichment_scores,
            color = colorRampPalette(c("navy", "white", "firebrick3"))(100),
            cluster_rows = TRUE,
            cluster_cols = TRUE,
            fontsize_row = 10,
            fontsize_col = 10,
            main = paste("Marker Enrichment Scores by", group_column),
            silent = TRUE  # Return the object instead of plotting
          )
          
          # Combine both heatmaps into a grid
          gridExtra::grid.arrange(
            heatmap_plot$gtable, 
            enrichment_heatmap$gtable, 
            ncol = 2
          )
        } else {
          return(heatmap_plot$gtable)
        }
      } else {
        # Fallback to basic heatmap using ggplot2
        expr_df <- reshape2::melt(scaled_expr, varnames = c("Marker", "Group"), value.name = "Expression")
        
        heatmap_plot <- ggplot(expr_df, aes(x = Group, y = Marker, fill = Expression)) +
          geom_tile() +
          scale_fill_gradient2(low = "navy", mid = "white", high = "firebrick3") +
          labs(title = paste("Scaled Marker Expression by", group_column)) +
          theme_minimal() +
          theme(
            panel.background = element_rect(fill = "white"),
            plot.background = element_rect(fill = "white", color = NA),
            axis.text.x = element_text(angle = 45, hjust = 1)
          )
        
        return(heatmap_plot)
      }
    }
  )
)

#' Spatial Distance Analysis Visualization
SpatialDistanceVisualization <- R6::R6Class("SpatialDistanceVisualization",
  inherit = Visualization,
  public = list(
    create_plot = function(target_column = "is_immune", img_id_column = "sample_id", 
                          distance_method = "min", ...) {
      # Extract data
      spe <- self$data
      
      # Check if required columns exist
      if (!(target_column %in% colnames(colData(spe)))) {
        stop(paste("Target column", target_column, "not found in SPE object"))
      }
      
      if (!(img_id_column %in% colnames(colData(spe)))) {
        # Try to find an alternative image ID column
        alternative_columns <- c("ImageNumber", "ImageID", "ROI", "Image")
        for (col in alternative_columns) {
          if (col %in% colnames(colData(spe))) {
            img_id_column <- col
            message(paste("Using", col, "as image ID column"))
            break
          }
        }
        if (!(img_id_column %in% colnames(colData(spe)))) {
          stop("No suitable image ID column found in SPE object")
        }
      }
      
      # Get unique ROIs
      rois <- unique(spe[[img_id_column]])
      
      # Create a list to store results for each ROI
      all_distances <- data.frame()
      
      # Calculate distances for each ROI
      for (roi in rois) {
        # Subset data for this ROI
        roi_cells <- spe[[img_id_column]] == roi
        coords <- SpatialExperiment::spatialCoords(spe)[roi_cells, ]
        target_status <- spe[[target_column]][roi_cells]
        
        # If we have cell types, include them
        if ("celltype" %in% colnames(colData(spe))) {
          cell_types <- spe$celltype[roi_cells]
        } else {
          cell_types <- rep("Unknown", length(target_status))
        }
        
        # Identify target and non-target cells
        target_indices <- which(target_status)
        non_target_indices <- which(!target_status)
        
        # Skip ROIs with too few cells of either type
        if (length(target_indices) < 2 || length(non_target_indices) < 2) {
          message(paste("Skipping ROI", roi, "due to insufficient cells"))
          next
        }
        
        # Calculate distances from each cell to nearest target cell
        if (distance_method == "min") {
          # For each non-target cell, find distance to nearest target cell
          distances <- numeric(length(non_target_indices))
          
          for (i in seq_along(non_target_indices)) {
            idx <- non_target_indices[i]
            cell_coords <- coords[idx, ]
            target_coords <- coords[target_indices, ]
            
            # Calculate Euclidean distances to all target cells
            dists <- sqrt(rowSums((target_coords - matrix(cell_coords, 
                                                          nrow = length(target_indices), 
                                                          ncol = 2, 
                                                          byrow = TRUE))^2))
            
            # Find minimum distance
            distances[i] <- min(dists)
          }
          
          # Prepare data frame
          roi_distances <- data.frame(
            ROI = roi,
            CellType = cell_types[non_target_indices],
            Distance = distances,
            TargetStatus = "Non-target"
          )
        } else if (distance_method == "center") {
          # Calculate center of target cells
          target_center <- colMeans(coords[target_indices, ])
          
          # Calculate distance from each cell to target center
          all_distances_to_center <- sqrt(rowSums((coords - matrix(target_center, 
                                                                  nrow = nrow(coords), 
                                                                  ncol = 2, 
                                                                  byrow = TRUE))^2))
          
          # Prepare data frame
          roi_distances <- data.frame(
            ROI = roi,
            CellType = cell_types,
            Distance = all_distances_to_center,
            TargetStatus = ifelse(target_status, "Target", "Non-target")
          )
        }
        
        # Add to overall results
        all_distances <- rbind(all_distances, roi_distances)
      }
      
      # If no valid distances were calculated, stop
      if (nrow(all_distances) == 0) {
        stop("No valid distances could be calculated")
      }
      
      # Create plots
      
      # 1. Density ridges plot of distances by cell type
      if (requireNamespace("ggridges", quietly = TRUE)) {
        density_plot <- ggplot(all_distances, aes(x = Distance, y = CellType, fill = CellType)) +
          ggridges::geom_density_ridges(alpha = 0.7) +
          labs(title = paste("Distance to", ifelse(distance_method == "min", "Nearest Target Cell", "Target Center")), 
               x = "Distance (μm)", 
               y = "Cell Type") +
          theme_minimal() +
          theme(
            panel.background = element_rect(fill = "white"),
            plot.background = element_rect(fill = "white", color = NA),
            legend.position = "none"
          )
      } else {
        # Fallback to standard density plot
        density_plot <- ggplot(all_distances, aes(x = Distance, color = CellType)) +
          geom_density() +
          labs(title = paste("Distance to", ifelse(distance_method == "min", "Nearest Target Cell", "Target Center")), 
               x = "Distance (μm)", 
               y = "Density") +
          theme_minimal() +
          theme(
            panel.background = element_rect(fill = "white"),
            plot.background = element_rect(fill = "white", color = NA)
          )
      }
      
      # 2. Boxplot of distances by ROI
      boxplot <- ggplot(all_distances, aes(x = ROI, y = Distance, fill = CellType)) +
        geom_boxplot() +
        labs(title = paste("Distance to", ifelse(distance_method == "min", "Nearest Target Cell", "Target Center"), "by ROI"), 
             x = "ROI", 
             y = "Distance (μm)") +
        theme_minimal() +
        theme(
          panel.background = element_rect(fill = "white"),
          plot.background = element_rect(fill = "white", color = NA),
          axis.text.x = element_text(angle = 45, hjust = 1)
        )
      
      # Arrange plots
      arranged_plot <- gridExtra::grid.arrange(density_plot, boxplot, ncol = 2)
      return(arranged_plot)
    }
  )
)

# Add the following new function to handle visualization of raw TIFF images and masks

visualize_raw_imc_data <- function(image_idx, images, masks, panel, imc_info, channels = NULL) {
  if (image_idx > length(images)) {
    stop("Image index is out of range")
  }
  
  # Debug prints
  message("Image dimensions:")
  print(dim(images[[image_idx]]))
  message("Range of image values:")
  print(range(images[[image_idx]], na.rm = TRUE))
  
  # Retrieve the image and corresponding mask
  img <- images[[image_idx]]
  msk <- masks[[image_idx]]
  
  # For multi-channel images, create a composite by taking the mean across selected channels
  if (length(dim(img)) == 3) {
    available_channels <- channelNames(images)
    message("Available channels: ", paste(available_channels, collapse = ", "))
    
    # If channels specified, validate and use them
    if (!is.null(channels)) {
      channel_indices <- match(channels, available_channels)
      if (any(is.na(channel_indices))) {
        warning("Some channels not found: ", 
                paste(channels[is.na(channel_indices)], collapse = ", "))
        channel_indices <- channel_indices[!is.na(channel_indices)]
      }
      if (length(channel_indices) == 0) stop("No valid channels specified")
      
      # Debug print to verify channel selection
      message("Using channels: ", paste(available_channels[channel_indices], collapse = ", "))
      message("Channel indices: ", paste(channel_indices, collapse = ", "))
      
      # Extract only the specified channels before averaging
      img <- img[,,channel_indices, drop = FALSE]
      img <- apply(img, c(1, 2), mean)
      
      # Update available_channels to only show selected ones
      available_channels <- available_channels[channel_indices]
    } else {
      message("Creating composite from all ", dim(img)[3], " channels")
      img <- apply(img, c(1, 2), mean)
    }
  }
  
  # Robust normalization using quantiles to handle outliers
  q_low <- quantile(img, 0.01, na.rm = TRUE)
  q_high <- quantile(img, 0.99, na.rm = TRUE)
  img_norm <- (img - q_low) / (q_high - q_low)
  img_norm[img_norm < 0] <- 0
  img_norm[img_norm > 1] <- 1
  
  # Convert to data frame for ggplot
  img_df <- expand.grid(x = 1:ncol(img), y = 1:nrow(img))
  img_df$intensity <- as.vector(img_norm)
  
  # Create mask data frame if mask exists
  # if (!is.null(msk)) {
  #   mask_df <- data.frame(
  #     x = rep(1:ncol(msk), each = nrow(msk)),
  #     y = rep(1:nrow(msk), times = ncol(msk)),
  #     mask = as.vector(msk)
  #   )
  # }
  
  # Get metadata for title
  filename <- names(images)[image_idx]
  
  # Create plot
  p <- ggplot() +
    # Plot image as tiles with viridis color scheme for better visualization
    geom_tile(data = img_df, aes(x = x, y = y, fill = intensity)) +
    scale_fill_viridis_c(
      option = "magma",
      name = "Intensity"
    ) +
    # Add mask contours if mask exists
    # {if (!is.null(msk)) 
    #   geom_contour(data = mask_df, aes(x = x, y = y, z = mask),
    #                breaks = 0.5, colour = "red", size = 0.8)
    # } +
    labs(
      title = sprintf("Image & Mask Overlay: %s", filename),
      subtitle = paste("Using channels:", paste(available_channels, collapse = ", "))
    ) +
    theme_minimal() +
    theme(
      axis.text = element_blank(),
      axis.title = element_blank(),
      panel.grid = element_blank(),
      plot.title = element_text(size = 10),
      plot.subtitle = element_text(size = 8)
    ) +
    coord_fixed(ratio = 1)
  
  print(p)
  return(p)
}