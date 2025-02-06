#' Base class for analysis modules
AnalysisBase <- R6::R6Class("AnalysisBase",
  public = list(
    data = NULL,
    results = NULL,
    config = NULL,
    
    initialize = function(data, config = NULL) {
      self$data <- data
      self$config <- config
      self$results <- list()
    },
    
    run = function(...) {
      stop("Abstract method: implement in subclass")
    },
    
    validate = function() {
      stop("Abstract method: implement in subclass")
    }
  )
)

#' Neighborhood analysis implementation
NeighborhoodAnalysis <- R6::R6Class("NeighborhoodAnalysis",
  inherit = AnalysisBase,
  public = list(
    run = function(k_neighbors = 6, distance_threshold = NULL, ...) {
      self$validate()
      
      # Extract coordinates and create initial neighbor data
      df_coords <- as.data.frame(spatialCoords(self$data$spe))
      
      # Calculate KNN
      if (is.null(distance_threshold)) {
        knn_results <- get.knn(df_coords, k = k_neighbors)
        neighbors <- knn_results$nn.index
        distances <- knn_results$nn.dist
      } else {
        # Use distance-based neighborhood definition
        dist_matrix <- as.matrix(dist(df_coords))
        neighbors <- apply(dist_matrix, 1, function(x) which(x <= distance_threshold))
        distances <- lapply(1:nrow(dist_matrix), function(i) dist_matrix[i, neighbors[[i]]])
      }
      
      # Calculate metrics and create plots
      self$results <- list(
        neighbors = neighbors,
        distances = distances,
        metrics = list(
          avg_distance = colMeans(distances),
          neighbor_density = sapply(neighbors, length)
        ),
        plots = self$create_plots(df_coords, neighbors, distances)
      )
      
      invisible(self)
    },
    
    validate = function() {
      if (is.null(self$data$spe)) {
        stop("No SPE data available for neighborhood analysis")
      }
      if (nrow(spatialCoords(self$data$spe)) == 0) {
        stop("No spatial coordinates found in SPE data")
      }
    }
  ),
  
  private = list(
    create_plots = function(df_coords, neighbors, distances) {
      # Create neighborhood network plot and other visualizations.
      # (Include only the relevant code for neighborhood analysis here.)
      # For example:
      edge_df <- do.call(rbind, lapply(1:nrow(df_coords), function(i) {
          data.frame(
            x1 = df_coords[i, 1],
            y1 = df_coords[i, 2],
            x2 = df_coords[neighbors[i, ], 1],
            y2 = df_coords[neighbors[i, ], 2]
          )
      }))
      
      network_plot <- ggplot() +
        geom_segment(data = edge_df,
                     aes(x = x1, y = y1, xend = x2, yend = y2),
                     alpha = 0.1) +
        geom_point(data = df_coords, aes(x = V1, y = V2), size = 1) +
        labs(title = "Neighborhood Network") +
        theme_minimal()
      
      return(network_plot)
    }
  )
)

#' Temporal analysis implementation
TemporalAnalysis <- R6::R6Class("TemporalAnalysis",
  inherit = AnalysisBase,
  public = list(
    run = function(...) {
      self$validate()
      
      # Extract timepoint information from sample IDs
      timepoints <- data.frame(
        sample_id = colData(self$data$spe)$sample_id,
        day = factor(gsub(".*D([1-7])_.*", "\\1", colData(self$data$spe)$sample_id)),
        mouse = gsub(".*_M([1-2])_.*", "\\1", colData(self$data$spe)$sample_id)
      )
      
      # Define key markers for cell types
      markers <- list(
        immune = "CD45",
        myeloid = "CD11b",
        endothelial = "CD31"
      )
      
      # Calculate cell populations over time
      cell_counts <- lapply(markers, function(marker) {
        if (marker %in% rownames(self$data$spe)) {
          table(timepoints$day, 
                assay(self$data$spe)[marker,] > median(assay(self$data$spe)[marker,]))
        } else {
          message(sprintf("Marker %s not found in data", marker))
          NULL
        }
      })
      
      # Calculate spatial distributions
      spatial_stats <- lapply(split(seq_len(ncol(self$data$spe)), timepoints$day), 
        function(idx) {
          coords <- spatialCoords(self$data$spe)[idx,]
          cells <- assay(self$data$spe)[,idx]
          
          # Calculate nearest neighbor distances for each cell type
          nn_dist <- lapply(markers, function(marker) {
            if (marker %in% rownames(cells)) {
              cells_of_type <- cells[marker,] > median(cells[marker,])
              if (sum(cells_of_type) > 1) {
                nndist(coords[cells_of_type,])
              } else {
                NULL
              }
            } else {
              NULL
            }
          })
          
          # Return summary statistics
          lapply(nn_dist[!sapply(nn_dist, is.null)], summary)
      })
      
      # Store results
      self$results <- list(
        timepoints = timepoints,
        cell_counts = cell_counts,
        spatial_stats = spatial_stats,
        spatial_df = do.call(rbind, lapply(names(spatial_stats), function(day) {
          stats <- spatial_stats[[day]]
          data.frame(
            day = day,
            cell_type = rep(names(stats), sapply(stats, length)),
            distance = unlist(stats)
          )
        }))
      )
      
      invisible(self)
    },
    
    validate = function() {
      if (is.null(self$data$spe)) {
        stop("No SPE data available for temporal analysis")
      }
      if (!("sample_id" %in% colnames(colData(self$data$spe)))) {
        stop("No sample_id column found in SPE data")
      }
    }
  )
)

#' Intensity analysis implementation
IntensityAnalysis <- R6::R6Class("IntensityAnalysis",
  inherit = AnalysisBase,
  public = list(
    run = function(max_points = 5000, distance_threshold = 50, ...) {
      self$validate()
      
      # Extract image data
      img_data <- imageData(self$data$images)[[1]]  # Analyze first image by default
      channels <- channelNames(self$data$images)
      
      # Initialize results matrices
      n_channels <- length(channels)
      spatial_correlation <- matrix(NA, 
        nrow = n_channels, 
        ncol = 3,
        dimnames = list(channels, c("Moran_I", "Geary_C", "Distance_Correlation"))
      )
      
      # Process each channel
      hotspot_analysis <- list()
      for (chan_idx in seq_len(n_channels)) {
        channel_name <- channels[chan_idx]
        intensity_values <- img_data[,,chan_idx]
        
        # Calculate spatial statistics
        if (!is.null(self$data$masks)) {
          # Use cell-level aggregation with masks
          mask_img <- as.matrix(self$data$masks[[1]])
          cell_intensities <- tapply(as.vector(intensity_values),
                                   as.vector(mask_img),
                                   mean,
                                   na.rm = TRUE)
          
          # Remove background (cell ID 0)
          cell_intensities <- cell_intensities[names(cell_intensities) != "0"]
          cell_coords <- spatialCoords(self$data$spe)
          
          # Calculate spatial statistics
          weights_list <- spdep::nb2listw(
            spdep::dnearneigh(cell_coords, 0, distance_threshold)
          )
          
          moran <- spdep::moran.test(cell_intensities, weights_list)
          geary <- spdep::geary.test(cell_intensities, weights_list)
          
          # Calculate local statistics
          local_moran <- spdep::localmoran(cell_intensities, weights_list)
          
          hotspot_analysis[[channel_name]] <- data.frame(
            coordinates = I(cell_coords),
            intensity = cell_intensities,
            local_moran_i = local_moran[,1],
            p_value = local_moran[,5]
          )
        } else {
          # Pixel-level analysis with sampling
          valid_idx <- which(!is.na(intensity_values))
          if (length(valid_idx) > max_points) {
            valid_idx <- sample(valid_idx, max_points)
          }
          
          pixel_coords <- expand.grid(
            x = 1:dim(intensity_values)[1],
            y = 1:dim(intensity_values)[2]
          )[valid_idx,]
          
          sampled_intensities <- intensity_values[valid_idx]
          
          # Calculate spatial statistics
          dist_matrix <- as.matrix(dist(pixel_coords))
          weights <- 1 / (dist_matrix + 1)
          diag(weights) <- 0
          
          moran <- list(statistic = ape::Moran.I(sampled_intensities, weights)$observed)
          geary <- list(statistic = ape::Geary.C(sampled_intensities, weights)$observed)
        }
        
        # Store results
        spatial_correlation[chan_idx,] <- c(
          moran$statistic,
          geary$statistic,
          energy::dcor(intensity_values, as.vector(dist_matrix))
        )
      }
      
      # Store all results
      self$results <- list(
        spatial_correlation = spatial_correlation,
        hotspot_analysis = hotspot_analysis,
        correlation_heatmap = self$create_correlation_heatmap(spatial_correlation)
      )
      
      invisible(self)
    },
    
    validate = function() {
      if (is.null(self$data$images)) {
        stop("No image data available for intensity analysis")
      }
      if (length(self$data$images) == 0) {
        stop("Empty image data")
      }
    }
  ),
  
  private = list(
    create_correlation_heatmap = function(correlation_matrix) {
      melted_correlations <- reshape2::melt(correlation_matrix)
      melted_correlations$value <- as.numeric(melted_correlations$value)
      
      ggplot(melted_correlations[!is.na(melted_correlations$value),],
             aes(x = Var2, y = Var1, fill = value)) +
        geom_tile() +
        scale_fill_gradient2(
          low = "blue",
          high = "red",
          mid = "white",
          midpoint = 0,
          na.value = "grey50"
        ) +
        labs(
          title = "Spatial-Intensity Correlation Metrics",
          x = "Metric",
          y = "Channel"
        ) +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
    }
  )
) 