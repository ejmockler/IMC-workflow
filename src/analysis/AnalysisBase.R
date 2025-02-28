#' Base class for analysis modules
AnalysisBase <- R6::R6Class("AnalysisBase",
  public = list(
    data = NULL,
    results = NULL,
    config = NULL,
    logger = NULL,
    
    initialize = function(data = NULL, config = NULL, logger = NULL) {
      self$data <- data
      self$config <- config
      self$logger <- logger
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
      
      # Calculate KNN or distance-based neighbors
      if (is.null(distance_threshold)) {
        # KNN-based neighborhood
        knn_results <- get.knn(df_coords, k = k_neighbors)
        neighbors <- knn_results$nn.index
        distances <- knn_results$nn.dist
        avg_distance <- rowMeans(distances) # Corrected for KNN
      } else {
        # Distance-based neighborhood using dbscan::frNN
        frnn_results <- dbscan::frNN(df_coords, eps = distance_threshold)
        neighbors <- frnn_results$id
        distances <- frnn_results$dist
        avg_distance <- unlist(lapply(distances, function(d) { # Corrected for distance threshold
          if(length(d) > 0) {
            mean(d)
          } else {
            NA # or 0, depending on how you want to handle no neighbors
          }
        }))
      }
      
      # Calculate metrics and create plots
      self$results <- list(
        neighbors = neighbors,
        distances = distances,
        metrics = list(
          avg_distance = avg_distance, # Using the corrected avg_distance calculation
          neighbor_density = sapply(neighbors, length)
        ),
        plots = private$create_plots(df_coords, neighbors, distances)
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
      edge_df_list <- lapply(1:nrow(df_coords), function(i) {
          neighbor_indices <- if(is.matrix(neighbors)) {
              neighbors[i, ] # Matrix indexing for KNN
          } else {
              neighbors[[i]] # List indexing for distance threshold
          }

          if(length(neighbor_indices) > 0) { # Check if there are neighbors
              data.frame(
                x1 = df_coords[i, 1],
                y1 = df_coords[i, 2],
                x2 = df_coords[neighbor_indices, 1],
                y2 = df_coords[neighbor_indices, 2]
              )
          } else {
              NULL # Return NULL if no neighbors
          }
      })

      # Remove NULL entries from the list before rbind
      edge_df <- do.call(rbind, Filter(Negate(is.null), edge_df_list))


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
          valid_coords <- which(!is.na(intensity_values), arr.ind = TRUE)
          sampled_intensities <- intensity_values[!is.na(intensity_values)]
          # No downsampling here – we work with all valid pixels

          # Use the FNN package to compute only the k-nearest neighbors for each pixel
          library(FNN)
          k_value <- 10  # choose a fixed small number for neighbors
          knn_results <- FNN::get.knn(valid_coords, k = k_value)
          neighbors <- knn_results$nn.index   # sparse neighbor indices
          distances <- knn_results$nn.dist    # corresponding distances

          epsilon <- 1e-6  # small constant to avoid division by zero
          # Calculate a list of weights based on neighbor distances (sparse approach)
          weights_list <- lapply(1:nrow(valid_coords), function(i) {
            1 / (distances[i, ] + epsilon)
          })

          # Now, compute Moran's I and Geary's C using only the sparse neighbor list.
          n <- nrow(valid_coords)
          S0 <- sum(sapply(weights_list, sum))
          mean_intensity <- mean(sampled_intensities)
          denom <- sum((sampled_intensities - mean_intensity)^2)
          num_moran <- 0
          num_geary <- 0

          # Loop over each valid pixel and its neighbors:
          for (i in 1:n) {
            if (length(neighbors[i, ]) > 0) {  # Check if there are any neighbors
              for (j in seq_along(neighbors[i, ])) {
                j_index <- neighbors[i, j]
                w_ij <- weights_list[[i]][j]
                num_moran <- num_moran + w_ij * (sampled_intensities[i] - mean_intensity) * (sampled_intensities[j_index] - mean_intensity)
                num_geary <- num_geary + w_ij * (sampled_intensities[i] - sampled_intensities[j_index])^2
              }
            }
          }
          moran_I <- (n / S0) * (num_moran / denom)
          geary_C <- ((n - 1) / (2 * S0)) * (num_geary / denom)

          # The computed 'moran_I' and 'geary_C' can now be stored as part of the results.
        }
        
        # Store results
        spatial_correlation[chan_idx,] <- c(
          moran_I,
          geary_C,
          energy::dcor(intensity_values, as.vector(distances))
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