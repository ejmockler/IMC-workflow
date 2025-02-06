#' Intensity analysis implementation
#' @description Handles intensity analysis while reconciling differences between
#' cell intensity measurements and spatial coordinate data.
IntensityAnalysis <- R6::R6Class("IntensityAnalysis",
  inherit = AnalysisBase,
  
  public = list(
    run = function(max_points = 5000, distance_threshold = NULL, ...) {
      self$validate()
      
      # Extract spatial experiment object and compute components
      spe <- self$data$spe
      coords <- spatialCoords(spe)
      intensities <- tryCatch(assay(spe, "intensities"), 
                              error = function(e) NULL)
      
      if (is.null(intensities)) {
        stop("Intensity assay is missing from the spe object.")
      }
      
      # Ensure that spatial coordinates and intensities match by cell IDs.
      if (nrow(coords) != nrow(intensities)) {
        common_ids <- intersect(rownames(coords), rownames(intensities))
        if (length(common_ids) == 0) {
          stop("No common cell IDs were found between spatial coordinates and intensities.")
        }
        warning(sprintf("Mismatch detected: %d coordinates vs %d intensities. Subsetting to %d common cell IDs.", 
                        nrow(coords), nrow(intensities), length(common_ids)))
        # Subset the SPE object to only include common cell IDs.
        spe <- spe[common_ids, ]
        self$data$spe <- spe  # update the data object
        coords <- spatialCoords(spe)
        intensities <- assay(spe, "intensities")
      }
      
      # Optionally, limit analysis to a maximum number of cells for speed.
      if (nrow(coords) > max_points) {
        sample_idx <- sample(seq_len(nrow(coords)), max_points)
        coords <- coords[sample_idx, , drop = FALSE]
        intensities <- intensities[sample_idx, , drop = FALSE]
        spe <- spe[sample_idx, ]
        self$data$spe <- spe
      }
      
      n_cells <- nrow(coords)
      
      epsilon <- 1e-6  # to avoid division by zero
      
      #--------------------------------------------------------------
      # NEW: Set default distance_threshold if not provided
      # This ensures we use a memory-efficient sparse neighbor computation.
      # We compute a default based on the median nearest neighbor distance.
      #--------------------------------------------------------------
      if (is.null(distance_threshold)) {
        require(FNN)
        # Compute the nearest neighbor distances (FNN excludes the point itself).
        knn_obj <- FNN::get.knn(coords, k = 1)
        default_threshold <- median(knn_obj$nn.dist) * 1.5
        message(sprintf("No distance_threshold provided; defaulting to %f based on median nearest neighbor distance.", default_threshold))
        distance_threshold <- default_threshold
      }
      
      #--------------------------------------------------------------
      # Build a sparse neighbor structure since distance_threshold is now provided.
      #--------------------------------------------------------------
      if (!is.null(distance_threshold)) {
        require(spdep)
        # Build neighbor list for cells within the given threshold.
        nb <- dnearneigh(as.matrix(coords), 0, distance_threshold)
        # Compute distances for each neighbor pair.
        dist_list <- nbdists(nb, as.matrix(coords))
        # Compute weights: inverse of distance plus epsilon.
        w_list <- lapply(dist_list, function(d) 1 / (d + epsilon))
        # Total sum of weights S0.
        S0 <- sum(unlist(w_list))
      } else {
        # Old full matrix approach (may be memory heavy)
        dist_mat <- as.matrix(dist(coords))
        w <- 1 / (dist_mat + epsilon)
        diag(w) <- 0
      }
      
      # Initialize the result matrix.
      channels <- colnames(intensities)
      if (is.null(channels)) { 
        channels <- paste0("Channel_", seq_len(ncol(intensities))) 
      }
      result_matrix <- matrix(NA, nrow = length(channels), ncol = 4)
      rownames(result_matrix) <- channels
      colnames(result_matrix) <- c("Moran_I", "Geary_C", "Distance_Correlation", "Hotspot_Ratio")
      
      # Iterate over each channel to compute the spatial statistics.
      for (j in seq_along(channels)) {
        x <- intensities[, j]
        x_bar <- mean(x)
        denom <- sum((x - x_bar)^2)
        
        if (!is.null(distance_threshold)) {
          # Compute Moran's I using the neighbor list.
          num_moran <- 0
          num_geary <- 0
          # Loop over each observation and its neighbors.
          for (i in seq_along(nb)) {
            if (length(nb[[i]]) > 0) {
              # Get the weights for neighbors of i.
              w_vec <- w_list[[i]]
              for (k in seq_along(nb[[i]])) {
                j_idx <- nb[[i]][k]
                num_moran <- num_moran + w_vec[k] * (x[i] - x_bar) * (x[j_idx] - x_bar)
                num_geary <- num_geary + w_vec[k] * (x[i] - x[j_idx])^2
              }
            }
          }
          moran_i <- (n_cells / S0) * (num_moran / denom)
          geary_c <- ((n_cells - 1) * num_geary) / (2 * S0 * denom)
        } else {
          # Full matrix approach (existing code)
          moran_obj <- ape::Moran.I(x, w)
          moran_i <- moran_obj$estimate
          # Helper function for Geary's C using the full matrix.
          calculate_geary_C <- function(x, w, n) {
            num <- sum(w * (outer(x, x, "-")^2))
            C <- ((n - 1) * num) / (2 * sum(w) * sum((x - mean(x))^2))
            return(C)
          }
          geary_c <- calculate_geary_C(x, w, n_cells)
        }
        
        # Compute the distance correlation between intensity values and spatial coordinates.
        dcor_val <- energy::dcor(x, coords)
        
        # Determine the hotspot ratio (cells with intensity above the 90th percentile).
        threshold <- quantile(x, 0.9, na.rm = TRUE)
        hotspot_ratio <- sum(x > threshold, na.rm = TRUE) / length(x)
        
        result_matrix[j, ] <- c(moran_i, geary_c, dcor_val, hotspot_ratio)
      }
      
      # -----------------------------
      # Append cell-level metadata for grouping (ROI, Time, Replicate)
      # -----------------------------
      cell_metadata <- as.data.frame(SummarizedExperiment::colData(spe))
      
      # Store the computed metrics along with metadata.
      self$results <- list(
        intensity_metrics = result_matrix,
        n_cells_used = n_cells,
        metadata = cell_metadata  # Added metadata for grouping in visualizations.
      )
    },
    
    validate = function() {
      if (is.null(self$data$spe)) {
        stop("No SPE object provided for intensity analysis.")
      }
      if (!("intensities" %in% assayNames(self$data$spe))) {
        stop("The SPE object does not contain an 'intensities' assay.")
      }
    }
  )
) 