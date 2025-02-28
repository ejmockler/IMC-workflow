#' Temporal analysis implementation
TemporalAnalysis <- R6::R6Class("TemporalAnalysis",
  inherit = AnalysisBase,
  public = list(
    initialize = function(data, config = NULL, logger = NULL) {
      super$initialize(data = data, config = config, logger = logger)
    },

    run = function(...) {
      self$validate()
      
      # Extract timepoint information from sample IDs
      # Expected sample_id format: ...D{day}_..._M{mouse}_...
      tp <- colData(self$data$spe)$sample_id
      timepoints <- data.frame(
        sample_id = tp,
        day = as.numeric(gsub(".*D([1-7])_.*", "\\1", tp)),
        mouse = as.numeric(gsub(".*_M([1-2])_.*", "\\1", tp)),
        stringsAsFactors = FALSE
      )
      
      # Log the detected unique timepoints using logger instead of message
      unique_days <- sort(unique(timepoints$day))
      unique_mice <- sort(unique(timepoints$mouse))
      if (!is.null(self$logger)) {
        self$logger$log_info(sprintf("Detected timepoints (days): %s", 
          paste(unique_days, collapse = ", ")))
        self$logger$log_info(sprintf("Detected replicates/mice: %s", 
          paste(unique_mice, collapse = ", ")))
      }
      
      # Calculate overall cell counts and proportions (each SPE column represents one cell)
      total_counts <- table(timepoints$day, timepoints$mouse)
      replicate_props <- prop.table(total_counts, margin = 2)  # per replicate (each column sums to 1)
      avg_overall_props <- rowMeans(replicate_props)
      
      # Retrieve key markers for cell types from configuration.
      # If not specified in self$config, fall back to defaults.
      # NOTE: The defaults here are placeholders.
      markers <- if (!is.null(self$config$markers)) {
                   self$config$markers
                 } else {
                   list(
                     macrophages = c("CD45", "CD11b", "CD206"),
                     neutrophils  = c("CD45", "Ly6G"),
                     endothelial  = c("CD31"),
                     fibroblasts  = c("CD140b")
                   )
                 }
      
      # Calculate marker-specific positive cell proportions
      # A cell is considered positive if its expression exceeds the median for that marker.
      marker_positive_proportions <- lapply(markers, function(marker_vec) {
        # marker_vec is assumed to be a vector of markers for the given cell type.
        valid_markers <- marker_vec[marker_vec %in% rownames(self$data$spe)]
        if (length(valid_markers) == 0) {
          if (!is.null(self$logger)) {
            self$logger$log_warning(sprintf("None of the markers %s found in SPE data",
                            paste(marker_vec, collapse = ", ")))
          }
          return(NULL)
        }
        # Compute positivity for each valid marker
        pos_matrix <- sapply(valid_markers, function(m) {
          expr_values <- assay(self$data$spe)[m, ]
          expr_values > median(expr_values, na.rm = TRUE)
        })
        # If only one marker, sapply returns a vector.
        if (is.null(dim(pos_matrix))) {
          pos <- pos_matrix
        } else {
          # Use logical AND so that a cell is positive only if all markers are positive.
          pos <- apply(pos_matrix, 1, all)
        }
        df <- data.frame(
          day = timepoints$day,
          mouse = timepoints$mouse,
          positive = pos,
          stringsAsFactors = FALSE
        )
        by_replicate <- aggregate(positive ~ day + mouse, data = df, FUN = mean)
        avg <- aggregate(positive ~ day, data = by_replicate, FUN = mean)
        list(by_replicate = by_replicate, avg = avg)
      })
      
      # Calculate spatial distributions.
      # For each day, we compute nearest neighbor distances for the cells in each marker that are positive.
      spatial_stats <- lapply(split(seq_len(ncol(self$data$spe)), timepoints$day), function(idx) {
        coords <- spatialCoords(self$data$spe)[idx, , drop = FALSE]
        cells <- assay(self$data$spe)[, idx, drop = FALSE]
        
        nn_dist <- lapply(markers, function(marker) {
          if (marker %in% rownames(cells)) {
            expr_values <- cells[marker, ]
            pos <- expr_values > median(expr_values, na.rm = TRUE)
            if (sum(pos) > 1) {
              nndist(coords[pos, , drop = FALSE])
            } else {
              NULL
            }
          } else {
            NULL
          }
        })
        
        # Return summary statistics for non-null nearest neighbor distances
        lapply(nn_dist[!sapply(nn_dist, is.null)], summary)
      })
      
      spatial_df <- do.call(rbind, lapply(names(spatial_stats), function(day) {
        stats <- spatial_stats[[day]]
        data.frame(
          day = day,
          cell_type = rep(names(stats), sapply(stats, length)),
          distance = unlist(stats),
          stringsAsFactors = FALSE
        )
      }))
      
      # Store results
      self$results <- list(
        timepoints = timepoints,
        overall_cell_counts = total_counts,
        overall_cell_proportions = list(by_replicate = replicate_props, avg = avg_overall_props),
        marker_positive_proportions = marker_positive_proportions,
        spatial_stats = spatial_stats,
        spatial_df = spatial_df
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