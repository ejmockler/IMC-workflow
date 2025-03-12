#' Spatial Context and Patch Analysis
#' 
#' @description Analyzes spatial contexts and patches in spatial data,
#' including context detection, patch detection, and distance calculations.
#'
#' @details Extends the InteractionBase class with methods for
#' spatial context detection and patch-related analyses.

library(R6)
library(SpatialExperiment)

source("src/analysis/spatial/interaction/InteractionBase.R")

SpatialContextAnalysis <- R6::R6Class("SpatialContextAnalysis",
  inherit = InteractionBase,
  
  public = list(
    #' @description Create a new SpatialContextAnalysis object
    #' @param spe SpatialExperiment object with spatial graphs
    #' @param logger Logger object
    #' @param n_cores Number of cores for parallel processing
    #' @param dependency_manager DependencyManager object
    initialize = function(spe, logger = NULL, n_cores = 1, dependency_manager = NULL) {
      super$initialize(spe, logger, n_cores, dependency_manager)
      
      # Load spatial context-specific dependencies
      private$loadSpatialContextDependencies()
      
      invisible(self)
    },
    
    #' @description Build a spatial context graph
    #' @param k Number of neighbors
    #' @param name Name for the graph
    #' @param img_id Column containing image identifiers
    #' @return Updated SpatialExperiment object
    buildSpatialContextGraph = function(
      k = 40, 
      name = "knn_spatialcontext_graph",
      img_id = "sample_id"
    ) {
      if (!is.null(self$logger)) self$logger$log_info("Building spatial context graph with k = %d", k)
      
      # Make sure imcRtools is available
      if (!requireNamespace("imcRtools", quietly = TRUE)) {
        if (!is.null(self$logger)) {
          self$logger$log_error("imcRtools package is required for building spatial graphs")
        }
        return(self$spe)
      }
      
      self$spe <- imcRtools::buildSpatialGraph(
        self$spe, 
        img_id = img_id, 
        type = "knn", 
        name = name, 
        k = k
      )
      
      return(self$spe)
    },
    
    #' @description Aggregate neighborhoods based on cellular neighborhoods
    #' @param colPairName Name of the spatial graph
    #' @param name Name for the aggregated neighborhood data
    #' @param count_by Column to count by for aggregation
    #' @return Updated SpatialExperiment object
    aggregateCNNeighbors = function(
      colPairName = "knn_spatialcontext_graph", 
      name = "aggregatedNeighborhood",
      count_by = "cn_celltypes"
    ) {
      if (!is.null(self$logger)) self$logger$log_info("Aggregating neighborhoods based on cellular neighborhoods")
      
      # Check if count_by column exists
      if (!count_by %in% colnames(SummarizedExperiment::colData(self$spe))) {
        if (!is.null(self$logger)) {
          self$logger$log_error("Column '%s' not found. Run aggregateCelltypeNeighbors first.", count_by)
        }
        return(self$spe)
      }
      
      # Make sure imcRtools is available
      if (!requireNamespace("imcRtools", quietly = TRUE)) {
        if (!is.null(self$logger)) {
          self$logger$log_error("imcRtools package is required for neighborhood aggregation")
        }
        return(self$spe)
      }
      
      self$spe <- imcRtools::aggregateNeighbors(
        self$spe, 
        colPairName = colPairName,
        aggregate_by = "metadata",
        count_by = count_by,
        name = name
      )
      
      return(self$spe)
    },
    
    #' @description Detect spatial contexts
    #' @param entry Column containing aggregated neighborhood data
    #' @param threshold Threshold for determining dominant CNs
    #' @param name Name for the spatial context column
    #' @return Updated SpatialExperiment object
    detectSpatialContext = function(
      entry = "aggregatedNeighborhood", 
      threshold = 0.9, 
      name = "spatial_context"
    ) {
      if (!is.null(self$logger)) self$logger$log_info("Detecting spatial contexts with threshold = %.2f", threshold)
      
      # Make sure entry column exists
      if (!entry %in% names(S4Vectors::metadata(self$spe))) {
        if (!is.null(self$logger)) {
          self$logger$log_error("Entry '%s' not found in metadata. Run aggregateCNNeighbors first.", entry)
        }
        return(self$spe)
      }
      
      # Make sure imcRtools is available
      if (!requireNamespace("imcRtools", quietly = TRUE)) {
        if (!is.null(self$logger)) {
          self$logger$log_error("imcRtools package is required for spatial context detection")
        }
        return(self$spe)
      }
      
      self$spe <- imcRtools::detectSpatialContext(
        self$spe, 
        entry = entry,
        threshold = threshold,
        name = name
      )
      
      return(self$spe)
    },
    
    #' @description Filter spatial contexts
    #' @param entry Column containing spatial context data
    #' @param group_by Column in colData to group by for filtering
    #' @param group_threshold Minimum number of groups that must have the spatial context
    #' @param cells_threshold Minimum number of cells per spatial context
    #' @return The SpatialExperiment object with filtered spatial contexts
    filterSpatialContext = function(
      entry = "spatial_context",
      group_by = "sample_id",
      group_threshold = 3,
      cells_threshold = 100
    ) {
      if (!is.null(self$logger)) {
        self$logger$log_info("Filtering spatial contexts (group threshold = %d, cells threshold = %d)", 
                          group_threshold, cells_threshold)
      }
      
      # Validate input parameters
      if (!group_by %in% colnames(SummarizedExperiment::colData(self$spe))) {
        if (!is.null(self$logger)) {
          self$logger$log_error("Column '%s' not found in colData", group_by)
        }
        return(self$spe)
      }
      
      # Make sure entry column exists
      if (!entry %in% colnames(SummarizedExperiment::colData(self$spe))) {
        if (!is.null(self$logger)) {
          self$logger$log_error("Entry '%s' not found in colData. Run detectSpatialContext first.", entry)
        }
        return(self$spe)
      }
      
      # Make sure imcRtools is available
      if (!requireNamespace("imcRtools", quietly = TRUE)) {
        if (!is.null(self$logger)) {
          self$logger$log_error("imcRtools package is required for filtering spatial contexts")
        }
        return(self$spe)
      }
      
      # Filter by number of group entries
      self$spe <- imcRtools::filterSpatialContext(
        self$spe, 
        entry = entry,
        group_by = group_by, 
        group_threshold = group_threshold
      )
      
      # Filter by number of cells if specified
      if (!is.null(cells_threshold)) {
        filtered_entry <- paste0(entry, "_filtered")
        self$spe <- imcRtools::filterSpatialContext(
          self$spe, 
          entry = filtered_entry,
          group_by = group_by, 
          cells_threshold = cells_threshold
        )
      }
      
      return(self$spe)
    },
    
    #' @description Detect patches of cells
    #' @param patch_cells Boolean vector indicating cells to include in patches
    #' @param expand_by Distance to expand patches
    #' @param min_patch_size Minimum number of cells per patch
    #' @param colPairName Name of the graph to use
    #' @param img_id Column containing image identifiers
    #' @return Updated SpatialExperiment object
    detectPatches = function(
      patch_cells, 
      expand_by = 1, 
      min_patch_size = 10, 
      colPairName = "neighborhood",
      img_id = "sample_id"
    ) {
      if (!is.null(self$logger)) {
        self$logger$log_info("Detecting patches (expand by = %d, min size = %d)", 
                           expand_by, min_patch_size)
      }
      
      # Check if any patch cells exist
      if (sum(patch_cells) < min_patch_size) {
        if (!is.null(self$logger)) {
          self$logger$log_warning("Not enough patch cells (%d) to meet minimum size requirement (%d)", 
                               sum(patch_cells), min_patch_size)
        }
        
        # Create dummy patch column to avoid downstream errors
        self$spe$patch_id <- NA
        self$spe$in_patch <- FALSE
        
        return(self$spe)
      }
      
      # Make sure imcRtools is available
      if (!requireNamespace("imcRtools", quietly = TRUE)) {
        if (!is.null(self$logger)) {
          self$logger$log_error("imcRtools package is required for patch detection")
        }
        # Create dummy patch column to avoid downstream errors
        self$spe$patch_id <- NA
        self$spe$in_patch <- FALSE
        return(self$spe)
      }
      
      # Try to detect patches
      tryCatch({
        self$spe <- imcRtools::patchDetection(
          self$spe, 
          patch_cells = patch_cells,
          img_id = img_id,
          expand_by = expand_by,
          min_patch_size = min_patch_size,
          colPairName = colPairName
        )
      }, error = function(e) {
        # If no connected components found, create dummy patch columns
        if (grepl("No connected components found", e$message)) {
          if (!is.null(self$logger)) {
            self$logger$log_warning("No connected components found for patch detection: %s", e$message)
          }
          
          # Create dummy patch column to avoid downstream errors
          self$spe$patch_id <- NA
          self$spe$in_patch <- FALSE
        } else {
          # For other errors, propagate them
          if (!is.null(self$logger)) {
            self$logger$log_error("Error in patch detection: %s", e$message)
          }
          # Create dummy patch column to avoid downstream errors
          self$spe$patch_id <- NA
          self$spe$in_patch <- FALSE
        }
      })
      
      # Check if patch_id exists and summarize patch information
      if ("patch_id" %in% colnames(SummarizedExperiment::colData(self$spe))) {
        valid_patches <- !is.na(self$spe$patch_id)
        patch_count <- length(unique(self$spe$patch_id[valid_patches]))
        
        if (!is.null(self$logger)) {
          if (patch_count > 0) {
            self$logger$log_info("Detected %d patches containing %d cells", 
                               patch_count, sum(valid_patches))
          } else {
            self$logger$log_warning("No valid patches detected")
          }
        }
      }
      
      return(self$spe)
    },
    
    #' @description Calculate minimum distance to patches
    #' @param img_id Column containing image identifiers
    #' @return Updated SpatialExperiment object
    calculateDistToPatches = function(img_id = "sample_id") {
      if (!is.null(self$logger)) self$logger$log_info("Calculating minimum distance to patches")
      
      # Check if patch_id exists
      if (!"patch_id" %in% colnames(SummarizedExperiment::colData(self$spe))) {
        if (!is.null(self$logger)) {
          self$logger$log_warning("Column 'patch_id' not found. Cannot calculate distances.")
        }
        
        # Create a dummy distance column to avoid downstream errors
        self$spe$dist_to_patch <- NA
        
        return(self$spe)
      }
      
      # Check if there are any valid patches
      valid_patches <- !is.na(self$spe$patch_id)
      if (sum(valid_patches) == 0) {
        if (!is.null(self$logger)) {
          self$logger$log_warning("No valid patches found. Cannot calculate distances.")
        }
        
        # Create a dummy distance column to avoid downstream errors
        self$spe$dist_to_patch <- NA
        
        return(self$spe)
      }
      
      # Make sure imcRtools is available
      if (!requireNamespace("imcRtools", quietly = TRUE)) {
        if (!is.null(self$logger)) {
          self$logger$log_error("imcRtools package is required for distance calculation")
        }
        # Create a dummy distance column to avoid downstream errors
        self$spe$dist_to_patch <- NA
        return(self$spe)
      }
      
      # Calculate minimum distance to patches
      tryCatch({
        self$spe <- imcRtools::minDistToCells(
          self$spe, 
          x_cells = !is.na(self$spe$patch_id), 
          img_id = img_id
        )
        
        if (!is.null(self$logger)) self$logger$log_info("Successfully calculated distances to patches")
      }, error = function(e) {
        if (!is.null(self$logger)) {
          self$logger$log_error("Error calculating distances to patches: %s", e$message)
        }
        
        # Create a dummy distance column to avoid downstream errors
        self$spe$dist_to_patch <- NA
      })
      
      return(self$spe)
    },
    
    #' @description Calculate patch sizes
    #' @param patch_id_col Column containing patch IDs
    #' @return DataFrame with patch sizes
    calculatePatchSizes = function(patch_id_col = "patch_id") {
      if (!is.null(self$logger)) self$logger$log_info("Calculating patch sizes")
      
      # Check if patch_id exists
      if (!patch_id_col %in% colnames(SummarizedExperiment::colData(self$spe))) {
        if (!is.null(self$logger)) {
          self$logger$log_warning("Column '%s' not found. Cannot calculate patch sizes.", patch_id_col)
        }
        return(data.frame())
      }
      
      # Make sure imcRtools is available
      if (!requireNamespace("imcRtools", quietly = TRUE)) {
        if (!is.null(self$logger)) {
          self$logger$log_error("imcRtools package is required for calculating patch sizes")
        }
        return(data.frame())
      }
      
      # Calculate patch sizes
      patch_size <- imcRtools::patchSize(self$spe, patch_id_col)
      
      # Merge with metadata if there are any patches
      if (nrow(patch_size) > 0) {
        patch_size <- merge(
          patch_size, 
          SummarizedExperiment::colData(self$spe)[match(patch_size$patch_id, self$spe[[patch_id_col]]),], 
          by = "patch_id"
        )
      }
      
      return(patch_size)
    },
    
    #' @description Run a complete spatial context analysis workflow
    #' @param k Number of neighbors for graph
    #' @param context_threshold Threshold for determining dominant CNs
    #' @param group_by Column to group by for filtering
    #' @param group_threshold Minimum number of groups for filtering
    #' @param cells_threshold Minimum number of cells for filtering
    #' @param count_by Column to count by for aggregation
    #' @param img_id Column containing image identifiers
    #' @return Updated SpatialExperiment object
    runSpatialContextWorkflow = function(
      k = 40,
      context_threshold = 0.9,
      group_by = "sample_id",
      group_threshold = 3,
      cells_threshold = 100,
      count_by = "cn_celltypes",
      img_id = "sample_id"
    ) {
      if (!is.null(self$logger)) self$logger$log_info("Running complete spatial context workflow")
      
      # Step 1: Build spatial context graph
      self$spe <- self$buildSpatialContextGraph(
        k = k,
        img_id = img_id
      )
      
      # Step 2: Aggregate cellular neighborhoods
      self$spe <- self$aggregateCNNeighbors(
        count_by = count_by
      )
      
      # Step 3: Detect spatial contexts
      self$spe <- self$detectSpatialContext(
        threshold = context_threshold
      )
      
      # Step 4: Filter spatial contexts
      self$spe <- self$filterSpatialContext(
        group_by = group_by,
        group_threshold = group_threshold,
        cells_threshold = cells_threshold
      )
      
      if (!is.null(self$logger)) self$logger$log_info("Spatial context workflow completed")
      
      return(self$spe)
    },
    
    #' @description Run a complete patch analysis workflow
    #' @param patch_cells Boolean vector indicating cells to include in patches
    #' @param expand_by Distance to expand patches
    #' @param min_patch_size Minimum number of cells per patch
    #' @param colPairName Name of the graph to use
    #' @param img_id Column containing image identifiers
    #' @return Updated SpatialExperiment object with patch information
    runPatchAnalysisWorkflow = function(
      patch_cells,
      expand_by = 1,
      min_patch_size = 10,
      colPairName = "neighborhood",
      img_id = "sample_id"
    ) {
      if (!is.null(self$logger)) self$logger$log_info("Running complete patch analysis workflow")
      
      # Step 1: Detect patches
      self$spe <- self$detectPatches(
        patch_cells = patch_cells,
        expand_by = expand_by,
        min_patch_size = min_patch_size,
        colPairName = colPairName,
        img_id = img_id
      )
      
      # Step 2: Calculate distance to patches
      self$spe <- self$calculateDistToPatches(
        img_id = img_id
      )
      
      # Step 3: Calculate patch sizes
      patch_sizes <- self$calculatePatchSizes()
      
      # Store patch sizes in metadata
      if (nrow(patch_sizes) > 0) {
        S4Vectors::metadata(self$spe)[["patch_sizes"]] <- patch_sizes
        
        if (!is.null(self$logger)) {
          self$logger$log_info("Stored patch size information for %d patches", nrow(patch_sizes))
        }
      }
      
      if (!is.null(self$logger)) self$logger$log_info("Patch analysis workflow completed")
      
      return(self$spe)
    },
    
    #' @description Plot spatial context graph
    #' @param entry Column containing spatial context assignments
    #' @param group_by Column to group by
    #' @param result_path Path to store visualization results (or NULL)
    #' @return List of ggplot objects
    plotSpatialContextGraph = function(
      entry = "spatial_context_filtered", 
      group_by = "sample_id",
      result_path = NULL
    ) {
      if (!is.null(self$logger)) self$logger$log_info("Plotting spatial context graph")
      
      # Check if entry exists
      if (!entry %in% colnames(SummarizedExperiment::colData(self$spe))) {
        if (!is.null(self$logger)) {
          self$logger$log_error("Entry '%s' not found in colData", entry)
        }
        return(NULL)
      }
      
      # Make sure required packages are available
      if (!requireNamespace("imcRtools", quietly = TRUE) || 
          !requireNamespace("ggplot2", quietly = TRUE)) {
        if (!is.null(self$logger)) {
          self$logger$log_error("Required packages (imcRtools, ggplot2) not available")
        }
        return(NULL)
      }
      
      # Create the context graphs
      tryCatch({
        # First plot - color by name
        p1 <- imcRtools::plotSpatialContext(
          self$spe, 
          entry = entry,
          group_by = group_by,
          node_color_by = "name",
          node_size_by = "n_cells",
          node_label_color_by = "name"
        )
        
        # Second plot - color by number of cells
        p2 <- imcRtools::plotSpatialContext(
          self$spe, 
          entry = entry,
          group_by = group_by,
          node_color_by = "n_cells",
          node_size_by = "n_group",
          node_label_color_by = "n_cells"
        ) + 
          ggplot2::scale_color_viridis_c()
        
        # Save if path provided
        if (!is.null(result_path)) {
          if (!is.null(self$logger)) {
            self$logger$log_info("Saving spatial context graphs to %s", result_path)
          }
          
          # Create directory if it doesn't exist
          dir.create(dirname(result_path), showWarnings = FALSE, recursive = TRUE)
          
          # Save first plot
          filename1 <- paste0(result_path, "_by_name.png")
          ggplot2::ggsave(filename1, p1, width = 10, height = 8)
          
          # Save second plot
          filename2 <- paste0(result_path, "_by_cells.png")
          ggplot2::ggsave(filename2, p2, width = 10, height = 8)
          
          if (!is.null(self$logger)) {
            self$logger$log_info("Saved plots to: %s and %s", filename1, filename2)
          }
        }
        
        return(list(by_name = p1, by_cells = p2))
      }, error = function(e) {
        if (!is.null(self$logger)) {
          self$logger$log_error("Error plotting spatial context graph: %s", e$message)
        }
        return(NULL)
      })
    },
    
    #' @description Plot patches on a spatial plot
    #' @param patch_id_column Column containing patch IDs
    #' @param img_id Column to filter by image ID
    #' @param image_id Specific image ID to plot (or NULL for all)
    #' @param spatial_x Column containing X coordinates
    #' @param spatial_y Column containing Y coordinates
    #' @param result_path Path to store visualization results (or NULL)
    #' @return ggplot object
    plotPatches = function(
      patch_id_column = "patch_id",
      img_id = "sample_id",
      image_id = NULL,
      spatial_x = "Pos_X",
      spatial_y = "Pos_Y",
      result_path = NULL
    ) {
      if (!is.null(self$logger)) self$logger$log_info("Plotting patches")
      
      # Check if patch_id exists
      if (!patch_id_column %in% colnames(SummarizedExperiment::colData(self$spe))) {
        if (!is.null(self$logger)) {
          self$logger$log_error("Column '%s' not found in colData", patch_id_column)
        }
        return(NULL)
      }
      
      # Make sure required packages are available
      if (!requireNamespace("ggplot2", quietly = TRUE)) {
        if (!is.null(self$logger)) {
          self$logger$log_error("Required package ggplot2 not available")
        }
        return(NULL)
      }
      
      # Filter to specific image if specified
      if (!is.null(image_id)) {
        subset_spe <- self$spe[, self$spe[[img_id]] == image_id]
        if (ncol(subset_spe) == 0) {
          if (!is.null(self$logger)) {
            self$logger$log_warning("No cells found for image: %s", image_id)
          }
          return(NULL)
        }
      } else {
        subset_spe <- self$spe
      }
      
      # Get patch information
      patch_id <- subset_spe[[patch_id_column]]
      in_patch <- !is.na(patch_id)
      
      # Check if any patches exist
      if (sum(in_patch) == 0) {
        if (!is.null(self$logger)) {
          self$logger$log_warning("No patches found to plot")
        }
        return(NULL)
      }
      
      # Create data frame for plotting
      plot_data <- data.frame(
        x = subset_spe[[spatial_x]],
        y = subset_spe[[spatial_y]],
        patch_id = patch_id,
        in_patch = in_patch,
        stringsAsFactors = FALSE
      )
      
      # Create the plot
      tryCatch({
        # Create base plot
        p <- ggplot2::ggplot(plot_data, ggplot2::aes(x = x, y = y)) +
          # Add points for cells not in patches
          ggplot2::geom_point(
            data = plot_data[!plot_data$in_patch, ],
            color = "gray80", 
            size = 0.5, 
            alpha = 0.3
          ) +
          # Add points for cells in patches with color by patch ID
          ggplot2::geom_point(
            data = plot_data[plot_data$in_patch, ],
            ggplot2::aes(color = factor(patch_id)), 
            size = 1
          ) +
          # Customize appearance
          ggplot2::theme_minimal() +
          ggplot2::labs(
            title = ifelse(is.null(image_id), "Patches", paste0("Patches - ", image_id)),
            x = spatial_x,
            y = spatial_y,
            color = "Patch ID"
          ) +
          ggplot2::theme(
            panel.grid = ggplot2::element_blank(),
            axis.text = ggplot2::element_text(size = 8),
            axis.title = ggplot2::element_text(size = 10),
            legend.position = "right"
          ) +
          ggplot2::coord_fixed()  # Equal aspect ratio
        
        # Save if path provided
        if (!is.null(result_path)) {
          if (!is.null(self$logger)) {
            self$logger$log_info("Saving patch plot to %s", result_path)
          }
          
          # Create directory if it doesn't exist
          dir.create(dirname(result_path), showWarnings = FALSE, recursive = TRUE)
          
          # Save plot
          ggplot2::ggsave(result_path, p, width = 10, height = 8)
        }
        
        return(p)
      }, error = function(e) {
        if (!is.null(self$logger)) {
          self$logger$log_error("Error plotting patches: %s", e$message)
        }
        return(NULL)
      })
    }
  ),
  
  private = list(
    #' @description Load spatial context-specific dependencies
    loadSpatialContextDependencies = function() {
      required_packages <- c("imcRtools", "igraph", "S4Vectors")
      
      if (!is.null(self$dependency_manager)) {
        for (pkg in required_packages) {
          self$dependency_manager$ensure_package(pkg)
        }
      } else {
        for (pkg in required_packages) {
          if (!requireNamespace(pkg, quietly = TRUE)) {
            warning(sprintf("Package '%s' is required but not installed", pkg))
          }
        }
      }
    }
  )
) 