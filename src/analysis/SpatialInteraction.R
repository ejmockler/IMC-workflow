#' SpatialInteraction class for analyzing spatial interactions
#' 
#' @description Analyzes spatial interactions between cells and cell types, 
#' including spatial context detection, patch detection, and interaction testing.
#'
#' @details Implements functionality from the "Spatial context analysis", "Patch detection", 
#' and "Interaction analysis" sections of the workshop.

SpatialInteraction <- R6::R6Class("SpatialInteraction",
  public = list(
    #' @field spe SpatialExperiment object
    spe = NULL,
    
    #' @field logger Logger object
    logger = NULL,
    
    #' @field n_cores Number of cores for parallel processing
    n_cores = NULL,
    
    #' @field dependency_manager DependencyManager object
    dependency_manager = NULL,
    
    #' @field img_id Column name for sample/image ID
    img_id = "sample_id",
    
    #' @description Create a new SpatialInteraction object
    #' @param spe SpatialExperiment object
    #' @param logger Logger object
    #' @param n_cores Number of cores for parallel processing
    #' @param dependency_manager DependencyManager object
    #' @param img_id Column name for sample/image ID
    initialize = function(spe, logger = NULL, n_cores = 1, dependency_manager = NULL, img_id = "sample_id") {
      self$spe <- spe
      self$logger <- logger
      self$n_cores <- n_cores
      self$img_id <- img_id
      
      # If no dependency manager is provided, create one
      if (is.null(dependency_manager)) {
        # Simply try to create the DependencyManager - if it fails, we'll get NULL
        tryCatch({
          self$dependency_manager <- DependencyManager$new(logger = logger)
        }, error = function(e) {
          if (!is.null(logger)) logger$log_warning("Could not create DependencyManager: %s", e$message)
        })
      } else {
        self$dependency_manager <- dependency_manager
      }
      
      # Load required packages
      private$loadDependencies()
      
      invisible(self)
    },
    
    #' @description Build a spatial context graph
    #' @param k Number of neighbors
    #' @param name Name for the graph
    #' @return Updated SpatialExperiment object
    buildSpatialContextGraph = function(k = 40, name = "knn_spatialcontext_graph") {
      if (!is.null(self$logger)) self$logger$log_info("Building spatial context graph with k = %d", k)
      
      self$spe <- imcRtools::buildSpatialGraph(
        self$spe, 
        img_id = self$img_id, 
        type = "knn", 
        name = name, 
        k = k
      )
      
      return(self$spe)
    },
    
    #' @description Aggregate neighborhoods based on cellular neighborhoods
    #' @param colPairName Name of the spatial graph
    #' @param name Name for the aggregated neighborhood data
    #' @return Updated SpatialExperiment object
    aggregateCNNeighbors = function(colPairName = "knn_spatialcontext_graph", name = "aggregatedNeighborhood") {
      if (!is.null(self$logger)) self$logger$log_info("Aggregating neighborhoods based on cellular neighborhoods")
      
      # Check if cn_celltypes exists
      if (!"cn_celltypes" %in% colnames(SummarizedExperiment::colData(self$spe))) {
        stop("Column 'cn_celltypes' not found. Run aggregateCelltypeNeighbors first.")
      }
      
      self$spe <- imcRtools::aggregateNeighbors(
        self$spe, 
        colPairName = colPairName,
        aggregate_by = "metadata",
        count_by = "cn_celltypes",
        name = name
      )
      
      return(self$spe)
    },
    
    #' @description Detect spatial contexts
    #' @param entry Column containing aggregated neighborhood data
    #' @param threshold Threshold for determining dominant CNs
    #' @param name Name for the spatial context column
    #' @return Updated SpatialExperiment object
    detectSpatialContext = function(entry = "aggregatedNeighborhood", threshold = 0.9, name = "spatial_context") {
      if (!is.null(self$logger)) self$logger$log_info("Detecting spatial contexts with threshold = %.2f", threshold)
      
      self$spe <- imcRtools::detectSpatialContext(
        self$spe, 
        entry = entry,
        threshold = threshold,
        name = name
      )
      
      return(self$spe)
    },
    
    #' @description Filter spatial contexts
    #' @param group_by Column in colData to group by for filtering
    #' @param group_threshold Minimum number of groups that must have the spatial context
    #' @param cells_threshold Minimum number of cells per spatial context
    #' @return The SpatialExperiment object with filtered spatial contexts
    filterSpatialContext = function(
      group_by = "sample_id",
      group_threshold = 3,
      cells_threshold = 100
    ) {
      self$logger$log_info("Filtering spatial contexts (group threshold = %d, cells threshold = %d)", 
                          group_threshold, cells_threshold)
      
      # Validate input parameters
      if (!group_by %in% colnames(SummarizedExperiment::colData(self$spe))) {
        self$logger$log_error("Column '%s' not found in colData", group_by)
        stop(sprintf("'%s' not in colData. Available columns: %s", 
                   group_by, 
                   paste(colnames(SummarizedExperiment::colData(self$spe)), collapse=", ")))
      }
      
      # Filter by number of group entries
      self$spe <- imcRtools::filterSpatialContext(
        self$spe, 
        entry = "spatial_context",
        group_by = group_by, 
        group_threshold = group_threshold
      )
      
      # Filter by number of cells
      if (!is.null(cells_threshold)) {
        self$spe <- imcRtools::filterSpatialContext(
          self$spe, 
          entry = "spatial_context",
          group_by = group_by, 
          group_threshold = group_threshold,
          cells_threshold = cells_threshold
        )
      }
      
      return(self$spe)
    },
    
    #' @description Plot spatial context graph
    #' @param entry Column containing spatial context assignments
    #' @param group_by Column to group by
    #' @param save_path Path to save the plot (or NULL to not save)
    #' @return ggplot object
    plotSpatialContextGraph = function(entry = "spatial_context_filtered", 
                                      group_by = "sample_id",
                                      save_path = NULL) {
      if (!is.null(self$logger)) self$logger$log_info("Plotting spatial context graph")
      
      # Create the context graph
      p1 <- imcRtools::plotSpatialContext(
        self$spe, 
        entry = entry,
        group_by = group_by,
        node_color_by = "name",
        node_size_by = "n_cells",
        node_label_color_by = "name"
      )
      
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
      if (!is.null(save_path)) {
        if (!is.null(self$logger)) self$logger$log_info("Saving spatial context graphs to %s", save_path)
        
        # Save first plot
        png(paste0(save_path, "_by_name.png"), width = 1000, height = 800, res = 150)
        print(p1)
        dev.off()
        
        # Save second plot
        png(paste0(save_path, "_by_cells.png"), width = 1000, height = 800, res = 150)
        print(p2)
        dev.off()
      }
      
      return(list(by_name = p1, by_cells = p2))
    },
    
    #' @description Detect patches of cells
    #' @param patch_cells Boolean vector indicating cells to include in patches
    #' @param expand_by Distance to expand patches
    #' @param min_patch_size Minimum number of cells per patch
    #' @param colPairName Name of the graph to use
    #' @return Updated SpatialExperiment object
    detectPatches = function(patch_cells, expand_by = 1, min_patch_size = 10, colPairName = "neighborhood") {
      if (!is.null(self$logger)) self$logger$log_info("Detecting patches (expand by = %d, min size = %d)", 
                                                    expand_by, min_patch_size)
      
      # Check if any patch cells exist
      if (sum(patch_cells) < min_patch_size) {
        if (!is.null(self$logger)) self$logger$log_warning("Not enough patch cells (%d) to meet minimum size requirement (%d)", 
                                                         sum(patch_cells), min_patch_size)
        
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
          img_id = self$img_id,
          expand_by = expand_by,
          min_patch_size = min_patch_size,
          colPairName = colPairName
        )
      }, error = function(e) {
        # If no connected components found, create dummy patch columns
        if (grepl("No connected components found", e$message)) {
          if (!is.null(self$logger)) self$logger$log_warning("No connected components found for patch detection: %s", e$message)
          
          # Create dummy patch column to avoid downstream errors
          self$spe$patch_id <- NA
          self$spe$in_patch <- FALSE
        } else {
          # For other errors, propagate them
          stop(e)
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
    #' @return Updated SpatialExperiment object
    calculateDistToPatches = function() {
      if (!is.null(self$logger)) self$logger$log_info("Calculating minimum distance to patches")
      
      # Check if patch_id exists
      if (!"patch_id" %in% colnames(SummarizedExperiment::colData(self$spe))) {
        if (!is.null(self$logger)) self$logger$log_warning("Column 'patch_id' not found. Cannot calculate distances.")
        
        # Create a dummy distance column to avoid downstream errors
        self$spe$dist_to_patch <- NA
        
        return(self$spe)
      }
      
      # Check if there are any valid patches
      valid_patches <- !is.na(self$spe$patch_id)
      if (sum(valid_patches) == 0) {
        if (!is.null(self$logger)) self$logger$log_warning("No valid patches found. Cannot calculate distances.")
        
        # Create a dummy distance column to avoid downstream errors
        self$spe$dist_to_patch <- NA
        
        return(self$spe)
      }
      
      # Calculate minimum distance to patches
      tryCatch({
        self$spe <- imcRtools::minDistToCells(
          self$spe, 
          x_cells = !is.na(self$spe$patch_id), 
          img_id = self$img_id
        )
        
        if (!is.null(self$logger)) self$logger$log_info("Successfully calculated distances to patches")
      }, error = function(e) {
        if (!is.null(self$logger)) self$logger$log_error("Error calculating distances to patches: %s", e$message)
        
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
      
      # Calculate patch sizes
      patch_size <- imcRtools::patchSize(self$spe, patch_id_col)
      
      # Merge with metadata
      patch_size <- merge(
        patch_size, 
        SummarizedExperiment::colData(self$spe)[match(patch_size$patch_id, self$spe[[patch_id_col]]),], 
        by = "patch_id"
      )
      
      return(patch_size)
    },
    
    #' @description Test cell-type interactions
    #' @param method Method to use ("classic" or "patch")
    #' @param group_by Column to group by
    #' @param label Column with cell type labels
    #' @param colPairName Name of the graph to use
    #' @param patch_size Number of patch cells (for patch method)
    #' @return DataFrame with interaction testing results
    testCelltypeInteractions = function(method = "classic", 
                                       group_by = "sample_id", 
                                       label = "celltype", 
                                       colPairName = "neighborhood",
                                       patch_size = 3) {
      if (!is.null(self$logger)) self$logger$log_info("Testing cell-type interactions using %s method", method)
      
      # Create BiocParallel param with seed
      bp_param <- BiocParallel::SerialParam(RNGseed = 221029)
      
      # Test interactions
      args <- list(
        self$spe, 
        group_by = group_by,
        label = label, 
        colPairName = colPairName,
        BPPARAM = bp_param
      )
      
      if (method == "patch") {
        args$method <- "patch"
        args$patch_size <- patch_size
      }
      
      results <- do.call(imcRtools::testInteractions, args)
      
      return(results)
    },
    
    #' @description Plot interaction testing results
    #' @param results DataFrame with interaction testing results
    #' @param save_path Path to save the plot (or NULL to not save)
    #' @return ggplot object
    plotInteractions = function(results, save_path = NULL) {
      if (!is.null(self$logger)) self$logger$log_info("Plotting interaction testing results")
      
      # Use dependency manager to ensure required packages
      private$ensurePackage("tidyverse")
      private$ensurePackage("scales")
      
      # Summarize results
      plot_data <- results %>% 
        tidyverse::as_tibble() %>%
        tidyverse::group_by(from_label, to_label) %>%
        tidyverse::summarize(
          significant = sum(sig == TRUE, na.rm = TRUE) / dplyr::n() > 0.5,
          interaction_type = factor(
            ifelse(
              significant,
              ifelse(mean(log_odds, na.rm = TRUE) > 0, "enriched", "depleted"),
              "not significant"
            ),
            levels = c("enriched", "depleted", "not significant")
          ),
          median_log_odds = median(log_odds, na.rm = TRUE),
          .groups = "drop"
        )
      
      # Create the plot
      p <- ggplot2::ggplot(
        plot_data,
        ggplot2::aes(
          x = from_label, 
          y = to_label, 
          fill = interaction_type,
          size = abs(median_log_odds)
        )
      ) +
        ggplot2::geom_point(shape = 21) +
        ggplot2::scale_fill_manual(
          values = c("enriched" = "red", "depleted" = "blue", "not significant" = "grey")
        ) +
        ggplot2::labs(
          title = "Cell Type Interactions",
          x = "From Cell Type",
          y = "To Cell Type",
          fill = "Interaction Type",
          size = "Absolute Log Odds"
        ) +
        ggplot2::theme_minimal() +
        ggplot2::theme(
          axis.text.x = ggplot2::element_text(angle = 45, hjust = 1),
          panel.grid.major = ggplot2::element_blank(),
          panel.grid.minor = ggplot2::element_blank()
        )
      
      # Save if path provided
      if (!is.null(save_path)) {
        if (!is.null(self$logger)) self$logger$log_info("Saving interaction plot to %s", save_path)
        ggplot2::ggsave(save_path, p, width = 10, height = 8)
      }
      
      return(p)
    }
  ),
  
  private = list(
    #' @description Load required dependencies
    loadDependencies = function() {
      # If dependency manager is available, use it
      if (!is.null(self$dependency_manager)) {
        self$dependency_manager$install_missing_packages()
        return()
      }
      
      # Otherwise, use direct installation
      if (!requireNamespace("imcRtools", quietly = TRUE)) {
        if (!is.null(self$logger)) self$logger$log_info("Installing imcRtools package")
        BiocManager::install("imcRtools")
      }
      
      if (!requireNamespace("BiocParallel", quietly = TRUE)) {
        if (!is.null(self$logger)) self$logger$log_info("Installing BiocParallel package")
        BiocManager::install("BiocParallel")
      }
      
      private$ensurePackage("ggplot2")
    },
    
    #' @description Ensure a package is installed
    ensurePackage = function(pkg_name) {
      # If dependency manager is available, use it
      if (!is.null(self$dependency_manager)) {
        return(self$dependency_manager$ensure_package(pkg_name))
      }
      
      # Otherwise, use direct installation
      if (!requireNamespace(pkg_name, quietly = TRUE)) {
        if (!is.null(self$logger)) self$logger$log_info("Installing package: %s", pkg_name)
        install.packages(pkg_name)
      }
      return(requireNamespace(pkg_name, quietly = TRUE))
    }
  )
)