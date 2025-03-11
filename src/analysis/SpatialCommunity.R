#' SpatialCommunity class for community and neighborhood detection
#' 
#' @description Analyzes spatial organization of cells by detecting communities
#' and neighborhoods based on spatial graphs using various methods.
#'
#' @details Implements functionality from the "Spatial community analysis" and
#' "Cellular neighborhood analysis" sections of the workshop.

SpatialCommunity <- R6::R6Class("SpatialCommunity",
  public = list(
    #' @field spe SpatialExperiment object
    spe = NULL,
    
    #' @field logger Logger object
    logger = NULL,
    
    #' @field n_cores Number of cores for parallel processing
    n_cores = NULL,
    
    #' @field dependency_manager DependencyManager object
    dependency_manager = NULL,
    
    #' @description Create a new SpatialCommunity object
    #' @param spe SpatialExperiment object
    #' @param logger Logger object
    #' @param n_cores Number of cores for parallel processing
    #' @param dependency_manager DependencyManager object
    initialize = function(spe, logger = NULL, n_cores = 1, dependency_manager = NULL) {
      self$spe <- spe
      self$logger <- logger
      self$n_cores <- n_cores
      
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
    
    #' @description Detect communities based on spatial graph
    #' @param colPairName Name of the spatial graph to use
    #' @param group_by Column to group by (e.g., tumor/stroma)
    #' @param size_threshold Minimum size of communities to keep
    #' @return Updated SpatialExperiment object
    detectGraphCommunities = function(colPairName = "neighborhood", group_by = "tumor_stroma", size_threshold = 10) {
      if (!is.null(self$logger)) self$logger$log_info("Detecting communities using graph-based approach")
      
      # Create the BiocParallel param object with seed for reproducibility
      bp_param <- BiocParallel::SerialParam(RNGseed = 220819)
      
      # Run community detection
      self$spe <- imcRtools::detectCommunity(
        self$spe, 
        colPairName = colPairName, 
        size_threshold = size_threshold,
        group_by = group_by,
        BPPARAM = bp_param
      )
      
      if (!is.null(self$logger)) self$logger$log_info("Communities detected and stored in 'spatial_community' column")
      
      return(self$spe)
    },
    
    #' @description Aggregate neighbors based on cell types
    #' @param colPairName Name of the spatial graph to use
    #' @param n_clusters Number of clusters for k-means
    #' @return Updated SpatialExperiment object
    aggregateCelltypeNeighbors = function(colPairName = "knn_interaction_graph", n_clusters = 6) {
      if (!is.null(self$logger)) self$logger$log_info("Aggregating neighbors by cell type")
      
      # Aggregate neighbor information based on cell types
      self$spe <- imcRtools::aggregateNeighbors(
        self$spe, 
        colPairName = colPairName, 
        aggregate_by = "metadata", 
        count_by = "celltype"
      )
      
      # Set seed for reproducibility
      set.seed(220705)
      
      # Run k-means clustering
      if (!is.null(self$logger)) self$logger$log_info("Running k-means clustering (k=%d) on aggregated neighbor data", n_clusters)
      
      cn_clusters <- stats::kmeans(self$spe$aggregatedNeighbors, centers = n_clusters)
      self$spe$cn_celltypes <- as.factor(cn_clusters$cluster)
      
      if (!is.null(self$logger)) self$logger$log_info("Cell type-based neighborhoods stored in 'cn_celltypes' column")
      
      return(self$spe)
    },
    
    #' @description Aggregate neighbors based on marker expression
    #' @param colPairName Name of the spatial graph to use
    #' @param assay_type Assay to use for expression values
    #' @param n_clusters Number of clusters for k-means
    #' @return Updated SpatialExperiment object
    aggregateExpressionNeighbors = function(colPairName = "knn_interaction_graph", 
                                           assay_type = "exprs", 
                                           n_clusters = 6) {
      if (!is.null(self$logger)) self$logger$log_info("Aggregating neighbors by expression")
      
      # Determine which markers to use
      subset_markers <- NULL
      if ("use_channel" %in% colnames(SummarizedExperiment::rowData(self$spe))) {
        subset_markers <- rownames(self$spe)[SummarizedExperiment::rowData(self$spe)$use_channel]
      }
      
      # Aggregate neighbor information based on expression
      self$spe <- imcRtools::aggregateNeighbors(
        self$spe, 
        colPairName = colPairName, 
        aggregate_by = "expression", 
        assay_type = assay_type,
        subset_row = subset_markers
      )
      
      # Set seed for reproducibility
      set.seed(220705)
      
      # Run k-means clustering
      if (!is.null(self$logger)) self$logger$log_info("Running k-means clustering (k=%d) on aggregated expression", n_clusters)
      
      cn_clusters <- stats::kmeans(self$spe$mean_aggregatedExpression, centers = n_clusters)
      self$spe$cn_expression <- as.factor(cn_clusters$cluster)
      
      if (!is.null(self$logger)) self$logger$log_info("Expression-based neighborhoods stored in 'cn_expression' column")
      
      return(self$spe)
    },
    
    #' @description Perform LISA-based spatial clustering
    #' @param radii Radii for LISA curve calculation
    #' @param n_clusters Number of clusters for k-means
    #' @param img_id Column name containing image identifiers
    #' @return Updated SpatialExperiment object
    performLisaClustering = function(radii = c(10, 20, 50), n_clusters = 6, img_id = "sample_id") {
      if (!is.null(self$logger)) self$logger$log_info("Performing LISA-based spatial clustering")
      
      # Check if required packages are installed
      if (!requireNamespace("lisaClust", quietly = TRUE)) {
        if (!is.null(self$logger)) self$logger$log_warning("Missing required package for LISA clustering")
        if (!is.null(self$logger)) self$logger$log_info("Please install lisaClust package using BiocManager::install('lisaClust')")
        
        # Create a dummy column and return
        self$spe$lisa_clusters <- factor(rep(1, ncol(self$spe)))
        if (!is.null(self$logger)) self$logger$log_info("Created dummy 'lisa_clusters' column")
        return(self$spe)
      }
      
      tryCatch({
        if (!is.null(self$logger)) self$logger$log_info("Preparing data for LISA clustering")
        
        # Ensure coordinates are in the right format and columns
        if (!all(c("Pos_X", "Pos_Y") %in% colnames(SpatialExperiment::spatialCoords(self$spe)))) {
          if (!is.null(self$logger)) self$logger$log_warning("Spatial coordinates missing expected columns (Pos_X, Pos_Y)")
          self$spe$lisa_clusters <- factor(rep(1, ncol(self$spe)))
          return(self$spe)
        }
        
        # Ensure celltype column exists
        if (!"celltype" %in% colnames(SummarizedExperiment::colData(self$spe))) {
          if (!is.null(self$logger)) self$logger$log_warning("celltype column not found in SpatialExperiment")
          self$spe$lisa_clusters <- factor(rep(1, ncol(self$spe)))
          return(self$spe)
        }
        
        # Calculate LISA curves directly from the SpatialExperiment object
        if (!is.null(self$logger)) self$logger$log_info("Calculating LISA curves with radii: %s", paste(radii, collapse = ", "))
        
        # Create a BiocParallel param object
        bp_param <- BiocParallel::SerialParam()
        
        # Call the updated lisa() function with the new signature
        lisaCurves <- lisaClust::lisa(
          cells = self$spe, 
          Rs = radii,
          BPPARAM = bp_param,
          spatialCoords = c("Pos_X", "Pos_Y"),
          cellType = "celltype",
          imageID = img_id
        )
        
        # Set NA to 0
        lisaCurves[is.na(lisaCurves)] <- 0
        
        # Perform clustering
        set.seed(220705)
        if (!is.null(self$logger)) self$logger$log_info("Running k-means clustering (k=%d) on LISA curves", n_clusters)
        
        lisa_clusters <- stats::kmeans(lisaCurves, centers = n_clusters)$cluster
        
        # Add results to SPE
        self$spe$lisa_clusters <- as.factor(lisa_clusters)
        
        if (!is.null(self$logger)) self$logger$log_info("LISA-based clusters stored in 'lisa_clusters' column")
        
        # Create visualization of cell type composition by LISA cluster
        if ("celltype_classified" %in% colnames(SummarizedExperiment::colData(self$spe))) {
          if (!is.null(self$logger)) self$logger$log_info("Creating visualization of cell type composition by LISA cluster")
          
          # Create output directory for LISA visualizations
          lisa_viz_dir <- file.path("output", "lisa_clusters")
          dir.create(lisa_viz_dir, recursive = TRUE, showWarnings = FALSE)
          
          # Generate and save the visualization
          self$visualizeCellTypesByLisaCluster(
            lisa_column = "lisa_clusters",
            celltype_column = "celltype_classified",
            save_path = file.path(lisa_viz_dir, "celltype_by_lisa_cluster.png")
          )
          
          if (!is.null(self$logger)) {
            self$logger$log_info("Saved cell type composition by LISA cluster visualization")
          }
        }
      }, error = function(e) {
        # If any error occurs during LISA clustering, log it and create a dummy column
        if (!is.null(self$logger)) self$logger$log_error("Error during LISA clustering: %s", e$message)
        if (!is.null(self$logger)) self$logger$log_info("Creating dummy 'lisa_clusters' column")
        self$spe$lisa_clusters <- factor(rep(1, ncol(self$spe)))
      })
      
      return(self$spe)
    },
    
    #' @description Generate cellular neighborhood composition heatmap
    #' @param cn_column Column containing neighborhood assignments
    #' @param save_path Path to save the heatmap (or NULL to not save)
    #' @return pheatmap object
    plotNeighborhoodComposition = function(cn_column = "cn_celltypes", save_path = NULL) {
      # Use dependency manager to ensure required packages
      private$ensurePackage("pheatmap")
      private$ensurePackage("magrittr")
      private$ensurePackage("dplyr")
      private$ensurePackage("tidyr")
      
      # Import the pipe operator explicitly
      `%>%` <- magrittr::`%>%`
      
      if (!is.null(self$logger)) self$logger$log_info("Generating neighborhood composition heatmap")
      
      # Get colData as a data frame
      col_data <- as.data.frame(SummarizedExperiment::colData(self$spe))
      
      # Process data for the heatmap
      for_plot <- col_data %>%
        dplyr::group_by(!!as.name(cn_column), celltype) %>%
        dplyr::summarize(count = dplyr::n(), .groups = "drop") %>%
        dplyr::group_by(!!as.name(cn_column)) %>%
        dplyr::mutate(freq = count / sum(count)) %>%
        tidyr::pivot_wider(
          id_cols = !!as.name(cn_column), 
          names_from = celltype, 
          values_from = freq, 
          values_fill = 0
        ) %>%
        dplyr::ungroup()
      
      # Remove the cn_column from the data for the heatmap
      for_plot <- for_plot[, !colnames(for_plot) %in% cn_column]
      
      # Create heatmap
      p <- pheatmap::pheatmap(
        for_plot, 
        color = colorRampPalette(c("dark blue", "white", "dark red"))(100), 
        scale = "column"
      )
      
      # Save if path provided
      if (!is.null(save_path)) {
        if (!is.null(self$logger)) self$logger$log_info("Saving heatmap to %s", save_path)
        png(save_path, width = 1000, height = 800, res = 150)
        print(p)
        dev.off()
      }
      
      return(p)
    },
    
    #' @description Analyze immune cell infiltration patterns
    #' @param immune_column Column containing immune cell annotations
    #' @param region_column Column containing region annotations
    #' @param condition_column Column containing condition annotations
    #' @param save_dir Directory to save visualizations
    #' @return List of immune infiltration analysis results
    analyzeImmuneInfiltration = function(immune_column = "is_immune", 
                                       region_column = NULL,
                                       condition_column = NULL,
                                       save_dir = NULL) {
      # Use dependency manager to ensure required packages
      private$ensurePackage("ggplot2")
      private$ensurePackage("magrittr")
      private$ensurePackage("dplyr")
      private$ensurePackage("tidyr")
      
      # Import the pipe operator explicitly
      `%>%` <- magrittr::`%>%`
      
      if (!is.null(self$logger)) self$logger$log_info("Analyzing immune cell infiltration patterns")
      
      # Validate immune column exists
      if (!immune_column %in% colnames(SummarizedExperiment::colData(self$spe))) {
        if (!is.null(self$logger)) self$logger$log_warning("Immune column '%s' not found", immune_column)
        return(NULL)
      }
      
      # Prepare results container
      results <- list()
      
      # Calculate overall immune infiltration statistics
      immune_count <- sum(self$spe[[immune_column]])
      total_count <- ncol(self$spe)
      immune_percent <- (immune_count / total_count) * 100
      
      if (!is.null(self$logger)) {
        self$logger$log_info("Overall immune infiltration: %.2f%% (%d/%d cells)", 
                           immune_percent, immune_count, total_count)
      }
      
      results$overall <- list(
        immune_count = immune_count,
        total_count = total_count,
        immune_percent = immune_percent
      )
      
      # Analyze by region if column provided
      if (!is.null(region_column) && region_column %in% colnames(SummarizedExperiment::colData(self$spe))) {
        if (!is.null(self$logger)) self$logger$log_info("Analyzing immune infiltration by region")
        
        # Get colData as a data frame
        col_data <- as.data.frame(SummarizedExperiment::colData(self$spe))
        
        # Group by region and summarize
        region_stats <- col_data %>%
          dplyr::group_by(dplyr::across(dplyr::all_of(region_column))) %>%
          dplyr::summarize(
            immune_count = sum(!!as.name(immune_column), na.rm = TRUE),
            total_count = dplyr::n(),
            immune_percent = (immune_count / total_count) * 100,
            .groups = "drop"
          )
        
        results$by_region <- region_stats
        
        # Create region visualization
        if (!is.null(save_dir)) {
          p_region <- ggplot2::ggplot(region_stats, ggplot2::aes(x = !!as.name(region_column), y = immune_percent)) +
            ggplot2::geom_bar(stat = "identity", fill = "steelblue") +
            ggplot2::labs(
              title = "Immune Cell Infiltration by Region",
              x = "Region",
              y = "Immune Cell Percentage (%)"
            ) +
            ggplot2::theme_minimal()
          
          if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)
          ggplot2::ggsave(file.path(save_dir, "immune_infiltration_by_region.png"), p_region, width = 8, height = 6)
        }
      }
      
      # Analyze by condition if column provided
      if (!is.null(condition_column) && condition_column %in% colnames(SummarizedExperiment::colData(self$spe))) {
        if (!is.null(self$logger)) self$logger$log_info("Analyzing immune infiltration by condition")
        
        # Get colData as a data frame
        col_data <- as.data.frame(SummarizedExperiment::colData(self$spe))
        
        # Group by condition and summarize
        condition_stats <- col_data %>%
          dplyr::group_by(dplyr::across(dplyr::all_of(condition_column))) %>%
          dplyr::summarize(
            immune_count = sum(!!as.name(immune_column), na.rm = TRUE),
            total_count = dplyr::n(),
            immune_percent = (immune_count / total_count) * 100,
            .groups = "drop"
          )
        
        results$by_condition <- condition_stats
        
        # Create condition visualization
        if (!is.null(save_dir)) {
          p_condition <- ggplot2::ggplot(condition_stats, ggplot2::aes(x = !!as.name(condition_column), y = immune_percent)) +
            ggplot2::geom_bar(stat = "identity", fill = "steelblue") +
            ggplot2::labs(
              title = "Immune Cell Infiltration by Condition",
              x = "Condition",
              y = "Immune Cell Percentage (%)"
            ) +
            ggplot2::theme_minimal()
          
          if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)
          ggplot2::ggsave(file.path(save_dir, "immune_infiltration_by_condition.png"), p_condition, width = 8, height = 6)
        }
        
        # If both region and condition are available, create interaction plot
        if (!is.null(region_column) && region_column %in% colnames(SummarizedExperiment::colData(self$spe))) {
          if (!is.null(self$logger)) self$logger$log_info("Analyzing region-condition interaction")
          
          # Get colData as a data frame
          col_data <- as.data.frame(SummarizedExperiment::colData(self$spe))
          
          # Group by region and condition and summarize
          interaction_stats <- col_data %>%
            dplyr::group_by(dplyr::across(dplyr::all_of(c(region_column, condition_column)))) %>%
            dplyr::summarize(
              immune_count = sum(!!as.name(immune_column), na.rm = TRUE),
              total_count = dplyr::n(),
              immune_percent = (immune_count / total_count) * 100,
              .groups = "drop"
            )
          
          results$interaction <- interaction_stats
          
          # Create interaction visualization
          if (!is.null(save_dir)) {
            p_interaction <- ggplot2::ggplot(interaction_stats, 
                                          ggplot2::aes(x = !!as.name(region_column), 
                                                     y = immune_percent, 
                                                     fill = !!as.name(condition_column))) +
              ggplot2::geom_bar(stat = "identity", position = "dodge") +
              ggplot2::labs(
                title = "Immune Cell Infiltration by Region and Condition",
                x = "Region",
                y = "Immune Cell Percentage (%)",
                fill = "Condition"
              ) +
              ggplot2::theme_minimal()
            
            ggplot2::ggsave(file.path(save_dir, "immune_infiltration_interaction.png"), 
                          p_interaction, width = 10, height = 7)
          }
        }
      }
      
      # Analyze immune cell proximity to different cell types (cell-centric approach)
      if (!is.null(self$logger)) self$logger$log_info("Analyzing immune cell proximity patterns")
      
      # Calculate proximity if neighborhood graph exists
      if ("neighborhood" %in% names(SingleCellExperiment::colPair(self$spe))) {
        proximity_results <- list()
        
        # Get the cell-cell interaction graph
        interaction_graph <- SingleCellExperiment::colPair(self$spe, "neighborhood")
        
        # For each cell type, calculate percentage of immune neighbors
        cell_types <- unique(as.character(self$spe$celltype))
        
        # Initialize results table
        proximity_data <- data.frame(
          celltype = character(),
          pct_immune_neighbors = numeric(),
          total_neighbors = numeric(),
          stringsAsFactors = FALSE
        )
        
        for (cell_type in cell_types) {
          # Get indices of cells of this type
          cell_indices <- which(self$spe$celltype == cell_type)
          
          if (length(cell_indices) > 0) {
            # Get all neighbors of these cells
            neighbor_indices <- integer()
            
            for (idx in cell_indices) {
              # Get row indices where idx is in the first column (from_idx)
              from_idx_rows <- which(interaction_graph$from_idx == idx)
              if (length(from_idx_rows) > 0) {
                neighbor_indices <- c(neighbor_indices, interaction_graph$to_idx[from_idx_rows])
              }
            }
            
            # Count immune neighbors
            if (length(neighbor_indices) > 0) {
              immune_neighbors <- sum(self$spe[[immune_column]][neighbor_indices])
              pct_immune_neighbors <- (immune_neighbors / length(neighbor_indices)) * 100
              
              proximity_data <- rbind(proximity_data, data.frame(
                celltype = cell_type,
                pct_immune_neighbors = pct_immune_neighbors,
                total_neighbors = length(neighbor_indices),
                stringsAsFactors = FALSE
              ))
            }
          }
        }
        
        results$proximity <- proximity_data
        
        # Create proximity visualization
        if (!is.null(save_dir) && nrow(proximity_data) > 0) {
          proximity_data <- proximity_data[order(-proximity_data$pct_immune_neighbors), ]
          
          p_proximity <- ggplot2::ggplot(proximity_data, 
                                      ggplot2::aes(x = reorder(celltype, pct_immune_neighbors), 
                                                 y = pct_immune_neighbors)) +
            ggplot2::geom_bar(stat = "identity", fill = "steelblue") +
            ggplot2::labs(
              title = "Immune Cell Proximity by Cell Type",
              x = "Cell Type",
              y = "Percentage of Immune Cell Neighbors (%)"
            ) +
            ggplot2::theme_minimal() +
            ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))
          
          ggplot2::ggsave(file.path(save_dir, "immune_cell_proximity.png"), 
                        p_proximity, width = 12, height = 8)
        }
      }
      
      return(results)
    },

    #' @description Create spatial visualizations of immune infiltration across ROI images
    #' @param immune_column Column containing immune cell annotations
    #' @param img_id_column Column containing image/ROI identifiers
    #' @param spatial_coords Names of coordinate columns in spatialCoords
    #' @param save_dir Directory to save visualizations
    #' @param plot_width Width of output plots
    #' @param plot_height Height of output plots
    #' @return List of spatial visualization results including plots and statistics
    visualizeImmuneInfiltrationSpatial = function(immune_column = "is_immune",
                                                img_id_column = "sample_id",
                                                spatial_coords = c("Pos_X", "Pos_Y"),
                                                save_dir = NULL,
                                                plot_width = 12,
                                                plot_height = 10) {
      # Use dependency manager to ensure required packages
      private$ensurePackage("ggplot2")
      private$ensurePackage("dplyr")
      private$ensurePackage("viridis")
      private$ensurePackage("ggpubr")
      private$ensurePackage("magrittr")
      
      # Import the pipe operator explicitly
      `%>%` <- magrittr::`%>%`
      
      if (!is.null(self$logger)) self$logger$log_info("Creating spatial visualizations of immune infiltration")
      
      # Validate immune column exists
      if (!immune_column %in% colnames(SummarizedExperiment::colData(self$spe))) {
        if (!is.null(self$logger)) self$logger$log_warning("Immune column '%s' not found", immune_column)
        return(NULL)
      }
      
      # Check spatial coordinates
      if (!all(spatial_coords %in% colnames(SpatialExperiment::spatialCoords(self$spe)))) {
        if (!is.null(self$logger)) {
          self$logger$log_warning("Required spatial coordinates not found in SPE object")
          self$logger$log_info("Available coordinates: %s", 
                               paste(colnames(SpatialExperiment::spatialCoords(self$spe)), collapse=", "))
        }
        
        # Use first two columns if specified columns not found
        spatial_coords <- colnames(SpatialExperiment::spatialCoords(self$spe))[1:2]
        if (!is.null(self$logger)) {
          self$logger$log_info("Using alternative coordinates: %s", paste(spatial_coords, collapse=", "))
        }
      }
      
      # Identify ROI/image identifier column
      if (!img_id_column %in% colnames(SummarizedExperiment::colData(self$spe))) {
        # Try common alternatives
        img_id_alternatives <- c("sample_id", "ImageNumber", "ImageID", "ROI", "Image")
        for (alt in img_id_alternatives) {
          if (alt %in% colnames(SummarizedExperiment::colData(self$spe))) {
            img_id_column <- alt
            if (!is.null(self$logger)) {
              self$logger$log_info("Using '%s' as ROI identifier column", img_id_column)
            }
            break
          }
        }
        
        if (!img_id_column %in% colnames(SummarizedExperiment::colData(self$spe))) {
          if (!is.null(self$logger)) {
            self$logger$log_warning("No suitable ROI identifier column found")
            self$logger$log_info("Creating a dummy ROI identifier")
          }
          # Create a dummy ROI identifier
          self$spe$ROI <- rep("ROI1", ncol(self$spe))
          img_id_column <- "ROI"
        }
      }
      
      # Prepare results container
      results <- list()
      
      # Create data frame for plotting
      plot_data <- data.frame(
        x = SpatialExperiment::spatialCoords(self$spe)[, spatial_coords[1]],
        y = SpatialExperiment::spatialCoords(self$spe)[, spatial_coords[2]],
        ROI = self$spe[[img_id_column]],
        is_immune = self$spe[[immune_column]]
      )
      
      # Add celltype information if available
      if ("celltype" %in% colnames(SummarizedExperiment::colData(self$spe))) {
        plot_data$celltype <- self$spe$celltype
      }
      
      # Add condition information if available
      cond_column <- NULL
      for (col in c("condition", "Condition", "treatment", "Treatment", "group", "Group")) {
        if (col %in% colnames(SummarizedExperiment::colData(self$spe))) {
          plot_data$condition <- self$spe[[col]]
          cond_column <- col
          break
        }
      }
      
      # Get unique ROIs
      rois <- unique(plot_data$ROI)
      if (!is.null(self$logger)) {
        self$logger$log_info("Found %d unique ROIs for visualization", length(rois))
      }
      
      # Initialize storage for plots and statistics
      roi_plots <- list()
      roi_stats <- data.frame(
        ROI = character(),
        total_cells = integer(),
        immune_cells = integer(),
        immune_percent = numeric(),
        stringsAsFactors = FALSE
      )
      
      # Calculate immune percentage per ROI
      roi_summary <- plot_data %>%
        dplyr::group_by(ROI) %>%
        dplyr::summarize(
          total_cells = dplyr::n(),
          immune_cells = sum(is_immune, na.rm = TRUE),
          immune_percent = (immune_cells / total_cells) * 100
        )
      
      # Store summary statistics
      results$roi_summary <- roi_summary
      
      # Create spatial plots for each ROI
      for (roi in rois) {
        roi_data <- plot_data[plot_data$ROI == roi, ]
        
        # Get statistics for this ROI
        roi_stat <- roi_summary[roi_summary$ROI == roi, ]
        
        # Create spatial plot
        p <- ggplot2::ggplot(roi_data, ggplot2::aes(x = x, y = y, color = is_immune)) +
          ggplot2::geom_point(alpha = 0.7, size = 1.5) +
          ggplot2::scale_color_manual(values = c("FALSE" = "gray80", "TRUE" = "red"), 
                                     labels = c("FALSE" = "Non-immune", "TRUE" = "Immune"),
                                     name = "Cell Type") +
          ggplot2::labs(
            title = paste0("Immune Cell Infiltration - ", roi),
            subtitle = sprintf("%.2f%% immune cells (%d/%d)", 
                              roi_stat$immune_percent, roi_stat$immune_cells, roi_stat$total_cells),
            x = "X Coordinate", y = "Y Coordinate"
          ) +
          ggplot2::theme_bw() +
          ggplot2::theme(
            plot.title = ggplot2::element_text(size = 14, face = "bold"),
            plot.subtitle = ggplot2::element_text(size = 12),
            legend.position = "right",
            panel.background = ggplot2::element_rect(fill = "white"),
            plot.background = ggplot2::element_rect(fill = "white", color = NA)
          )
        
        # Store the plot
        roi_plots[[roi]] <- p
        
        # Save the plot if a directory is specified
        if (!is.null(save_dir)) {
          if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)
          
          # Create a safe filename
          safe_roi_name <- gsub("[^a-zA-Z0-9]", "_", roi)
          filename <- file.path(save_dir, paste0("immune_infiltration_spatial_", safe_roi_name, ".png"))
          
          ggplot2::ggsave(filename, p, width = plot_width, height = plot_height, dpi = 300)
          
          if (!is.null(self$logger)) {
            self$logger$log_info("Saved spatial infiltration plot for ROI '%s' to %s", roi, filename)
          }
        }
      }
      
      # Store plots in results
      results$plots <- roi_plots
      
      # Create summary overview plot
      if (!is.null(save_dir)) {
        # Sort ROIs by immune percentage
        sorted_summary <- roi_summary[order(roi_summary$immune_percent, decreasing = TRUE), ]
        
        # Create bar plot of immune percentage by ROI
        p_summary <- ggplot2::ggplot(sorted_summary, 
                                   ggplot2::aes(x = reorder(ROI, -immune_percent), 
                                              y = immune_percent)) +
          ggplot2::geom_bar(stat = "identity", fill = "steelblue") +
          ggplot2::geom_text(ggplot2::aes(label = sprintf("%.1f%%", immune_percent)), 
                           vjust = -0.5, size = 3.5) +
          ggplot2::labs(
            title = "Immune Cell Infiltration Across ROIs",
            x = "ROI",
            y = "Percentage of Immune Cells (%)"
          ) +
          ggplot2::theme_bw() +
          ggplot2::theme(
            axis.text.x = ggplot2::element_text(angle = 45, hjust = 1, size = 10),
            plot.title = ggplot2::element_text(size = 14, face = "bold"),
            panel.background = ggplot2::element_rect(fill = "white"),
            plot.background = ggplot2::element_rect(fill = "white", color = NA),
            plot.margin = ggplot2::margin(b = 60, l = 20, r = 20, t = 20, unit = "pt")
          )
        
        # Save summary plot with increased bottom margin for labels
        summary_filename <- file.path(save_dir, "immune_infiltration_summary.png")
        ggplot2::ggsave(summary_filename, p_summary, 
                      width = plot_width, height = plot_height / 1.5, 
                      dpi = 300)
        
        if (!is.null(self$logger)) {
          self$logger$log_info("Saved immune infiltration summary plot to %s", summary_filename)
        }
        
        # Save summary data
        summary_csv <- file.path(save_dir, "immune_infiltration_by_roi.csv")
        write.csv(sorted_summary, summary_csv, row.names = FALSE)
        
        if (!is.null(self$logger)) {
          self$logger$log_info("Saved immune infiltration statistics to %s", summary_csv)
        }
        
        # If we have multiple ROIs, create a composite figure with representative ROIs
        if (length(rois) > 1) {
          if (!is.null(self$logger)) {
            self$logger$log_info("Creating composite visualization of representative ROIs")
          }
          
          # Choose ROIs to show:
          # 1. Highest immune infiltration
          # 2. Median immune infiltration
          # 3. Lowest immune infiltration
          if (length(rois) >= 3) {
            sorted_rois <- sorted_summary$ROI
            rep_rois <- c(
              sorted_rois[1],  # highest
              sorted_rois[ceiling(length(sorted_rois)/2)],  # median
              sorted_rois[length(sorted_rois)]  # lowest
            )
            
            rep_titles <- c(
              "Highest Infiltration",
              "Median Infiltration",
              "Lowest Infiltration"
            )
          } else {
            # If we have only 2 ROIs, show both
            rep_rois <- sorted_summary$ROI
            rep_titles <- rep("", length(rep_rois))
          }
          
          # Create list of plots to arrange
          plots_to_arrange <- list()
          for (i in seq_along(rep_rois)) {
            roi <- rep_rois[i]
            p <- roi_plots[[roi]] + 
                 ggplot2::ggtitle(rep_titles[i]) +
                 ggplot2::theme(legend.position = if(i == length(rep_rois)) "right" else "none")
            plots_to_arrange[[i]] <- p
          }
          
          # Add summary plot at the top
          plots_to_arrange <- c(list(p_summary), plots_to_arrange)
          
          # Arrange plots
          combined_plot <- ggpubr::ggarrange(
            plotlist = plots_to_arrange,
            ncol = 2, nrow = ceiling(length(plots_to_arrange)/2),
            common.legend = TRUE, legend = "right"
          )
          
          # Save combined plot
          combined_filename <- file.path(save_dir, "immune_infiltration_composite.png")
          ggplot2::ggsave(combined_filename, combined_plot, 
                        width = plot_width, height = plot_height, dpi = 300)
          
          if (!is.null(self$logger)) {
            self$logger$log_info("Saved composite visualization to %s", combined_filename)
          }
        }
      }
      
      # Add additional analysis: Nearest neighbor distances between immune and non-immune cells
      if (!is.null(self$logger)) {
        self$logger$log_info("Analyzing spatial distribution patterns of immune cells")
      }
      
      # If neighborhood graph exists, use it to analyze immune cell clustering
      if ("neighborhood" %in% names(SingleCellExperiment::colPair(self$spe))) {
        # Perform spatial analysis using the neighborhood graph
        interaction_graph <- SingleCellExperiment::colPair(self$spe, "neighborhood")
        
        # Calculate immune cell clustering metrics
        clustering_stats <- data.frame(
          ROI = character(),
          immune_isolation_score = numeric(),
          immune_dispersion = numeric(),
          stringsAsFactors = FALSE
        )
        
        for (roi in rois) {
          # Get cells for this ROI
          roi_indices <- which(self$spe[[img_id_column]] == roi)
          
          if (length(roi_indices) > 0) {
            # Get immune cells in this ROI
            immune_indices <- roi_indices[self$spe[[immune_column]][roi_indices]]
            
            if (length(immune_indices) > 0) {
              # Calculate isolation score (average % of immune neighbors for immune cells)
              immune_neighbor_pcts <- numeric(length(immune_indices))
              
              for (i in seq_along(immune_indices)) {
                idx <- immune_indices[i]
                # Get neighbors of this immune cell
                neighbor_rows <- which(interaction_graph$from_idx == idx)
                
                if (length(neighbor_rows) > 0) {
                  neighbors <- interaction_graph$to_idx[neighbor_rows]
                  # Calculate % of neighbors that are immune
                  immune_neighbor_pct <- sum(self$spe[[immune_column]][neighbors]) / length(neighbors) * 100
                  immune_neighbor_pcts[i] <- immune_neighbor_pct
                }
              }
              
              # Calculate average isolation score and dispersion
              isolation_score <- mean(immune_neighbor_pcts, na.rm = TRUE)
              dispersion <- sd(immune_neighbor_pcts, na.rm = TRUE)
              
              # Add to results
              clustering_stats <- rbind(clustering_stats, data.frame(
                ROI = roi,
                immune_isolation_score = isolation_score,
                immune_dispersion = dispersion,
                stringsAsFactors = FALSE
              ))
            }
          }
        }
        
        # Add clustering statistics to results
        results$clustering_stats <- clustering_stats
        
        # Create visualization of clustering patterns
        if (!is.null(save_dir) && nrow(clustering_stats) > 0) {
          p_cluster <- ggplot2::ggplot(clustering_stats, 
                                     ggplot2::aes(x = reorder(ROI, -immune_isolation_score), 
                                                y = immune_isolation_score)) +
            ggplot2::geom_bar(stat = "identity", fill = "steelblue") +
            ggplot2::geom_errorbar(ggplot2::aes(ymin = immune_isolation_score - immune_dispersion,
                                              ymax = immune_isolation_score + immune_dispersion),
                                 width = 0.2) +
            ggplot2::labs(
              title = "Immune Cell Clustering by ROI",
              subtitle = "Higher values indicate stronger immune cell clustering",
              x = "ROI",
              y = "Immune Cell Isolation Score (%)"
            ) +
            ggplot2::theme_bw() +
            ggplot2::theme(
              axis.text.x = ggplot2::element_text(angle = 45, hjust = 1),
              plot.title = ggplot2::element_text(size = 14, face = "bold"),
              panel.background = ggplot2::element_rect(fill = "white"),
              plot.background = ggplot2::element_rect(fill = "white", color = NA),
              plot.margin = ggplot2::margin(b = 60, l = 20, r = 20, t = 20, unit = "pt")
            )
          
          cluster_filename <- file.path(save_dir, "immune_cell_clustering.png")
          ggplot2::ggsave(cluster_filename, p_cluster, 
                        width = plot_width, height = plot_height / 1.5, 
                        dpi = 300)
          
          if (!is.null(self$logger)) {
            self$logger$log_info("Saved immune cell clustering analysis to %s", cluster_filename)
          }
          
          # Save clustering statistics
          cluster_csv <- file.path(save_dir, "immune_clustering_by_roi.csv")
          write.csv(clustering_stats, cluster_csv, row.names = FALSE)
        }
      }
      
      return(results)
    },

    #' @description Classify cell types based on marker expression
    #' @param output_column Column to store classified cell types
    #' @param expression_assay Assay to use for expression values
    #' @param use_config Boolean indicating whether to use marker definitions from config
    #' @return Updated SpatialExperiment object
    classifyCellTypesByMarkers = function(output_column = "celltype_classified",
                                         expression_assay = "exprs",
                                         use_config = TRUE) {
      private$ensurePackage("magrittr")
      
      # Explicitly import the pipe operator
      `%>%` <- magrittr::`%>%`
      
      if (!is.null(self$logger)) self$logger$log_info("Classifying cell types based on marker expression")
      
      # Get expression data
      expr_data <- SummarizedExperiment::assay(self$spe, expression_assay)
      
      # Get marker definitions either from config or use defaults
      marker_defs <- NULL
      if (use_config) {
        tryCatch({
          config_manager <- ConfigurationManager$new()
          marker_defs <- config_manager$config$community_analysis$marker_definitions
        }, error = function(e) {
          if (!is.null(self$logger)) self$logger$log_warning("Could not load marker definitions from config: %s", e$message)
        })
      }
      
      # Use default definitions if not available from config
      if (is.null(marker_defs)) {
        if (!is.null(self$logger)) self$logger$log_info("Using default marker definitions")
        marker_defs <- list(
          "Endothelial" = list(
            required = c("CD31", "CD34", "vWF", "PECAM1"),
            optional = c("TIE1", "TIE2", "VEGFR"),
            exclude = c("CD45", "CD140a", "CD140b", "SMA", "FAP"),
            threshold = 0.8
          ),
          "Fibroblast" = list(
            required = c("CD140a", "CD140b", "FAP", "SMA", "PDGFRa", "PDGFRb"),
            optional = c("CD44", "Vimentin", "FSP1"),
            exclude = c("CD31", "CD45", "CD3"),
            threshold = 0.5  # Lowered threshold to be more sensitive
          ),
          "Immune" = list(
            required = c("CD45", "CD11b", "CD11c", "CD3", "CD4", "CD8", "CD20"),
            optional = c("Ly6G", "CD206", "CD68", "MHCII"),
            exclude = NULL,
            threshold = 0.6  # Lowered threshold to be more sensitive
          ),
          "Other" = list(
            required = c("CD44", "Keratin", "KRT", "E-cadherin", "CDH1"),
            optional = NULL,
            exclude = c("CD31", "CD45", "CD140a", "CD140b"),
            threshold = 0.5
          )
        )
      }
      
      # Normalize marker names to match the expression data
      rownames_expr <- rownames(expr_data)
      
      # Log available markers for debugging
      if (!is.null(self$logger)) {
        self$logger$log_info("Available markers in expression data: %s", 
                            paste(head(rownames_expr, 15), collapse=", "))
      }
      
      # Function to find markers in expression data
      find_markers <- function(markers) {
        if (is.null(markers)) return(NULL)
        
        found_markers <- c()
        for (m in markers) {
          # Try exact match first
          if (m %in% rownames_expr) {
            found_markers <- c(found_markers, m)
            next
          }
          
          # Try partial match with more variations
          # This handles cases like "CD45" matching "CD45_RO", "CD45RA", "CD45.marker", etc.
          patterns <- c(
            paste0("^", m, "$"),                # Exact match
            paste0("^", m, "[_.\\-]"),          # Marker at start
            paste0("[_.\\-]", m, "$"),          # Marker at end
            paste0("[_.\\-]", m, "[_.\\-]"),    # Marker in middle
            paste0("^", m, "[0-9A-Za-z]*$")     # Marker followed by alphanumeric (e.g., CD45RA)
          )
          
          pattern_str <- paste(patterns, collapse="|")
          partial_matches <- grep(pattern_str, rownames_expr, value = TRUE)
          
          if (length(partial_matches) > 0) {
            found_markers <- c(found_markers, partial_matches[1])
            
            # Log the match for debugging
            if (!is.null(self$logger)) {
              self$logger$log_info("Matched marker '%s' to '%s' in expression data", 
                                  m, partial_matches[1])
            }
          } else {
            # Log failed match
            if (!is.null(self$logger)) {
              self$logger$log_info("No match found for marker '%s' in expression data", m)
            }
          }
        }
        
        return(unique(found_markers))
      }
      
      # Find available markers for each cell type
      cell_type_markers <- list()
      for (cell_type in names(marker_defs)) {
        def <- marker_defs[[cell_type]]
        req_markers <- find_markers(def$required)
        opt_markers <- find_markers(def$optional)
        excl_markers <- find_markers(def$exclude)
        
        cell_type_markers[[cell_type]] <- list(
          required = req_markers,
          optional = opt_markers,
          exclude = excl_markers
        )
        
        # Log available markers for each cell type
        if (!is.null(self$logger)) {
          self$logger$log_info("Markers for %s:", cell_type)
          self$logger$log_info("  Required: %s", paste(req_markers, collapse=", "))
          if (length(opt_markers) > 0) {
            self$logger$log_info("  Optional: %s", paste(opt_markers, collapse=", "))
          }
          if (length(excl_markers) > 0) {
            self$logger$log_info("  Exclude: %s", paste(excl_markers, collapse=", "))
          }
        }
      }
      
      # Initialize with "Unknown" cell type
      n_cells <- ncol(self$spe)
      cell_types <- rep("Unknown", n_cells)
      cell_type_scores <- rep(0, n_cells)
      
      # For each cell, check the marker criteria for each cell type
      for (i in 1:n_cells) {
        cell_expr <- expr_data[, i]
        
        for (cell_type in names(marker_defs)) {
          def <- marker_defs[[cell_type]]
          threshold <- if(!is.null(def$threshold)) def$threshold else 1.0
          
          # Get matched markers for this cell type
          matched_markers <- cell_type_markers[[cell_type]]
          req_markers <- matched_markers$required
          
          # Skip if no required markers were found in the data
          if (length(req_markers) == 0) next
          
          # Check if required markers are expressed
          req_expr <- cell_expr[req_markers]
          
          # More flexible expression criteria: 
          # At least 50% of required markers should be above threshold
          min_required <- max(1, ceiling(length(req_markers) * 0.5))
          req_satisfied <- sum(req_expr >= threshold) >= min_required
          
          if (!req_satisfied) next
          
          # Check exclusion markers
          exclude_markers <- matched_markers$exclude
          if (length(exclude_markers) > 0) {
            excl_expr <- cell_expr[exclude_markers]
            # Only exclude if ALL exclusion markers are high
            if (sum(excl_expr >= threshold) == length(excl_expr)) next
          }
          
          # Calculate score based on required + optional markers
          opt_markers <- matched_markers$optional
          
          # Base score on proportion of markers expressed and their intensity
          req_score <- sum(req_expr >= threshold) / length(req_markers) * mean(req_expr)
          
          opt_score <- if(length(opt_markers) > 0) {
            opt_expr <- cell_expr[opt_markers]
            sum(opt_expr >= threshold * 0.75) / length(opt_markers) * mean(opt_expr)
          } else {
            0
          }
          
          total_score <- req_score + opt_score
          
          # Assign cell type if score is better than current
          if (total_score > cell_type_scores[i]) {
            cell_types[i] <- cell_type
            cell_type_scores[i] <- total_score
          }
        }
      }
      
      # Add results to SPE object
      self$spe[[output_column]] <- factor(cell_types)
      
      # Summarize results
      if (!is.null(self$logger)) {
        type_counts <- table(cell_types)
        self$logger$log_info("Cell type classification results: %s", 
                           paste(names(type_counts), type_counts, sep="=", collapse=", "))
      }
      
      # Create summary for each phenograph cluster
      if ("phenograph_corrected" %in% colnames(SummarizedExperiment::colData(self$spe))) {
        pheno_clusters <- self$spe$phenograph_corrected
        summary_df <- data.frame(
          Cluster = unique(pheno_clusters)
        )
        
        # For each cluster, calculate cell type compositions
        for (cluster in summary_df$Cluster) {
          cluster_cells <- pheno_clusters == cluster
          cluster_types <- table(cell_types[cluster_cells])
          total_cells <- sum(cluster_cells)
          
          # Add dominant cell type
          if (length(cluster_types) > 0) {
            dominant_type <- names(cluster_types)[which.max(cluster_types)]
            dominant_pct <- max(cluster_types) / total_cells * 100
            
            summary_df$DominantType[summary_df$Cluster == cluster] <- dominant_type
            summary_df$DominantPct[summary_df$Cluster == cluster] <- round(dominant_pct, 1)
            
            # Add composition as a string
            composition <- paste(
              names(cluster_types), 
              paste0(round(cluster_types / total_cells * 100, 1), "%"), 
              sep=":", collapse=", "
            )
            summary_df$Composition[summary_df$Cluster == cluster] <- composition
          }
        }
        
        # Log cluster compositions
        if (!is.null(self$logger)) {
          self$logger$log_info("Cluster to cell type mapping:")
          for (i in 1:nrow(summary_df)) {
            self$logger$log_info("Cluster %s: %s (%s%%); Composition: %s", 
                               summary_df$Cluster[i], summary_df$DominantType[i], 
                               summary_df$DominantPct[i], summary_df$Composition[i])
          }
        }
        
        # Create a more reliable cluster-based cell type
        # For each cell, use the dominant cell type of its cluster
        cluster_to_celltype <- setNames(
          summary_df$DominantType,
          summary_df$Cluster
        )
        
        # Apply the cluster-based mapping
        self$spe[[paste0(output_column, "_cluster")]] <- 
          cluster_to_celltype[as.character(self$spe$phenograph_corrected)]
      }
      
      return(self$spe)
    },

    #' @description Create visualization of cell type distribution by phenograph cluster
    #' @param phenograph_column Column containing phenograph cluster assignments
    #' @param celltype_column Column containing cell type assignments
    #' @param save_path Path to save the visualization (or NULL to not save)
    #' @return ggplot object
    visualizeCellTypesByCluster = function(phenograph_column = "phenograph_corrected",
                                         celltype_column = "celltype_classified",
                                         save_path = NULL) {
      
      private$ensurePackage("dplyr")
      private$ensurePackage("RColorBrewer")
      private$ensurePackage("magrittr")
      
      # Explicitly import the pipe operator
      `%>%` <- magrittr::`%>%`
      
      if (!is.null(self$logger)) self$logger$log_info("Creating visualization of cell type distribution by phenograph cluster")
      
      # Extract data
      plot_data <- data.frame(
        Cluster = self$spe[[phenograph_column]],
        CellType = self$spe[[celltype_column]]
      )
      
      # Calculate proportions
      plot_summary <- plot_data %>%
        dplyr::group_by(Cluster, CellType) %>%
        dplyr::summarize(Count = dplyr::n(), .groups = "drop") %>%
        dplyr::group_by(Cluster) %>%
        dplyr::mutate(Proportion = Count / sum(Count)) %>%
        dplyr::ungroup()
      
      # Prepare a color palette
      cell_types <- unique(plot_summary$CellType)
      n_types <- length(cell_types)
      palette <- if (n_types <= 8) {
        RColorBrewer::brewer.pal(max(3, n_types), "Set1")
      } else {
        colorRampPalette(RColorBrewer::brewer.pal(8, "Set1"))(n_types)
      }
      color_map <- setNames(palette[1:n_types], cell_types)
      
      # Create stacked bar plot
      p <- ggplot2::ggplot(plot_summary, 
                     ggplot2::aes(x = factor(Cluster), y = Proportion, fill = CellType)) +
        ggplot2::geom_bar(stat = "identity") +
        ggplot2::scale_fill_manual(values = color_map) +
        ggplot2::labs(
          title = "Cell Type Composition by Phenograph Cluster",
          x = "Phenograph Cluster",
          y = "Proportion",
          fill = "Cell Type"
        ) +
        ggplot2::theme_bw() +
        ggplot2::theme(
          axis.text.x = ggplot2::element_text(angle = 45, hjust = 1),
          panel.background = ggplot2::element_rect(fill = "white"),
          plot.background = ggplot2::element_rect(fill = "white", color = NA)
        )
      
      # Save plot if requested
      if (!is.null(save_path)) {
        ggplot2::ggsave(save_path, p, width = 12, height = 8, dpi = 300)
        
        if (!is.null(self$logger)) {
          self$logger$log_info("Saved cell type distribution visualization to %s", save_path)
        }
      }
      
      return(p)
    },
    
    #' @description Create visualization of cell type distribution by LISA cluster
    #' @param lisa_column Column containing LISA cluster assignments
    #' @param celltype_column Column containing cell type assignments
    #' @param save_path Path to save the visualization (or NULL to not save)
    #' @return ggplot object
    visualizeCellTypesByLisaCluster = function(lisa_column = "lisa_clusters",
                                             celltype_column = "celltype_classified",
                                             save_path = NULL) {
      
      private$ensurePackage("dplyr")
      private$ensurePackage("RColorBrewer")
      private$ensurePackage("magrittr")
      
      # Explicitly import the pipe operator
      `%>%` <- magrittr::`%>%`
      
      if (!is.null(self$logger)) self$logger$log_info("Creating visualization of cell type distribution by LISA cluster")
      
      # Check if lisa clusters exist
      if (!(lisa_column %in% colnames(SummarizedExperiment::colData(self$spe)))) {
        if (!is.null(self$logger)) self$logger$log_warning("LISA cluster column '%s' not found in SPE object", lisa_column)
        return(NULL)
      }
      
      # Extract data
      plot_data <- data.frame(
        Cluster = self$spe[[lisa_column]],
        CellType = self$spe[[celltype_column]]
      )
      
      # Calculate proportions
      plot_summary <- plot_data %>%
        dplyr::group_by(Cluster, CellType) %>%
        dplyr::summarize(Count = dplyr::n(), .groups = "drop") %>%
        dplyr::group_by(Cluster) %>%
        dplyr::mutate(Proportion = Count / sum(Count)) %>%
        dplyr::ungroup()
      
      # Prepare a color palette
      cell_types <- unique(plot_summary$CellType)
      n_types <- length(cell_types)
      palette <- if (n_types <= 8) {
        RColorBrewer::brewer.pal(max(3, n_types), "Set1")
      } else {
        colorRampPalette(RColorBrewer::brewer.pal(8, "Set1"))(n_types)
      }
      color_map <- setNames(palette[1:n_types], cell_types)
      
      # Create stacked bar plot
      p <- ggplot2::ggplot(plot_summary, 
                     ggplot2::aes(x = factor(Cluster), y = Proportion, fill = CellType)) +
        ggplot2::geom_bar(stat = "identity") +
        ggplot2::scale_fill_manual(values = color_map) +
        ggplot2::labs(
          title = "Cell Type Composition by LISA Cluster",
          x = "LISA Cluster",
          y = "Proportion",
          fill = "Cell Type"
        ) +
        ggplot2::theme_bw() +
        ggplot2::theme(
          axis.text.x = ggplot2::element_text(angle = 45, hjust = 1),
          panel.background = ggplot2::element_rect(fill = "white"),
          plot.background = ggplot2::element_rect(fill = "white", color = NA)
        )
      
      # Save plot if requested
      if (!is.null(save_path)) {
        ggplot2::ggsave(save_path, p, width = 12, height = 8, dpi = 300)
        
        if (!is.null(self$logger)) {
          self$logger$log_info("Saved LISA cluster cell type distribution visualization to %s", save_path)
        }
      }
      
      return(p)
    },

    #' @description Create visualization of marker expression by phenograph cluster
    #' @param phenograph_column Column containing phenograph cluster assignments
    #' @param expression_assay Assay to use for expression values
    #' @param top_markers Number of top markers to include in heatmap
    #' @param save_path Path to save the visualization (or NULL to not save)
    #' @return pheatmap object
    visualizeClusterMarkerExpression = function(phenograph_column = "phenograph_corrected",
                                           expression_assay = "exprs",
                                           top_markers = 25,
                                           save_path = NULL) {
      private$ensurePackage("pheatmap")
      private$ensurePackage("RColorBrewer")
      private$ensurePackage("dplyr")
      private$ensurePackage("magrittr")
      
      # Explicitly import the pipe operator
      `%>%` <- magrittr::`%>%`
      
      if (!is.null(self$logger)) self$logger$log_info("Creating marker expression heatmap by phenograph cluster")
      
      # Check if the phenograph column exists
      if (!(phenograph_column %in% colnames(SummarizedExperiment::colData(self$spe)))) {
        if (!is.null(self$logger)) self$logger$log_error("Column '%s' not found in colData", phenograph_column)
        return(NULL)
      }
      
      # Check if the expression assay exists
      if (!(expression_assay %in% SummarizedExperiment::assayNames(self$spe))) {
        if (!is.null(self$logger)) self$logger$log_error("Assay '%s' not found", expression_assay)
        return(NULL)
      }
      
      # Get expression data
      expr_data <- SummarizedExperiment::assay(self$spe, expression_assay)
      
      # Get cluster assignments
      clusters <- SummarizedExperiment::colData(self$spe)[[phenograph_column]]
      
      # Calculate mean expression per cluster
      cluster_means <- t(sapply(rownames(expr_data), function(marker) {
        tapply(expr_data[marker,], clusters, mean)
      }))
      
      # Calculate variance across clusters for each marker
      marker_variances <- apply(cluster_means, 1, var)
      
      # Select top markers by variance
      top_marker_indices <- order(marker_variances, decreasing = TRUE)[1:min(top_markers, length(marker_variances))]
      top_markers_matrix <- cluster_means[top_marker_indices, ]
      
      # Scale the data for better visualization
      scaled_data <- t(scale(t(top_markers_matrix)))
      
      # Create annotation for clusters
      cluster_anno <- data.frame(Cluster = colnames(scaled_data))
      
      # Create color palettes
      heatmap_colors <- colorRampPalette(c("navy", "white", "firebrick3"))(100)
      
      # Create the heatmap
      heatmap_obj <- pheatmap::pheatmap(
        scaled_data,
        color = heatmap_colors,
        cluster_rows = TRUE,
        cluster_cols = TRUE,
        fontsize_row = 8,
        fontsize_col = 10,
        main = "Marker Expression by Phenograph Cluster",
        angle_col = 45,
        cellwidth = 15,
        cellheight = 12,
        border_color = NA,
        silent = TRUE
      )
      
      # Save the plot if requested
      if (!is.null(save_path)) {
        # Save as PNG
        grDevices::png(save_path, width = 10, height = 10, units = "in", res = 300)
        pheatmap::pheatmap(
          scaled_data,
          color = heatmap_colors,
          cluster_rows = TRUE,
          cluster_cols = TRUE,
          fontsize_row = 8,
          fontsize_col = 10,
          main = "Marker Expression by Phenograph Cluster",
          angle_col = 45,
          cellwidth = 15,
          cellheight = 12,
          border_color = NA
        )
        grDevices::dev.off()
        
        if (!is.null(self$logger)) {
          self$logger$log_info("Saved marker expression heatmap to %s", save_path)
        }
      }
      
      # Also create a marker enrichment score heatmap for clearer cell type signatures
      # Calculate enrichment score (log fold change vs other clusters)
      enrichment_scores <- t(apply(cluster_means, 1, function(marker_expr) {
        sapply(1:length(marker_expr), function(i) {
          other_expr <- marker_expr[-i]
          log2((marker_expr[i] + 0.1) / (mean(other_expr) + 0.1))
        })
      }))
      colnames(enrichment_scores) <- colnames(cluster_means)
      rownames(enrichment_scores) <- rownames(cluster_means)
      
      # Select top markers by enrichment score
      marker_max_enrichment <- apply(enrichment_scores, 1, max)
      top_enriched_indices <- order(marker_max_enrichment, decreasing = TRUE)[1:min(top_markers, length(marker_max_enrichment))]
      top_enriched_matrix <- enrichment_scores[top_enriched_indices, ]
      
      # If save_path is provided, create enrichment score heatmap too
      if (!is.null(save_path)) {
        enrichment_path <- sub("\\.png$", "_enrichment.png", save_path)
        
        grDevices::png(enrichment_path, width = 10, height = 10, units = "in", res = 300)
        pheatmap::pheatmap(
          top_enriched_matrix,
          color = colorRampPalette(c("navy", "white", "firebrick3"))(100),
          cluster_rows = TRUE,
          cluster_cols = TRUE,
          fontsize_row = 8,
          fontsize_col = 10,
          main = "Marker Enrichment by Phenograph Cluster",
          angle_col = 45,
          cellwidth = 15,
          cellheight = 12,
          border_color = NA
        )
        grDevices::dev.off()
        
        if (!is.null(self$logger)) {
          self$logger$log_info("Saved marker enrichment heatmap to %s", enrichment_path)
        }
      }
      
      return(heatmap_obj)
    },
    
    #' @description Visualize communities in spatial context
    #' @param community_column Column containing community IDs
    #' @param img_id_column Column containing image/ROI IDs
    #' @param save_dir Directory to save the visualizations
    #' @param plot_width Width of the plot in inches
    #' @param plot_height Height of the plot in inches
    #' @return List of ggplot objects
    visualizeCommunities = function(community_column = "community_id", 
                                   img_id_column = "sample_id",
                                   save_dir = NULL,
                                   plot_width = 10,
                                   plot_height = 8) {
      # Load visualization functions
      source("src/visualization/VisualizationFunctions.R")
      
      if (!is.null(self$logger)) self$logger$log_info("Visualizing communities in spatial context")
      
      # Create visualization factory
      viz_factory <- VisualizationFactory$new()
      
      # Create community visualization
      viz <- viz_factory$create_visualization(
        "community", 
        data = self$spe,
        results = NULL,
        community_column = community_column,
        img_id_column = img_id_column
      )
      
      # Save if requested
      if (!is.null(save_dir)) {
        if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)
        save_path <- file.path(save_dir, "spatial_communities.png")
        ggsave(save_path, viz$plot, width = plot_width, height = plot_height, dpi = 300)
        if (!is.null(self$logger)) self$logger$log_info("Saved community visualization to %s", save_path)
      }
      
      return(viz$plot)
    },
    
    #' @description Visualize cell types in spatial context
    #' @param celltype_column Column containing cell type assignments
    #' @param img_id_column Column containing image/ROI identifiers
    #' @param density_plot Whether to include a density plot
    #' @param phenograph_column Column containing phenograph cluster assignments
    #' @param save_dir Directory to save visualizations (or NULL to not save)
    #' @return Visualization object
    visualizeCellTypes = function(celltype_column = "celltype_classified", 
                                  img_id_column = NULL,
                                  density_plot = TRUE,
                                  phenograph_column = "phenograph_corrected",
                                  save_dir = NULL) {
      private$ensurePackage("R6")
      private$ensurePackage("SpatialExperiment")
      private$ensurePackage("SummarizedExperiment")
      
      if (!is.null(self$logger)) self$logger$log_info("Creating cell type spatial visualizations")
      
      # Auto-detect image ID column if not specified
      if (is.null(img_id_column)) {
        for (col in c("sample_id", "ImageNumber", "ImageID", "ROI", "Image")) {
          if (col %in% colnames(SummarizedExperiment::colData(self$spe))) {
            img_id_column <- col
            if (!is.null(self$logger)) self$logger$log_info("Using '%s' as image ID column", img_id_column)
            break
          }
        }
        if (is.null(img_id_column)) {
          stop("No suitable image ID column found. Please specify the img_id_column parameter.")
        }
      }
      
      # Check if we should use a different cell type column
      available_columns <- colnames(SummarizedExperiment::colData(self$spe))
      
      # Try to find the best cell type column
      if (!(celltype_column %in% available_columns)) {
        for (col in c("celltype_classified_cluster", "celltype", "celltype_manual")) {
          if (col %in% available_columns) {
            celltype_column <- col
            if (!is.null(self$logger)) self$logger$log_info("Using '%s' column for cell types", celltype_column)
            break
          }
        }
        if (!(celltype_column %in% available_columns)) {
          stop("No suitable cell type column found. Please specify the celltype_column parameter.")
        }
      }
      
      # Source the visualization functions
      source("src/visualization/VisualizationFunctions.R")
      
      # Create the visualization factory
      viz_factory <- VisualizationFactory$new()
      
      # Create the cell type spatial visualization
      viz <- viz_factory$create_visualization(
        viz_type = "celltype_spatial",
        data = self$spe,
        celltype_column = celltype_column,
        img_id_column = img_id_column,
        show_density = density_plot,
        cluster_column = phenograph_column
      )
      
      # Save the visualization if requested
      if (!is.null(save_dir)) {
        # Create the directory if it doesn't exist
        if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)
        
        # Save the visualization
        filename <- file.path(save_dir, "cell_type_spatial.png")
        viz$save(filename, width = 16, height = 12)
        
        if (!is.null(self$logger)) self$logger$log_info("Saved cell type spatial visualization to %s", filename)
      }
      
      return(viz)
    },
    
    #' @description Visualize cell-cell interactions using heatmaps
    #' @param colPairName Name of the spatial graph to use
    #' @param save_dir Directory to save the visualizations
    #' @param plot_width Width of the plot in inches
    #' @param plot_height Height of the plot in inches
    #' @return ggplot object
    visualizeCellInteractions = function(colPairName = "neighborhood",
                                        save_dir = NULL,
                                        plot_width = 10,
                                        plot_height = 8) {
      # Load visualization functions
      source("src/visualization/VisualizationFunctions.R")
      private$ensurePackage("imcRtools")
      private$ensurePackage("dplyr")
      
      if (!is.null(self$logger)) self$logger$log_info("Analyzing and visualizing cell-cell interactions")
      
      # Check if we have the required graph
      if (!(colPairName %in% SingleCellExperiment::colPairNames(self$spe))) {
        if (!is.null(self$logger)) self$logger$log_error("Spatial graph %s not found in SPE object", colPairName)
        stop(paste("Spatial graph", colPairName, "not found in SPE object"))
      }
      
      # Find the best cell type column to use
      celltype_column <- "celltype"
      possible_columns <- c("celltype_classified", "celltype_classified_cluster", "celltype")
      for (col in possible_columns) {
        if (col %in% colnames(SummarizedExperiment::colData(self$spe))) {
          celltype_column <- col
          if (!is.null(self$logger)) self$logger$log_info("Using '%s' column for cell type interactions", celltype_column)
          break
        }
      }
      
      # Identify the appropriate image ID column
      img_id_column <- NULL
      for (col_name in c("sample_id", "ImageNumber", "ImageID", "ROI", "Image")) {
        if (col_name %in% colnames(SummarizedExperiment::colData(self$spe))) {
          img_id_column <- col_name
          break
        }
      }
      
      if (is.null(img_id_column)) {
        if (!is.null(self$logger)) self$logger$log_error("No suitable image ID column found in SPE object")
        stop("No suitable image ID column found in SPE object")
      }
      
      # Calculate interaction statistics using imcRtools
      if (!is.null(self$logger)) self$logger$log_info("Calculating cell-cell interactions using testInteractions")
      
      set.seed(42)  # For reproducible permutation tests
      interaction_results <- imcRtools::testInteractions(
        self$spe,
        group_by = img_id_column,
        label = celltype_column,
        colPairName = colPairName
      )
      
      # Create visualization factory
      viz_factory <- VisualizationFactory$new()
      
      # Create interaction visualization
      viz <- viz_factory$create_visualization(
        "interaction", 
        data = self$spe,
        results = interaction_results
      )
      
      # Save if requested
      if (!is.null(save_dir)) {
        if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)
        save_path <- file.path(save_dir, "cell_interactions.png")
        ggsave(save_path, viz$plot, width = plot_width, height = plot_height, dpi = 300)
        if (!is.null(self$logger)) self$logger$log_info("Saved cell interaction visualization to %s", save_path)
      }
      
      return(viz$plot)
    },
    
    #' @description Visualize the spatial proximity between different cell types
    #' @param celltype_column Name of the column containing cell type information
    #' @param max_cells_per_type Maximum number of cells to sample per cell type
    #' @param save_dir Directory to save the visualization
    #' @param plot_width Width of the plot in inches
    #' @param plot_height Height of the plot in inches
    #' @return ggplot object with the cell type proximity visualization
    visualizeCellTypeProximity = function(celltype_column = "celltype",
                                        max_cells_per_type = 5000,
                                        save_dir = NULL,
                                        plot_width = 10,
                                        plot_height = 10) {
      # Ensure required packages are loaded
      require(ggplot2)
      
      # Find the best cell type column to use if the provided one doesn't exist
      if (!celltype_column %in% colnames(SummarizedExperiment::colData(self$spe))) {
        possible_columns <- c("celltype_classified", "celltype_classified_cluster", "celltype")
        for (col in possible_columns) {
          if (col %in% colnames(SummarizedExperiment::colData(self$spe))) {
            if (!is.null(self$logger)) self$logger$log_info("Using %s for cell type proximity analysis", col)
            celltype_column <- col
            break
          }
        }
      }
      
      # Check if we have a valid column
      if (!celltype_column %in% colnames(SummarizedExperiment::colData(self$spe))) {
        if (!is.null(self$logger)) self$logger$log_error("No valid cell type column found for proximity analysis")
        stop("No valid cell type column found for proximity analysis")
      }
      
      # Create visualization factory
      viz_factory <- VisualizationFactory$new()
      
      # Create cell type proximity visualization
      if (!is.null(self$logger)) self$logger$log_info("Calculating cell type proximity relationships")
      
      viz <- viz_factory$create_visualization(
        "celltype_proximity", 
        data = self$spe,
        celltype_column = celltype_column,
        max_cells_per_type = max_cells_per_type
      )
      
      # Save if requested
      if (!is.null(save_dir)) {
        if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)
        save_path <- file.path(save_dir, "cell_type_proximity.png")
        ggsave(save_path, viz$plot, width = plot_width, height = plot_height, dpi = 300)
        if (!is.null(self$logger)) self$logger$log_info("Saved cell type proximity visualization to %s", save_path)
      }
      
      return(viz$plot)
    },
    
    #' @description Create comprehensive spatial visualizations of immune infiltration
    #' @param immune_column Column indicating immune cells (TRUE/FALSE)
    #' @param img_id_column Column containing image/ROI IDs
    #' @param include_metrics Whether to include metrics plots
    #' @param save_dir Directory to save the visualizations
    #' @param plot_width Width of the plot in inches
    #' @param plot_height Height of the plot in inches
    #' @return List of ggplot objects
    visualizeImmuneInfiltrationComprehensive = function(immune_column = "is_immune",
                                               img_id_column = "sample_id",
                                               include_metrics = TRUE,
                                               save_dir = NULL,
                                               plot_width = 12,
                                               plot_height = 10) {
      # Load visualization functions
      source("src/visualization/VisualizationFunctions.R")
      
      if (!is.null(self$logger)) self$logger$log_info("Creating comprehensive immune infiltration visualizations")
      
      # Check if immune column exists
      if (!(immune_column %in% colnames(SummarizedExperiment::colData(self$spe)))) {
        if (!is.null(self$logger)) self$logger$log_error("Immune column %s not found in SPE object", immune_column)
        stop(paste("Immune column", immune_column, "not found in SPE object"))
      }
      
      # Create visualization factory
      viz_factory <- VisualizationFactory$new()
      
      # Create immune infiltration visualization
      infiltration_viz <- viz_factory$create_visualization(
        "immune_infiltration", 
        data = self$spe,
        results = NULL,
        immune_column = immune_column,
        img_id_column = img_id_column,
        include_metrics = include_metrics
      )
      
      # Create distance visualization
      distance_viz <- viz_factory$create_visualization(
        "spatial_distance", 
        data = self$spe,
        results = NULL,
        target_column = immune_column,
        img_id_column = img_id_column,
        distance_method = "min"
      )
      
      # Save if requested
      if (!is.null(save_dir)) {
        if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)
        
        # Save infiltration visualization
        inf_save_path <- file.path(save_dir, "immune_infiltration_spatial.png")
        ggsave(inf_save_path, infiltration_viz$plot, width = plot_width, height = plot_height, dpi = 300)
        if (!is.null(self$logger)) self$logger$log_info("Saved immune infiltration visualization to %s", inf_save_path)
        
        # Save distance visualization
        dist_save_path <- file.path(save_dir, "immune_distance_analysis.png")
        ggsave(dist_save_path, distance_viz$plot, width = plot_width, height = plot_height, dpi = 300)
        if (!is.null(self$logger)) self$logger$log_info("Saved immune distance visualization to %s", dist_save_path)
      }
      
      return(list(
        infiltration = infiltration_viz$plot,
        distance = distance_viz$plot
      ))
    },
    
    #' @description Create marker expression heatmap by group
    #' @param group_column Column to group cells by
    #' @param assay_name Assay to use for expression values
    #' @param top_markers Number of top markers to include in heatmap
    #' @param show_enrichment Whether to show enrichment scores
    #' @param save_dir Directory to save the visualizations
    #' @param plot_width Width of the plot in inches
    #' @param plot_height Height of the plot in inches
    #' @return pheatmap object
    visualizeMarkerExpression = function(group_column = "celltype",
                                        assay_name = "exprs",
                                        top_markers = 25,
                                        show_enrichment = TRUE,
                                        save_dir = NULL,
                                        plot_width = 12,
                                        plot_height = 8) {
      # Load visualization functions
      source("src/visualization/VisualizationFunctions.R")
      
      if (!is.null(self$logger)) self$logger$log_info("Creating marker expression heatmap by %s", group_column)
      
      # Create visualization factory
      viz_factory <- VisualizationFactory$new()
      
      # Create marker expression visualization
      viz <- viz_factory$create_visualization(
        "marker_expression", 
        data = self$spe,
        results = NULL,
        group_column = group_column,
        assay_name = assay_name,
        top_markers = top_markers,
        show_enrichment = show_enrichment
      )
      
      # Save if requested
      if (!is.null(save_dir)) {
        if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)
        save_path <- file.path(save_dir, paste0("marker_expression_by_", group_column, ".png"))
        ggsave(save_path, viz$plot, width = plot_width, height = plot_height, dpi = 300)
        if (!is.null(self$logger)) self$logger$log_info("Saved marker expression visualization to %s", save_path)
      }
      
      return(viz$plot)
    },
    
    #' @description Wrapper to create and save all visualizations
    #' @param output_dir Directory to save all visualizations
    #' @return List of visualization results
    createAllVisualizations = function(output_dir = "output/visualizations") {
      if (!is.null(self$logger)) self$logger$log_info("Creating all visualizations and saving to %s", output_dir)
      
      # Create output directory
      if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
      
      # Check if we have the required data
      if (is.null(self$spe)) {
        if (!is.null(self$logger)) self$logger$log_error("SPE object not available for visualization")
        stop("SPE object not available. Run analysis methods first.")
      }
      
      # Create subdirectories for different visualization types
      communities_dir <- file.path(output_dir, "communities")
      celltypes_dir <- file.path(output_dir, "cell_types")
      interactions_dir <- file.path(output_dir, "interactions")
      immune_dir <- file.path(output_dir, "immune_infiltration")
      markers_dir <- file.path(output_dir, "marker_expression")
      
      # Initialize results list
      results <- list()
      
      # 1. Visualize communities if community_id column exists
      if ("community_id" %in% colnames(SummarizedExperiment::colData(self$spe))) {
        if (!is.null(self$logger)) self$logger$log_info("Creating community visualizations")
        results$communities <- self$visualizeCommunities(
          community_column = "community_id",
          save_dir = communities_dir
        )
      }
      
      # 2. Visualize cell types
      if ("celltype" %in% colnames(SummarizedExperiment::colData(self$spe))) {
        if (!is.null(self$logger)) self$logger$log_info("Creating cell type visualizations")
        results$celltypes <- self$visualizeCellTypes(
          celltype_column = "celltype",
          save_dir = celltypes_dir
        )
      }
      
      # 3. Visualize cell-cell interactions
      if ("neighborhood" %in% SingleCellExperiment::colPairNames(self$spe) &&
          "celltype" %in% colnames(SummarizedExperiment::colData(self$spe))) {
        if (!is.null(self$logger)) self$logger$log_info("Creating cell interaction visualizations")
        results$interactions <- self$visualizeCellInteractions(
          colPairName = "neighborhood",
          save_dir = interactions_dir
        )
      }
      
      # 4. Visualize immune infiltration if is_immune column exists
      if ("is_immune" %in% colnames(SummarizedExperiment::colData(self$spe))) {
        if (!is.null(self$logger)) self$logger$log_info("Creating immune infiltration visualizations")
        results$immune <- self$visualizeImmuneInfiltrationComprehensive(
          immune_column = "is_immune",
          save_dir = immune_dir
        )
      }
      
      # 5. Visualize marker expression by different grouping variables
      if (any(c("exprs", "counts") %in% SummarizedExperiment::assayNames(self$spe))) {
        if (!is.null(self$logger)) self$logger$log_info("Creating marker expression visualizations")
        
        # By cell type
        if ("celltype" %in% colnames(SummarizedExperiment::colData(self$spe))) {
          results$markers_by_celltype <- self$visualizeMarkerExpression(
            group_column = "celltype",
            save_dir = markers_dir
          )
        }
        
        # By community_id
        if ("community_id" %in% colnames(SummarizedExperiment::colData(self$spe))) {
          results$markers_by_community <- self$visualizeMarkerExpression(
            group_column = "community_id",
            save_dir = markers_dir
          )
        }
        
        # By phenograph cluster
        for (cluster_col in c("phenograph_cluster", "phenograph_corrected", "cluster")) {
          if (cluster_col %in% colnames(SummarizedExperiment::colData(self$spe))) {
            results[[paste0("markers_by_", cluster_col)]] <- self$visualizeMarkerExpression(
              group_column = cluster_col,
              save_dir = markers_dir
            )
            break
          }
        }
      }
      
      # Create a summary HTML with all visualizations
      if (requireNamespace("htmltools", quietly = TRUE)) {
        if (!is.null(self$logger)) self$logger$log_info("Creating summary HTML")
        html_file <- file.path(output_dir, "visualization_summary.html")
        
        # Get all PNG files
        png_files <- list.files(
          path = output_dir,
          pattern = ".png$",
          recursive = TRUE,
          full.names = TRUE
        )
        
        # Create HTML content
        html_content <- htmltools::tags$html(
          htmltools::tags$head(
            htmltools::tags$title("Spatial Analysis Visualization Summary"),
            htmltools::tags$style("
              body { font-family: Arial, sans-serif; margin: 20px; }
              h1 { color: #333366; }
              h2 { color: #336699; margin-top: 30px; }
              img { max-width: 900px; border: 1px solid #ddd; margin: 10px 0; }
            ")
          ),
          htmltools::tags$body(
            htmltools::tags$h1("Spatial Analysis Visualization Summary"),
            htmltools::tags$p(paste("Generated on", Sys.time())),
            htmltools::tags$hr(),
            lapply(png_files, function(file) {
              section_name <- gsub(".*/", "", dirname(file))
              file_name <- gsub(".png$", "", basename(file))
              
              htmltools::tagList(
                htmltools::tags$h2(paste(section_name, "-", file_name)),
                htmltools::tags$img(src = file, alt = file_name)
              )
            })
          )
        )
        
        # Write HTML to file
        htmltools::save_html(html_content, file = html_file)
        if (!is.null(self$logger)) self$logger$log_info("Saved visualization summary to %s", html_file)
      }
      
      if (!is.null(self$logger)) self$logger$log_info("All visualizations created successfully")
      return(results)
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
      
      private$ensurePackage("pheatmap")
      private$ensurePackage("magrittr")
      private$ensurePackage("dplyr")
      private$ensurePackage("tidyr")
      private$ensurePackage("ggplot2")
      private$ensurePackage("viridis")
      private$ensurePackage("ggpubr")
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
