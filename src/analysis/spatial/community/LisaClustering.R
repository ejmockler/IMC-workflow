#' LISA-based spatial clustering
#' 
#' @description Implements LISA (Local Indicators of Spatial Association) based clustering
#' for detecting spatial patterns in cellular organization.
#'
#' @details Extends the CommunityBase class with methods specific to LISA-based
#' spatial pattern detection using the lisaClust package.

library(R6)
library(SpatialExperiment)

source("src/analysis/spatial/community/CommunityBase.R")

LisaClustering <- R6::R6Class("LisaClustering",
  inherit = CommunityBase,
  
  public = list(
    #' @description Create a new LisaClustering object
    #' @param spe SpatialExperiment object with spatial coordinates and cell types
    #' @param logger Logger object for status updates
    #' @param n_cores Number of cores for parallel processing
    #' @param dependency_manager DependencyManager object for package management
    initialize = function(spe, logger = NULL, n_cores = 1, dependency_manager = NULL) {
      super$initialize(spe, logger, n_cores, dependency_manager)
      
      # Load LISA-specific dependencies
      private$loadLisaDependencies()
      
      invisible(self)
    },
    
    #' @description Perform LISA-based spatial clustering
    #' @param radii Radii for LISA curve calculation
    #' @param n_clusters Number of clusters for k-means
    #' @param img_id Column name containing image identifiers
    #' @param celltype_column Name of column containing cell type information
    #' @param result_column Name of column to store LISA cluster assignments
    #' @param save_visualizations Whether to save visualizations
    #' @param visualization_dir Directory to save visualizations
    #' @return Updated SpatialExperiment object
    performLisaClustering = function(
      radii = c(10, 20, 50), 
      n_clusters = 6, 
      img_id = "sample_id",
      celltype_column = "celltype",
      result_column = "lisa_clusters",
      save_visualizations = TRUE,
      visualization_dir = file.path("output", "lisa_clusters")
    ) {
      if (!is.null(self$logger)) self$logger$log_info("Performing LISA-based spatial clustering")
      
      # Check if required packages are installed
      if (!requireNamespace("lisaClust", quietly = TRUE)) {
        if (!is.null(self$logger)) self$logger$log_warning("Missing required package for LISA clustering")
        if (!is.null(self$logger)) self$logger$log_info("Please install lisaClust package using BiocManager::install('lisaClust')")
        
        # Create a dummy column and return
        self$spe[[result_column]] <- factor(rep(1, ncol(self$spe)))
        if (!is.null(self$logger)) self$logger$log_info("Created dummy '%s' column", result_column)
        return(self$spe)
      }
      
      tryCatch({
        if (!is.null(self$logger)) self$logger$log_info("Preparing data for LISA clustering")
        
        # Find coordinate columns
        coord_cols <- colnames(SpatialExperiment::spatialCoords(self$spe))
        spatial_coords <- c("Pos_X", "Pos_Y")
        
        # Try to match standard coordinate column names
        if (!all(c("Pos_X", "Pos_Y") %in% coord_cols)) {
          # Use the first two columns as X and Y coordinates
          spatial_coords <- coord_cols[1:2]
          if (!is.null(self$logger)) {
            self$logger$log_warning("Standard coordinate columns (Pos_X, Pos_Y) not found")
            self$logger$log_info("Using %s as spatial coordinates", paste(spatial_coords, collapse=", "))
          }
        }
        
        # Ensure celltype column exists
        if (!(celltype_column %in% colnames(SummarizedExperiment::colData(self$spe)))) {
          if (!is.null(self$logger)) self$logger$log_warning("Cell type column '%s' not found in SpatialExperiment", celltype_column)
          self$spe[[result_column]] <- factor(rep(1, ncol(self$spe)))
          return(self$spe)
        }
        
        # Calculate LISA curves
        if (!is.null(self$logger)) self$logger$log_info("Calculating LISA curves with radii: %s", paste(radii, collapse=", "))
        
        # Create a BiocParallel param object with appropriate number of cores
        bp_param <- if (self$n_cores > 1) {
          BiocParallel::MulticoreParam(workers = self$n_cores, RNGseed = 42)
        } else {
          BiocParallel::SerialParam(RNGseed = 42)
        }
        
        # Call the lisaClust function
        lisaCurves <- lisaClust::lisa(
          cells = self$spe, 
          Rs = radii,
          BPPARAM = bp_param,
          spatialCoords = spatial_coords,
          cellType = celltype_column,
          imageID = img_id
        )
        
        # Replace NAs with 0
        lisaCurves[is.na(lisaCurves)] <- 0
        
        # Perform clustering
        set.seed(42)  # For reproducibility
        if (!is.null(self$logger)) self$logger$log_info("Running k-means clustering (k=%d) on LISA curves", n_clusters)
        
        lisa_clusters <- stats::kmeans(lisaCurves, centers = n_clusters)$cluster
        
        # Add results to SPE
        self$spe[[result_column]] <- as.factor(lisa_clusters)
        
        if (!is.null(self$logger)) self$logger$log_info("LISA-based clusters stored in '%s' column", result_column)
        
        # Create visualization if requested
        if (save_visualizations) {
          # Create visualization output directory
          if (!dir.exists(visualization_dir)) {
            dir.create(visualization_dir, recursive = TRUE, showWarnings = FALSE)
          }
          
          # Create and save visualization
          self$visualizeLisaClusters(
            lisa_column = result_column,
            celltype_column = celltype_column,
            save_path = file.path(visualization_dir, "celltype_by_lisa_cluster.png")
          )
        }
      }, error = function(e) {
        # If any error occurs during LISA clustering, log it and create a dummy column
        if (!is.null(self$logger)) self$logger$log_error("Error during LISA clustering: %s", e$message)
        if (!is.null(self$logger)) self$logger$log_info("Creating dummy '%s' column", result_column)
        self$spe[[result_column]] <- factor(rep(1, ncol(self$spe)))
      })
      
      return(self$spe)
    },
    
    #' @description Create visualization of cell type distribution by LISA cluster
    #' @param lisa_column Column containing LISA cluster assignments
    #' @param celltype_column Column containing cell type assignments
    #' @param save_path Path to save the visualization (or NULL to not save)
    #' @return ggplot object
    visualizeLisaClusters = function(
      lisa_column = "lisa_clusters",
      celltype_column = "celltype",
      save_path = NULL
    ) {
      # Ensure required packages are available
      private$ensurePackage("ggplot2")
      private$ensurePackage("dplyr")
      private$ensurePackage("RColorBrewer")
      private$ensurePackage("magrittr")
      
      # Import the pipe operator explicitly
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
    
    #' @description Create spatial visualization of LISA clusters
    #' @param lisa_column Column containing LISA cluster assignments
    #' @param img_id_column Column containing image identifiers
    #' @param save_path Path to save the visualization
    #' @return ggplot object
    visualizeSpatialLisaClusters = function(
      lisa_column = "lisa_clusters",
      img_id_column = "sample_id",
      save_path = NULL
    ) {
      # Ensure required packages are available
      private$ensurePackage("ggplot2")
      private$ensurePackage("dplyr")
      private$ensurePackage("viridis")
      
      if (!is.null(self$logger)) self$logger$log_info("Creating spatial visualization of LISA clusters")
      
      # Check if lisa clusters exist
      if (!(lisa_column %in% colnames(SummarizedExperiment::colData(self$spe)))) {
        if (!is.null(self$logger)) self$logger$log_warning("LISA cluster column '%s' not found in SPE object", lisa_column)
        return(NULL)
      }
      
      # Extract spatial coordinates
      coords <- SpatialExperiment::spatialCoords(self$spe)
      
      # Create data frame for plotting
      plot_data <- data.frame(
        x = coords[,1],
        y = coords[,2],
        Cluster = self$spe[[lisa_column]]
      )
      
      # If img_id_column exists, add it to the data
      if (img_id_column %in% colnames(SummarizedExperiment::colData(self$spe))) {
        plot_data$ImageID <- self$spe[[img_id_column]]
        
        # Create a faceted plot if multiple images
        if (length(unique(plot_data$ImageID)) > 1) {
          p <- ggplot2::ggplot(plot_data, ggplot2::aes(x = x, y = y, color = Cluster)) +
            ggplot2::geom_point(size = 2, alpha = 0.7) +
            ggplot2::scale_color_viridis_d(option = "D", end = 0.9) +
            ggplot2::labs(
              title = "LISA Clusters - Spatial Distribution",
              subtitle = paste("Number of clusters:", length(unique(plot_data$Cluster)))
            ) +
            ggplot2::facet_wrap(~ImageID, scales = "free") +
            ggplot2::theme_bw() +
            ggplot2::theme(
              panel.background = ggplot2::element_rect(fill = "white"),
              plot.background = ggplot2::element_rect(fill = "white", color = NA)
            )
          
          # Save if requested
          if (!is.null(save_path)) {
            ggplot2::ggsave(save_path, p, width = 15, height = 12, dpi = 300)
            if (!is.null(self$logger)) self$logger$log_info("Saved LISA cluster spatial visualization to %s", save_path)
          }
          
          return(p)
        }
      }
      
      # Default single plot if no faceting
      p <- ggplot2::ggplot(plot_data, ggplot2::aes(x = x, y = y, color = Cluster)) +
        ggplot2::geom_point(size = 2, alpha = 0.7) +
        ggplot2::scale_color_viridis_d(option = "D", end = 0.9) +
        ggplot2::labs(
          title = "LISA Clusters - Spatial Distribution",
          subtitle = paste("Number of clusters:", length(unique(plot_data$Cluster)))
        ) +
        ggplot2::theme_bw() +
        ggplot2::theme(
          panel.background = ggplot2::element_rect(fill = "white"),
          plot.background = ggplot2::element_rect(fill = "white", color = NA)
        )
      
      # Save if requested
      if (!is.null(save_path)) {
        ggplot2::ggsave(save_path, p, width = 12, height = 10, dpi = 300)
        if (!is.null(self$logger)) self$logger$log_info("Saved LISA cluster spatial visualization to %s", save_path)
      }
      
      return(p)
    }
  ),
  
  private = list(
    #' @description Load LISA-specific dependencies
    loadLisaDependencies = function() {
      if (!is.null(self$logger)) self$logger$log_info("Loading LISA clustering dependencies")
      
      # Try to load lisaClust package using BiocManager if missing
      if (!requireNamespace("lisaClust", quietly = TRUE)) {
        if (!is.null(self$logger)) self$logger$log_info("Installing lisaClust package")
        
        # Use dependency manager if available
        if (!is.null(self$dependency_manager)) {
          self$dependency_manager$install_bioc_package("lisaClust")
        } else {
          # Direct installation
          if (!requireNamespace("BiocManager", quietly = TRUE)) {
            install.packages("BiocManager")
          }
          BiocManager::install("lisaClust")
        }
      }
      
      # Also ensure other required packages are available
      private$ensurePackage("BiocParallel")
      private$ensurePackage("ggplot2")
      private$ensurePackage("viridis")
      private$ensurePackage("RColorBrewer")
    },
    
    #' @description Ensure a specific package is installed
    ensurePackage = function(pkg_name) {
      # Use dependency manager if available
      if (!is.null(self$dependency_manager)) {
        return(self$dependency_manager$ensure_package(pkg_name))
      }
      
      # Otherwise install directly
      if (!requireNamespace(pkg_name, quietly = TRUE)) {
        if (!is.null(self$logger)) self$logger$log_info("Installing package: %s", pkg_name)
        install.packages(pkg_name)
      }
      
      return(requireNamespace(pkg_name, quietly = TRUE))
    }
  )
) 