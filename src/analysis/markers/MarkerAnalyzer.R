# MarkerAnalyzer integrates all marker analysis components
# Main entry point for segmentation-free marker analysis

library(cytomapper)
library(R6)

# Load all component files
source("src/analysis/markers/PixelSampler.R")
source("src/analysis/markers/DataTransformer.R")
source("src/analysis/markers/CorrelationAnalyzer.R")
source("src/analysis/markers/CooccurrenceAnalyzer.R")
source("src/analysis/markers/DiffusionAnalyzer.R")
source("src/analysis/markers/PixelClusterAnalyzer.R")
source("src/analysis/markers/VisualizationManager.R")
source("src/analysis/markers/MarkerAnalysisUtils.R")

#' MarkerAnalyzer class for segmentation-free marker analysis
#' 
#' Integrates multiple analytical approaches to understand
#' marker relationships directly from pixel data
MarkerAnalyzer <- R6::R6Class("MarkerAnalyzer",
  private = list(
    .images = NULL,
    .output_dir = NULL,
    .logger = NULL,
    .dependency_manager = NULL,
    
    # Component instances
    .pixel_sampler = NULL,
    .data_transformer = NULL,
    .correlation_analyzer = NULL,
    .cooccurrence_analyzer = NULL,
    .diffusion_analyzer = NULL,
    .pixel_cluster_analyzer = NULL,
    .visualization_manager = NULL,
    
    # Helper method to track performance
    .trackPerformance = function(start_time, stage_name) {
      current_time <- Sys.time()
      elapsed <- as.numeric(difftime(current_time, start_time, units = "secs"))
      mem_used <- utils::memory.size()
      message(sprintf("[PERF] %s completed in %.2f seconds, memory usage: %.2f MB", 
                    stage_name, elapsed, mem_used))
      return(current_time)
    }
  ),
  
  public = list(
    #' Initialize a MarkerAnalyzer with images or a file path
    #' 
    #' @param images A CytoImageList object from cytomapper containing images (optional)
    #' @param input_file Path to a saved CytoImageList RDS file (optional)
    #' @param output_dir Output directory for results
    #' @param n_pixels Number of pixels to sample from each image
    #' @param transform_data Whether to automatically transform the data
    #' @param transform_method Method for data transformation
    #' @param logger Optional logger object
    #' @param dependency_manager Optional dependency manager
    initialize = function(
      images = NULL, 
      input_file = NULL, 
      output_dir = "output", 
      n_pixels = 100000, 
      transform_data = TRUE,
      transform_method = "standard",
      logger = NULL,
      dependency_manager = NULL
    ) {
      start_time <- Sys.time()
      
      # Store logger and dependency manager
      private$.logger <- logger
      private$.dependency_manager <- dependency_manager
      private$.output_dir <- output_dir
      
      # Ensure output directory exists
      if (!dir.exists(private$.output_dir)) {
        dir.create(private$.output_dir, recursive = TRUE)
      }
      
      # Prioritize images if provided, otherwise load from file
      if (is.null(images) && !is.null(input_file)) {
        # Load the CytoImageList from file
        if (!file.exists(input_file)) {
          stop(sprintf("Input file does not exist: %s", input_file))
        }
        
        message(sprintf("Loading CytoImageList from %s", input_file))
        images <- readRDS(input_file)
        
        # Validate that it's a CytoImageList
        if (!inherits(images, "CytoImageList")) {
          stop("Input file must contain a CytoImageList object")
        }
      } else if (is.null(images)) {
        stop("Either 'images' or 'input_file' must be provided")
      }
      
      # Store the images
      private$.images <- images
      
      # Initialize components
      message("Initializing analysis components...")
      
      # 1. Create the pixel sampler
      private$.pixel_sampler <- PixelSampler$new(images)
      
      # 2. Sample pixels
      private$.pixel_sampler$samplePixels(n_pixels)
      
      # 3. Create other components
      private$.data_transformer <- DataTransformer$new(logger)
      private$.correlation_analyzer <- CorrelationAnalyzer$new(logger)
      private$.cooccurrence_analyzer <- CooccurrenceAnalyzer$new(logger)
      private$.diffusion_analyzer <- DiffusionAnalyzer$new(logger)
      private$.pixel_cluster_analyzer <- PixelClusterAnalyzer$new(logger)
      private$.visualization_manager <- VisualizationManager$new(output_dir, logger)
      
      # 4. Apply standard transformations if requested
      if (transform_data) {
        message(sprintf("Applying %s data transformation...", transform_method))
        pixel_data <- private$.pixel_sampler$getPixelData(transformed = FALSE)
        transformed_data <- private$.data_transformer$transform(pixel_data, method = transform_method)
        private$.pixel_sampler$setTransformedData(transformed_data)
      }
      
      message("MarkerAnalyzer initialized successfully")
      private$.trackPerformance(start_time, "Initialization")
      
      return(invisible(self))
    },
    
    #' Run comprehensive image-aware marker analysis
    #' 
    #' This method performs all analytical approaches:
    #' 1. Correlation analysis
    #' 2. Co-occurrence analysis
    #' 3. Diffusion map analysis
    #' 4. Pixel clustering and profiling
    #' 
    #' @param n_cores Number of cores to use for parallel processing
    #' @param transform_method Method for data transformation if not already transformed
    #' @param threshold_method Method for co-occurrence thresholding
    #' @param k_clusters Number of clusters for pixel clustering (NULL for automatic)
    #' @param visualize Whether to create visualizations
    #' @return List with comprehensive analysis results
    runImageAwareMarkerAnalysis = function(
      n_cores = NULL,
      transform_method = "standard",
      threshold_method = "median",
      k_clusters = NULL,
      visualize = TRUE
    ) {
      start_time <- Sys.time()
      
      # Ensure data is transformed
      pixel_data <- private$.pixel_sampler$getPixelData(transformed = TRUE)
      if (is.null(pixel_data)) {
        message(sprintf("Data not yet transformed. Applying %s transformation...", transform_method))
        raw_data <- private$.pixel_sampler$getPixelData(transformed = FALSE)
        pixel_data <- private$.data_transformer$transform(raw_data, method = transform_method)
        private$.pixel_sampler$setTransformedData(pixel_data)
      }
      
      # Get image indices and names
      pixel_image_index <- private$.pixel_sampler$getPixelImageIndices()
      pixel_image_names <- private$.pixel_sampler$getPixelImageNames()
      marker_names <- private$.pixel_sampler$getMarkerNames()
      
      # Determine number of cores to use if not specified
      if (is.null(n_cores)) {
        n_cores <- MarkerUtils$calculate_cores(max_percentage = 0.3)
      }
      
      message(sprintf("Running comprehensive image-aware marker analysis using %d cores...", n_cores))
      
      # Step 1: Correlation Analysis
      message("\n=== STEP 1: CORRELATION ANALYSIS ===")
      correlation_results <- private$.correlation_analyzer$analyze(
        pixel_data = pixel_data,
        pixel_image_index = pixel_image_index,
        marker_names = marker_names,
        n_cores = n_cores
      )
      
      # Step 2: Co-occurrence Analysis
      message("\n=== STEP 2: CO-OCCURRENCE ANALYSIS ===")
      cooccurrence_results <- private$.cooccurrence_analyzer$analyze(
        pixel_data = pixel_data,
        pixel_image_index = pixel_image_index,
        marker_names = marker_names,
        threshold_method = threshold_method,
        n_cores = n_cores
      )
      
      # Step 3: Diffusion Map Analysis
      message("\n=== STEP 3: DIFFUSION MAP ANALYSIS ===")
      diffusion_results <- private$.diffusion_analyzer$analyze(
        pixel_data = pixel_data,
        marker_names = marker_names
      )
      
      # Step 4: Pixel Clustering
      message("\n=== STEP 4: PIXEL CLUSTERING ===")
      pixel_cluster_results <- private$.pixel_cluster_analyzer$analyze(
        pixel_data = pixel_data,
        pixel_image_names = pixel_image_names,
        marker_names = marker_names,
        k = k_clusters
      )
      
      # Create visualizations if requested
      if (visualize) {
        message("\n=== CREATING VISUALIZATIONS ===")
        
        # Correlation heatmap
        correlation_heatmap <- private$.visualization_manager$createCorrelationHeatmap(
          correlation_results = correlation_results
        )
        
        # Co-occurrence heatmap
        cooccurrence_heatmap <- private$.visualization_manager$createCooccurrenceHeatmap(
          cooccurrence_results = cooccurrence_results
        )
        
        # Diffusion map visualization
        diffusion_heatmap <- private$.visualization_manager$createDiffusionVisualization(
          diffusion_results = diffusion_results
        )
        
        # Marker network visualization
        network_viz <- private$.visualization_manager$createNetworkVisualization(
          similarity_matrix = diffusion_results$similarity
        )
        
        # Cluster profile heatmap
        cluster_profile_heatmap <- private$.visualization_manager$createClusterProfileHeatmap(
          cluster_profiles = pixel_cluster_results$cluster_profiles
        )
        
        # Cluster distribution heatmap
        cluster_dist_heatmap <- private$.visualization_manager$createClusterDistributionHeatmap(
          image_cluster_dist = pixel_cluster_results$image_distribution
        )
        
        # Pixel-level heatmap
        pixel_heatmap <- private$.visualization_manager$createPixelHeatmap(
          pixel_data = pixel_data[pixel_cluster_results$sample_indices, ],
          pixel_clusters = pixel_cluster_results$cluster_assignments,
          pixel_image_names = pixel_image_names[pixel_cluster_results$sample_indices],
          marker_names = marker_names
        )
      }
      
      # Assemble final results
      results <- list(
        correlation = correlation_results,
        cooccurrence = cooccurrence_results,
        diffusion = diffusion_results,
        pixel_clusters = pixel_cluster_results
      )
      
      # Add visualization results if created
      if (visualize) {
        results$visualizations <- list(
          correlation_heatmap = if(exists("correlation_heatmap")) correlation_heatmap else NULL,
          cooccurrence_heatmap = if(exists("cooccurrence_heatmap")) cooccurrence_heatmap else NULL,
          diffusion_heatmap = if(exists("diffusion_heatmap")) diffusion_heatmap else NULL,
          network_viz = if(exists("network_viz")) network_viz else NULL,
          cluster_profile_heatmap = if(exists("cluster_profile_heatmap")) cluster_profile_heatmap else NULL,
          cluster_dist_heatmap = if(exists("cluster_dist_heatmap")) cluster_dist_heatmap else NULL,
          pixel_heatmap = if(exists("pixel_heatmap")) pixel_heatmap else NULL
        )
      }
      
      # Save the complete results to an RDS file
      results_file <- file.path(private$.output_dir, 
                              paste0(format(Sys.time(), "%Y%m%d"), "_marker_analysis_results.rds"))
      saveRDS(results, results_file)
      message(sprintf("Complete analysis results saved to %s", results_file))
      
      total_time <- MarkerUtils$execution_time(start_time)
      message(sprintf("Total analysis time: %.2f seconds", total_time))
      
      return(results)
    },
    
    #' Apply data transformation to pixel data
    #' 
    #' @param method Transformation method
    #' @param ... Additional parameters for the transformation
    #' @return Self (for method chaining)
    transformData = function(method = "standard", ...) {
      pixel_data <- private$.pixel_sampler$getPixelData(transformed = FALSE)
      transformed_data <- private$.data_transformer$transform(pixel_data, method = method, ...)
      private$.pixel_sampler$setTransformedData(transformed_data)
      return(invisible(self))
    },
    
    #' Add visual context to marker analysis results
    #' 
    #' This method adds image thumbnails and spatial context to analysis results
    #' @param output_subdir Subdirectory for visual context output
    #' @return Path to the context directory
    addVisualContext = function(output_subdir = "image_context") {
      # Create a directory for image context visualizations
      context_dir <- file.path(private$.output_dir, output_subdir)
      if (!dir.exists(context_dir)) {
        dir.create(context_dir, recursive = TRUE)
      }
      
      # Access the original images
      images <- private$.images
      
      # Loop through each image and create visual context
      for (img_idx in seq_along(images)) {
        img <- images[[img_idx]]
        img_name <- names(images)[img_idx]
        message(sprintf("  Processing image context for %s", img_name))
        
        # Create RGB projection for visual reference
        # Use first 3 markers or create a meaningful combination
        channels_to_use <- min(3, dim(img)[3])
        rgb_image <- array(0, dim = c(dim(img)[1:2], 3))
        
        # Create a meaningful RGB projection
        for (i in 1:channels_to_use) {
          channel_idx <- i
          # Normalize the channel
          channel_data <- img[,,channel_idx]
          if (max(channel_data) > min(channel_data)) {
            normalized <- (channel_data - min(channel_data)) / (max(channel_data) - min(channel_data))
            rgb_image[,,i] <- normalized
          }
        }
        
        # Save RGB projection
        rgb_file <- file.path(context_dir, paste0(img_name, "_rgb_projection.png"))
        png(rgb_file, width = 800, height = 800)
        par(mar = c(0.5, 0.5, 2, 0.5))
        image(rgb_image[,,1], col = hcl.colors(100, "Blues"), axes = FALSE, main = paste(img_name, "- Channel 1"))
        dev.off()
        
        # Create a multi-panel visualization showing each marker channel
        channel_file <- file.path(context_dir, paste0(img_name, "_channel_gallery.png"))
        n_markers <- dim(img)[3]
        
        # Calculate grid dimensions
        ncols <- min(4, n_markers)
        nrows <- ceiling(n_markers / ncols)
        
        png(channel_file, width = 250 * ncols, height = 250 * nrows)
        par(mfrow = c(nrows, ncols), mar = c(1, 1, 2, 1))
        
        for (ch in 1:n_markers) {
          channel_data <- img[,,ch]
          # Normalize for visualization
          if (max(channel_data) > min(channel_data)) {
            normalized <- (channel_data - min(channel_data)) / (max(channel_data) - min(channel_data))
            image(normalized, col = viridis(100), axes = FALSE, 
                 main = paste(channelNames(images)[ch]))
          } else {
            # Handle case where channel is constant
            image(channel_data, col = "gray", axes = FALSE, 
                 main = paste(channelNames(images)[ch]))
          }
        }
        dev.off()
      }
      
      message("Visual context added. See the context directory for results.")
      return(invisible(context_dir))
    },
    
    #' Get the sampled pixel data
    #' 
    #' @param transformed Whether to return transformed data
    #' @return Matrix of pixel data
    getPixelData = function(transformed = TRUE) {
      return(private$.pixel_sampler$getPixelData(transformed))
    },
    
    #' Get the marker names
    #' 
    #' @return Character vector of marker names
    getMarkerNames = function() {
      return(private$.pixel_sampler$getMarkerNames())
    },
    
    #' Get component instances for direct access
    #' 
    #' @return List of component instances
    getComponents = function() {
      return(list(
        pixel_sampler = private$.pixel_sampler,
        data_transformer = private$.data_transformer,
        correlation_analyzer = private$.correlation_analyzer,
        cooccurrence_analyzer = private$.cooccurrence_analyzer,
        diffusion_analyzer = private$.diffusion_analyzer,
        pixel_cluster_analyzer = private$.pixel_cluster_analyzer,
        visualization_manager = private$.visualization_manager
      ))
    },
    
    #' Factory method to create MarkerAnalyzer from a file
    #' 
    #' @param input_file Path to a saved CytoImageList RDS file
    #' @param output_dir Output directory for results
    #' @param n_pixels Number of pixels to sample
    #' @param transform_data Whether to automatically transform the data
    #' @return A new MarkerAnalyzer instance
    create_from_file = function(input_file, output_dir = "output", 
                               n_pixels = 100000, transform_data = TRUE) {
      return(MarkerAnalyzer$new(
        input_file = input_file,
        output_dir = output_dir,
        n_pixels = n_pixels,
        transform_data = transform_data
      ))
    }
  )
)
