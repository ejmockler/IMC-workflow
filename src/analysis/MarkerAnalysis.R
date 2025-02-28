# Core functionality for marker-based analysis without segmentation

library(cytomapper)
library(ComplexHeatmap)
library(dendextend)
library(viridis)
library(igraph)
# Explicitly load parallel processing packages
library(foreach)
library(doParallel)
library(parallel)

#' MarkerAnalyzer class for segmentation-free marker analysis
#' 
#' Provides methods for various analytical approaches to understand
#' marker relationships directly from pixel data
MarkerAnalyzer <- R6::R6Class("MarkerAnalyzer",
  private = list(
    .images = NULL,
    .pixel_data = NULL,
    .pixel_data_transformed = NULL,
    .marker_names = NULL,
    .n_markers = NULL,
    .output_dir = NULL,
    .pixel_image_index = NULL,
    .pixel_image_names = NULL,
    
    # Helper function to save heatmap with more descriptive names
    .saveHeatmap = function(heatmap_obj, filename, width = 10, height = 8) {
      # Create more descriptive filename with date stamp
      timestamp <- format(Sys.time(), "%Y%m%d")
      descriptive_filename <- paste0(timestamp, "_", filename)
      
      # PDF version
      pdf_file <- file.path(private$.output_dir, paste0(descriptive_filename, ".pdf"))
      pdf(pdf_file, width = width, height = height)
      draw(heatmap_obj)
      dev.off()
      
      # PNG version
      png_file <- file.path(private$.output_dir, paste0(descriptive_filename, ".png"))
      png(png_file, width = 1200, height = 1000, res = 150)
      draw(heatmap_obj)
      dev.off()
      
      return(list(pdf = pdf_file, png = png_file))
    },
    
    # Sample pixels from images, keeping track of source image - OPTIMIZED VERSION
    .samplePixels = function(n_pixels) {
      # First calculate total available pixels across all images
      total_available_pixels <- 0
      image_dimensions <- list()
      
      for (i in seq_along(private$.images)) {
        img <- private$.images[[i]]
        img_dims <- dim(img)
        image_dimensions[[i]] <- img_dims
        total_available_pixels <- total_available_pixels + prod(img_dims[1:2])
      }
      
      message(sprintf("Total available pixels across all images: %d", total_available_pixels))
      
      # Determine how many pixels we can actually use
      n_pixels_to_use <- min(n_pixels, total_available_pixels)
      
      # Provide feedback if requested pixels exceeds available
      if (n_pixels > total_available_pixels) {
        message(sprintf("Warning: Requested %d pixels, but only %d are available. Using all available pixels.", 
                       n_pixels, total_available_pixels))
      }
      
      # Pre-allocate result matrices - significant speedup
      pixel_data <- matrix(0, ncol = private$.n_markers, nrow = n_pixels_to_use)
      colnames(pixel_data) <- private$.marker_names
      pixel_image_index <- integer(n_pixels_to_use)
      pixel_image_names <- character(n_pixels_to_use)
      
      # Counter for filled rows
      current_row <- 1
      
      # Loop through images and sample pixels - with vectorized operations
      for (i in seq_along(private$.images)) {
        # Track progress through images
        message(sprintf("Processing image %d of %d...", i, length(private$.images)))
        
        # Get current image and dimensions
        img <- private$.images[[i]]
        img_dims <- image_dimensions[[i]]
        total_img_pixels <- prod(img_dims[1:2])
        
        # Calculate pixels to sample from this image proportionally
        img_fraction <- total_img_pixels / total_available_pixels
        n_to_sample <- min(ceiling(n_pixels_to_use * img_fraction), total_img_pixels)
        
        # If we're sampling all pixels, provide feedback
        if (n_to_sample == total_img_pixels) {
          message(sprintf("Using all %d pixels from image %d", n_to_sample, i))
        } else {
          message(sprintf("Sampling %d pixels from image %d (%.1f%% of image)", 
                         n_to_sample, i, 100 * n_to_sample / total_img_pixels))
        }
        
        # Calculate end row for this batch
        end_row <- current_row + n_to_sample - 1
        if (end_row > n_pixels_to_use) {
          n_to_sample <- n_pixels_to_use - current_row + 1
          end_row <- n_pixels_to_use
        }
        
        # Sample random pixel positions without replacement
        sampled_indices <- sample(total_img_pixels, n_to_sample)
        
        # Convert to row, col coordinates (vectorized)
        img_width <- img_dims[2]
        row_indices <- ((sampled_indices - 1) %/% img_width) + 1
        col_indices <- ((sampled_indices - 1) %% img_width) + 1
        
        # VECTORIZED APPROACH: Process in chunks of 10,000 pixels at a time
        # This avoids memory issues with very large images
        chunk_size <- 10000
        for (chunk_start in seq(1, n_to_sample, by=chunk_size)) {
          chunk_end <- min(chunk_start + chunk_size - 1, n_to_sample)
          chunk_rows <- chunk_start:chunk_end
          
          # Track progress for large images
          if (n_to_sample > 100000 && chunk_start %% 100000 == 1) {
            message(sprintf("  Processing pixels %d-%d of %d...", 
                           chunk_start, chunk_end, n_to_sample))
          }
          
          # Get row indices for this chunk
          chunk_row_idx <- row_indices[chunk_rows]
          chunk_col_idx <- col_indices[chunk_rows]
          
          # Process each marker channel for this chunk of pixels
          for (k in 1:private$.n_markers) {
            # Extract all pixels for this marker in one operation
            pixel_data[current_row:(current_row + length(chunk_rows) - 1), k] <- 
              img[cbind(chunk_row_idx, chunk_col_idx, rep(k, length(chunk_rows)))]
          }
          
          # Update image indices for this chunk
          pixel_image_index[current_row:(current_row + length(chunk_rows) - 1)] <- i
          pixel_image_names[current_row:(current_row + length(chunk_rows) - 1)] <- names(private$.images)[i]
          
          # Update row counter
          current_row <- current_row + length(chunk_rows)
        }
        
        # Force garbage collection every few images to prevent memory buildup
        if (i %% 5 == 0) {
          gc(verbose = FALSE)
        }
      }
      
      # Store the sampled pixel data and image indices
      private$.pixel_data <- pixel_data
      private$.pixel_image_index <- pixel_image_index
      private$.pixel_image_names <- pixel_image_names
      
      message(sprintf("Sampled %d pixels. Data matrix dimensions: %d rows x %d columns",
                     nrow(pixel_data), nrow(pixel_data), ncol(pixel_data)))
      
      return(invisible(pixel_data))
    },
    
    #' Monitor performance metrics during analysis
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
    initialize = function(images = NULL, input_file = NULL, output_dir = "output", 
                         n_pixels = 100000, transform_data = TRUE) {
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
      
      # Initialize private fields
      private$.images <- images
      private$.output_dir <- output_dir
      
      # Ensure output directory exists
      if (!dir.exists(private$.output_dir)) {
        dir.create(private$.output_dir, recursive = TRUE)
      }
      
      # Extract marker names from the images (CytoImageList has channelNames)
      private$.marker_names <- channelNames(images)
      private$.n_markers <- length(private$.marker_names)
      
      # Log basic image information
      message(sprintf("Initialized MarkerAnalyzer with %d images and %d markers",
                     length(images), private$.n_markers))
      for (i in seq_along(images)) {
        message(sprintf("  Image %d: %s, dimensions: %s", 
                       i, names(images)[i], paste(dim(images[[i]]), collapse="x")))
      }
      
      # Sample pixels from images
      message(sprintf("Sampling %d pixels from images...", n_pixels))
      private$.samplePixels(n_pixels)
      
      # Apply standard transformations if requested
      if (transform_data) {
        self$transformData()
      }
      
      return(invisible(self))
    },
    
    #' Transform pixel data for analysis
    #' 
    #' Applies appropriate transformations to prepare data for analysis
    transformData = function() {
      message("Transforming pixel data...")
      
      # Create a copy of the original data for transformation
      private$.pixel_data_transformed <- private$.pixel_data
      
      # Apply log transformation (with small offset to handle zeros)
      private$.pixel_data_transformed <- log1p(private$.pixel_data_transformed)
      
      # Scale each marker to [0,1] range
      for (i in 1:private$.n_markers) {
        col_min <- min(private$.pixel_data_transformed[, i])
        col_max <- max(private$.pixel_data_transformed[, i])
        
        if (col_max > col_min) {
          private$.pixel_data_transformed[, i] <- 
            (private$.pixel_data_transformed[, i] - col_min) / (col_max - col_min)
        }
      }
      
      message("Data transformation complete")
      return(invisible(self))
    },
    
    #' Run image-aware marker analysis with parallelization
    #' 
    #' This method performs all analytical approaches with image context:
    #' 1. Correlation analysis by image
    #' 2. Co-occurrence analysis by image
    #' 3. Diffusion map analysis by image
    #' 4. Comprehensive pixel heatmap with image annotations
    runImageAwareMarkerAnalysis = function(n_cores = NULL) {
      # Start timing
      analysis_start_time <- Sys.time()
      
      # Ensure required packages are loaded
      if (!requireNamespace("circlize", quietly = TRUE)) {
        install.packages("circlize")
      }
      library(circlize)  # Load circlize for colorRamp2 function
      
      # Memory management - run garbage collection
      gc(verbose = FALSE)
      
      message("Starting comprehensive image-aware marker analysis")
      
      # Determine number of cores to use
      if (is.null(n_cores)) {
        # Use a conservative 30% of available cores to avoid memory issues
        n_cores <- max(1, floor(parallel::detectCores() * 0.3))
      }
      message(sprintf("Using %d cores for parallel processing", n_cores))
      
      # Use transformed pixel data
      pixel_data <- private$.pixel_data_transformed
      marker_names <- private$.marker_names
      n_markers <- private$.n_markers
      
      # Extract these values for parallel processing
      pixel_image_index <- private$.pixel_image_index
      pixel_image_names <- private$.pixel_image_names
      image_names <- names(private$.images)
      
      # Define minimum pixels per image for analysis
      min_pixels_per_image <- 50
      
      # Calculate per-image correlations
      # Get unique image indices and store them for later use
      unique_image_indices <- sort(unique(pixel_image_index))
      
      message(sprintf("Calculating correlations for each of %d images...", length(unique_image_indices)))
      
      # IMPORTANT: Create and register cluster ONLY for the correlation calculation
      # then immediately stop it before moving to the next step
      message("Calculating per-image correlations...")
      
      cl <- parallel::makeCluster(n_cores, type = "PSOCK")
      doParallel::registerDoParallel(cl)
      
      # Explicitly export only what's needed for correlation
      parallel::clusterExport(cl, varlist = c("pixel_data", "pixel_image_index", 
                                          "marker_names", "n_markers", "min_pixels_per_image"), 
                           envir = environment())
                           
      # Run the parallel correlation calculation - FIXED indexing issues
      image_correlations <- parallel::parLapply(cl, unique_image_indices, function(img_idx) {
        # Get pixels from this image
        img_pixels <- which(pixel_image_index == img_idx)
        
        # Skip if too few pixels
        if (length(img_pixels) < min_pixels_per_image) {
          return(NULL)
        }
        
        # Get image data
        img_data <- pixel_data[img_pixels, ]
        
        # Handle potential issues
        if (any(is.na(img_data))) {
          img_data[is.na(img_data)] <- 0
        }
        
        # Calculate correlation matrix for this image
        tryCatch({
          result <- cor(img_data, use = "pairwise.complete.obs")
          # Validate result
          if (any(is.na(result)) || !all(is.finite(result))) {
            return(NULL)
          }
          return(result)
        }, error = function(e) {
          return(NULL)
        })
      })
      
      # IMPORTANT: Stop the cluster as soon as we're done with it
      parallel::stopCluster(cl)
      rm(cl) # Remove the cluster reference
      gc(verbose = FALSE) # Force garbage collection
      
      # Continue with correlation processing...
      
      # Filter out NULL results
      image_correlations <- image_correlations[!sapply(image_correlations, is.null)]
      
      message("Calculating overall correlation...")
      if (length(image_correlations) > 0) {
        # Calculate average correlation across all images
        combined_corr <- matrix(0, nrow = n_markers, ncol = n_markers)
        colnames(combined_corr) <- marker_names
        rownames(combined_corr) <- marker_names
        
        # Better validation before combining correlations
        valid_correlations <- 0
        for (i in 1:length(image_correlations)) {
          corr_matrix <- image_correlations[[i]]
          # Check if the result is a proper matrix with correct dimensions
          if (!is.null(corr_matrix) && is.matrix(corr_matrix) && 
              nrow(corr_matrix) == n_markers && ncol(corr_matrix) == n_markers) {
            combined_corr <- combined_corr + corr_matrix
            valid_correlations <- valid_correlations + 1
          }
        }
        
        # Only divide if we have valid correlations
        if (valid_correlations > 0) {
          avg_corr <- combined_corr / valid_correlations
        } else {
          avg_corr <- matrix(NA, nrow = n_markers, ncol = n_markers)
          rownames(avg_corr) <- marker_names
          colnames(avg_corr) <- marker_names
        }
        
        # Calculate correlation variance across images
        message("Calculating correlation variance across images...")
        
        # Create matrix for variances
        corr_var <- matrix(0, nrow = n_markers, ncol = n_markers)
        colnames(corr_var) <- marker_names
        rownames(corr_var) <- marker_names
        
        # Calculate variance for each cell in the correlation matrix
        for (i in 1:n_markers) {
          for (j in 1:n_markers) {
            # Extract correlation values for this marker pair across images
            corr_values <- sapply(image_correlations, function(x) {
              if (is.null(x)) return(NA)
              if (is.matrix(x) && nrow(x) == n_markers && ncol(x) == n_markers) {
                return(x[i, j])
              } else {
                return(NA)
              }
            })
            
            # Calculate variance (excluding NAs)
            corr_var[i, j] <- var(corr_values, na.rm = TRUE)
          }
        }
        
        # Store results
        correlation_results <- list(
          by_image = image_correlations,
          overall = avg_corr,
          variance = corr_var
        )
      } else {
        message("Warning: No valid correlations calculated. Check image data.")
        correlation_results <- list(
          by_image = list(),
          overall = matrix(NA, nrow = n_markers, ncol = n_markers, 
                         dimnames = list(marker_names, marker_names)),
          variance = matrix(NA, nrow = n_markers, ncol = n_markers, 
                          dimnames = list(marker_names, marker_names))
        )
      }
      
      # Run garbage collection to free memory
      gc(verbose = FALSE)
      
      # Perform hierarchical clustering on overall correlation
      if (all(is.na(correlation_results$overall))) {
        message("Warning: Cannot perform clustering, correlation matrix contains only NA values")
        dend <- NULL
      } else {
        # Remove NA values for clustering if present
        corr_for_clustering <- correlation_results$overall
        corr_for_clustering[is.na(corr_for_clustering)] <- 0  # Replace NA with 0 for clustering
        hc <- hclust(as.dist(1 - corr_for_clustering), method = "ward.D2")
        dend <- as.dendrogram(hc)
      }
      
      # Create a combined heatmap with overall correlation and variance
      if (!is.null(dend) && !all(is.na(correlation_results$overall))) {
        # Create correlation heatmap
        hm_corr <- Heatmap(correlation_results$overall, 
                          name = "Correlation",
                          cluster_rows = dend,
                          cluster_columns = dend,
                          col = viridis(100),
                          column_title = "Overall Marker Correlation")
        
        # Create variance heatmap
        hm_var <- Heatmap(correlation_results$variance,
                          name = "Variance",
                          cluster_rows = dend,
                          cluster_columns = dend,
                          col = colorRamp2(
                            c(0, max(correlation_results$variance)/2, max(correlation_results$variance)),
                            c("blue", "white", "red")
                          ),
                          column_title = "Correlation Variance Across Images")
        
        # Combine heatmaps
        hm_corr_combined <- hm_corr + hm_var
        
        # Save heatmap
        private$.saveHeatmap(hm_corr_combined, "image_aware_marker_correlation", width = 15, height = 8)
        
      } else {
        message("Warning: Cannot create heatmaps due to invalid correlation data")
        hm_corr_combined <- NULL
      }
      
      correlation_end <- Sys.time()
      message(sprintf("Correlation analysis completed in %.2f seconds", 
                     as.numeric(difftime(correlation_end, analysis_start_time, units = "secs"))))
      
      ##########################################
      # PART 2: CO-OCCURRENCE ANALYSIS BY IMAGE
      ##########################################
      cooccur_start_time <- Sys.time()
      message("Running co-occurrence analysis by image...")
      
      # Create new cluster specifically for co-occurrence analysis
      co_cl <- parallel::makeCluster(n_cores, type = "PSOCK")
      on.exit({
        # Ensure cluster is properly closed when function exits
        if(!is.null(co_cl) && inherits(co_cl, "cluster")) {
          parallel::stopCluster(co_cl)
          message("Co-occurrence cluster stopped")
        }
      }, add = TRUE)
      
      doParallel::registerDoParallel(co_cl)
      
      # Export necessary variables for co-occurrence analysis
      parallel::clusterExport(co_cl, varlist = c("pixel_data", "pixel_image_index", 
                                           "marker_names", "n_markers", "min_pixels_per_image"), 
                           envir = environment())
      
      # Binary thresholds for co-occurrence
      message("Calculating marker expression thresholds for co-occurrence...")
      
      # Calculate median for each marker as threshold
      marker_thresholds <- apply(pixel_data, 2, median, na.rm = TRUE)
      
      # Export thresholds to cluster
      parallel::clusterExport(co_cl, "marker_thresholds", envir = environment())
      
      # Calculate co-occurrence
      image_cooccurrences <- parallel::parLapply(co_cl, unique_image_indices, function(img_idx) {
        # Get pixels from this image
        img_pixels <- which(pixel_image_index == img_idx)
        
        # Skip if too few pixels
        if (length(img_pixels) < min_pixels_per_image) {
          return(NULL)
        }
        
        # Get image data
        img_data <- pixel_data[img_pixels, ]
        
        # Handle NAs
        if (any(is.na(img_data))) {
          img_data[is.na(img_data)] <- 0
        }
        
        # Calculate binary presence
        binary_presence <- matrix(FALSE, nrow = nrow(img_data), ncol = ncol(img_data))
        for (m in 1:ncol(img_data)) {
          binary_presence[, m] <- img_data[, m] > marker_thresholds[m]
        }
        
        # Calculate co-occurrence matrix
        cooc_matrix <- matrix(0, nrow = n_markers, ncol = n_markers)
        
        # Loop through each marker pair
        for (i in 1:n_markers) {
          for (j in 1:n_markers) {
            # Count co-occurrence (both markers present)
            cooc_matrix[i, j] <- sum(binary_presence[, i] & binary_presence[, j]) / 
                                nrow(binary_presence)
          }
        }
        
        # Set dimension names
        colnames(cooc_matrix) <- marker_names
        rownames(cooc_matrix) <- marker_names
        
        return(cooc_matrix)
      })
      
      # Close co-occurrence cluster
      parallel::stopCluster(co_cl)
      co_cl <- NULL  # Clear reference
      gc(verbose = FALSE)  # Force garbage collection
      
      # Remove null entries
      image_cooccurrences <- image_cooccurrences[!sapply(image_cooccurrences, is.null)]
      
      # Calculate overall co-occurrence
      overall_cooccurrence <- matrix(0, nrow = n_markers, ncol = n_markers)
      rownames(overall_cooccurrence) <- marker_names
      colnames(overall_cooccurrence) <- marker_names
      
      # Average co-occurrence across images
      for (i in 1:length(image_cooccurrences)) {
        overall_cooccurrence <- overall_cooccurrence + image_cooccurrences[[i]]
      }
      overall_cooccurrence <- overall_cooccurrence / length(image_cooccurrences)
      
      # Calculate variance of co-occurrences across images
      cooccurrence_variance <- matrix(0, nrow=n_markers, ncol=n_markers)
      rownames(cooccurrence_variance) <- marker_names
      colnames(cooccurrence_variance) <- marker_names
      
      for (i in 1:n_markers) {
        for (j in 1:n_markers) {
          cooc_values <- sapply(image_cooccurrences, function(m) m[i,j])
          cooccurrence_variance[i,j] <- var(cooc_values)
        }
      }
      
      # Hierarchical clustering on co-occurrence
      hc_cooc <- hclust(as.dist(1 - overall_cooccurrence), method = "ward.D2")
      dend_cooc <- as.dendrogram(hc_cooc)
      
      # Create heatmap for co-occurrence with variance subplot
      hm_cooc <- Heatmap(overall_cooccurrence, 
                        name = "Co-occurrence",
                        cluster_rows = dend_cooc,
                        cluster_columns = dend_cooc,
                        col = viridis(100),
                        column_title = "Overall Marker Co-occurrence")
      
      # Add variance heatmap for co-occurrence
      hm_cooc_var <- Heatmap(cooccurrence_variance,
                           name = "Variance",
                           cluster_rows = dend_cooc,
                           cluster_columns = dend_cooc,
                           col = colorRamp2(
                             c(0, max(cooccurrence_variance)/2, max(cooccurrence_variance)),
                             c("blue", "white", "red")
                           ),
                           column_title = "Co-occurrence Variance Across Images")
      
      # Combine heatmaps
      hm_cooc_combined <- hm_cooc + hm_cooc_var
      
      # Save heatmap
      private$.saveHeatmap(hm_cooc_combined, "image_aware_marker_cooccurrence", width = 15, height = 8)
      
      cooccur_end_time <- Sys.time()
      message(sprintf("Co-occurrence analysis completed in %.2f seconds", 
                      as.numeric(difftime(cooccur_end_time, cooccur_start_time, units = "secs"))))
      
      ##########################################
      # PART 3: DIFFUSION MAP ANALYSIS BY IMAGE
      ##########################################
      diffusion_start_time <- Sys.time()
      message("Running diffusion map analysis by image...")
      
      # Calculate diffusion maps with optimized parameters
      tryCatch({
        # Create a transpose matrix for dimension reduction across all images
        marker_similarity <- t(pixel_data) %*% pixel_data
        marker_similarity <- marker_similarity / nrow(pixel_data)
        
        # Create distance matrix from similarity
        marker_dist <- as.dist(1 - (marker_similarity / max(marker_similarity)))
        
        # Create heatmap from distance matrix
        hc_diff <- hclust(marker_dist, method = "ward.D2")
        dend_diff <- as.dendrogram(hc_diff)
        diffusion_similarity <- 1 - (as.matrix(marker_dist) / max(marker_dist))
        
        # Visualize
        heatmap_diffusion <- Heatmap(diffusion_similarity, 
                            name = "Diffusion\nSimilarity",
                            cluster_rows = dend_diff,
                            cluster_columns = dend_diff,
                            col = viridis(100),
                            row_title = "Markers",
                            column_title = "Marker Relationships (Diffusion)")
        
        # Save outputs with clearer name
        private$.saveHeatmap(heatmap_diffusion, "diffusion_marker_similarity_heatmap")
        
        # Create network visualization if we have enough markers
        if (n_markers >= 5) {
          # Use adaptive thresholding
          similarity_vals <- diffusion_similarity[lower.tri(diffusion_similarity)]
          network_threshold <- quantile(similarity_vals, 0.7)
          
          g <- graph_from_adjacency_matrix(
            (diffusion_similarity > network_threshold) * diffusion_similarity,
            mode = "undirected",
            weighted = TRUE
          )
          
          # Calculate community structure if we have edges
          if (ecount(g) > 0) {
            comm <- cluster_louvain(g)
            comm_colors <- rainbow(max(membership(comm)))
            V(g)$color <- comm_colors[membership(comm)]
          } else {
            V(g)$color <- "skyblue"
          }
          
          V(g)$size <- 15
          if (ecount(g) > 0) {
            E(g)$width <- E(g)$weight * 5
          }
          
          # If too many edges, keep only the strongest ones for visualization
          if (ecount(g) > 30 && n_markers > 8) {
            weights <- E(g)$weight
            keep_idx <- order(weights, decreasing=TRUE)[1:30]
            g <- subgraph_from_edges(g, keep_idx, delete.vertices=FALSE)
          }
          
          # Save network visualization with improved name
          png(file.path(private$.output_dir, 
              paste0(format(Sys.time(), "%Y%m%d"), "_diffusion_marker_network.png")), 
              width = 1200, height = 1000, res = 150)
          plot(g, 
               vertex.label = V(g)$name,
               vertex.label.cex = 0.8,
               layout = layout_with_fr(g),
               main = "Marker Similarity Network")
          dev.off()
        }
      }, error = function(e) {
        message(sprintf("Error in diffusion analysis: %s", e$message))
      })
      
      diffusion_end_time <- Sys.time()
      message(sprintf("Diffusion analysis completed in %.2f seconds", 
                     as.numeric(difftime(diffusion_end_time, diffusion_start_time, units = "secs"))))
      
      ################################################
      # PART 4: DIRECT PIXEL CLUSTERING AND HEATMAP
      ################################################
      heatmap_start_time <- Sys.time()
      message("Creating pixel-level clustering and heatmap...")
      
      # Sample pixels for clustering (adjust sample size based on memory constraints)
      message("Sampling pixels for direct clustering...")
      max_pixels_for_clustering <- min(100000, nrow(pixel_data))
      if (nrow(pixel_data) > max_pixels_for_clustering) {
        sampled_indices <- sample(1:nrow(pixel_data), max_pixels_for_clustering)
        pixel_subset <- pixel_data[sampled_indices, ]
        pixel_image_subset <- pixel_image_index[sampled_indices]
        pixel_image_names_subset <- pixel_image_names[sampled_indices]
      } else {
        pixel_subset <- pixel_data
        pixel_image_subset <- pixel_image_index
        pixel_image_names_subset <- pixel_image_names
      }
      
      # Normalize data for better clustering (scale to 0-1 range)
      message("Normalizing pixel data...")
      pixel_subset_normalized <- pixel_subset
      
      # Calculate 99th percentile for each marker to avoid outlier influence
      upper_limits <- apply(pixel_subset, 2, function(x) quantile(x, 0.99, na.rm=TRUE))
      
      # Scale each marker to 0-1 range capped at 99th percentile
      for (i in 1:ncol(pixel_subset_normalized)) {
        pixel_subset_normalized[,i] <- pmin(pixel_subset_normalized[,i] / upper_limits[i], 1)
      }
      
      # Apply direct k-means clustering to pixels
      message("Performing k-means clustering directly on pixels...")
      k_clusters <- min(15, ncol(pixel_subset) * 2)  # Adjust number of clusters based on markers
      
      # Try using a more memory-efficient implementation if available
      if (requireNamespace("ClusterR", quietly = TRUE)) {
        message("Using ClusterR for more efficient k-means...")
        pixel_km <- ClusterR::KMeans_rcpp(pixel_subset_normalized, k_clusters, num_init = 5, max_iters = 100)
        pixel_clusters <- pixel_km$clusters
      } else {
        pixel_km <- kmeans(pixel_subset_normalized, centers = k_clusters, iter.max = 50, nstart = 5)
        pixel_clusters <- pixel_km$cluster
      }
      
      # Calculate cluster profiles (mean expression of each marker in each cluster)
      message("Calculating cluster profiles...")
      cluster_profiles <- matrix(0, nrow = k_clusters, ncol = ncol(pixel_subset))
      colnames(cluster_profiles) <- colnames(pixel_subset)
      rownames(cluster_profiles) <- paste0("Cluster", 1:k_clusters)
      
      for (i in 1:k_clusters) {
        if (sum(pixel_clusters == i) > 0) {
          cluster_profiles[i,] <- colMeans(pixel_subset_normalized[pixel_clusters == i,, drop=FALSE])
        }
      }
      
      # Determine top markers for each cluster for better labeling
      top_cluster_markers <- apply(cluster_profiles, 1, function(x) {
        ordered_markers <- names(sort(x, decreasing=TRUE))
        top_markers <- ordered_markers[1:min(3, length(ordered_markers))]
        return(paste(top_markers, collapse="+"))
      })
      
      # Create more informative row labels
      cluster_labels <- paste0("Cluster ", 1:k_clusters, " (", top_cluster_markers, ")")
      rownames(cluster_profiles) <- cluster_labels
      
      # Get marker clustering for visualization
      marker_dist <- dist(t(cluster_profiles))
      marker_hc <- hclust(marker_dist, method="ward.D2")
      marker_dend_viz <- as.dendrogram(marker_hc)
      
      # Create a heatmap of cluster profiles
      message("Creating cluster profile heatmap...")
      expression_colors <- colorRampPalette(c("#F7FBFF", "#DEEBF7", "#C6DBEF", "#9ECAE1", 
                                             "#6BAED6", "#4292C6", "#2171B5", "#08519C", "#08306B"))(100)
      
      cluster_profile_hm <- Heatmap(
        cluster_profiles,
        name = "Mean\nExpression",
        col = expression_colors,
        cluster_rows = TRUE,
        cluster_columns = marker_dend_viz,
        column_title = "Pixel Cluster Profiles",
        show_row_names = TRUE,
        show_column_names = TRUE,
        column_names_rot = 45,
        row_names_side = "left",
        row_names_gp = gpar(fontsize = 10),
        heatmap_legend_param = list(
          title = "Expression",
          at = c(0, 0.25, 0.5, 0.75, 1), 
          labels = c("Low", "", "Medium", "", "High"),
          legend_height = unit(4, "cm")
        )
      )
      
      # Save the cluster profile heatmap
      private$.saveHeatmap(cluster_profile_hm, "pixel_cluster_profile_heatmap", width = 12, height = 8)
      
      # Calculate what percentage of pixels from each image falls into each cluster
      message("Analyzing cluster distribution across images...")
      unique_images <- unique(pixel_image_names_subset)
      image_cluster_dist <- matrix(0, nrow = length(unique_images), ncol = k_clusters)
      rownames(image_cluster_dist) <- unique_images
      colnames(image_cluster_dist) <- paste0("Cluster", 1:k_clusters)
      
      for (img in unique_images) {
        img_pixels <- which(pixel_image_names_subset == img)
        if (length(img_pixels) > 0) {
          for (k in 1:k_clusters) {
            image_cluster_dist[img, k] <- sum(pixel_clusters[img_pixels] == k) / length(img_pixels)
          }
        }
      }
      
      # Create heatmap showing cluster distribution across images
      image_cluster_hm <- Heatmap(
        image_cluster_dist,
        name = "Percentage",
        col = colorRampPalette(c("white", "red"))(100),
        cluster_rows = TRUE,
        cluster_columns = FALSE,
        column_title = "Cluster Distribution Across Images",
        row_title = "Images",
        column_title_side = "top",
        row_names_gp = gpar(fontsize = 8),
        column_names_gp = gpar(fontsize = 8),
        column_names_rot = 45
      )
      
      # Save image cluster distribution heatmap
      private$.saveHeatmap(image_cluster_hm, "image_cluster_distribution", width = 14, height = 10)
      
      # Sample pixels for heatmap visualization (if too many clusters/pixels)
      message("Creating sample heatmap of pixels...")
      max_pixels_for_viz <- min(10000, length(pixel_clusters))
      if (length(pixel_clusters) > max_pixels_for_viz) {
        # Sample evenly from each cluster
        sampled_viz_indices <- unlist(lapply(1:k_clusters, function(k) {
          cluster_pixels <- which(pixel_clusters == k)
          if (length(cluster_pixels) > 0) {
            sample(cluster_pixels, min(ceiling(max_pixels_for_viz/k_clusters), length(cluster_pixels)))
          } else {
            integer(0)
          }
        }))
      } else {
        sampled_viz_indices <- 1:length(pixel_clusters)
      }
      
      # Sort by cluster for visualization
      viz_order <- order(pixel_clusters[sampled_viz_indices])
      viz_indices <- sampled_viz_indices[viz_order]
      
      # Create annotation for image source
      img_colors <- setNames(
        rainbow(length(unique(pixel_image_names_subset))),
        unique(pixel_image_names_subset)
      )
      
      row_anno <- rowAnnotation(
        image = pixel_image_names_subset[viz_indices],
        col = list(image = img_colors),
        show_annotation_name = TRUE,
        annotation_legend_param = list(image = list(title = "Image")),
        annotation_name_gp = gpar(fontsize = 10)
      )
      
      # Create pixel heatmap
      pixel_hm <- Heatmap(
        as.matrix(pixel_subset_normalized[viz_indices,]),
        name = "Expression",
        col = expression_colors,
        cluster_rows = FALSE,
        cluster_columns = marker_dend_viz,
        show_row_names = FALSE,
        show_column_names = TRUE,
        column_names_rot = 45,
        column_title = "Pixel-level Expression",
        row_split = factor(pixel_clusters[viz_indices]),
        right_annotation = row_anno
      )
      
      # Save the pixel heatmap
      private$.saveHeatmap(pixel_hm, "pixel_level_heatmap", width = 14, height = 12)
      
      heatmap_end_time <- Sys.time()
      message(sprintf("Pixel clustering and heatmaps created in %.2f seconds", 
                    as.numeric(difftime(heatmap_end_time, heatmap_start_time, units = "secs"))))
      
      # Return the whole set of results
      return(list(
        correlation = list(
          overall = if(exists("correlation_results") && !is.null(correlation_results$overall)) 
                   correlation_results$overall else NULL,
          variance = if(exists("correlation_results") && !is.null(correlation_results$variance)) 
                   correlation_results$variance else NULL,
          by_image = if(exists("image_correlations")) image_correlations else list(),
          dendrogram = dend,
          heatmap = hm_corr_combined
        ),
        cooccurrence = list(
          overall = overall_cooccurrence,
          variance = cooccurrence_variance,
          by_image = image_cooccurrences,
          dendrogram = dend_cooc,
          heatmap = hm_cooc_combined
        ),
        diffusion = list(
          similarity = if(exists("diffusion_similarity")) diffusion_similarity else NULL,
          dendrogram = if(exists("dend_diff")) dend_diff else NULL,
          heatmap = if(exists("heatmap_diffusion")) heatmap_diffusion else NULL
        ),
        pixel_clusters = list(
          cluster_profiles = cluster_profiles,
          cluster_heatmap = cluster_profile_hm,
          image_distribution = image_cluster_dist,
          image_heatmap = image_cluster_hm,
          pixel_heatmap = pixel_hm,
          cluster_assignments = setNames(pixel_clusters, rownames(pixel_subset))
        )
      ))
    },
    
    #' Get the sampled pixel data
    getPixelData = function(transformed = TRUE) {
      if (transformed) {
        return(private$.pixel_data_transformed)
      } else {
        return(private$.pixel_data)
      }
    },
    
    #' Get the marker names
    getMarkerNames = function() {
      return(private$.marker_names)
    },
    
    #' Create a comprehensive pixel-level heatmap visualization similar to phenograph
    #' 
    #' @param n_pixels Number of pixels to sample for visualization (default: 5000)
    #' @return A ComplexHeatmap object
    createComprehensiveHeatmap = function(n_pixels = 5000) {
      # This is now included in the runImageAwareMarkerAnalysis method
      message("This method is deprecated. Use runImageAwareMarkerAnalysis instead.")
      # Redirect to the comprehensive analysis
      results <- self$runImageAwareMarkerAnalysis()
      return(results$pixel_heatmap)
    },
    
    #' Add visual context to marker analysis results
    #' 
    #' This method adds image thumbnails and spatial context to analysis results
    addVisualContext = function() {
      message("Adding visual context to marker analysis...")
      
      # Create a directory for image context visualizations
      context_dir <- file.path(private$.output_dir, "image_context")
      if (!dir.exists(context_dir)) {
        dir.create(context_dir, recursive = TRUE)
      }
      
      # Loop through each image and create visual context
      for (img_idx in seq_along(private$.images)) {
        img <- private$.images[[img_idx]]
        img_name <- names(private$.images)[img_idx]
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
          if (max(channel_data) > 0) {
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
                  main = paste(private$.marker_names[ch]))
          } else {
            # Handle case where channel is constant
            image(channel_data, col = "gray", axes = FALSE, 
                  main = paste(private$.marker_names[ch]))
          }
        }
        dev.off()
        
        # Create a context heatmap for this image
        image_pixels <- which(private$.pixel_image_index == img_idx)
        if (length(image_pixels) > 0) {
          # Sample a subset of pixels if there are too many
          if (length(image_pixels) > 5000) {
            sampled_pixels <- sample(image_pixels, 5000)
          } else {
            sampled_pixels <- image_pixels
          }
          
          pixel_subset <- private$.pixel_data_transformed[sampled_pixels, ]
          
          # Calculate positions in the original image
          # This requires additional information storage - not available
          # in current implementation
          
          # Create heatmap with marker expression
          context_heatmap_file <- file.path(context_dir, paste0(img_name, "_context_heatmap.png"))
          png(context_heatmap_file, width = 1000, height = 600)
          Heatmap(as.matrix(pixel_subset),
                 name = "Expression",
                 col = viridis(100),
                 show_row_names = FALSE,
                 column_names_rot = 45,
                 column_title = paste("Marker Expression in", img_name))
          dev.off()
        }
      }
      
      # Create a combined dashboard
      message("Creating spatial context dashboard...")
      dashboard_file <- file.path(private$.output_dir, 
                                paste0(format(Sys.time(), "%Y%m%d"), "_spatial_context_dashboard.html"))
      
      # Use htmlwidgets or similar to create a dashboard
      # This would require additional packages
      
      message("Visual context added. See the 'image_context' directory for results.")
      return(invisible(context_dir))
    },
    
    #' Factory method to create MarkerAnalyzer from a file
    #' 
    #' @param input_file Path to a saved CytoImageList RDS file
    #' @param output_dir Output directory for results
    #' @param n_pixels Number of pixels to sample
    #' @param transform_data Whether to automatically transform the data
    #' @return A new MarkerAnalyzer instance
    create_from_file = function(input_file, output_dir = "output", n_pixels = 100000, transform_data = TRUE) {
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
      
      # Create the analyzer
      analyzer <- MarkerAnalyzer$new(images = images, output_dir = output_dir, pixel_sample_size = n_pixels)
      
      # Transform data if requested
      if (transform_data) {
        analyzer$transformData()
      }
      
      return(analyzer)
    }
  )
) 