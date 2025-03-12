# Visualization manager for marker analysis
# Handles creating and saving visualizations

library(ComplexHeatmap)
library(dendextend)
library(viridis)
library(circlize)

#' VisualizationManager class
#' 
#' Creates and saves visualizations for marker analysis results
VisualizationManager <- R6::R6Class("VisualizationManager",
  private = list(
    .output_dir = NULL,
    .logger = NULL,
    
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
    #' Initialize a new VisualizationManager
    #' 
    #' @param output_dir Directory to save visualizations
    #' @param logger Optional logger object
    initialize = function(output_dir = "output", logger = NULL) {
      private$.output_dir <- output_dir
      private$.logger <- logger
      
      # Ensure output directory exists
      if (!dir.exists(private$.output_dir)) {
        dir.create(private$.output_dir, recursive = TRUE)
      }
      
      return(invisible(self))
    },
    
    #' Create a correlation heatmap with variance subplot
    #' 
    #' @param correlation_results List with correlation analysis results
    #' @param save Logical indicating whether to save the heatmap
    #' @return Heatmap object
    createCorrelationHeatmap = function(correlation_results, save = TRUE) {
      start_time <- Sys.time()
      message("Creating correlation heatmap...")
      
      # Extract data from results
      overall_corr <- correlation_results$overall
      corr_var <- correlation_results$variance
      
      # Perform hierarchical clustering on overall correlation
      if (all(is.na(overall_corr))) {
        message("Warning: Cannot perform clustering, correlation matrix contains only NA values")
        return(NULL)
      }
      
      # Remove NA values for clustering if present
      corr_for_clustering <- overall_corr
      corr_for_clustering[is.na(corr_for_clustering)] <- 0  # Replace NA with 0 for clustering
      hc <- hclust(as.dist(1 - corr_for_clustering), method = "ward.D2")
      dend <- as.dendrogram(hc)
      
      # Create correlation heatmap
      hm_corr <- Heatmap(overall_corr, 
                       name = "Correlation",
                       cluster_rows = dend,
                       cluster_columns = dend,
                       col = viridis(100),
                       column_title = "Overall Marker Correlation")
      
      # Create variance heatmap
      hm_var <- Heatmap(corr_var,
                       name = "Variance",
                       cluster_rows = dend,
                       cluster_columns = dend,
                       col = colorRamp2(
                         c(0, max(corr_var, na.rm = TRUE)/2, max(corr_var, na.rm = TRUE)),
                         c("blue", "white", "red")
                       ),
                       column_title = "Correlation Variance Across Images")
      
      # Combine heatmaps
      hm_corr_combined <- hm_corr + hm_var
      
      # Save heatmap if requested
      if (save) {
        private$.saveHeatmap(hm_corr_combined, "marker_correlation", width = 15, height = 8)
      }
      
      private$.trackPerformance(start_time, "Correlation heatmap creation")
      return(hm_corr_combined)
    },
    
    #' Create a co-occurrence heatmap with variance subplot
    #' 
    #' @param cooccurrence_results List with co-occurrence analysis results
    #' @param save Logical indicating whether to save the heatmap
    #' @return Heatmap object
    createCooccurrenceHeatmap = function(cooccurrence_results, save = TRUE) {
      start_time <- Sys.time()
      message("Creating co-occurrence heatmap...")
      
      # Extract data from results
      overall_cooc <- cooccurrence_results$overall
      cooc_var <- cooccurrence_results$variance
      
      # Perform hierarchical clustering on co-occurrence
      if (all(is.na(overall_cooc))) {
        message("Warning: Cannot perform clustering, co-occurrence matrix contains only NA values")
        return(NULL)
      }
      
      # Remove NA values for clustering if present
      cooc_for_clustering <- overall_cooc
      cooc_for_clustering[is.na(cooc_for_clustering)] <- 0  # Replace NA with 0 for clustering
      hc_cooc <- hclust(as.dist(1 - cooc_for_clustering), method = "ward.D2")
      dend_cooc <- as.dendrogram(hc_cooc)
      
      # Create co-occurrence heatmap
      hm_cooc <- Heatmap(overall_cooc, 
                       name = "Co-occurrence",
                       cluster_rows = dend_cooc,
                       cluster_columns = dend_cooc,
                       col = viridis(100),
                       column_title = "Overall Marker Co-occurrence")
      
      # Create variance heatmap
      hm_cooc_var <- Heatmap(cooc_var,
                           name = "Variance",
                           cluster_rows = dend_cooc,
                           cluster_columns = dend_cooc,
                           col = colorRamp2(
                             c(0, max(cooc_var, na.rm = TRUE)/2, max(cooc_var, na.rm = TRUE)),
                             c("blue", "white", "red")
                           ),
                           column_title = "Co-occurrence Variance Across Images")
      
      # Combine heatmaps
      hm_cooc_combined <- hm_cooc + hm_cooc_var
      
      # Save heatmap if requested
      if (save) {
        private$.saveHeatmap(hm_cooc_combined, "marker_cooccurrence", width = 15, height = 8)
      }
      
      private$.trackPerformance(start_time, "Co-occurrence heatmap creation")
      return(hm_cooc_combined)
    },
    
    #' Create a diffusion map visualization
    #' 
    #' @param diffusion_results List with diffusion analysis results
    #' @param save Logical indicating whether to save the visualization
    #' @return Heatmap object
    createDiffusionVisualization = function(diffusion_results, save = TRUE) {
      start_time <- Sys.time()
      message("Creating diffusion map visualization...")
      
      # Extract data from results
      diffusion_similarity <- diffusion_results$similarity
      dend_diff <- diffusion_results$dendrogram
      
      # Create heatmap
      heatmap_diffusion <- Heatmap(diffusion_similarity, 
                                  name = "Diffusion\nSimilarity",
                                  cluster_rows = dend_diff,
                                  cluster_columns = dend_diff,
                                  col = viridis(100),
                                  row_title = "Markers",
                                  column_title = "Marker Relationships (Diffusion)")
      
      # Save heatmap if requested
      if (save) {
        private$.saveHeatmap(heatmap_diffusion, "diffusion_marker_similarity", width = 12, height = 10)
      }
      
      private$.trackPerformance(start_time, "Diffusion visualization")
      return(heatmap_diffusion)
    },
    
    #' Create a network visualization of marker relationships
    #' 
    #' @param similarity_matrix Matrix of marker similarities
    #' @param threshold Threshold for including edges in the network
    #' @param save Logical indicating whether to save the network
    #' @return Network visualization file path
    createNetworkVisualization = function(similarity_matrix, threshold = NULL, save = TRUE) {
      start_time <- Sys.time()
      message("Creating network visualization...")
      
      if (!requireNamespace("igraph", quietly = TRUE)) {
        install.packages("igraph")
      }
      
      require(igraph)
      
      # Use adaptive thresholding if threshold is not provided
      if (is.null(threshold)) {
        similarity_vals <- similarity_matrix[lower.tri(similarity_matrix)]
        threshold <- quantile(similarity_vals, 0.7, na.rm = TRUE)
      }
      
      # Create graph
      g <- graph_from_adjacency_matrix(
        (similarity_matrix > threshold) * similarity_matrix,
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
      
      # Set node size and edge width
      V(g)$size <- 15
      if (ecount(g) > 0) {
        E(g)$width <- E(g)$weight * 5
      }
      
      # If too many edges, keep only the strongest ones for visualization
      if (ecount(g) > 30 && vcount(g) > 8) {
        weights <- E(g)$weight
        keep_idx <- order(weights, decreasing=TRUE)[1:30]
        g <- subgraph_from_edges(g, keep_idx, delete.vertices=FALSE)
      }
      
      # Save network visualization if requested
      if (save) {
        filename <- paste0(format(Sys.time(), "%Y%m%d"), "_marker_network.png")
        filepath <- file.path(private$.output_dir, filename)
        
        png(filepath, width = 1200, height = 1000, res = 150)
        plot(g, 
            vertex.label = V(g)$name,
            vertex.label.cex = 0.8,
            layout = layout_with_fr(g),
            main = "Marker Similarity Network")
        dev.off()
        
        message(sprintf("Network visualization saved to %s", filepath))
        private$.trackPerformance(start_time, "Network visualization")
        
        return(filepath)
      } else {
        private$.trackPerformance(start_time, "Network visualization")
        return(g)
      }
    },
    
    #' Create a cluster profile heatmap
    #' 
    #' @param cluster_profiles Matrix of cluster profiles
    #' @param save Logical indicating whether to save the heatmap
    #' @return Heatmap object
    createClusterProfileHeatmap = function(cluster_profiles, save = TRUE) {
      start_time <- Sys.time()
      message("Creating cluster profile heatmap...")
      
      # Get marker clustering for visualization
      marker_dist <- dist(t(cluster_profiles))
      marker_hc <- hclust(marker_dist, method="ward.D2")
      marker_dend_viz <- as.dendrogram(marker_hc)
      
      # Define expression colors
      expression_colors <- colorRampPalette(c("#F7FBFF", "#DEEBF7", "#C6DBEF", "#9ECAE1", 
                                           "#6BAED6", "#4292C6", "#2171B5", "#08519C", "#08306B"))(100)
      
      # Create heatmap
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
      
      # Save heatmap if requested
      if (save) {
        private$.saveHeatmap(cluster_profile_hm, "pixel_cluster_profile", width = 12, height = 8)
      }
      
      private$.trackPerformance(start_time, "Cluster profile heatmap")
      return(cluster_profile_hm)
    },
    
    #' Create a heatmap showing cluster distribution across images
    #' 
    #' @param image_cluster_dist Matrix of cluster distributions per image
    #' @param save Logical indicating whether to save the heatmap
    #' @return Heatmap object
    createClusterDistributionHeatmap = function(image_cluster_dist, save = TRUE) {
      start_time <- Sys.time()
      message("Creating cluster distribution heatmap...")
      
      # Create heatmap
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
      
      # Save heatmap if requested
      if (save) {
        private$.saveHeatmap(image_cluster_hm, "image_cluster_distribution", width = 14, height = 10)
      }
      
      private$.trackPerformance(start_time, "Cluster distribution heatmap")
      return(image_cluster_hm)
    },
    
    #' Create a pixel-level heatmap with annotations
    #' 
    #' @param pixel_data Matrix of pixel data
    #' @param pixel_clusters Vector of cluster assignments
    #' @param pixel_image_names Vector of image names for each pixel
    #' @param marker_names Vector of marker names
    #' @param max_pixels Maximum number of pixels to include in the heatmap
    #' @param save Logical indicating whether to save the heatmap
    #' @return Heatmap object
    createPixelHeatmap = function(pixel_data, pixel_clusters, pixel_image_names, 
                                marker_names, max_pixels = 10000, save = TRUE) {
      start_time <- Sys.time()
      message("Creating pixel-level heatmap...")
      
      # Get marker dendrogram
      marker_dist <- dist(t(pixel_data))
      marker_hc <- hclust(marker_dist, method="ward.D2")
      marker_dend_viz <- as.dendrogram(marker_hc)
      
      # Sample pixels for heatmap visualization if needed
      if (length(pixel_clusters) > max_pixels) {
        # Sample evenly from each cluster
        k_clusters <- max(pixel_clusters)
        sampled_viz_indices <- unlist(lapply(1:k_clusters, function(k) {
          cluster_pixels <- which(pixel_clusters == k)
          if (length(cluster_pixels) > 0) {
            sample(cluster_pixels, min(ceiling(max_pixels/k_clusters), length(cluster_pixels)))
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
        rainbow(length(unique(pixel_image_names))),
        unique(pixel_image_names)
      )
      
      row_anno <- rowAnnotation(
        image = pixel_image_names[viz_indices],
        col = list(image = img_colors),
        show_annotation_name = TRUE,
        annotation_legend_param = list(image = list(title = "Image")),
        annotation_name_gp = gpar(fontsize = 10)
      )
      
      # Create heatmap
      expression_colors <- colorRampPalette(c("#F7FBFF", "#DEEBF7", "#C6DBEF", "#9ECAE1", 
                                           "#6BAED6", "#4292C6", "#2171B5", "#08519C", "#08306B"))(100)
      
      pixel_hm <- Heatmap(
        as.matrix(pixel_data[viz_indices, ]),
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
      
      # Save heatmap if requested
      if (save) {
        private$.saveHeatmap(pixel_hm, "pixel_level_heatmap", width = 14, height = 12)
      }
      
      private$.trackPerformance(start_time, "Pixel heatmap")
      return(pixel_hm)
    }
  )
) 