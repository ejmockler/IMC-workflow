# Diffusion analyzer for marker analysis
# Calculates diffusion maps for marker similarity

#' DiffusionAnalyzer class
#' 
#' Analyzes marker relationships using diffusion map approaches
DiffusionAnalyzer <- R6::R6Class("DiffusionAnalyzer",
  private = list(
    .logger = NULL,
    
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
    #' Initialize a new DiffusionAnalyzer
    #' 
    #' @param logger Optional logger object
    initialize = function(logger = NULL) {
      private$.logger <- logger
      return(invisible(self))
    },
    
    #' Calculate marker similarity using diffusion approach
    #' 
    #' @param pixel_data Matrix of pixel data
    #' @param marker_names Names of markers
    #' @return Matrix of marker similarities
    calculateMarkerSimilarity = function(pixel_data, marker_names) {
      start_time <- Sys.time()
      message("Calculating marker similarity using diffusion approach...")
      
      # Create a transpose matrix for dimension reduction across all images
      marker_similarity <- t(pixel_data) %*% pixel_data
      marker_similarity <- marker_similarity / nrow(pixel_data)
      
      # Add row and column names
      colnames(marker_similarity) <- marker_names
      rownames(marker_similarity) <- marker_names
      
      private$.trackPerformance(start_time, "Marker similarity calculation")
      return(marker_similarity)
    },
    
    #' Calculate diffusion distance between markers
    #' 
    #' @param similarity_matrix Matrix of marker similarities
    #' @return Distance matrix between markers
    calculateDiffusionDistance = function(similarity_matrix) {
      start_time <- Sys.time()
      message("Calculating diffusion distance...")
      
      # Normalize similarity to create a proper diffusion matrix
      row_sums <- rowSums(similarity_matrix)
      normalized_similarity <- similarity_matrix / row_sums
      
      # Create distance matrix from similarity
      max_sim <- max(similarity_matrix)
      if (max_sim == 0) max_sim <- 1  # Handle edge case
      
      marker_dist <- as.dist(1 - (similarity_matrix / max_sim))
      
      private$.trackPerformance(start_time, "Diffusion distance calculation")
      return(marker_dist)
    },
    
    #' Calculate hierarchical clustering on diffusion distances
    #' 
    #' @param diffusion_dist Distance object from diffusion analysis
    #' @param method Hierarchical clustering method
    #' @return Dendrogram object
    calculateClustering = function(diffusion_dist, method = "ward.D2") {
      start_time <- Sys.time()
      message(sprintf("Calculating hierarchical clustering using %s method...", method))
      
      # Check if we have valid distances
      if (any(is.na(diffusion_dist)) || any(is.infinite(as.vector(diffusion_dist)))) {
        message("Warning: Distance matrix contains NA or infinite values")
        # Replace problematic values
        attr_names <- attributes(diffusion_dist)
        dist_vector <- as.vector(diffusion_dist)
        dist_vector[is.na(dist_vector) | is.infinite(dist_vector)] <- max(dist_vector[!is.na(dist_vector) & !is.infinite(dist_vector)], 0) + 0.1
        diffusion_dist <- stats::as.dist(matrix(dist_vector, nrow = attr(diffusion_dist, "Size")))
        attributes(diffusion_dist) <- attr_names
      }
      
      # Calculate hierarchical clustering
      hc <- hclust(diffusion_dist, method = method)
      
      # Convert to dendrogram
      dend <- as.dendrogram(hc)
      
      private$.trackPerformance(start_time, "Hierarchical clustering")
      return(dend)
    },
    
    #' Analyze marker relationships using diffusion approach
    #' 
    #' @param pixel_data Matrix of pixel data
    #' @param marker_names Names of markers
    #' @param clustering_method Hierarchical clustering method
    #' @return List with diffusion analysis results
    analyze = function(pixel_data, marker_names, clustering_method = "ward.D2") {
      start_time <- Sys.time()
      message("Starting diffusion map analysis...")
      
      # Calculate marker similarity
      similarity_matrix <- self$calculateMarkerSimilarity(pixel_data, marker_names)
      
      # Calculate diffusion distance
      diffusion_dist <- self$calculateDiffusionDistance(similarity_matrix)
      
      # Calculate hierarchical clustering
      dend <- self$calculateClustering(diffusion_dist, method = clustering_method)
      
      # Convert distance matrix to similarity matrix for visualization
      max_dist <- max(diffusion_dist)
      if (max_dist == 0) max_dist <- 1  # Handle edge case
      diffusion_similarity <- 1 - (as.matrix(diffusion_dist) / max_dist)
      colnames(diffusion_similarity) <- marker_names
      rownames(diffusion_similarity) <- marker_names
      
      # Assemble results
      diffusion_results <- list(
        similarity = diffusion_similarity,
        distance = diffusion_dist,
        dendrogram = dend
      )
      
      private$.trackPerformance(start_time, "Diffusion analysis")
      return(diffusion_results)
    }
  )
) 