# NOTE: This file contains a standalone function that was previously used for neighborhood analysis.
# To avoid fragmentation, please use the NeighborhoodAnalysis R6 class defined in AnalysisBase.R.
# run_neighborhood_analysis <- function(...) { ... }

run_neighborhood_analysis <- function(spe, k_neighbors = 6, 
                                    distance_threshold = NULL,
                                    marker_of_interest = NULL) {
    # Extract coordinates and create initial neighbor data
    df_coords <- as.data.frame(spatialCoords(spe))
    
    # Calculate KNN
    if (is.null(distance_threshold)) {
        knn_results <- get.knn(df_coords, k = k_neighbors)
        neighbors <- knn_results$nn.index
        distances <- knn_results$nn.dist
    } else {
        # Use distance-based neighborhood definition
        dist_matrix <- as.matrix(dist(df_coords))
        neighbors <- apply(dist_matrix, 1, function(x) which(x <= distance_threshold))
        distances <- lapply(1:nrow(dist_matrix), function(i) dist_matrix[i, neighbors[[i]]])
    }
    
    # Initialize results list
    results <- list(
        neighbors = neighbors,
        distances = distances,
        metrics = list(),
        plots = list()
    )
    
    # Calculate neighborhood metrics
    results$metrics$avg_distance <- colMeans(distances)
    results$metrics$neighbor_density <- sapply(neighbors, length)
    
    # If marker data is provided, analyze marker-specific patterns
    if (!is.null(marker_of_interest) && marker_of_interest %in% rownames(spe)) {
        marker_values <- assay(spe)[marker_of_interest,]
        
        # Calculate spatial autocorrelation for marker
        results$metrics$marker_autocorr <- sapply(1:ncol(spe), function(i) {
            neighbor_vals <- marker_values[neighbors[i,]]
            cor(marker_values[i], neighbor_vals)
        })
        
        # Create marker-based neighborhood visualization
        results$plots$marker_neighborhood <- ggplot(data.frame(
            x = df_coords[,1],
            y = df_coords[,2],
            value = marker_values,
            avg_neighbor = sapply(1:length(neighbors), function(i) 
                mean(marker_values[neighbors[i,]]))
        )) +
        geom_point(aes(x = x, y = y, color = value)) +
        geom_point(aes(x = x, y = y, size = avg_neighbor), alpha = 0.3) +
        scale_color_gradient(low = "blue", high = "red") +
        labs(title = paste("Neighborhood Analysis -", marker_of_interest),
             color = "Marker Value",
             size = "Avg Neighbor Value") +
        theme_minimal()
    }
    
    # Create neighborhood network visualization
    edge_df <- do.call(rbind, lapply(1:nrow(df_coords), function(i) {
        data.frame(
            x1 = df_coords[i,1],
            y1 = df_coords[i,2],
            x2 = df_coords[neighbors[i,],1],
            y2 = df_coords[neighbors[i,],2]
        )
    }))
    
    results$plots$network <- ggplot() +
        geom_segment(data = edge_df,
                    aes(x = x1, y = y1, xend = x2, yend = y2),
                    alpha = 0.1) +
        geom_point(data = df_coords,
                  aes(x = V1, y = V2),
                  size = 1) +
        labs(title = "Neighborhood Network") +
        theme_minimal()
    
    # Create distance distribution plot
    results$plots$distance_dist <- ggplot(data.frame(
        distance = as.vector(distances)
    )) +
        geom_histogram(aes(x = distance), bins = 30) +
        labs(title = "Distribution of Neighbor Distances",
             x = "Distance",
             y = "Count") +
        theme_minimal()
    
    # Create neighbor density heatmap
    results$plots$density <- ggplot(data.frame(
        x = df_coords[,1],
        y = df_coords[,2],
        density = results$metrics$neighbor_density
    )) +
        geom_point(aes(x = x, y = y, color = density)) +
        scale_color_gradient(low = "blue", high = "red") +
        labs(title = "Neighborhood Density",
             color = "Number of Neighbors") +
        theme_minimal()
    
    return(results)
}

# Function to evaluate neighborhood stability
evaluate_knn_stability <- function(spe, k_range = 2:10) {
    df_coords <- as.data.frame(spatialCoords(spe))
    
    # Calculate stability metrics across different k values
    stability_results <- lapply(k_range, function(k) {
        knn_res <- get.knn(df_coords, k = k)
        list(
            k = k,
            avg_distance = mean(knn_res$nn.dist),
            std_distance = sd(knn_res$nn.dist),
            neighbor_consistency = mean(apply(knn_res$nn.index, 1, function(x) 
                length(unique(x)) / length(x))))
    })
    
    # Convert to data frame
    stability_df <- do.call(rbind, lapply(stability_results, as.data.frame))
    
    # Create stability plot
    stability_plot <- ggplot(stability_df) +
        geom_line(aes(x = k, y = avg_distance)) +
        geom_ribbon(aes(x = k, 
                       ymin = avg_distance - std_distance,
                       ymax = avg_distance + std_distance),
                   alpha = 0.2) +
        labs(title = "KNN Stability Analysis",
             x = "Number of Neighbors (k)",
             y = "Average Distance") +
        theme_minimal()
    
    return(list(
        metrics = stability_df,
        plot = stability_plot
    ))
}