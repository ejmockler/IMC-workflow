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