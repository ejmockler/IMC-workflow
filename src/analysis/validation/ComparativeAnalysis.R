# Script: comparativeAnalysis.R
# Description: Functions for comparing gated vs. unsupervised cell typing results
# Author: Your Name
# Date: Current Date

#' Calculate agreement metrics between two cell type classifications
#'
#' @param labels1 First set of cell type labels
#' @param labels2 Second set of cell type labels
#' @return List with agreement metrics
calculateAgreementMetrics <- function(labels1, labels2) {
  # Create confusion matrix
  confusion_matrix <- table(Labels1 = labels1, Labels2 = labels2)
  
  # Calculate various agreement metrics
  # (implementation of metrics like Rand Index, Adjusted Rand Index, etc.)
  
  return(list(
    confusion_matrix = confusion_matrix,
    rand_index = calculate_rand_index(labels1, labels2),
    adjusted_rand_index = calculate_adjusted_rand_index(labels1, labels2)
  ))

}
#' Compare cell type proportions between two classifications
#'
#' @param spe1 First SpatialExperiment object
#' @param spe2 Second SpatialExperiment object
#' @param celltype_col1 Column name for cell types in spe1
#' @param celltype_col2 Column name for cell types in spe2
#' @return Data frame with cell type proportion comparison
compareProportions <- function(spe1, spe2, celltype_col1, celltype_col2) {
  # Calculate proportions in each dataset
  props1 <- prop.table(table(spe1[[celltype_col1]])) * 100
  props2 <- prop.table(table(spe2[[celltype_col2]])) * 100
  
  # Format results
  result <- data.frame(
    CellType = unique(c(names(props1), names(props2))),
    Classification1 = NA,
    Classification2 = NA
  )
  
  result$Classification1[match(names(props1), result$CellType)] <- props1
  result$Classification2[match(names(props2), result$CellType)] <- props2
  
  return(result)
}

#' Compare spatial community composition between two approaches
#'
#' @param spe1 First SpatialExperiment object with community assignments
#' @param spe2 Second SpatialExperiment object with community assignments
#' @param celltype_col1 Column name for cell types in spe1
#' @param celltype_col2 Column name for cell types in spe2
#' @param community_col1 Column name for communities in spe1
#' @param community_col2 Column name for communities in spe2
#' @return List with community composition comparison metrics
compareCommunityComposition <- function(
  spe1, spe2, 
  celltype_col1, celltype_col2,
  community_col1, community_col2
) {
  # Calculate community composition for each approach
  comp1 <- calculateCommunityComposition(spe1, celltype_col1, community_col1)
  comp2 <- calculateCommunityComposition(spe2, celltype_col2, community_col2)
  
  # Compare diversity metrics
  div1 <- calculateCommunityDiversity(spe1, celltype_col1, community_col1)
  div2 <- calculateCommunityDiversity(spe2, celltype_col2, community_col2)
  
  # Return comparison results
  return(list(
    composition1 = comp1,
    composition2 = comp2,
    diversity1 = div1,
    diversity2 = div2,
    comparison_metrics = compareDiversityMetrics(div1, div2)
  ))
}

# Add these helper functions to implement the metrics
calculate_rand_index <- function(labels1, labels2) {
  # Load required packages
  if (!requireNamespace("fossil", quietly = TRUE)) {
    stop("Package 'fossil' is needed for Rand Index calculation. Please install it.")
  }
  
  # Convert labels to numeric if they're not already
  if (!is.numeric(labels1)) labels1 <- as.numeric(as.factor(labels1))
  if (!is.numeric(labels2)) labels2 <- as.numeric(as.factor(labels2))
  
  # Calculate Rand Index
  return(fossil::rand.index(labels1, labels2))
}

calculate_adjusted_rand_index <- function(labels1, labels2) {
  # Load required packages
  if (!requireNamespace("mclust", quietly = TRUE)) {
    stop("Package 'mclust' is needed for Adjusted Rand Index calculation. Please install it.")
  }
  
  # Calculate Adjusted Rand Index
  return(mclust::adjustedRandIndex(labels1, labels2))
}

calculateCommunityComposition <- function(spe, celltype_col, community_col) {
  # Create composition table
  composition <- table(
    Community = spe[[community_col]],
    CellType = spe[[celltype_col]]
  )
  
  # Convert to percentages within each community
  composition_pct <- sweep(composition, 1, rowSums(composition), "/") * 100
  
  return(list(
    counts = composition,
    percentages = composition_pct
  ))
}

calculateCommunityDiversity <- function(spe, celltype_col, community_col) {
  # Load required packages
  if (!requireNamespace("vegan", quietly = TRUE)) {
    stop("Package 'vegan' is needed for diversity calculations. Please install it.")
  }
  
  # Get unique communities
  communities <- unique(spe[[community_col]])
  
  # Initialize results
  results <- data.frame(
    Community = communities,
    CellCount = NA,
    CellTypeCount = NA,
    Shannon = NA,
    Simpson = NA,
    Evenness = NA
  )
  
  # Calculate diversity metrics for each community
  for (i in seq_along(communities)) {
    comm <- communities[i]
    
    # Get cell types in this community
    comm_cells <- spe[[community_col]] == comm
    cell_types <- table(spe[comm_cells, ][[celltype_col]])
    
    # Calculate diversity metrics
    shannon <- vegan::diversity(cell_types, index = "shannon")
    simpson <- vegan::diversity(cell_types, index = "simpson")
    evenness <- shannon / log(length(cell_types))
    
    # Store results
    results$CellCount[i] <- sum(comm_cells)
    results$CellTypeCount[i] <- length(cell_types)
    results$Shannon[i] <- shannon
    results$Simpson[i] <- simpson
    results$Evenness[i] <- evenness
  }
  
  return(results)
}

compareDiversityMetrics <- function(div1, div2) {
  # Assuming div1 and div2 are outputs of calculateCommunityDiversity
  
  # Create unified community IDs
  all_communities <- unique(c(div1$Community, div2$Community))
  
  # Create comparison dataframe
  comparison <- data.frame(
    Community = all_communities,
    Shannon_1 = NA,
    Shannon_2 = NA,
    Shannon_Diff = NA,
    Simpson_1 = NA,
    Simpson_2 = NA,
    Simpson_Diff = NA,
    Evenness_1 = NA,
    Evenness_2 = NA,
    Evenness_Diff = NA
  )
  
  # Fill in values for each community
  for (i in seq_along(all_communities)) {
    comm <- all_communities[i]
    
    # Get indices in each dataset
    idx1 <- which(div1$Community == comm)
    idx2 <- which(div2$Community == comm)
    
    # Fill Shannon values
    if (length(idx1) > 0) comparison$Shannon_1[i] <- div1$Shannon[idx1]
    if (length(idx2) > 0) comparison$Shannon_2[i] <- div2$Shannon[idx2]
    
    # Fill Simpson values
    if (length(idx1) > 0) comparison$Simpson_1[i] <- div1$Simpson[idx1]
    if (length(idx2) > 0) comparison$Simpson_2[i] <- div2$Simpson[idx2]
    
    # Fill Evenness values
    if (length(idx1) > 0) comparison$Evenness_1[i] <- div1$Evenness[idx1]
    if (length(idx2) > 0) comparison$Evenness_2[i] <- div2$Evenness[idx2]
  }
  
  # Calculate differences
  comparison$Shannon_Diff <- comparison$Shannon_2 - comparison$Shannon_1
  comparison$Simpson_Diff <- comparison$Simpson_2 - comparison$Simpson_1
  comparison$Evenness_Diff <- comparison$Evenness_2 - comparison$Evenness_1
  
  return(comparison)
}

#' ValidationAnalyzer Class
#'
#' @description A class for comparing and validating cell type classifications
#' and community detection approaches.
#'
#' @details This class provides methods to quantitatively compare different
#' cell typing approaches, calculate agreement metrics, and visualize the results.
#'
#' @examples
#' # Create a ValidationAnalyzer object
#' validator <- ValidationAnalyzer$new(reference_spe, test_spe)
#'
#' # Compare classifications
#' metrics <- validator$compareClassifications("manual_gating", "phenograph_clusters")
#'
#' # Visualize confusion matrix
#' validator$visualizeConfusionMatrix(metrics)
#'
#' # Compare cell type proportions
#' validator$visualizeProportions()
#'
#' @export
ValidationAnalyzer <- R6::R6Class("ValidationAnalyzer",
  public = list(
    #' @field spe_reference Reference SpatialExperiment object
    spe_reference = NULL,
    
    #' @field spe_test Test SpatialExperiment object to validate
    spe_test = NULL,
    
    #' @description Create a new ValidationAnalyzer object
    #' @param spe_reference Reference SpatialExperiment object
    #' @param spe_test Test SpatialExperiment object to validate
    initialize = function(spe_reference, spe_test) {
      self$spe_reference <- spe_reference
      self$spe_test <- spe_test
    },
    
    #' @description Calculate agreement between cell type classifications
    compareClassifications = function(ref_col = "celltype", test_col = "celltype") {
      # Check if columns exist
      if (!ref_col %in% colnames(SummarizedExperiment::colData(self$spe_reference))) {
        stop(sprintf("Column '%s' not found in reference dataset", ref_col))
      }
      if (!test_col %in% colnames(SummarizedExperiment::colData(self$spe_test))) {
        stop(sprintf("Column '%s' not found in test dataset", test_col))
      }
      
      # Get cell type labels
      ref_labels <- self$spe_reference[[ref_col]]
      test_labels <- self$spe_test[[test_col]]
      
      # Check for mismatched lengths
      if (length(ref_labels) != length(test_labels)) {
        warning("Label vectors have different lengths. Using the minimum length.")
        min_length <- min(length(ref_labels), length(test_labels))
        ref_labels <- ref_labels[1:min_length]
        test_labels <- test_labels[1:min_length]
      }
      
      # Calculate agreement metrics
      return(calculateAgreementMetrics(ref_labels, test_labels))
    },
    
    #' @description Compare cell type proportions between reference and test data
    #' @param ref_col Column containing cell types in reference data
    #' @param test_col Column containing cell types in test data
    #' @return Data frame with proportion comparison
    compareCellTypeProportions = function(ref_col = "celltype", test_col = "celltype") {
      return(compareProportions(
        self$spe_reference, self$spe_test,
        ref_col, test_col
      ))
    },
    
    #' @description Compare community composition between reference and test data
    #' @param ref_celltype_col Column for cell types in reference data
    #' @param test_celltype_col Column for cell types in test data
    #' @param ref_community_col Column for communities in reference data
    #' @param test_community_col Column for communities in test data
    #' @return List with community composition comparison
    compareCommunities = function(
      ref_celltype_col = "celltype", 
      test_celltype_col = "celltype",
      ref_community_col = "community_id", 
      test_community_col = "community_id"
    ) {
      return(compareCommunityComposition(
        self$spe_reference, self$spe_test,
        ref_celltype_col, test_celltype_col,
        ref_community_col, test_community_col
      ))
    },
    
    #' @description Visualize confusion matrix as a heatmap
    #' @param metrics Output from compareClassifications()
    #' @param title Plot title
    #' @param save_path Path to save the plot (or NULL to not save)
    #' @return ggplot2 object
    visualizeConfusionMatrix = function(metrics = NULL, title = "Cell Type Classification Comparison", save_path = NULL) {
      # If metrics not provided, calculate them
      if (is.null(metrics)) {
        metrics <- self$compareClassifications()
      }
      
      # Load required packages
      if (!requireNamespace("ggplot2", quietly = TRUE)) {
        stop("Package 'ggplot2' is needed for visualization. Please install it.")
      }
      
      # Convert confusion matrix to data frame for plotting
      conf_df <- as.data.frame(as.table(metrics$confusion_matrix))
      colnames(conf_df) <- c("Reference", "Test", "Count")
      
      # Create the plot
      p <- ggplot2::ggplot(conf_df, ggplot2::aes(x = Test, y = Reference, fill = Count)) +
        ggplot2::geom_tile() +
        ggplot2::scale_fill_viridis_c() +
        ggplot2::labs(
          title = title,
          subtitle = sprintf("Rand Index: %.3f, Adjusted Rand Index: %.3f", 
                             metrics$rand_index, metrics$adjusted_rand_index)
        ) +
        ggplot2::theme_minimal() +
        ggplot2::theme(
          axis.text.x = ggplot2::element_text(angle = 45, hjust = 1),
          panel.grid = ggplot2::element_blank()
        )
      
      # Save if path provided
      if (!is.null(save_path)) {
        ggplot2::ggsave(save_path, p, width = 10, height = 8)
      }
      
      return(p)
    },
    
    #' @description Visualize cell type proportion comparison
    #' @param proportion_data Output from compareCellTypeProportions()
    #' @param title Plot title
    #' @param save_path Path to save the plot (or NULL to not save)
    #' @return ggplot2 object
    visualizeProportions = function(proportion_data = NULL, title = "Cell Type Proportion Comparison", save_path = NULL) {
      # If proportion data not provided, calculate it
      if (is.null(proportion_data)) {
        proportion_data <- self$compareCellTypeProportions()
      }
      
      # Load required packages
      if (!requireNamespace("ggplot2", quietly = TRUE)) {
        stop("Package 'ggplot2' is needed for visualization. Please install it.")
      }
      
      # Reshape data for plotting
      plot_data <- reshape2::melt(proportion_data, id.vars = "CellType", 
                                  variable.name = "Dataset", value.name = "Percentage")
      
      # Create the plot
      p <- ggplot2::ggplot(plot_data, ggplot2::aes(x = CellType, y = Percentage, fill = Dataset)) +
        ggplot2::geom_bar(stat = "identity", position = "dodge") +
        ggplot2::labs(
          title = title,
          y = "Percentage (%)",
          x = "Cell Type"
        ) +
        ggplot2::theme_minimal() +
        ggplot2::theme(
          axis.text.x = ggplot2::element_text(angle = 45, hjust = 1)
        )
      
      # Save if path provided
      if (!is.null(save_path)) {
        ggplot2::ggsave(save_path, p, width = 10, height = 6)
      }
      
      return(p)
    }
  )
)

#' Demonstrate usage of ValidationAnalyzer with example data
#'
#' @param spe1 First SpatialExperiment object
#' @param spe2 Second SpatialExperiment object
#' @param output_dir Directory to save visualizations
#' @return ValidationAnalyzer object
runValidationExample <- function(spe1, spe2, output_dir = "results/validation") {
  # Create output directory if it doesn't exist
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  # Create validator
  validator <- ValidationAnalyzer$new(spe1, spe2)
  
  # Run classification comparison
  metrics <- validator$compareClassifications("celltype", "celltype")
  
  # Create visualizations
  validator$visualizeConfusionMatrix(
    metrics, 
    save_path = file.path(output_dir, "confusion_matrix.png")
  )
  
  validator$visualizeProportions(
    save_path = file.path(output_dir, "proportion_comparison.png")
  )
  
  # Compare communities if available
  if ("community_id" %in% colnames(SummarizedExperiment::colData(spe1)) &&
      "community_id" %in% colnames(SummarizedExperiment::colData(spe2))) {
    
    community_comparison <- validator$compareCommunities()
    # Save community comparison results
    saveRDS(
      community_comparison, 
      file.path(output_dir, "community_comparison.rds")
    )
  }
  
  # Print summary
  cat("Validation Results Summary:\n")
  cat("-------------------------\n")
  cat("Rand Index:", round(metrics$rand_index, 3), "\n")
  cat("Adjusted Rand Index:", round(metrics$adjusted_rand_index, 3), "\n")
  cat("Visualizations saved to:", output_dir, "\n")
  
  return(validator)
}
