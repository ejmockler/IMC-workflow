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
