#' Column and data structure utilities for the IMC analysis pipeline
#'
#' This file contains utility functions for detecting, validating, and mapping columns 
#' across different data structures used in the IMC analysis pipeline.

#' Find a column matching one of several potential names
#'
#' @param data A SpatialExperiment object or data frame
#' @param candidates Vector of potential column names to search for
#' @param logger Optional logger object for logging the selected column
#' @param description Description of what the column represents (for logging)
#' @param default Default value to return if no matching column is found
#'
#' @return The name of the first matching column, or the default value if none match
findMatchingColumn <- function(data, candidates, logger = NULL, description = "column", default = NULL) {
  if (is(data, "SummarizedExperiment")) {
    available_columns <- colnames(SummarizedExperiment::colData(data))
  } else {
    available_columns <- colnames(data)
  }
  
  for (candidate in candidates) {
    if (candidate %in% available_columns) {
      if (!is.null(logger)) {
        logger$log_info("Using '%s' as %s", candidate, description)
      }
      return(candidate)
    }
  }
  
  if (!is.null(logger)) {
    if (!is.null(default)) {
      logger$log_warning("No %s column found among candidates: %s. Using default: %s", 
                         description, paste(candidates, collapse = ", "), default)
    } else {
      logger$log_warning("No %s column found among candidates: %s", 
                         description, paste(candidates, collapse = ", "))
    }
  }
  
  return(default)
}

#' Ensure a directory exists, creating it if necessary
#'
#' @param dir_path Path to the directory
#' @param recursive Whether to recursively create parent directories
#' @param logger Optional logger object for logging
#'
#' @return TRUE if the directory exists or was created, FALSE otherwise
ensureDirectory <- function(dir_path, recursive = TRUE, logger = NULL) {
  if (!dir.exists(dir_path)) {
    if (!is.null(logger)) {
      logger$log_info("Creating directory: %s", dir_path)
    }
    dir.create(dir_path, recursive = recursive, showWarnings = FALSE)
    return(dir.exists(dir_path))
  }
  return(TRUE)
}

#' Find the image ID column in a dataset
#'
#' @param data A SpatialExperiment object or data frame
#' @param logger Optional logger object for logging
#'
#' @return The name of the image ID column, or NULL if none found
findImageIdColumn <- function(data, logger = NULL) {
  id_candidates <- c("sample_id", "ImageNumber", "ImageID", "ROI", "Image", "Details")
  return(findMatchingColumn(data, id_candidates, logger, "image/sample ID"))
}

#' Find the cell type column in a dataset
#'
#' @param data A SpatialExperiment object or data frame
#' @param logger Optional logger object for logging
#'
#' @return The name of the cell type column, or NULL if none found
findCellTypeColumn <- function(data, logger = NULL) {
  celltype_candidates <- c("celltype", "celltype_classified", "celltype_manual", "cell_type")
  return(findMatchingColumn(data, celltype_candidates, logger, "cell type"))
}

#' Find the phenograph/clustering column in a dataset
#'
#' @param data A SpatialExperiment object or data frame
#' @param logger Optional logger object for logging
#'
#' @return The name of the phenograph column, or NULL if none found
findPhenographColumn <- function(data, logger = NULL) {
  phenograph_candidates <- c("phenograph_corrected", "phenograph_raw", "phenograph", "clusters", "cluster")
  return(findMatchingColumn(data, phenograph_candidates, logger, "clustering/phenograph"))
}

#' Create path for saving visualizations
#'
#' @param base_dir Base output directory
#' @param sub_dir Subdirectory for specific visualization type
#' @param filename Filename for the visualization
#' @param logger Optional logger object for logging
#'
#' @return Full path to save the visualization
createVisualizationPath <- function(base_dir, sub_dir = NULL, filename, logger = NULL) {
  if (!is.null(sub_dir)) {
    full_dir <- file.path(base_dir, sub_dir)
  } else {
    full_dir <- base_dir
  }
  
  ensureDirectory(full_dir, TRUE, logger)
  return(file.path(full_dir, filename))
} 