#' Harmonize External Metadata into a SpatialExperiment object.
#'
#' This function provides a flexible mechanism to merge external CSV metadata files
#' into the colData or rowData slots of a SpatialExperiment object based on specified join keys.
#'
#' @param spe A SpatialExperiment (or SingleCellExperiment) object.
#' @param metadata_configs A list of configurations. Each element should be a list with:
#'   \item{path}{File path to the CSV containing external metadata.}
#'   \item{target}{Target slot for merging ("colData" or "rowData").}
#'   \item{join_key_spe}{The column name in \code{spe}'s target data used for joining.}
#'   \item{join_key_csv}{The column name in the CSV used for joining.}
#'
#' @return A SpatialExperiment object updated with the external metadata merged.
#'
#' @examples
#' metadata_configs <- list(
#'   list(
#'     path = "data/Data_annotations_Karen/Metadata-Table 1.csv",
#'     target = "colData",
#'     join_key_spe = "sample_id",   # the column in spe that identifies the sample
#'     join_key_csv = "File Name"      # the CSV column that matches the sample id
#'   ),
#'   list(
#'     path = "data/Data_annotations_Karen/Channels and Cell types-Table 1.csv",
#'     target = "rowData",
#'     join_key_spe = "channel",       # the rowData key (you might need to rename this field)
#'     join_key_csv = "Channel"        # the CSV column with channel identifiers
#'   )
#' )
#' spe <- harmonizeMetadata(spe, metadata_configs)
harmonizeMetadata <- function(spe, metadata_configs) {
  library(dplyr)
  
  for(config in metadata_configs) {
    # Read the external metadata CSV.
    ext_meta <- read.csv(config$path, stringsAsFactors = FALSE, check.names = FALSE)
    
    # Ensure external metadata column names are unique.
    if (any(duplicated(colnames(ext_meta)))) {
      warning("Duplicate column names found in external metadata file '", config$path,
              "'. Generating unique column names.")
      colnames(ext_meta) <- make.unique(colnames(ext_meta))
    }
    
    # Deduplicate the join key column to ensure uniqueness.
    join_col <- config$join_key_csv
    if (any(duplicated(ext_meta[[join_col]]))) {
      warning("Duplicate values found in column '", join_col, "' in file: ", config$path,
              ". Retaining only the first occurrence.")
      ext_meta <- ext_meta[!duplicated(ext_meta[[join_col]]), ]
    }
    
    # Determine whether we are merging into colData or rowData.
    if (config$target == "colData") {
      base_meta <- as.data.frame(SummarizedExperiment::colData(spe))
    } else if (config$target == "rowData") {
      base_meta <- as.data.frame(SpatialExperiment::rowData(spe))
    } else {
      stop("Unsupported target: ", config$target)
    }
    
    # Merge the external metadata into base_meta using a left join.
    joined <- left_join(
      base_meta,
      ext_meta,
      by = setNames(join_col, config$join_key_spe)
    )
    
    # Update the spe slot with the merged metadata.
    if (config$target == "colData") {
      SummarizedExperiment::colData(spe) <- DataFrame(joined)
    } else if (config$target == "rowData") {
      SpatialExperiment::rowData(spe) <- DataFrame(joined)
    }
  }
  
  return(spe)
}

#' @description Harmonize SPE metadata with additional table(s)
#' @param spe SpatialExperiment to harmonize
#' @param metadata_tables List of data.frames with metadata
#' @param join_column Column to join on
#' @param join_type Type of join ('left', 'right', 'inner', 'full')
#' @return SpatialExperiment with harmonized metadata
harmonize = function(spe, metadata_tables, join_column = "sample_id", join_type = "left") {
  # Extract colData as data.frame
  base_meta <- as.data.frame(SummarizedExperiment::colData(spe))
  
  # Convert metadata_tables to list if it's a single data.frame
  if (is.data.frame(metadata_tables)) {
    metadata_tables <- list(metadata_tables)
  }
  
  # Join each metadata table with the base metadata
  joined <- base_meta
  for (meta_table in metadata_tables) {
    if (!join_column %in% colnames(joined) || !join_column %in% colnames(meta_table)) {
      stop(sprintf("Join column '%s' not found in one of the tables", join_column))
    }
    
    joined <- switch(join_type,
                   "left" = dplyr::left_join(joined, meta_table, by = join_column),
                   "right" = dplyr::right_join(joined, meta_table, by = join_column),
                   "inner" = dplyr::inner_join(joined, meta_table, by = join_column),
                   "full" = dplyr::full_join(joined, meta_table, by = join_column),
                   stop(sprintf("Invalid join type: %s", join_type)))
  }
  
  # Update colData
  SummarizedExperiment::colData(spe) <- S4Vectors::DataFrame(joined)
  
  return(spe)
} 