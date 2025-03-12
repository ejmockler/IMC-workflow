#' Metadata Harmonization for IMC Analysis
#' @description Handles merging of external metadata into SpatialExperiment objects.
#'
#' @import R6
#'
MetadataHarmonizer <- R6::R6Class("MetadataHarmonizer",
  public = list(
    #' @field config Configuration parameters
    config = NULL,
    
    #' @field logger Logger object
    logger = NULL,
    
    #' @description
    #' Initialize a new MetadataHarmonizer object
    #' @param config Configuration parameters
    #' @param logger Logger object
    initialize = function(config = NULL, logger = NULL) {
      self$config <- config
      self$logger <- logger %||% Logger$new("MetadataHarmonizer")
      
      # Ensure required packages
      if (!requireNamespace("dplyr", quietly = TRUE)) {
        self$logger$warn("dplyr package not available, harmonization functionality may be limited")
      }
      
      if (!requireNamespace("S4Vectors", quietly = TRUE)) {
        self$logger$warn("S4Vectors package not available, harmonization functionality may be limited")
      }
    },
    
    #' @description
    #' Harmonize metadata using config-specified metadata tables
    #' @param spe SpatialExperiment object to harmonize
    #' @return Updated SpatialExperiment object with harmonized metadata
    harmonize = function(spe) {
      if (is.null(self$config$paths$metadata)) {
        self$logger$info("No metadata file specified in config, skipping harmonization")
        return(spe)
      }
      
      self$logger$info("Harmonizing metadata from configured sources")
      
      # Get metadata configurations from config if available
      metadata_configs <- NULL
      if (!is.null(self$config$metadata$configs)) {
        metadata_configs <- self$config$metadata$configs
        spe <- self$harmonizeMetadata(spe, metadata_configs)
      } else if (file.exists(self$config$paths$metadata)) {
        # Use simple metadata path approach if configs not available
        metadata_table <- tryCatch({
          read.csv(self$config$paths$metadata, stringsAsFactors = FALSE)
        }, error = function(e) {
          self$logger$warn(paste("Error reading metadata file:", e$message))
          return(NULL)
        })
        
        if (!is.null(metadata_table)) {
          join_column <- self$config$metadata$join_column %||% "sample_id"
          join_type <- self$config$metadata$join_type %||% "left"
          
          spe <- self$harmonizeSimple(
            spe, 
            metadata_table, 
            join_column = join_column,
            join_type = join_type
          )
        }
      } else {
        self$logger$warn(paste("Metadata file not found:", self$config$paths$metadata))
      }
      
      return(spe)
    },
    
    #' @description
    #' Harmonize SPE metadata with additional table(s)
    #' @param spe SpatialExperiment to harmonize
    #' @param metadata_tables List of data.frames with metadata
    #' @param join_column Column to join on
    #' @param join_type Type of join ('left', 'right', 'inner', 'full')
    #' @return SpatialExperiment with harmonized metadata
    harmonizeSimple = function(spe, metadata_tables, join_column = "sample_id", join_type = "left") {
      self$logger$info(paste("Harmonizing metadata using", join_column, "column with", join_type, "join"))
      
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
          self$logger$warn(sprintf("Join column '%s' not found in one of the tables", join_column))
          next
        }
        
        joined <- switch(join_type,
                       "left" = dplyr::left_join(joined, meta_table, by = join_column),
                       "right" = dplyr::right_join(joined, meta_table, by = join_column),
                       "inner" = dplyr::inner_join(joined, meta_table, by = join_column),
                       "full" = dplyr::full_join(joined, meta_table, by = join_column),
                       {
                         self$logger$warn(sprintf("Invalid join type: %s, using left join", join_type))
                         dplyr::left_join(joined, meta_table, by = join_column)
                       })
      }
      
      # Update colData
      SummarizedExperiment::colData(spe) <- S4Vectors::DataFrame(joined)
      
      return(spe)
    },
    
    #' @description
    #' Harmonize metadata with configuration-based approach
    #' @param spe A SpatialExperiment (or SingleCellExperiment) object
    #' @param metadata_configs A list of configurations for merging metadata
    #' @return Updated SpatialExperiment object
    harmonizeMetadata = function(spe, metadata_configs) {
      self$logger$info("Harmonizing metadata with multiple configurations")
      
      for(config in metadata_configs) {
        # Read the external metadata CSV.
        tryCatch({
          ext_meta <- read.csv(config$path, stringsAsFactors = FALSE, check.names = FALSE)
          
          # Ensure external metadata column names are unique.
          if (any(duplicated(colnames(ext_meta)))) {
            self$logger$warn(paste("Duplicate column names found in external metadata file", 
                                   config$path, "generating unique column names"))
            colnames(ext_meta) <- make.unique(colnames(ext_meta))
          }
          
          # Deduplicate the join key column to ensure uniqueness.
          join_col <- config$join_key_csv
          if (any(duplicated(ext_meta[[join_col]]))) {
            self$logger$warn(paste("Duplicate values found in column", join_col, 
                                   "in file", config$path, "retaining only first occurrence"))
            ext_meta <- ext_meta[!duplicated(ext_meta[[join_col]]), ]
          }
          
          # Determine whether we are merging into colData or rowData.
          if (config$target == "colData") {
            base_meta <- as.data.frame(SummarizedExperiment::colData(spe))
          } else if (config$target == "rowData") {
            base_meta <- as.data.frame(SpatialExperiment::rowData(spe))
          } else {
            self$logger$error(paste("Unsupported target:", config$target))
            next
          }
          
          # Merge the external metadata into base_meta using a left join.
          joined <- dplyr::left_join(
            base_meta,
            ext_meta,
            by = setNames(join_col, config$join_key_spe)
          )
          
          # Update the spe slot with the merged metadata.
          if (config$target == "colData") {
            SummarizedExperiment::colData(spe) <- S4Vectors::DataFrame(joined)
            self$logger$info(paste("Updated colData with metadata from", config$path))
          } else if (config$target == "rowData") {
            SpatialExperiment::rowData(spe) <- S4Vectors::DataFrame(joined)
            self$logger$info(paste("Updated rowData with metadata from", config$path))
          }
        }, error = function(e) {
          self$logger$error(paste("Error processing metadata config:", e$message))
        })
      }
      
      return(spe)
    }
  ),
  
  private = list(
    # Helper methods can be added here
  )
)

# Backwards compatibility function to maintain existing code
harmonizeMetadata <- function(spe, metadata_configs) {
  harmonizer <- MetadataHarmonizer$new()
  return(harmonizer$harmonizeMetadata(spe, metadata_configs))
}

# Backwards compatibility function to maintain existing code
harmonize <- function(spe, metadata_tables, join_column = "sample_id", join_type = "left") {
  harmonizer <- MetadataHarmonizer$new()
  return(harmonizer$harmonizeSimple(spe, metadata_tables, join_column, join_type))
} 