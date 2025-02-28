#' Annotate Processed Steinbock Data with External Metadata
#'
#' This entrypoint loads the previously imported SpatialExperiment object,
#' assigns unique cell identifiers, and merges in external metadata using a
#' configuration‐driven file path.
#'
#' @return The annotated SpatialExperiment object.
#'
#' @example
#'   spe_annotated <- runAnnotateSteinbockData()

# Load Core Dependencies
source("src/core/ConfigurationManager.R")
source("src/core/Logger.R")
source("src/core/MetadataHarmonizer.R")

runAnnotateSteinbockData <- function() {
  # Initialize configuration and logger.
  configManager <- ConfigurationManager$new()
  logger <- Logger$new(log_file = "logs/annotateSteinbockData.log", log_level = "INFO")
  
  # Load the imported spe (produced in importSteinbockData.R) using config values.
  spe <- readRDS(file.path(configManager$config$output$dir, "spe.rds"))
  
  # Assign unique cell identifiers (e.g., sample_id_ObjectNumber).
  colnames(spe) <- paste0(spe$sample_id, "_", spe$ObjectNumber)
  logger$log_info("Unique cell identifiers assigned to spe.")
  
  # Obtain external metadata file path from configuration.
  # Make sure to include 'metadata_annotation' in your configuration defaults.
  external_metadata_path <- configManager$config$paths$metadata_annotation
  
  # Optionally, if you have a channels metadata file, you might add it, for instance:
  # external_channels_path <- configManager$config$paths$channels_annotation
  
  # Prepare external metadata configuration.
  metadata_configs <- list(
    list(
      path = external_metadata_path,
      target = "colData",
      join_key_spe = "sample_id",   # Adjust based on your spe field.
      join_key_csv = "File Name"      # Adjust based on the CSV header.
    )
    # Uncomment the following list element if channel metadata is needed:
    # list(
    #   path = external_channels_path,
    #   target = "rowData",
    #   join_key_spe = "channel",       # Adjust the corresponding field in spe.
    #   join_key_csv = "Channel"
    # )
  )
  
  # Merge external metadata into spe using the MetadataHarmonizer abstraction.
  spe <- harmonizeMetadata(spe, metadata_configs)
  logger$log_info("External metadata merged into spe colData.")
  
  # Persist the annotated object using the output directory from configuration.
  saveRDS(spe, file = file.path(configManager$config$output$dir, "spe_annotated.rds"))
  logger$log_info("Annotated spe saved to output directory: %s", configManager$config$output$dir)
  
  invisible(spe)
}

if (interactive() || identical(environment(), globalenv())) {
  spe_annotated <- runAnnotateSteinbockData()
} 