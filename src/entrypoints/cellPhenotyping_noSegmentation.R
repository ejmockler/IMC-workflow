#' Perform Marker Relationship Analysis Without Cell Segmentation
#'
#' This entrypoint analyzes marker relationships directly from pixel data without
#' relying on potentially biased cell segmentation. It uses multiple advanced
#' analytical techniques while preserving image context to provide comprehensive 
#' marker insights across samples.
#'
#' @return A MarkerAnalyzer object containing analysis results and methods for further exploration.
#'
#' @example
#'   analyzer <- runMarkerAnalysisNoSegmentation()

# Load required libraries
library(R6)

# Source necessary files
source("src/core/DependencyManager.R")
source("src/core/ConfigurationManager.R")
source("src/core/Logger.R")
source("src/analysis/MarkerAnalysis.R")
source("src/core/ImageIO.R")

runMarkerAnalysisNoSegmentation <- function(
  input_file = NULL,
  output_dir = NULL,
  n_pixels = NULL,
  transformation = NULL,
  save_plots = NULL,
  memory_limit = 0,
  n_cores = NULL
) {
  # Initialize configuration and logger
  configManager <- ConfigurationManager$new()
  logger <- Logger$new(log_file = "logs/cellPhenotyping_noSegmentation.log", log_level = "INFO")
  
  # Initialize dependencies
  dependencyManager <- DependencyManager$new(logger = logger)
  dependencyManager$install_missing_packages()
  
  # Get parameters from config or use provided values
  input_file <- input_file %||% configManager$config$marker_analysis$input_file %||% "output/images_processed.rds"
  output_dir <- output_dir %||% configManager$config$output$dir %||% "output"
  n_pixels <- n_pixels %||% configManager$config$marker_analysis$n_pixels %||% 50000000
  transformation <- transformation %||% configManager$config$marker_analysis$transformation %||% TRUE
  save_plots <- save_plots %||% configManager$config$marker_analysis$save_plots %||% TRUE
  
  # Initialize cores if not specified
  if (is.null(n_cores)) {
    n_cores <- configManager$config$marker_analysis$n_cores %||% max(1, floor(parallel::detectCores() * 0.7))
  }
  
  logger$log_info("Starting image-aware marker analysis")
  logger$log_info("Using input file: %s", input_file)
  logger$log_info("Using %d pixels for analysis", n_pixels)
  logger$log_info("Using %d cores for parallel processing", n_cores)
  
  # Ensure parallel processing packages are loaded
  dependencyManager$ensure_package("foreach")
  dependencyManager$ensure_package("doParallel")
  
  # Create MarkerAnalyzer object
  analyzer <- MarkerAnalyzer$new(
    input_file = input_file,
    output_dir = output_dir,
    n_pixels = n_pixels,
    transform_data = transformation,
    logger = logger
  )
  
  # Run image-aware analysis
  logger$log_info("Running image-aware marker analysis...")
  results <- analyzer$runImageAwareMarkerAnalysis(n_cores = n_cores)
  
  # Save combined results
  results_file <- file.path(output_dir, "marker_analysis_results.rds")
  saveRDS(results, results_file)
  logger$log_info("Saved comprehensive analysis results to: %s", results_file)
  
  # Return the analyzer object for further exploration
  logger$log_info("Image-aware marker analysis completed successfully")
  
  if (memory_limit > 0) {
    gc_limit <- memory_limit * 1024 * 1024  # Convert to bytes
    message(sprintf("Setting memory limit to %d MB", memory_limit))
    utils::mem.limit(gc_limit)
  }
  
  invisible(analyzer)
}

if (interactive() || identical(environment(), globalenv())) {
  analyzer <- runMarkerAnalysisNoSegmentation()
}
