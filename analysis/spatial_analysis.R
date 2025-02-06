#----------------------------------
# Spatial Analysis of IMC Data Processed by the Steinbock Pipeline
#
# NOTE: Many sections (package installation, data loading, panel integration,
# and visualization function definitions) have been removed. These tasks are
# now handled by DependencyManager, DataLoaderFactory, and VisualizationFactory,
# respectively.
#----------------------------------

# Load core, factory, and analysis components. Ensure that all analysis modules
source("analysis/core/DependencyManager.R")
source("analysis/core/ConfigurationManager.R")
source("analysis/core/Logger.R")
source("analysis/core/ProgressTracker.R")
source("analysis/core/ResultsManager.R")
source("analysis/factories/DataLoaderFactory.R")
source("analysis/analysis/AnalysisBase.R")          # loads NeighborhoodAnalysis
source("analysis/analysis/NeighborhoodAnalysis.R")   # if additional functions are needed
source("analysis/builders/SpatialAnalysisBuilder.R")
source("analysis/visualization/VisualizationFunctions.R")

library(R6)
library(SpatialExperiment)
library(cytomapper)
library(ggplot2)
library(dplyr)
library(FNN)
library(grid)
library(reshape2)
library(gridExtra)
library(spatstat)
library(ape)
library(energy)
library(spdep)

#--------------------------------------------------------
# Entry point: run_spatial_analysis()
#--------------------------------------------------------
run_spatial_analysis <- function(user_config = NULL) {
  # 1. Core Component Initialization --------------------------------------
  logger <- Logger$new(
    log_file = file.path("logs", format(Sys.time(), "analysis_%Y%m%d_%H%M%S.log"))
  )
  logger$log_info("Starting spatial analysis pipeline")
  
  progress <- ProgressTracker$new(logger)
  results_manager <- ResultsManager$new(logger)
  
  # 2. Setup Phase: Validate environment and load configuration -----------
  progress$start_phase("setup")
  tryCatch({
    dep_manager <- DependencyManager$new(logger)
    dep_manager$validate_environment()
    logger$log_info("Environment validated")
    
    config_manager <- ConfigurationManager$new()
    config <- config_manager$merge_with_defaults(user_config)$config
    logger$log_info("Configuration loaded and validated")
    
    progress$update_progress(50, "Setup completed")
  }, error = function(e) {
    logger$log_error(sprintf("Setup failed: %s", e$message))
    stop(e)
  })
  progress$complete_phase()
  
  # 3. Analysis Phase: Load data and run analyses --------------------------
  # Instantiate SpatialAnalysisBuilder and load data using DataLoaderFactory.
  analysis_builder <- SpatialAnalysisBuilder$new(
    visualization_factory = VisualizationFactory$new(),
    logger = logger,
    progress = progress,
    results_manager = results_manager
  )
  
  # Load all data (spe, images, masks, panel)
  analysis_builder$load_data(config$paths)
  
  # Run analyses (e.g., intensity analysis attaches metadata for visualization)
  for (analysis_type in c("neighborhood", "temporal", "intensity")) {
    analysis_builder$run_analysis(
      analysis_type,
      max_points = config$analysis_params$max_points,
      distance_threshold = config$analysis_params$distance_threshold
    )
    progress$update_progress(
      50 + which(analysis_type == c("neighborhood", "temporal", "intensity")) * 15,
      sprintf("%s analysis complete", analysis_type)
    )
  }
  progress$complete_phase()
  
  # 4. Visualization Phase: Create visual outputs using VisualizationFactory --
  progress$start_phase("visualization")
  tryCatch({
    for (viz_type in c("processed_data", "neighborhood", "temporal", "intensity_hotspots")) {
      logger$log_info(sprintf("Generating %s visualization", viz_type))
      analysis_builder$visualize(viz_type)
      progress$update_progress(
        25 * which(viz_type == c("processed_data", "neighborhood", "temporal", "intensity_hotspots")),
        sprintf("%s visualization complete", viz_type)
      )
    }
  }, error = function(e) {
    logger$log_error(sprintf("Visualization failed: %s", e$message))
    stop(e)
  })
  progress$complete_phase()
  
  # 5. Results Phase: Save final outputs using ResultsManager ---------------
  progress$start_phase("saving")
  tryCatch({
    results <- analysis_builder$get_results()
    results_manager$save_results(config$output$dir)
    progress$update_progress(100, "Results saved")
    logger$log_info("Analysis pipeline completed successfully")
    return(results)
  }, error = function(e) {
    logger$log_error(sprintf("Saving results failed: %s", e$message))
    stop(e)
  })
  progress$complete_phase()
}


run_spatial_analysis()
