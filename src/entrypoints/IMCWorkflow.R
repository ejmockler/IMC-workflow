runIMCWorkflow <- function(
  config_file = NULL,
  workflow_type = c("unsupervised", "gated", "both"),
  steps = c("load", "spatial", "visualize", "report"),
  output_dir = NULL
) {
  # Initialize
  workflow_type <- match.arg(workflow_type)
  config <- ConfigurationManager$new()
  if (!is.null(config_file)) config$merge_with_defaults(config_file)
  
  # Override output directory if provided
  if (!is.null(output_dir)) config$config$paths$output_dir <- output_dir
  
  # Create logger
  logger <- Logger$new("IMCWorkflow")
  
  # Initialize data loader
  data_loader <- DataLoader$new(config$config, logger)
  
  # Run selected workflow
  spe_objects <- list()
  
  if (workflow_type %in% c("unsupervised", "both") && "load" %in% steps) {
    logger$info("Running unsupervised cell phenotyping workflow")
    spe_objects$unsupervised <- data_loader$loadUnsupervisedData()
  }
  
  if (workflow_type %in% c("gated", "both") && "load" %in% steps) {
    logger$info("Running gated cell workflow")
    spe_objects$gated <- data_loader$loadGatedCellData()
  }
  
  # Spatial analysis
  if ("spatial" %in% steps) {
    logger$info("Running spatial analysis")
    spatial_analyzer <- SpatialAnalyzer$new(config$config, logger)
    
    if (!is.null(spe_objects$unsupervised)) {
      spe_objects$unsupervised <- spatial_analyzer$analyze(spe_objects$unsupervised)
    }
    
    if (!is.null(spe_objects$gated)) {
      spe_objects$gated <- spatial_analyzer$analyze(spe_objects$gated)
    }
  }
  
  # Visualization
  if ("visualize" %in% steps) {
    logger$info("Generating visualizations")
    visualizer <- Visualizer$new(config$config, logger)
    
    for (workflow in names(spe_objects)) {
      visualizer$createVisualizations(spe_objects[[workflow]], workflow)
    }
  }
  
  # Reporting
  if ("report" %in% steps && "both" %in% workflow_type) {
    logger$info("Generating comparative report")
    reporter <- Reporter$new(config$config, logger)
    reporter$generateReport(spe_objects)
  }
  
  return(spe_objects)
}
