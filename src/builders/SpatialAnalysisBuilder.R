#' Builder for spatial analysis pipeline
SpatialAnalysisBuilder <- R6::R6Class("SpatialAnalysisBuilder",
  private = list(
    .data = NULL,
    .config = NULL,
    .results = NULL,
    .visualization_factory = NULL,
    .logger = NULL,
    .progress = NULL,
    .results_manager = NULL
  ),
  
  public = list(
    initialize = function(visualization_factory = NULL,
                         logger = NULL,
                         progress = NULL,
                         results_manager = NULL) {
      private$.visualization_factory <- visualization_factory %||% 
        VisualizationFactory$new()
      private$.logger <- logger
      private$.progress <- progress
      private$.results_manager <- results_manager
      private$.results <- list()
    },
    
    with_config = function(config) {
      private$.config <- config
      self
    },
    
    load_data = function(paths) {
      loader_factory <- DataLoaderFactory$new()
      private$.data <- loader_factory$load_multiple(
        types = c("spe", "images", "masks", "panel"),
        paths = paths
      )
      self
    },
    
    run_analysis = function(analysis_type, ...) {
      # Create appropriate analysis class
      analysis_class <- switch(analysis_type,
        "neighborhood" = NeighborhoodAnalysis,
        "temporal" = TemporalAnalysis,
        "intensity" = IntensityAnalysis,
        stop("Unknown analysis type: ", analysis_type)
      )
      
      # Initialize and run analysis
      analysis <- analysis_class$new(
        data = private$.data,
        config = private$.config,
        logger = private$.logger
      )
      analysis$run(...)
      
      # Store results
      private$.results[[analysis_type]] <- analysis$results
      
      self
    },
    
    visualize = function(viz_type, ...) {
      if (is.null(private$.visualization_factory)) {
        stop("No visualization factory available")
      }
      
      viz <- private$.visualization_factory$create_visualization(
        viz_type,
        data = private$.data,
        results = private$.results,
        ...
      )
      
      private$.results$visualization[[viz_type]] <- viz
      self
    },
    
    get_results = function() {
      private$.results
    }
  )
) 