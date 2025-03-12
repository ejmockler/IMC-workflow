# SpatialAnalyzer.R
# High-level manager for spatial analysis components

# Load all required spatial components
source("src/analysis/spatial/graph/SpatialGraph.R")
source("src/analysis/spatial/community/CommunityBase.R")
source("src/analysis/spatial/community/GraphCommunity.R")
source("src/analysis/spatial/community/CellTypeNeighborhood.R")
source("src/analysis/spatial/community/SpatialCommunityManager.R")
source("src/analysis/spatial/interaction/InteractionBase.R")
source("src/analysis/spatial/interaction/PairwiseInteraction.R")
source("src/analysis/spatial/interaction/NeighborhoodEnrichment.R")
source("src/analysis/spatial/interaction/SpatialInteractionManager.R")

#' @import R6
#' @import SpatialExperiment

# Make sure we have the null-coalescing operator available
if (!exists("%||%")) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
}

SpatialAnalyzer <- R6::R6Class(
  "SpatialAnalyzer",
  
  public = list(
    #' @field config Configuration parameters
    config = NULL,
    
    #' @field logger Logger object
    logger = NULL,
    
    #' @field graph_builder SpatialGraph object
    graph_builder = NULL,
    
    #' @field community_manager SpatialCommunityManager object
    community_manager = NULL,
    
    #' @field interaction_manager SpatialInteractionManager object 
    interaction_manager = NULL,
    
    #' @description
    #' Initialize a new SpatialAnalyzer object
    #' @param config Configuration parameters
    #' @param logger Logger object
    initialize = function(config = NULL, logger = NULL) {
      self$config <- config
      self$logger <- logger %||% Logger$new("SpatialAnalyzer")
      
      # Load components
      self$logger$info("Loading spatial analysis components")
      
      # Initialize spatial graph builder
      self$graph_builder <- SpatialGraph$new(config, self$logger)
      
      # Initialize community manager if available
      if (exists("SpatialCommunityManager")) {
        self$community_manager <- SpatialCommunityManager$new(config, self$logger)
      } else {
        self$logger$warn("SpatialCommunityManager not available")
      }
      
      # Initialize interaction manager if available
      if (exists("SpatialInteractionManager")) {
        self$interaction_manager <- SpatialInteractionManager$new(config, self$logger)
      } else {
        self$logger$warn("SpatialInteractionManager not available")
      }
    },
    
    #' @description
    #' Analyze spatial data using specified steps
    #' @param spe SpatialExperiment object
    #' @param steps Vector of analysis steps to run
    #' @param is_gated Whether the data is from gated cells
    #' @return Updated SpatialExperiment object
    analyze = function(spe, 
                       steps = c("graphs", "communities", "interactions"),
                       is_gated = FALSE) {
      self$logger$info(paste("Running spatial analysis on", ifelse(is_gated, "gated", "unsupervised"), "data"))
      
      # Build spatial graphs if needed
      if ("graphs" %in% steps) {
        self$logger$info("Building spatial graphs")
        
        # Get graph parameters from config
        graph_params <- list(
          knn_k = self$config$spatial_analysis$knn_k %||% 10,
          expansion_radius = self$config$spatial_analysis$expansion_threshold %||% 30
        )
        
        # Build graphs
        graph_types <- self$config$spatial_analysis$graph_types %||% c("expansion", "knn", "delaunay")
        spe <- self$graph_builder$buildGraphs(spe, methods = graph_types, params = graph_params)
        
        # Save intermediate result
        if (!is.null(self$config$paths$output_dir)) {
          output_file <- file.path(
            self$config$paths$output_dir, 
            paste0("spe_spatial_graphs", ifelse(is_gated, "_gated", ""), ".rds")
          )
          saveRDS(spe, output_file)
          self$logger$info(paste("Saved spatial graphs to", output_file))
        }
      }
      
      # Run community detection if needed
      if ("communities" %in% steps && !is.null(self$community_manager)) {
        self$logger$info("Running community detection")
        
        # Check if we have the phenotyping column
        phenotyping_col <- self$config$community_analysis$phenotyping_column %||% "phenograph"
        if (!phenotyping_col %in% colnames(colData(spe))) {
          self$logger$warn(paste("Phenotyping column", phenotyping_col, "not found. Community detection requires cell phenotypes."))
        } else {
          # Run community detection
          spe <- self$community_manager$runCommunityDetection(spe, is_gated = is_gated)
          
          # Save intermediate result
          if (!is.null(self$config$paths$output_dir)) {
            output_file <- file.path(
              self$config$paths$output_dir, 
              paste0("spe_communities", ifelse(is_gated, "_gated", ""), ".rds")
            )
            saveRDS(spe, output_file)
            self$logger$info(paste("Saved communities to", output_file))
          }
        }
      }
      
      # Run interaction analysis if needed
      if ("interactions" %in% steps && !is.null(self$interaction_manager)) {
        self$logger$info("Running interaction analysis")
        
        # Check if we have the phenotyping column
        phenotyping_col <- self$config$community_analysis$phenotyping_column %||% "phenograph"
        if (!phenotyping_col %in% colnames(colData(spe))) {
          self$logger$warn(paste("Phenotyping column", phenotyping_col, "not found. Interaction analysis requires cell phenotypes."))
        } else {
          # Run interaction analysis
          spe <- self$interaction_manager$runInteractionAnalysis(spe, is_gated = is_gated)
          
          # Save intermediate result
          if (!is.null(self$config$paths$output_dir)) {
            output_file <- file.path(
              self$config$paths$output_dir, 
              paste0("spe_interactions", ifelse(is_gated, "_gated", ""), ".rds")
            )
            saveRDS(spe, output_file)
            self$logger$info(paste("Saved interaction analysis to", output_file))
          }
        }
      }
      
      return(spe)
    },
    
    #' @description
    #' Run batch analysis on multiple samples
    #' @param spe_list List of SpatialExperiment objects
    #' @param steps Vector of analysis steps to run
    #' @param is_gated Whether the data is from gated cells
    #' @return List of updated SpatialExperiment objects
    batchAnalyze = function(spe_list, 
                           steps = c("graphs", "communities", "interactions"),
                           is_gated = FALSE) {
      self$logger$info(paste("Running batch spatial analysis on", length(spe_list), "samples"))
      
      # Process each sample
      result_list <- lapply(names(spe_list), function(sample_name) {
        self$logger$info(paste("Processing sample:", sample_name))
        spe_list[[sample_name]] <- self$analyze(spe_list[[sample_name]], steps, is_gated)
        return(spe_list[[sample_name]])
      })
      
      names(result_list) <- names(spe_list)
      return(result_list)
    },
    
    #' @description
    #' Run comparative analysis between samples
    #' @param spe_list List of analyzed SpatialExperiment objects
    #' @param comparison_type Type of comparison to run
    #' @return Comparison results
    compareResults = function(spe_list, 
                              comparison_type = c("interactions", "communities")) {
      comparison_type <- match.arg(comparison_type)
      self$logger$info(paste("Running comparative", comparison_type, "analysis"))
      
      # Delegate to the appropriate manager
      if (comparison_type == "communities" && !is.null(self$community_manager)) {
        return(self$community_manager$compareCommunities(spe_list))
      } else if (comparison_type == "interactions" && !is.null(self$interaction_manager)) {
        return(self$interaction_manager$compareInteractions(spe_list))
      } else {
        self$logger$error(paste("Cannot run comparative analysis:", comparison_type))
        return(NULL)
      }
    }
  ),
  
  private = list(
    # Helper methods can be added here
  )
) 