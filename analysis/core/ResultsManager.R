#' Results management for spatial analysis pipeline
#' @description Manages storage and export of analysis results
ResultsManager <- R6::R6Class("ResultsManager",
  private = list(
    .results = list(),
    .logger = NULL,
    
    validate_output_dir = function(dir) {
      if (!dir.exists(dir)) {
        dir.create(dir, recursive = TRUE)
      }
    }
  ),
  
  public = list(
    initialize = function(logger = NULL) {
      private$.logger <- logger
    },
    
    add_result = function(type, result) {
      private$.results[[type]] <- result
      
      if (!is.null(private$.logger)) {
        private$.logger$log_info(sprintf("Added result: %s", type))
      }
      
      invisible(self)
    },
    
    get_result = function(type) {
      if (!type %in% names(private$.results)) {
        stop(sprintf("No results found for type: %s", type))
      }
      private$.results[[type]]
    },
    
    save_results = function(output_dir) {
      private$validate_output_dir(output_dir)
      
      # Save each result type
      for (type in names(private$.results)) {
        result <- private$.results[[type]]
        
        # Handle different result types
        if (inherits(result, "ggplot")) {
          # Save plots
          filename <- file.path(output_dir, paste0(type, ".pdf"))
          ggsave(filename, plot = result)
        } else if (is.list(result) || is.data.frame(result)) {
          # Save data objects
          filename <- file.path(output_dir, paste0(type, ".rds"))
          saveRDS(result, filename)
        }
        
        if (!is.null(private$.logger)) {
          private$.logger$log_info(sprintf("Saved %s result", type))
        }
      }
      
      invisible(self)
    },
    
    export_to_format = function(format = c("rds", "csv", "json")) {
      format <- match.arg(format)
      # Implementation for different export formats
      # ...
    }
  )
) 