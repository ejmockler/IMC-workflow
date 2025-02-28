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
      
      plots_dir <- file.path(output_dir, "plots")
      data_dir <- file.path(output_dir, "data")
      private$validate_output_dir(plots_dir)
      private$validate_output_dir(data_dir)
      
      # Save each result type
      for (type in names(private$.results)) {
        result <- private$.results[[type]]
        
        if (inherits(result, "ggplot")) {
          # Save plots in plots directory
          filename <- file.path(plots_dir, paste0(type, ".pdf"))
          ggsave(filename, plot = result)
          
          # Also save as PNG for easy viewing
          png_filename <- file.path(plots_dir, paste0(type, ".png"))
          ggsave(png_filename, plot = result, device = "png", dpi = 300)
          
          if (!is.null(private$.logger)) {
            private$.logger$log_info(sprintf("Saved plot %s as PDF and PNG", type))
          }
        } else if (is.list(result) || is.data.frame(result)) {
          # Save data objects in data directory
          filename <- file.path(data_dir, paste0(type, ".rds"))
          saveRDS(result, filename)
          
          if (!is.null(private$.logger)) {
            private$.logger$log_info(sprintf("Saved data for %s", type))
          }
        }
      }
      
      invisible(self)
    },
    
    export_to_format = function(format = c("rds", "csv", "json"), output_dir) {
      format <- match.arg(format)
      private$validate_output_dir(output_dir)
      
      # Loop over the stored results
      for (type in names(private$.results)) {
        result <- private$.results[[type]]
        filename <- file.path(output_dir, paste0(type, ".", format))
        
        if (inherits(result, "ggplot")) {
          # For ggplot objects, we'll only export the RDS; alternatively, export as PDF is handled in save_results()
          if (format == "rds") {
            saveRDS(result, filename)
          } else {
            warning(sprintf("Export for ggplot objects to format '%s' is not supported. Skipping %s.", format, type))
          }
        } else if (is.data.frame(result)) {
          if (format == "rds") {
            saveRDS(result, filename)
          } else if (format == "csv") {
            write.csv(result, filename, row.names = FALSE)
          } else if (format == "json") {
            if (!requireNamespace("jsonlite", quietly = TRUE)) {
              stop("Package 'jsonlite' is required for JSON export. Please install it.")
            }
            jsonlite::write_json(result, filename, pretty = TRUE, auto_unbox = TRUE)
          }
        } else if (is.list(result)) {
          if (format == "rds") {
            saveRDS(result, filename)
          } else if (format == "csv") {
            # Attempt to coerce list to data frame if possible
            tryCatch({
              df <- do.call(rbind, lapply(result, as.data.frame))
              write.csv(df, filename, row.names = FALSE)
            }, error = function(e) {
              warning(sprintf("Could not export result '%s' to CSV: %s. Skipping.", type, e$message))
            })
          } else if (format == "json") {
            if (!requireNamespace("jsonlite", quietly = TRUE)) {
              stop("Package 'jsonlite' is required for JSON export. Please install it.")
            }
            jsonlite::write_json(result, filename, pretty = TRUE, auto_unbox = TRUE)
          }
        } else {
          # For any other object types, we default to RDS export
          if (format == "rds") {
            saveRDS(result, filename)
          } else {
            warning(sprintf("Export for result '%s' of type '%s' is not supported in format '%s'.", type, class(result), format))
          }
        }
      }
      
      invisible(self)
    }
  )
) 