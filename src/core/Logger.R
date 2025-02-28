#' Logging system for spatial analysis pipeline
#' @description Handles logging of messages, warnings, and errors
Logger <- R6::R6Class("Logger",
  private = list(
    .log_file = NULL,
    .log_level = NULL,
    
    write_log = function(level, msg) {
      timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
      log_entry <- sprintf("[%s] %s: %s\n", timestamp, level, msg)
      
      # Write to console
      cat(log_entry)
      
      # Write to file if configured
      if (!is.null(private$.log_file)) {
        cat(log_entry, file = private$.log_file, append = TRUE)
      }
    }
  ),
  
  public = list(
    initialize = function(log_file = NULL, log_level = "INFO") {
      private$.log_file <- log_file
      private$.log_level <- log_level
      
      if (!is.null(log_file)) {
        dir.create(dirname(log_file), recursive = TRUE, showWarnings = FALSE)
        cat("", file = log_file) # Initialize log file
      }
      
      self$log_info("Logger initialized")
    },
    
    log_info = function(msg, ...) {
      # Accept additional arguments for sprintf formatting.
      formatted_msg <- if (length(list(...)) > 0) sprintf(msg, ...) else msg
      private$write_log("INFO", formatted_msg)
    },
    
    log_warning = function(msg, ...) {
      formatted_msg <- if (length(list(...)) > 0) sprintf(msg, ...) else msg
      private$write_log("WARNING", formatted_msg)
    },
    
    log_error = function(msg, ...) {
      formatted_msg <- if (length(list(...)) > 0) sprintf(msg, ...) else msg
      private$write_log("ERROR", formatted_msg)
    },
    
    log_debug = function(msg, ...) {
      if (private$.log_level == "DEBUG") {
        formatted_msg <- if (length(list(...)) > 0) sprintf(msg, ...) else msg
        private$write_log("DEBUG", formatted_msg)
      }
    }
  )
) 