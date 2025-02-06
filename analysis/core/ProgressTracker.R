#' Progress tracking for spatial analysis pipeline
#' @description Tracks progress of analysis phases and steps
ProgressTracker <- R6::R6Class("ProgressTracker",
  private = list(
    .current_phase = NULL,
    .phases = list(),
    .start_time = NULL,
    .logger = NULL
  ),
  
  public = list(
    initialize = function(logger = NULL) {
      private$.logger <- logger
      private$.start_time <- Sys.time()
    },
    
    start_phase = function(phase_name) {
      private$.current_phase <- phase_name
      private$.phases[[phase_name]] <- list(
        start_time = Sys.time(),
        progress = 0
      )
      
      if (!is.null(private$.logger)) {
        private$.logger$log_info(sprintf("Starting phase: %s", phase_name))
      }
      
      invisible(self)
    },
    
    update_progress = function(percent, message = NULL) {
      if (is.null(private$.current_phase)) {
        stop("No phase currently active")
      }
      
      private$.phases[[private$.current_phase]]$progress <- percent
      
      if (!is.null(message) && !is.null(private$.logger)) {
        private$.logger$log_info(
          sprintf("[%s] %d%%: %s", 
                 private$.current_phase, percent, message)
        )
      }
      
      invisible(self)
    },
    
    complete_phase = function() {
      if (is.null(private$.current_phase)) {
        stop("No phase currently active")
      }
      
      phase_name <- private$.current_phase
      private$.phases[[phase_name]]$end_time <- Sys.time()
      private$.phases[[phase_name]]$duration <- 
        difftime(private$.phases[[phase_name]]$end_time,
                private$.phases[[phase_name]]$start_time,
                units = "secs")
      
      if (!is.null(private$.logger)) {
        private$.logger$log_info(sprintf(
          "Completed phase: %s (Duration: %.2f seconds)",
          phase_name,
          private$.phases[[phase_name]]$duration
        ))
      }
      
      private$.current_phase <- NULL
      invisible(self)
    },
    
    get_summary = function() {
      total_duration <- difftime(Sys.time(), private$.start_time, units = "secs")
      
      list(
        phases = private$.phases,
        total_duration = total_duration
      )
    }
  )
) 