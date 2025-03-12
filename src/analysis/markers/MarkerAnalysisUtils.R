# Utility functions for marker analysis
# Contains shared functions used across marker analysis classes

#' MarkerAnalysisUtils class
#' 
#' Provides utility functions for marker analysis
MarkerAnalysisUtils <- R6::R6Class("MarkerAnalysisUtils",
  public = list(
    #' Check if a package is installed and install if needed
    #' 
    #' @param package_name Name of the package to check
    #' @param bioconductor Whether the package is from Bioconductor
    #' @return TRUE if package is available, FALSE otherwise
    ensure_package = function(package_name, bioconductor = FALSE) {
      if (!requireNamespace(package_name, quietly = TRUE)) {
        message(sprintf("Package '%s' not found. Installing...", package_name))
        
        if (bioconductor) {
          if (!requireNamespace("BiocManager", quietly = TRUE)) {
            install.packages("BiocManager")
          }
          BiocManager::install(package_name, update = FALSE, ask = FALSE)
        } else {
          install.packages(package_name)
        }
        
        if (!requireNamespace(package_name, quietly = TRUE)) {
          message(sprintf("Failed to install package '%s'", package_name))
          return(FALSE)
        }
      }
      
      return(TRUE)
    },
    
    #' Calculate percentage of memory in use
    #' 
    #' @return Percentage of memory in use
    memory_usage = function() {
      mem_used <- utils::memory.size()
      mem_limit <- utils::memory.limit()
      return(mem_used / mem_limit * 100)
    },
    
    #' Format file size in human-readable format
    #' 
    #' @param size Size in bytes
    #' @return Human-readable size string
    format_size = function(size) {
      units <- c("B", "KB", "MB", "GB", "TB")
      power <- floor(log(size, 1024))
      power <- min(power, length(units) - 1)
      
      if (size == 0) {
        return("0 B")
      }
      
      value <- size / (1024 ^ power)
      return(sprintf("%.2f %s", value, units[power + 1]))
    },
    
    #' Create a timestamp string
    #' 
    #' @param format Format string for timestamp
    #' @return Formatted timestamp
    timestamp = function(format = "%Y%m%d_%H%M%S") {
      return(format(Sys.time(), format))
    },
    
    #' Calculate execution time
    #' 
    #' @param start_time Start time from Sys.time()
    #' @param units Time units ("secs", "mins", "hours", "days")
    #' @return Time difference in specified units
    execution_time = function(start_time, units = "secs") {
      return(as.numeric(difftime(Sys.time(), start_time, units = units)))
    },
    
    #' Generate a progress message
    #' 
    #' @param current Current progress value
    #' @param total Total value
    #' @param prefix Prefix string
    #' @param suffix Suffix string
    #' @param length Length of progress bar
    #' @return Progress message string
    progress_message = function(current, total, prefix = "", suffix = "", length = 30) {
      percent <- current / total
      filled_length <- floor(length * percent)
      bar <- paste0(
        paste(rep("█", filled_length), collapse = ""),
        paste(rep("░", length - filled_length), collapse = "")
      )
      return(sprintf("%s [%s] %d%% %s", prefix, bar, percent * 100, suffix))
    },
    
    #' Calculate number of cores to use for parallel processing
    #' 
    #' @param max_percentage Maximum percentage of cores to use (0-1)
    #' @param min_cores Minimum number of cores
    #' @param max_cores Maximum number of cores
    #' @return Number of cores to use
    calculate_cores = function(max_percentage = 0.5, min_cores = 1, max_cores = NULL) {
      available_cores <- parallel::detectCores()
      
      if (is.null(max_cores)) {
        max_cores <- available_cores
      }
      
      cores_to_use <- max(min_cores, floor(available_cores * max_percentage))
      cores_to_use <- min(cores_to_use, max_cores)
      
      return(cores_to_use)
    },
    
    #' Validate marker names across objects
    #' 
    #' @param ... Objects with marker names
    #' @return TRUE if all marker names are consistent, FALSE otherwise
    validate_marker_names = function(...) {
      objects <- list(...)
      if (length(objects) < 2) {
        return(TRUE)
      }
      
      reference <- NULL
      
      for (obj in objects) {
        # Extract marker names based on object type
        if (is.matrix(obj) || is.data.frame(obj)) {
          current <- colnames(obj)
        } else if (is.list(obj) && !is.null(obj$markers)) {
          current <- obj$markers
        } else if (is.character(obj)) {
          current <- obj
        } else {
          next
        }
        
        if (is.null(reference)) {
          reference <- current
          next
        }
        
        # Check if marker names match
        if (length(reference) != length(current) || !all(sort(reference) == sort(current))) {
          return(FALSE)
        }
      }
      
      return(TRUE)
    },
    
    #' Generate a hash code for data
    #' 
    #' @param data Data to hash
    #' @return Hash code as character string
    hash_data = function(data) {
      if (!requireNamespace("digest", quietly = TRUE)) {
        install.packages("digest")
      }
      
      require(digest)
      return(digest::digest(data, algo = "md5"))
    }
  )
)

# Create a singleton instance for easy access
MarkerUtils <- MarkerAnalysisUtils$new() 