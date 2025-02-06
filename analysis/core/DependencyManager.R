#' Dependency management for spatial analysis pipeline
#' @description Handles package dependencies and environment validation
DependencyManager <- R6::R6Class("DependencyManager",
  private = list(
    .logger = NULL,
    
    .required_packages = list(
      cran = c(
        "ggplot2", "dplyr", "FNN", "grid", "reshape2", "gridExtra",
        "spatstat", "ape", "energy", "spdep", "R6"
      ),
      bioc = c(
        "SpatialExperiment", "cytomapper"
      )
    ),
    
    install_cran_package = function(pkg) {
      if (!requireNamespace(pkg, quietly = TRUE)) {
        private$.logger$log_info(sprintf("Installing CRAN package: %s", pkg))
        install.packages(pkg, repos = "http://cran.us.r-project.org")
      }
    },
    
    install_bioc_package = function(pkg) {
      if (!requireNamespace(pkg, quietly = TRUE)) {
        private$.logger$log_info(sprintf("Installing Bioconductor package: %s", pkg))
        BiocManager::install(pkg, update = FALSE)
      }
    }
  ),
  
  public = list(
    initialize = function(logger = NULL) {
      private$.logger <- logger %||% Logger$new()
    },
    
    check_required_packages = function() {
      missing_cran <- private$.required_packages$cran[
        !sapply(private$.required_packages$cran, requireNamespace, quietly = TRUE)
      ]
      
      missing_bioc <- private$.required_packages$bioc[
        !sapply(private$.required_packages$bioc, requireNamespace, quietly = TRUE)
      ]
      
      list(
        cran = missing_cran,
        bioc = missing_bioc
      )
    },
    
    install_missing_packages = function() {
      missing <- self$check_required_packages()
      
      if (length(missing$cran) > 0) {
        private$.logger$log_info("Installing missing CRAN packages...")
        sapply(missing$cran, private$install_cran_package)
      }
      
      if (length(missing$bioc) > 0) {
        private$.logger$log_info("Installing missing Bioconductor packages...")
        if (!requireNamespace("BiocManager", quietly = TRUE)) {
          install.packages("BiocManager")
        }
        sapply(missing$bioc, private$install_bioc_package)
      }
      
      invisible(self)
    },
    
    validate_environment = function() {
      # Check R version
      r_version <- getRversion()
      if (r_version < "4.0.0") {
        stop("R version 4.0.0 or higher is required")
      }
      
      # Check package versions
      missing <- self$check_required_packages()
      if (length(c(missing$cran, missing$bioc)) > 0) {
        self$install_missing_packages()
      }
      
      # Validate memory
      mem_info <- gc()
      available_mem <- mem_info[2,4] # Available memory in MB
      if (available_mem < 1000) {
        private$.logger$log_warning("Low memory available (<1GB)")
      }
      
      invisible(self)
    }
  )
) 