# Define a built-in Logger adapter that uses base R functions.
Logger <- R6::R6Class("Logger",
  public = list(
    log_info = function(msg, ...) {
      message(sprintf(msg, ...))
    },
    log_warning = function(msg, ...) {
      warning(sprintf(msg, ...), call. = FALSE)
    }
  )
)

#' Dependency management for spatial analysis pipeline
#' @description Handles package dependencies and environment validation
DependencyManager <- R6::R6Class("DependencyManager",
  private = list(
    .logger = NULL,
    
    .required_packages = list(
      cran = c(
        "ggplot2", "dplyr", "FNN", "grid", "reshape2", "gridExtra",
        "spatstat", "ape", "energy", "spdep", "R6", "dbscan", "viridis",
        "devtools", "RcppAnnoy", "dendextend", "entropy", "infotheo", "pheatmap",
        "foreach", "doParallel", "magrittr", "tidyr", "scales", "circlize",
        "tidyverse", "umap", "fossil", "mclust", "vegan", "reshape2"
      ),
      bioc = c(
        "SpatialExperiment", "cytomapper", "imcRtools", "CATALYST", "scater", 
        "dittoSeq", "batchelor", "scater", "ComplexHeatmap", "lisaClust"
      ),
      github = list(
        rphenoannoy = list(repo = "stuchly/Rphenoannoy", ref = NULL),
        spicyR = list(repo = "SydneyBioX/spicyR", ref = NULL)
      )
    ),
    
    install_cran_package = function(pkg) {
      if (!requireNamespace(pkg, quietly = TRUE)) {
        private$.logger$log_info("Installing CRAN package: %s", pkg)
        install.packages(pkg, repos = "http://cran.us.r-project.org")
      }
    },
    
    install_bioc_package = function(pkg) {
      if (!requireNamespace(pkg, quietly = TRUE)) {
        private$.logger$log_info("Installing Bioconductor package: %s", pkg)
        BiocManager::install(pkg, update = FALSE)
      }
    },
    
    install_github_package = function(pkg_name, repo, ref = NULL) {
      if (!requireNamespace(pkg_name, quietly = TRUE)) {
        private$.logger$log_info("Installing GitHub package: %s from %s", pkg_name, repo)
        devtools::install_github(repo, ref = ref)
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
      
      missing_github <- names(private$.required_packages$github)[
        !sapply(names(private$.required_packages$github), requireNamespace, quietly = TRUE)
      ]
      
      list(
        cran = missing_cran,
        bioc = missing_bioc,
        github = missing_github
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
      
      if (length(missing$github) > 0) {
        private$.logger$log_info("Installing missing GitHub packages...")
        for (pkg in missing$github) {
          pkg_info <- private$.required_packages$github[[pkg]]
          private$install_github_package(pkg, pkg_info$repo, pkg_info$ref)
        }
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
      if (length(c(missing$cran, missing$bioc, missing$github)) > 0) {
        self$install_missing_packages()
      }
      
      # Validate memory
      mem_info <- gc()
      available_mem <- mem_info[2,4] # Available memory in MB
      if (available_mem < 1000) {
        private$.logger$log_warning("Low memory available (<1GB)")
      }
      
      invisible(self)
    },
    
    # Helper method to ensure a package is installed
    ensure_package = function(pkg_name) {
      if (!requireNamespace(pkg_name, quietly = TRUE)) {
        if (pkg_name %in% private$.required_packages$cran) {
          private$install_cran_package(pkg_name)
        } else if (pkg_name %in% private$.required_packages$bioc) {
          private$install_bioc_package(pkg_name)
        } else if (pkg_name %in% names(private$.required_packages$github)) {
          pkg_info <- private$.required_packages$github[[pkg_name]]
          private$install_github_package(pkg_name, pkg_info$repo, pkg_info$ref)
        } else {
          private$.logger$log_warning("Package %s not in the known dependencies list", pkg_name)
          # Default to CRAN
          private$install_cran_package(pkg_name)
        }
      }
      return(requireNamespace(pkg_name, quietly = TRUE))
    }
  )
)
