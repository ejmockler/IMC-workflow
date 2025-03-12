#' Pairwise cell type interaction analysis
#' 
#' @description Analyzes pairwise interactions between cell types in spatial data
#' using methods such as permutation testing and odds ratios.
#'
#' @details Extends the InteractionBase class with methods for quantifying the 
#' significance of interactions between pairs of cell types in spatial data.

library(R6)
library(SpatialExperiment)

source("src/analysis/spatial/interaction/InteractionBase.R")

PairwiseInteraction <- R6::R6Class("PairwiseInteraction",
  inherit = InteractionBase,
  
  public = list(
    #' @description Create a new PairwiseInteraction object
    #' @param spe SpatialExperiment object with spatial graphs
    #' @param logger Logger object
    #' @param n_cores Number of cores for parallel processing
    #' @param dependency_manager DependencyManager object
    initialize = function(spe, logger = NULL, n_cores = 1, dependency_manager = NULL) {
      super$initialize(spe, logger, n_cores, dependency_manager)
      
      # Load pairwise interaction-specific dependencies
      private$loadPairwiseDependencies()
      
      invisible(self)
    },
    
    #' @description Calculate interaction strengths between cell types
    #' @param colPairName Name of the spatial graph to use
    #' @param celltype_column Column containing cell type information
    #' @param img_id Column containing image identifiers
    #' @param result_column Name of the column to store interaction results
    #' @param auto_build_graph Whether to automatically build the graph if missing
    #' @return Updated SpatialExperiment object
    calculateInteractions = function(
      colPairName = "knn_interaction_graph",
      celltype_column = "celltype",
      img_id = "sample_id",
      result_column = "pairwise_interactions",
      auto_build_graph = TRUE
    ) {
      if (!is.null(self$logger)) self$logger$log_info("Calculating pairwise cell type interactions")
      
      # Check phenotype data
      if (!self$checkPhenotypeData(celltype_column)) {
        if (!is.null(self$logger)) self$logger$log_error("Invalid phenotype data")
        return(self$spe)
      }
      
      # Check if the required graph exists
      if (!self$checkSpatialGraphs(colPairName, auto_build_graph)) {
        if (!is.null(self$logger)) self$logger$log_error("Required graph '%s' not found", colPairName)
        return(self$spe)
      }
      
      # Make sure imcRtools is available
      if (!requireNamespace("imcRtools", quietly = TRUE)) {
        if (!is.null(self$logger)) {
          self$logger$log_error("imcRtools package is required for interaction analysis")
        }
        return(self$spe)
      }
      
      # Get all images
      images <- self$getImages(img_id)
      if (is.null(images)) {
        if (!is.null(self$logger)) self$logger$log_error("No valid images found")
        return(self$spe)
      }
      
      # Calculate interactions for each image
      interaction_results <- list()
      
      for (img in images) {
        if (!is.null(self$logger)) self$logger$log_info("Processing image: %s", img)
        
        # Subset to current image
        img_cells <- self$spe[[img_id]] == img
        if (sum(img_cells) == 0) {
          if (!is.null(self$logger)) self$logger$log_warning("No cells found in image: %s", img)
          next
        }
        
        # Create a subset SPE for this image
        img_spe <- self$spe[, img_cells]
        
        # Calculate interactions using countPairs from imcRtools
        tryCatch({
          # Count the occurrences of cell type pairs
          interactions <- imcRtools::countPairs(
            img_spe,
            colPairName = colPairName,
            groupBy = celltype_column
          )
          
          # Store the results
          interaction_results[[img]] <- interactions
          
          if (!is.null(self$logger)) {
            self$logger$log_info("Calculated interactions for image: %s", img)
          }
        }, error = function(e) {
          if (!is.null(self$logger)) {
            self$logger$log_error("Error calculating interactions for image %s: %s", img, e$message)
          }
        })
      }
      
      # Store the results in the SPE object
      if (length(interaction_results) > 0) {
        S4Vectors::metadata(self$spe)[[result_column]] <- interaction_results
        
        if (!is.null(self$logger)) {
          self$logger$log_info("Pairwise interactions calculated for %d images", length(interaction_results))
        }
      }
      
      return(self$spe)
    },
    
    #' @description Calculate significance of interactions using permutation testing
    #' @param colPairName Name of the spatial graph to use
    #' @param celltype_column Column containing cell type information
    #' @param img_id Column containing image identifiers
    #' @param n_permutations Number of permutations for significance testing
    #' @param significance_level Significance level for hypothesis testing
    #' @param result_column Name of the column to store significance results
    #' @param seed Random seed for reproducibility
    #' @return Updated SpatialExperiment object
    testInteractionSignificance = function(
      colPairName = "knn_interaction_graph",
      celltype_column = "celltype",
      img_id = "sample_id",
      n_permutations = 1000,
      significance_level = 0.05,
      result_column = "interaction_significance",
      seed = 42
    ) {
      if (!is.null(self$logger)) {
        self$logger$log_info("Testing interaction significance (%d permutations)", n_permutations)
      }
      
      # Check phenotype data
      if (!self$checkPhenotypeData(celltype_column)) {
        if (!is.null(self$logger)) self$logger$log_error("Invalid phenotype data")
        return(self$spe)
      }
      
      # Check if the required graph exists
      if (!self$checkSpatialGraphs(colPairName, FALSE)) {
        if (!is.null(self$logger)) self$logger$log_error("Required graph '%s' not found", colPairName)
        return(self$spe)
      }
      
      # Set seed for reproducibility
      set.seed(seed)
      
      # Get all images
      images <- self$getImages(img_id)
      if (is.null(images)) {
        if (!is.null(self$logger)) self$logger$log_error("No valid images found")
        return(self$spe)
      }
      
      # Test significance for each image
      significance_results <- list()
      
      for (img in images) {
        if (!is.null(self$logger)) self$logger$log_info("Processing image: %s", img)
        
        # Subset to current image
        img_cells <- self$spe[[img_id]] == img
        if (sum(img_cells) == 0) {
          if (!is.null(self$logger)) self$logger$log_warning("No cells found in image: %s", img)
          next
        }
        
        # Create a subset SPE for this image
        img_spe <- self$spe[, img_cells]
        
        # Test interaction significance
        tryCatch({
          # Use imcRtools for permutation testing
          sig_results <- imcRtools::testPairs(
            img_spe,
            colPairName = colPairName,
            groupBy = celltype_column,
            alternative = "two.sided",
            numPermutations = n_permutations,
            BPPARAM = BiocParallel::MulticoreParam(workers = self$n_cores)
          )
          
          # Add significance flag
          sig_results$significant <- sig_results$pvalue < significance_level
          
          # Store the results
          significance_results[[img]] <- sig_results
          
          if (!is.null(self$logger)) {
            n_sig <- sum(sig_results$significant, na.rm = TRUE)
            self$logger$log_info(
              "Found %d significant interactions in image: %s",
              n_sig, img
            )
          }
        }, error = function(e) {
          if (!is.null(self$logger)) {
            self$logger$log_error(
              "Error testing interactions for image %s: %s",
              img, e$message
            )
          }
        })
      }
      
      # Store the results in the SPE object
      if (length(significance_results) > 0) {
        S4Vectors::metadata(self$spe)[[result_column]] <- significance_results
        
        if (!is.null(self$logger)) {
          self$logger$log_info(
            "Interaction significance tested for %d images",
            length(significance_results)
          )
        }
      }
      
      return(self$spe)
    },
    
    #' @description Calculate interaction strength as odds ratios
    #' @param interactions_column Name of the column containing interaction counts
    #' @param celltype_column Column containing cell type information
    #' @param img_id Column containing image identifiers
    #' @param result_column Name of the column to store odds ratio results
    #' @return Updated SpatialExperiment object
    calculateOddsRatios = function(
      interactions_column = "pairwise_interactions",
      celltype_column = "celltype",
      img_id = "sample_id",
      result_column = "interaction_odds_ratios"
    ) {
      if (!is.null(self$logger)) self$logger$log_info("Calculating interaction odds ratios")
      
      # Check if interactions have been calculated
      if (!interactions_column %in% names(S4Vectors::metadata(self$spe))) {
        if (!is.null(self$logger)) {
          self$logger$log_warning("Interactions not found, calculating them first")
        }
        self$spe <- self$calculateInteractions(
          celltype_column = celltype_column,
          img_id = img_id,
          result_column = interactions_column
        )
      }
      
      # Get the interaction results
      interaction_results <- S4Vectors::metadata(self$spe)[[interactions_column]]
      if (is.null(interaction_results) || length(interaction_results) == 0) {
        if (!is.null(self$logger)) self$logger$log_error("No interaction results found")
        return(self$spe)
      }
      
      # Calculate odds ratios for each image
      odds_ratio_results <- list()
      
      for (img in names(interaction_results)) {
        if (!is.null(self$logger)) self$logger$log_info("Processing image: %s", img)
        
        # Get the interactions for this image
        interactions <- interaction_results[[img]]
        
        # Calculate odds ratios
        tryCatch({
          # Get unique cell types
          cell_types <- unique(c(interactions$group_from, interactions$group_to))
          
          # Create a matrix to store odds ratios
          odds_matrix <- matrix(1, nrow = length(cell_types), ncol = length(cell_types))
          rownames(odds_matrix) <- cell_types
          colnames(odds_matrix) <- cell_types
          
          # Calculate odds ratio for each pair
          for (i in 1:nrow(interactions)) {
            from_type <- interactions$group_from[i]
            to_type <- interactions$group_to[i]
            
            # Skip self-interactions for simplicity
            if (from_type == to_type) next
            
            # Count of this specific interaction
            observed <- interactions$count[i]
            
            # Get total interactions for these cell types
            total_from <- sum(interactions$count[interactions$group_from == from_type])
            total_to <- sum(interactions$count[interactions$group_to == to_type])
            
            # Total interactions in the image
            total_interactions <- sum(interactions$count)
            
            # Calculate expected count under independence
            expected <- (total_from * total_to) / total_interactions
            
            # Calculate odds ratio
            odds_ratio <- observed / expected
            
            # Store in the matrix
            odds_matrix[from_type, to_type] <- odds_ratio
          }
          
          # Store the results
          odds_ratio_results[[img]] <- odds_matrix
          
          if (!is.null(self$logger)) {
            self$logger$log_info("Calculated odds ratios for image: %s", img)
          }
        }, error = function(e) {
          if (!is.null(self$logger)) {
            self$logger$log_error(
              "Error calculating odds ratios for image %s: %s",
              img, e$message
            )
          }
        })
      }
      
      # Store the results in the SPE object
      if (length(odds_ratio_results) > 0) {
        S4Vectors::metadata(self$spe)[[result_column]] <- odds_ratio_results
        
        if (!is.null(self$logger)) {
          self$logger$log_info(
            "Interaction odds ratios calculated for %d images",
            length(odds_ratio_results)
          )
        }
      }
      
      return(self$spe)
    },
    
    #' @description Run the complete pairwise interaction analysis pipeline
    #' @param colPairName Name of the spatial graph to use
    #' @param celltype_column Column containing cell type information
    #' @param img_id Column containing image identifiers
    #' @param n_permutations Number of permutations for significance testing
    #' @param auto_build_graph Whether to automatically build the graph if missing
    #' @param seed Random seed for reproducibility
    #' @return Updated SpatialExperiment object
    runInteractionAnalysis = function(
      colPairName = "knn_interaction_graph",
      celltype_column = "celltype",
      img_id = "sample_id",
      n_permutations = 1000,
      auto_build_graph = TRUE,
      seed = 42
    ) {
      if (!is.null(self$logger)) self$logger$log_info("Running complete pairwise interaction analysis")
      
      # Step 1: Calculate pairwise interactions
      self$spe <- self$calculateInteractions(
        colPairName = colPairName,
        celltype_column = celltype_column,
        img_id = img_id,
        auto_build_graph = auto_build_graph
      )
      
      # Step 2: Test interaction significance
      self$spe <- self$testInteractionSignificance(
        colPairName = colPairName,
        celltype_column = celltype_column,
        img_id = img_id,
        n_permutations = n_permutations,
        seed = seed
      )
      
      # Step 3: Calculate odds ratios
      self$spe <- self$calculateOddsRatios(
        celltype_column = celltype_column,
        img_id = img_id
      )
      
      if (!is.null(self$logger)) self$logger$log_info("Pairwise interaction analysis completed")
      
      return(self$spe)
    }
  ),
  
  private = list(
    #' @description Load pairwise interaction-specific dependencies
    loadPairwiseDependencies = function() {
      required_packages <- c("imcRtools", "BiocParallel")
      
      if (!is.null(self$dependency_manager)) {
        for (pkg in required_packages) {
          self$dependency_manager$ensure_package(pkg)
        }
      } else {
        for (pkg in required_packages) {
          if (!requireNamespace(pkg, quietly = TRUE)) {
            warning(sprintf("Package '%s' is required but not installed", pkg))
          }
        }
      }
    }
  )
) 