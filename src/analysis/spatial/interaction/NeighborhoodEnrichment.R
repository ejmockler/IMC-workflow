#' Neighborhood enrichment analysis for spatial data
#' 
#' @description Analyzes the enrichment or depletion of cell types in
#' spatial neighborhoods, identifying preferential co-localization patterns.
#'
#' @details Extends the InteractionBase class with methods for quantifying
#' the enrichment of cell types around other cell types in spatial data.

library(R6)
library(SpatialExperiment)

source("src/analysis/spatial/interaction/InteractionBase.R")

NeighborhoodEnrichment <- R6::R6Class("NeighborhoodEnrichment",
  inherit = InteractionBase,
  
  public = list(
    #' @description Create a new NeighborhoodEnrichment object
    #' @param spe SpatialExperiment object with spatial graphs
    #' @param logger Logger object
    #' @param n_cores Number of cores for parallel processing
    #' @param dependency_manager DependencyManager object
    initialize = function(spe, logger = NULL, n_cores = 1, dependency_manager = NULL) {
      super$initialize(spe, logger, n_cores, dependency_manager)
      
      # Load neighborhood-specific dependencies
      private$loadNeighborhoodDependencies()
      
      invisible(self)
    },
    
    #' @description Calculate enrichment of cell types in neighborhoods
    #' @param colPairName Name of the spatial graph to use
    #' @param celltype_column Column containing cell type information
    #' @param img_id Column containing image identifiers
    #' @param result_column Name of the column to store enrichment results
    #' @param auto_build_graph Whether to automatically build the graph if missing
    #' @return Updated SpatialExperiment object
    calculateEnrichment = function(
      colPairName = "knn_interaction_graph",
      celltype_column = "celltype",
      img_id = "sample_id",
      result_column = "neighborhood_enrichment",
      auto_build_graph = TRUE
    ) {
      if (!is.null(self$logger)) self$logger$log_info("Calculating neighborhood enrichment")
      
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
      
      # Get all images
      images <- self$getImages(img_id)
      if (is.null(images)) {
        if (!is.null(self$logger)) self$logger$log_error("No valid images found")
        return(self$spe)
      }
      
      # Calculate enrichment for each image
      enrichment_results <- list()
      
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
        
        # Calculate neighborhood enrichment
        tryCatch({
          # Get cell types
          cell_types <- unique(img_spe[[celltype_column]])
          
          # Create a matrix to store enrichment values
          n_types <- length(cell_types)
          enrichment_matrix <- matrix(1, nrow = n_types, ncol = n_types)
          rownames(enrichment_matrix) <- cell_types
          colnames(enrichment_matrix) <- cell_types
          
          # Calculate enrichment for each cell type pair
          for (central_type in cell_types) {
            # Cells of the central type
            central_cells <- img_spe[[celltype_column]] == central_type
            if (sum(central_cells) == 0) next
            
            # Get indices of central cells
            central_indices <- which(central_cells)
            
            # Get the spatial graph
            spatial_graph <- SpatialExperiment::adjacencyMatrix(img_spe, colPairName)
            
            # For each central cell, get its neighbors
            all_neighbors <- vector("list", length(central_indices))
            
            for (i in seq_along(central_indices)) {
              # Get neighbors of this central cell
              neighbor_indices <- which(spatial_graph[central_indices[i], ] > 0)
              
              # Get cell types of neighbors
              if (length(neighbor_indices) > 0) {
                all_neighbors[[i]] <- img_spe[[celltype_column]][neighbor_indices]
              }
            }
            
            # Combine all neighbors
            all_neighbors <- unlist(all_neighbors)
            
            # If no neighbors found, skip
            if (length(all_neighbors) == 0) next
            
            # Count cell types in the neighborhood
            observed_counts <- table(all_neighbors)
            
            # Calculate total cells of each type
            total_counts <- table(img_spe[[celltype_column]])
            
            # Calculate expected counts based on overall frequency
            expected_props <- total_counts / sum(total_counts)
            
            # For each neighbor type, calculate enrichment
            for (neighbor_type in names(observed_counts)) {
              # Observed count
              observed <- observed_counts[neighbor_type]
              
              # Expected count
              expected <- expected_props[neighbor_type] * length(all_neighbors)
              
              # Calculate enrichment (observed/expected)
              enrichment <- observed / expected
              
              # Store in the matrix
              enrichment_matrix[central_type, neighbor_type] <- enrichment
            }
          }
          
          # Store the results
          enrichment_results[[img]] <- enrichment_matrix
          
          if (!is.null(self$logger)) {
            self$logger$log_info("Calculated enrichment for image: %s", img)
          }
        }, error = function(e) {
          if (!is.null(self$logger)) {
            self$logger$log_error(
              "Error calculating enrichment for image %s: %s",
              img, e$message
            )
          }
        })
      }
      
      # Store the results in the SPE object
      if (length(enrichment_results) > 0) {
        S4Vectors::metadata(self$spe)[[result_column]] <- enrichment_results
        
        if (!is.null(self$logger)) {
          self$logger$log_info(
            "Neighborhood enrichment calculated for %d images",
            length(enrichment_results)
          )
        }
      }
      
      return(self$spe)
    },
    
    #' @description Calculate significance of enrichment using permutation testing
    #' @param colPairName Name of the spatial graph to use
    #' @param celltype_column Column containing cell type information
    #' @param img_id Column containing image identifiers
    #' @param n_permutations Number of permutations for significance testing
    #' @param significance_level Significance level for hypothesis testing
    #' @param result_column Name of the column to store significance results
    #' @param seed Random seed for reproducibility
    #' @return Updated SpatialExperiment object
    testEnrichmentSignificance = function(
      colPairName = "knn_interaction_graph",
      celltype_column = "celltype",
      img_id = "sample_id",
      n_permutations = 1000,
      significance_level = 0.05,
      result_column = "enrichment_significance",
      seed = 42
    ) {
      if (!is.null(self$logger)) {
        self$logger$log_info("Testing enrichment significance (%d permutations)", n_permutations)
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
        
        # Test enrichment significance
        tryCatch({
          # Get cell types
          cell_types <- unique(img_spe[[celltype_column]])
          
          # Create matrices to store p-values
          n_types <- length(cell_types)
          pvalue_matrix <- matrix(1, nrow = n_types, ncol = n_types)
          significant_matrix <- matrix(FALSE, nrow = n_types, ncol = n_types)
          rownames(pvalue_matrix) <- cell_types
          colnames(pvalue_matrix) <- cell_types
          rownames(significant_matrix) <- cell_types
          colnames(significant_matrix) <- cell_types
          
          # Get the spatial graph
          spatial_graph <- SpatialExperiment::adjacencyMatrix(img_spe, colPairName)
          
          # Original cell types
          original_cell_types <- img_spe[[celltype_column]]
          
          # For each cell type pair, perform permutation test
          for (central_type in cell_types) {
            # Progress reporting
            if (!is.null(self$logger)) {
              self$logger$log_info("Testing central type: %s", central_type)
            }
            
            # Cells of the central type
            central_cells <- original_cell_types == central_type
            if (sum(central_cells) < 3) {
              # Skip if too few cells of this type
              if (!is.null(self$logger)) {
                self$logger$log_warning("Too few cells of type %s, skipping", central_type)
              }
              next
            }
            
            # Get indices of central cells
            central_indices <- which(central_cells)
            
            # Calculate observed enrichment for each neighbor type
            observed_enrichment <- private$calculateObservedEnrichment(
              img_spe, spatial_graph, central_indices, celltype_column
            )
            
            # Skip if no valid enrichment values
            if (length(observed_enrichment) == 0) next
            
            # Perform permutation test
            for (p in 1:n_permutations) {
              # Permute cell types
              permuted_types <- sample(original_cell_types)
              
              # Calculate enrichment with permuted types
              permuted_enrichment <- private$calculateObservedEnrichment(
                img_spe, spatial_graph, central_indices, 
                celltype_column, permuted_types = permuted_types
              )
              
              # Skip if no valid enrichment values
              if (length(permuted_enrichment) == 0) next
              
              # For each neighbor type, update p-value
              for (neighbor_type in names(observed_enrichment)) {
                if (!neighbor_type %in% names(permuted_enrichment)) next
                
                # If permuted enrichment is more extreme than observed, increment p-value
                if (permuted_enrichment[neighbor_type] >= observed_enrichment[neighbor_type]) {
                  pvalue_matrix[central_type, neighbor_type] <- 
                    pvalue_matrix[central_type, neighbor_type] + (1 / n_permutations)
                }
              }
            }
            
            # Mark significant enrichments
            for (neighbor_type in cell_types) {
              significant_matrix[central_type, neighbor_type] <- 
                pvalue_matrix[central_type, neighbor_type] < significance_level
            }
          }
          
          # Store the results
          significance_results[[img]] <- list(
            pvalue = pvalue_matrix,
            significant = significant_matrix
          )
          
          if (!is.null(self$logger)) {
            n_sig <- sum(significant_matrix)
            self$logger$log_info(
              "Found %d significant enrichments in image: %s",
              n_sig, img
            )
          }
        }, error = function(e) {
          if (!is.null(self$logger)) {
            self$logger$log_error(
              "Error testing enrichment for image %s: %s",
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
            "Enrichment significance tested for %d images",
            length(significance_results)
          )
        }
      }
      
      return(self$spe)
    },
    
    #' @description Run the complete neighborhood enrichment analysis pipeline
    #' @param colPairName Name of the spatial graph to use
    #' @param celltype_column Column containing cell type information
    #' @param img_id Column containing image identifiers
    #' @param n_permutations Number of permutations for significance testing
    #' @param auto_build_graph Whether to automatically build the graph if missing
    #' @param seed Random seed for reproducibility
    #' @return Updated SpatialExperiment object
    runEnrichmentAnalysis = function(
      colPairName = "knn_interaction_graph",
      celltype_column = "celltype",
      img_id = "sample_id",
      n_permutations = 1000,
      auto_build_graph = TRUE,
      seed = 42
    ) {
      if (!is.null(self$logger)) self$logger$log_info("Running complete neighborhood enrichment analysis")
      
      # Step 1: Calculate neighborhood enrichment
      self$spe <- self$calculateEnrichment(
        colPairName = colPairName,
        celltype_column = celltype_column,
        img_id = img_id,
        auto_build_graph = auto_build_graph
      )
      
      # Step 2: Test enrichment significance
      self$spe <- self$testEnrichmentSignificance(
        colPairName = colPairName,
        celltype_column = celltype_column,
        img_id = img_id,
        n_permutations = n_permutations,
        seed = seed
      )
      
      if (!is.null(self$logger)) self$logger$log_info("Neighborhood enrichment analysis completed")
      
      return(self$spe)
    }
  ),
  
  private = list(
    #' @description Load neighborhood-specific dependencies
    loadNeighborhoodDependencies = function() {
      required_packages <- c("imcRtools", "igraph")
      
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
    },
    
    #' @description Calculate observed enrichment for a set of central cells
    #' @param spe SpatialExperiment object
    #' @param spatial_graph Adjacency matrix of the spatial graph
    #' @param central_indices Indices of central cells
    #' @param celltype_column Column containing cell type information
    #' @param permuted_types Optional vector of permuted cell types
    #' @return Named vector of enrichment values
    calculateObservedEnrichment = function(
      spe, spatial_graph, central_indices, celltype_column, permuted_types = NULL
    ) {
      # If permuted types provided, use them; otherwise use original types
      if (!is.null(permuted_types)) {
        cell_types <- permuted_types
      } else {
        cell_types <- spe[[celltype_column]]
      }
      
      # For each central cell, get its neighbors
      all_neighbors <- vector("list", length(central_indices))
      
      for (i in seq_along(central_indices)) {
        # Get neighbors of this central cell
        neighbor_indices <- which(spatial_graph[central_indices[i], ] > 0)
        
        # Get cell types of neighbors
        if (length(neighbor_indices) > 0) {
          all_neighbors[[i]] <- cell_types[neighbor_indices]
        }
      }
      
      # Combine all neighbors
      all_neighbors <- unlist(all_neighbors)
      
      # If no neighbors found, return empty result
      if (length(all_neighbors) == 0) return(c())
      
      # Count cell types in the neighborhood
      observed_counts <- table(all_neighbors)
      
      # Calculate total cells of each type
      total_counts <- table(cell_types)
      
      # Calculate expected counts based on overall frequency
      expected_props <- total_counts / sum(total_counts)
      
      # For each neighbor type, calculate enrichment
      enrichment_values <- c()
      
      for (neighbor_type in names(observed_counts)) {
        # Observed count
        observed <- observed_counts[neighbor_type]
        
        # Expected count
        expected <- expected_props[neighbor_type] * length(all_neighbors)
        
        # Calculate enrichment (observed/expected)
        enrichment <- observed / expected
        
        # Store in the result
        enrichment_values[neighbor_type] <- enrichment
      }
      
      return(enrichment_values)
    }
  )
) 