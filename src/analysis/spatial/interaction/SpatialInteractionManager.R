#' Spatial Interaction Manager
#' 
#' @description Orchestrates various spatial interaction analysis approaches
#' and serves as the main entry point for interaction analysis functionality.
#'
#' @details This class manages the different interaction analysis methods and provides
#' a unified interface for the analysis workflow. It delegates to specialized classes
#' for specific analysis algorithms.

library(R6)
library(SpatialExperiment)

source("src/analysis/spatial/interaction/InteractionBase.R")
source("src/analysis/spatial/interaction/PairwiseInteraction.R")
source("src/analysis/spatial/interaction/NeighborhoodEnrichment.R")
source("src/analysis/spatial/interaction/SpatialContextAnalysis.R")

SpatialInteractionManager <- R6::R6Class("SpatialInteractionManager",
  public = list(
    #' @field spe SpatialExperiment object
    spe = NULL,
    
    #' @field logger Logger object
    logger = NULL,
    
    #' @field n_cores Number of cores for parallel processing
    n_cores = NULL,
    
    #' @field dependency_manager DependencyManager object
    dependency_manager = NULL,
    
    #' @field pairwise_interaction PairwiseInteraction instance
    pairwise_interaction = NULL,
    
    #' @field neighborhood_enrichment NeighborhoodEnrichment instance
    neighborhood_enrichment = NULL,
    
    #' @field spatial_context_analysis SpatialContextAnalysis instance
    spatial_context_analysis = NULL,
    
    #' @description Create a new SpatialInteractionManager object
    #' @param spe SpatialExperiment object
    #' @param logger Logger object
    #' @param n_cores Number of cores for parallel processing
    #' @param dependency_manager DependencyManager object
    initialize = function(spe, logger = NULL, n_cores = 1, dependency_manager = NULL) {
      self$spe <- spe
      self$logger <- logger
      self$n_cores <- n_cores
      self$dependency_manager <- dependency_manager
      
      # Initialize component instances as needed
      
      invisible(self)
    },
    
    #' @description Run interaction analysis using the specified methods
    #' @param methods Vector of methods to run ("pairwise", "neighborhood", "autocorrelation", "spatial_context", "patch")
    #' @param colPairName Name of the graph to use
    #' @param celltype_column Column containing cell type information
    #' @param img_id Column containing image identifiers
    #' @param n_permutations Number of permutations for significance testing
    #' @param auto_build_graph Whether to automatically build the graph if missing
    #' @param seed Random seed for reproducibility
    #' @param patch_cells Boolean vector indicating cells to include in patches (for patch method)
    #' @return Updated SpatialExperiment object
    runInteractionAnalysis = function(
      methods = c("pairwise", "neighborhood"),
      colPairName = "knn_interaction_graph",
      celltype_column = "celltype",
      img_id = "sample_id",
      n_permutations = 1000,
      auto_build_graph = TRUE,
      seed = 42,
      patch_cells = NULL
    ) {
      if (!is.null(self$logger)) {
        self$logger$log_info("Running spatial interaction analysis with methods: %s", 
                           paste(methods, collapse = ", "))
      }
      
      # Run pairwise interaction analysis
      if ("pairwise" %in% methods) {
        if (!is.null(self$logger)) self$logger$log_info("Running pairwise interaction analysis")
        
        # Initialize PairwiseInteraction if not already done
        if (is.null(self$pairwise_interaction)) {
          self$pairwise_interaction <- PairwiseInteraction$new(
            self$spe, self$logger, self$n_cores, self$dependency_manager
          )
        } else {
          # Update SPE in the existing instance
          self$pairwise_interaction$spe <- self$spe
        }
        
        # Run pairwise interaction analysis
        self$spe <- self$pairwise_interaction$runInteractionAnalysis(
          colPairName = colPairName,
          celltype_column = celltype_column,
          img_id = img_id,
          n_permutations = n_permutations,
          auto_build_graph = auto_build_graph,
          seed = seed
        )
      }
      
      # Run neighborhood enrichment analysis
      if ("neighborhood" %in% methods) {
        if (!is.null(self$logger)) self$logger$log_info("Running neighborhood enrichment analysis")
        
        # Initialize NeighborhoodEnrichment if not already done
        if (is.null(self$neighborhood_enrichment)) {
          self$neighborhood_enrichment <- NeighborhoodEnrichment$new(
            self$spe, self$logger, self$n_cores, self$dependency_manager
          )
        } else {
          # Update SPE in the existing instance
          self$neighborhood_enrichment$spe <- self$spe
        }
        
        # Run neighborhood enrichment analysis
        self$spe <- self$neighborhood_enrichment$runEnrichmentAnalysis(
          colPairName = colPairName,
          celltype_column = celltype_column,
          img_id = img_id,
          n_permutations = n_permutations,
          auto_build_graph = auto_build_graph,
          seed = seed
        )
      }
      
      # Run spatial context analysis
      if ("spatial_context" %in% methods) {
        if (!is.null(self$logger)) self$logger$log_info("Running spatial context analysis")
        
        # Initialize SpatialContextAnalysis if not already done
        if (is.null(self$spatial_context_analysis)) {
          self$spatial_context_analysis <- SpatialContextAnalysis$new(
            self$spe, self$logger, self$n_cores, self$dependency_manager
          )
        } else {
          # Update SPE in the existing instance
          self$spatial_context_analysis$spe <- self$spe
        }
        
        # Check if we have the required cellular neighborhood data
        if (!"cn_celltypes" %in% colnames(SummarizedExperiment::colData(self$spe))) {
          if (!is.null(self$logger)) {
            self$logger$log_warning("Column 'cn_celltypes' not found. Spatial context analysis may fail.")
          }
        }
        
        # Run spatial context workflow
        self$spe <- self$spatial_context_analysis$runSpatialContextWorkflow(
          img_id = img_id,
          count_by = "cn_celltypes"
        )
      }
      
      # Run patch analysis
      if ("patch" %in% methods) {
        if (!is.null(self$logger)) self$logger$log_info("Running patch analysis")
        
        # Check if patch_cells is provided
        if (is.null(patch_cells)) {
          if (!is.null(self$logger)) {
            self$logger$log_warning("No patch_cells provided. Patch analysis will be skipped.")
          }
        } else {
          # Initialize SpatialContextAnalysis if not already done
          if (is.null(self$spatial_context_analysis)) {
            self$spatial_context_analysis <- SpatialContextAnalysis$new(
              self$spe, self$logger, self$n_cores, self$dependency_manager
            )
          } else {
            # Update SPE in the existing instance
            self$spatial_context_analysis$spe <- self$spe
          }
          
          # Run patch analysis workflow
          self$spe <- self$spatial_context_analysis$runPatchAnalysisWorkflow(
            patch_cells = patch_cells,
            colPairName = colPairName,
            img_id = img_id
          )
        }
      }
      
      # Add spatial autocorrelation functionality if needed
      if ("autocorrelation" %in% methods) {
        if (!is.null(self$logger)) {
          self$logger$log_info("Spatial autocorrelation not yet implemented")
        }
      }
      
      if (!is.null(self$logger)) {
        self$logger$log_info("Spatial interaction analysis completed")
      }
      
      return(self$spe)
    },
    
    #' @description Run a complete interaction analysis workflow
    #' @param img_id Column containing image identifiers
    #' @param celltype_column Column containing cell type information
    #' @param all_methods Whether to run all available methods
    #' @param colPairName Name of the graph to use
    #' @param auto_build_graph Whether to automatically build the graph if missing
    #' @param seed Random seed for reproducibility
    #' @param patch_cells Boolean vector indicating cells to include in patches (for patch method)
    #' @return Updated SpatialExperiment object
    runCompleteWorkflow = function(
      img_id = "sample_id",
      celltype_column = "celltype",
      all_methods = FALSE,
      colPairName = "knn_interaction_graph",
      auto_build_graph = TRUE,
      seed = 42,
      patch_cells = NULL
    ) {
      if (!is.null(self$logger)) {
        self$logger$log_info("Running complete interaction analysis workflow")
      }
      
      # Determine which methods to run
      methods <- c("pairwise", "neighborhood")
      if (all_methods) {
        methods <- c(methods, "autocorrelation", "spatial_context")
        # Only include patch method if patch_cells is provided
        if (!is.null(patch_cells)) {
          methods <- c(methods, "patch")
        }
      }
      
      # Check requirements
      if (!celltype_column %in% colnames(SummarizedExperiment::colData(self$spe))) {
        if (!is.null(self$logger)) {
          self$logger$log_warning("Cell type column '%s' not found, analysis may fail", 
                                celltype_column)
        }
      }
      
      # Run interaction analysis
      self$spe <- self$runInteractionAnalysis(
        methods = methods,
        celltype_column = celltype_column,
        img_id = img_id,
        colPairName = colPairName,
        auto_build_graph = auto_build_graph,
        seed = seed,
        patch_cells = patch_cells
      )
      
      return(self$spe)
    },
    
    #' @description Get pairwise interaction statistics
    #' @param significance_column Column containing significance results
    #' @param odds_ratio_column Column containing odds ratio results
    #' @param img_id Column containing image identifiers
    #' @return List of data frames with interaction statistics
    getPairwiseStats = function(
      significance_column = "interaction_significance",
      odds_ratio_column = "interaction_odds_ratios",
      img_id = NULL
    ) {
      if (!is.null(self$logger)) {
        self$logger$log_info("Retrieving pairwise interaction statistics")
      }
      
      # Check if results exist
      if (!significance_column %in% names(S4Vectors::metadata(self$spe)) ||
          !odds_ratio_column %in% names(S4Vectors::metadata(self$spe))) {
        if (!is.null(self$logger)) {
          self$logger$log_warning("Interaction results not found")
        }
        return(NULL)
      }
      
      # Get the results
      significance_results <- S4Vectors::metadata(self$spe)[[significance_column]]
      odds_ratio_results <- S4Vectors::metadata(self$spe)[[odds_ratio_column]]
      
      # If no results, return NULL
      if (is.null(significance_results) || is.null(odds_ratio_results)) {
        if (!is.null(self$logger)) {
          self$logger$log_warning("Interaction results are empty")
        }
        return(NULL)
      }
      
      # Filter to specific image if requested
      if (!is.null(img_id)) {
        if (!img_id %in% names(significance_results)) {
          if (!is.null(self$logger)) {
            self$logger$log_warning("Image '%s' not found in results", img_id)
          }
          return(NULL)
        }
        
        significance_results <- significance_results[img_id]
        odds_ratio_results <- odds_ratio_results[img_id]
      }
      
      # Combine results for each image
      result_list <- list()
      
      for (img in names(significance_results)) {
        sig_data <- significance_results[[img]]
        odds_data <- odds_ratio_results[[img]]
        
        # Convert to data frame for easier handling
        interactions_df <- data.frame()
        
        # Get unique cell types
        cell_types <- rownames(odds_data)
        
        # For each pair of cell types, extract statistics
        for (from_type in cell_types) {
          for (to_type in cell_types) {
            # Skip self-interactions if needed
            if (from_type == to_type) next
            
            # Find this pair in the significance results
            pair_index <- which(sig_data$group_from == from_type & 
                               sig_data$group_to == to_type)
            
            if (length(pair_index) > 0) {
              # Add to data frame
              interactions_df <- rbind(interactions_df, data.frame(
                from_type = from_type,
                to_type = to_type,
                odds_ratio = odds_data[from_type, to_type],
                pvalue = sig_data$pvalue[pair_index],
                significant = sig_data$significant[pair_index],
                observed = sig_data$from_count[pair_index],
                expected = sig_data$to_count[pair_index],
                stringsAsFactors = FALSE
              ))
            }
          }
        }
        
        # Sort by significance and odds ratio
        if (nrow(interactions_df) > 0) {
          interactions_df <- interactions_df[order(
            -interactions_df$significant, 
            -interactions_df$odds_ratio
          ), ]
          
          result_list[[img]] <- interactions_df
        }
      }
      
      return(result_list)
    },
    
    #' @description Get neighborhood enrichment statistics
    #' @param enrichment_column Column containing enrichment results
    #' @param significance_column Column containing significance results
    #' @param img_id Column containing image identifiers
    #' @return List of data frames with enrichment statistics
    getEnrichmentStats = function(
      enrichment_column = "neighborhood_enrichment",
      significance_column = "enrichment_significance",
      img_id = NULL
    ) {
      if (!is.null(self$logger)) {
        self$logger$log_info("Retrieving neighborhood enrichment statistics")
      }
      
      # Check if results exist
      if (!enrichment_column %in% names(S4Vectors::metadata(self$spe)) ||
          !significance_column %in% names(S4Vectors::metadata(self$spe))) {
        if (!is.null(self$logger)) {
          self$logger$log_warning("Enrichment results not found")
        }
        return(NULL)
      }
      
      # Get the results
      enrichment_results <- S4Vectors::metadata(self$spe)[[enrichment_column]]
      significance_results <- S4Vectors::metadata(self$spe)[[significance_column]]
      
      # If no results, return NULL
      if (is.null(enrichment_results) || is.null(significance_results)) {
        if (!is.null(self$logger)) {
          self$logger$log_warning("Enrichment results are empty")
        }
        return(NULL)
      }
      
      # Filter to specific image if requested
      if (!is.null(img_id)) {
        if (!img_id %in% names(enrichment_results)) {
          if (!is.null(self$logger)) {
            self$logger$log_warning("Image '%s' not found in results", img_id)
          }
          return(NULL)
        }
        
        enrichment_results <- enrichment_results[img_id]
        significance_results <- significance_results[img_id]
      }
      
      # Combine results for each image
      result_list <- list()
      
      for (img in names(enrichment_results)) {
        enrichment_matrix <- enrichment_results[[img]]
        sig_data <- significance_results[[img]]
        
        # Convert to data frame for easier handling
        enrichment_df <- data.frame()
        
        # Get unique cell types
        cell_types <- rownames(enrichment_matrix)
        
        # For each pair of cell types, extract statistics
        for (central_type in cell_types) {
          for (neighbor_type in cell_types) {
            # Add to data frame
            enrichment_df <- rbind(enrichment_df, data.frame(
              central_type = central_type,
              neighbor_type = neighbor_type,
              enrichment = enrichment_matrix[central_type, neighbor_type],
              pvalue = sig_data$pvalue[central_type, neighbor_type],
              significant = sig_data$significant[central_type, neighbor_type],
              stringsAsFactors = FALSE
            ))
          }
        }
        
        # Sort by significance and enrichment
        if (nrow(enrichment_df) > 0) {
          enrichment_df <- enrichment_df[order(
            -enrichment_df$significant, 
            -enrichment_df$enrichment
          ), ]
          
          result_list[[img]] <- enrichment_df
        }
      }
      
      return(result_list)
    },
    
    #' @description Get spatial context statistics
    #' @param context_column Column containing spatial context information
    #' @param img_id Column containing image identifiers
    #' @return List of data frames with spatial context statistics
    getSpatialContextStats = function(
      context_column = "spatial_context_filtered",
      img_id = "sample_id"
    ) {
      if (!is.null(self$logger)) {
        self$logger$log_info("Retrieving spatial context statistics")
      }
      
      # Check if results exist
      if (!context_column %in% colnames(SummarizedExperiment::colData(self$spe))) {
        if (!is.null(self$logger)) {
          self$logger$log_warning("Spatial context results not found")
        }
        return(NULL)
      }
      
      # Get the unique images
      images <- unique(self$spe[[img_id]])
      
      # Calculate statistics for each image
      result_list <- list()
      
      for (img in images) {
        # Get cells for this image
        img_cells <- self$spe[[img_id]] == img
        
        # Skip if no cells found
        if (sum(img_cells) == 0) {
          if (!is.null(self$logger)) {
            self$logger$log_warning("No cells found for image: %s", img)
          }
          next
        }
        
        # Get spatial contexts for this image
        contexts <- self$spe[, img_cells][[context_column]]
        
        # Skip if no valid contexts
        if (all(is.na(contexts))) {
          if (!is.null(self$logger)) {
            self$logger$log_warning("No valid spatial contexts found for image: %s", img)
          }
          next
        }
        
        # Count cells per context
        context_counts <- table(contexts, useNA = "ifany")
        
        # Convert to data frame
        context_df <- data.frame(
          context = names(context_counts),
          n_cells = as.numeric(context_counts),
          percentage = as.numeric(context_counts) / sum(as.numeric(context_counts)) * 100,
          image = img,
          stringsAsFactors = FALSE
        )
        
        # Add to result list
        result_list[[img]] <- context_df
      }
      
      return(result_list)
    },
    
    #' @description Get patch statistics
    #' @param patch_id_column Column containing patch IDs
    #' @param img_id Column containing image identifiers
    #' @return List of data frames with patch statistics
    getPatchStats = function(
      patch_id_column = "patch_id",
      img_id = "sample_id"
    ) {
      if (!is.null(self$logger)) {
        self$logger$log_info("Retrieving patch statistics")
      }
      
      # Check if patch sizes exist in metadata
      if ("patch_sizes" %in% names(S4Vectors::metadata(self$spe))) {
        # Return directly from metadata
        return(S4Vectors::metadata(self$spe)[["patch_sizes"]])
      }
      
      # Check if patch_id exists
      if (!patch_id_column %in% colnames(SummarizedExperiment::colData(self$spe))) {
        if (!is.null(self$logger)) {
          self$logger$log_warning("Patch results not found")
        }
        return(NULL)
      }
      
      # Create SpatialContextAnalysis instance if needed
      if (is.null(self$spatial_context_analysis)) {
        self$spatial_context_analysis <- SpatialContextAnalysis$new(
          self$spe, self$logger, self$n_cores, self$dependency_manager
        )
      } else {
        # Update SPE in the existing instance
        self$spatial_context_analysis$spe <- self$spe
      }
      
      # Calculate patch sizes
      patch_sizes <- self$spatial_context_analysis$calculatePatchSizes(patch_id_column)
      
      # Store in metadata for future use
      if (nrow(patch_sizes) > 0) {
        S4Vectors::metadata(self$spe)[["patch_sizes"]] <- patch_sizes
      }
      
      return(patch_sizes)
    },
    
    #' @description Plot spatial context graph
    #' @param entry Column containing spatial context assignments
    #' @param group_by Column to group by
    #' @param result_path Path to store visualization results (or NULL)
    #' @return List of ggplot objects
    plotSpatialContextGraph = function(
      entry = "spatial_context_filtered", 
      group_by = "sample_id",
      result_path = NULL
    ) {
      if (!is.null(self$logger)) {
        self$logger$log_info("Plotting spatial context graph")
      }
      
      # Initialize SpatialContextAnalysis if not already done
      if (is.null(self$spatial_context_analysis)) {
        self$spatial_context_analysis <- SpatialContextAnalysis$new(
          self$spe, self$logger, self$n_cores, self$dependency_manager
        )
      } else {
        # Update SPE in the existing instance
        self$spatial_context_analysis$spe <- self$spe
      }
      
      # Call the plot function
      return(self$spatial_context_analysis$plotSpatialContextGraph(
        entry = entry,
        group_by = group_by,
        result_path = result_path
      ))
    },
    
    #' @description Plot patches on a spatial plot
    #' @param patch_id_column Column containing patch IDs
    #' @param img_id Column to filter by image ID
    #' @param image_id Specific image ID to plot (or NULL for all)
    #' @param spatial_x Column containing X coordinates
    #' @param spatial_y Column containing Y coordinates
    #' @param result_path Path to store visualization results (or NULL)
    #' @return ggplot object
    plotPatches = function(
      patch_id_column = "patch_id",
      img_id = "sample_id",
      image_id = NULL,
      spatial_x = "Pos_X",
      spatial_y = "Pos_Y",
      result_path = NULL
    ) {
      if (!is.null(self$logger)) {
        self$logger$log_info("Plotting patches")
      }
      
      # Initialize SpatialContextAnalysis if not already done
      if (is.null(self$spatial_context_analysis)) {
        self$spatial_context_analysis <- SpatialContextAnalysis$new(
          self$spe, self$logger, self$n_cores, self$dependency_manager
        )
      } else {
        # Update SPE in the existing instance
        self$spatial_context_analysis$spe <- self$spe
      }
      
      # Call the plot function
      return(self$spatial_context_analysis$plotPatches(
        patch_id_column = patch_id_column,
        img_id = img_id,
        image_id = image_id,
        spatial_x = spatial_x,
        spatial_y = spatial_y,
        result_path = result_path
      ))
    }
  )
) 