Reporter <- R6::R6Class("Reporter",
  public = list(
    config = NULL,
    logger = NULL,
    
    initialize = function(config, logger) {
      self$config <- config
      self$logger <- logger
    },
    
    generateReport = function(spe_objects) {
      self$logger$info("Generating comprehensive analysis report")
      
      # Create report sections based on available data
      sections <- list()
      
      # Add dataset summary
      sections$summary <- self$createDatasetSummary(spe_objects)
      
      # Add phenotyping results if available
      if (!is.null(spe_objects$unsupervised)) {
        sections$unsupervised <- self$createUnsupervisedAnalysisSection(spe_objects$unsupervised)
      }
      
      # Add gated cell results if available
      if (!is.null(spe_objects$gated)) {
        sections$gated <- self$createGatedAnalysisSection(spe_objects$gated)
      }
      
      # Add comparative analysis if both types are available
      if (!is.null(spe_objects$unsupervised) && !is.null(spe_objects$gated)) {
        sections$comparison <- self$createComparativeSection(
          spe_objects$unsupervised, 
          spe_objects$gated
        )
      }
      
      # Compile and save the report
      report_path <- self$compileReport(sections)
      
      return(report_path)
    },
    
    createDatasetSummary = function(spe_objects) {
      self$logger$info("Creating dataset summary")
      
      summary_info <- list()
      
      # Process each workflow type
      for (workflow in names(spe_objects)) {
        spe <- spe_objects[[workflow]]
        
        # Get basic counts
        cell_count <- ncol(spe)
        marker_count <- nrow(spe)
        
        # Get sample information if available
        sample_column <- NULL
        for (col in c("sample_id", "ImageNumber", "ROI")) {
          if (col %in% colnames(colData(spe))) {
            sample_column <- col
            break
          }
        }
        
        sample_count <- NA
        if (!is.null(sample_column)) {
          sample_count <- length(unique(spe[[sample_column]]))
        }
        
        # Get cell type information if available
        celltype_column <- NULL
        if (workflow == "gated") {
          if ("gated_celltype" %in% colnames(colData(spe))) {
            celltype_column <- "gated_celltype"
          }
        } else {
          if ("phenograph_corrected" %in% colnames(colData(spe))) {
            celltype_column <- "phenograph_corrected"
          }
        }
        
        celltype_count <- NA
        if (!is.null(celltype_column)) {
          celltype_count <- length(unique(spe[[celltype_column]]))
        }
        
        # Store summary information
        summary_info[[workflow]] <- list(
          workflow = workflow,
          cell_count = cell_count,
          marker_count = marker_count,
          sample_count = sample_count,
          celltype_count = celltype_count
        )
      }
      
      return(summary_info)
    },
    
    createUnsupervisedAnalysisSection = function(spe) {
      self$logger$info("Creating unsupervised analysis section")
      
      # Generate description of unsupervised clustering
      if ("phenograph_corrected" %in% colnames(colData(spe))) {
        clusters <- table(spe$phenograph_corrected)
        
        section <- list(
          title = "Unsupervised Cell Phenotyping Results",
          num_clusters = length(clusters),
          cluster_sizes = clusters,
          cluster_proportions = prop.table(clusters) * 100
        )
      } else {
        section <- list(
          title = "Unsupervised Cell Phenotyping Results",
          message = "No clustering results available"
        )
      }
      
      return(section)
    },
    
    createGatedAnalysisSection = function(spe) {
      self$logger$info("Creating gated cell analysis section")
      
      # Generate description of gated cell types
      if ("gated_celltype" %in% colnames(colData(spe))) {
        celltypes <- table(spe$gated_celltype)
        
        section <- list(
          title = "Gated Cell Type Results",
          num_celltypes = length(celltypes),
          celltype_counts = celltypes,
          celltype_proportions = prop.table(celltypes) * 100
        )
      } else {
        section <- list(
          title = "Gated Cell Type Results",
          message = "No gated cell type information available"
        )
      }
      
      return(section)
    },
    
    createComparativeSection = function(unsupervised_spe, gated_spe) {
      self$logger$info("Creating comparative analysis section")
      
      # Check if we have the necessary columns for comparison
      if (!("phenograph_corrected" %in% colnames(colData(unsupervised_spe))) || 
          !("gated_celltype" %in% colnames(colData(gated_spe)))) {
        return(list(
          title = "Comparative Analysis",
          message = "Missing required data for comparison"
        ))
      }
      
      # In a real implementation, this would perform more sophisticated comparison
      # For now, just create a placeholder
      section <- list(
        title = "Comparative Analysis: Unsupervised vs. Gated",
        message = "Comparison between unsupervised clustering and gated cell types"
      )
      
      return(section)
    },
    
    compileReport = function(sections) {
      self$logger$info("Compiling final report")
      
      # In a real implementation, this would generate an HTML or PDF report
      # For now, just save the sections as an RDS file
      
      output_path <- file.path(self$config$paths$output_dir, "reports")
      dir.create(output_path, recursive = TRUE, showWarnings = FALSE)
      
      report_file <- file.path(output_path, "analysis_report.rds")
      saveRDS(sections, report_file)
      
      return(report_file)
    }
  )
)
