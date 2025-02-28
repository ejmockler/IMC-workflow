#' Process Steinbock Data: Transformation, QC, Feature Selection, and Visualization
#'
#' This entrypoint loads the annotated SpatialExperiment object, applies
#' transformations to the counts, conducts quality control filtering, flags channels,
#' runs dimensionality reduction, and then (optionally) generates exploratory plots.
#'
#' @return A processed SpatialExperiment object.
#'
#' @example
#'   spe_processed <- runProcessSteinbockData()

source("src/core/ConfigurationManager.R")
source("src/core/Logger.R")

runProcessSteinbockData <- function() {
  # Initialize configuration and logger.
  configManager <- ConfigurationManager$new()
  logger <- Logger$new(log_file = "logs/processSteinbockData.log", log_level = "INFO")
  
  # Load the annotated spe.
  spe <- readRDS(file.path(configManager$config$output$dir, "spe_annotated.rds"))
  logger$log_info("Annotated spe loaded.")
  
  # ------------------------ Data Transformation ------------------------
  # Example transformation: apply asinh transformation to counts.
  counts <- assay(spe, "counts")
  assay(spe, "exprs") <- asinh(counts)
  logger$log_info("Counts transformed using asinh.")
  
  # ------------------------ Debug: Check Available Metadata ------------------------
  # Log the available names in colData(spe) to identify the correct cell area attribute.
  available_cols <- colnames(colData(spe))
  logger$log_info("Available columns in colData(spe): %s", paste(available_cols, collapse = ", "))
  
  # Check if "cell_area" exists; if not, fall back to "area"
  if ("cell_area" %in% available_cols) {
    areas <- spe$cell_area
    area_label <- "cell_area"
  } else if ("area" %in% available_cols) {
    areas <- spe$area
    area_label <- "area"
  } else {
    stop("Neither 'cell_area' nor 'area' is available in colData(spe).")
  }
  
  logger$log_info("Using '%s' for filtering cell areas.", area_label)
  
  # ------------------------ Dynamic Quality Control Filtering with Debugging ------------------------
  # Print cell area summary and (if interactive) plot its histogram.
  logger$log_info("Cell %s summary before filtering:", area_label)
  logger$log_info("%s", paste(capture.output(summary(areas)), collapse = "\n"))
  
  if (interactive()) {
    tryCatch({
      library(ggplot2)
      p_area <- ggplot2::ggplot(as.data.frame(areas), ggplot2::aes(x = areas)) +
                 ggplot2::geom_histogram(bins = 50, fill = "steelblue", color = "black") +
                 ggplot2::theme_minimal() +
                 ggplot2::ggtitle(paste("Distribution of", area_label))
      print(p_area)
    }, error = function(e) {
      logger$log_warning("Could not generate histogram: %s", e$message)
    })
  }
  
  # ---- Option 1: Using Median/MAD Approach ----
  median_area <- median(areas, na.rm = TRUE)
  mad_area <- mad(areas, na.rm = TRUE)
  multiplier_mad <- 5   # Adjust multiplier if 3 is too strict.
  lower_bound_mad <- max(0, median_area - multiplier_mad * mad_area)
  upper_bound_mad <- median_area + multiplier_mad * mad_area
  logger$log_info("Using median/MAD with multiplier %d: lower bound = %.2f, upper bound = %.2f", 
                  multiplier_mad, lower_bound_mad, upper_bound_mad)
  
  # ---- Option 2: Using IQR Approach ----
  q1 <- quantile(areas, 0.25, na.rm = TRUE)
  q3 <- quantile(areas, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  lower_bound_iqr <- max(0, q1 - 1.5 * iqr)
  upper_bound_iqr <- q3 + 1.5 * iqr
  logger$log_info("Using IQR method: lower bound = %.2f, upper bound = %.2f", 
                  lower_bound_iqr, upper_bound_iqr)
  
  # Log the number of cells that would be retained using each method.
  n_cells_original <- ncol(spe)
  cells_mad <- which(areas >= lower_bound_mad & areas <= upper_bound_mad)
  cells_iqr <- which(areas >= lower_bound_iqr & areas <= upper_bound_iqr)
  
  logger$log_info("After median/MAD filtering, %d cells remain.", length(cells_mad))
  logger$log_info("After IQR filtering, %d cells remain.", length(cells_iqr))
  
  # ---- Choose a Filtering Method ----
  # For this example, we choose the median/MAD method if any cell remains.
  # Otherwise, fall back to the IQR method.
  if (length(cells_mad) > 0) {
    chosen_method <- "median/MAD"
    spe <- spe[, cells_mad]
  } else {
    chosen_method <- "IQR"
    spe <- spe[, cells_iqr]
  }
  
  logger$log_info("Filtering using the %s method, resulting in %d cells (from %d).",
                  chosen_method, ncol(spe), n_cells_original)
  
  # ------------------------ Feature Selection ------------------------
  # Flag channels that are of interest (exclude segmentation markers like DNA/Histone)
  rowData(spe)$use_channel <- !grepl("DNA|Histone", rownames(spe))
  logger$log_info("Flagged channels for downstream analysis.")
  
  # ------------------------ Dimensionality Reduction ------------------------
  # Perform UMAP and t-SNE using the selected channels.
  library(scater)
  library(BiocSingular)  # Ensure BiocSingular is available
  set.seed(220225)
  
  # Precompute PCA on the selected channels using full SVD via ExactParam()
  spe <- runPCA(spe, 
                subset_row = rowData(spe)$use_channel, 
                exprs_values = "exprs",
                BSPARAM = ExactParam())
  
  # Determine the number of available PCA dimensions.
  pca_dims <- ncol(reducedDim(spe, "PCA"))
  # Use the minimum of 50 or the available PCA dimensions.
  n_dims <- min(50, pca_dims)
  logger$log_info("Using %d PCA dimensions (from %d available) for UMAP.", n_dims, pca_dims)
  
  # Run UMAP using the precomputed PCA.
  spe <- runUMAP(spe, 
                 dimred = "PCA", 
                 n_dimred = n_dims)
                 
  # Continue with t-SNE as before.
  spe <- runTSNE(spe, 
                 subset_row = rowData(spe)$use_channel, 
                 exprs_values = "exprs")
  
  logger$log_info("Dimensionality reduction completed (UMAP and t-SNE).")
  
  # Debug: Check variance explained for each PCA component.
  pca_results <- reducedDim(spe, "PCA")
  if(!is.null(attr(pca_results, "percentVar"))) {
    percent_var <- attr(pca_results, "percentVar")
    logger$log_info("Variance explained by PCA: %s", paste(round(percent_var, 2), collapse = ", "))
  } else {
    logger$log_info("PCA variance explained information is not available.")
  }
  
  # ------------------------ Save Processed Data ------------------------
  saveRDS(spe, file = file.path(configManager$config$output$dir, "spe_processed.rds"))
  logger$log_info("Processed spe saved to output directory: %s", configManager$config$output$dir)
  
  # ------------------------ Exploratory Visualization (Optional) ------------------------
  if (interactive()) {
    message("Generating exploratory plots...")
    
    # Load additional required packages.
    library(dittoSeq)
    library(ggplot2)
    library(viridis)
    
    if (is.null(colnames(spe)) || all(colnames(spe) == "")) {
      colnames(spe) <- paste0("cell", seq_len(ncol(spe)))
      logger$log_info("No cell names detected; default names assigned.")
    } else {
      logger$log_info("Cell names are present. First ten: %s", paste(colnames(spe)[1:10], collapse=", "))
    }
    
    # UMAP plot colored by Mouse (instead of patient_id)
    p1 <- dittoDimPlot(spe, var = "Mouse", reduction.use = "UMAP", size = 0.2) +
      scale_color_viridis(discrete = TRUE) +
      ggtitle("Mouse ID on UMAP")
    
    # UMAP plot colored by expression of a marker (e.g., CD45)
    p2 <- dittoDimPlot(spe, var = "CD45", reduction.use = "UMAP", assay = "exprs", size = 0.2) +
      scale_color_viridis(name = "CD45") +
      ggtitle("CD45 Expression on UMAP")
    
    # UMAP plot colored by condition
    p <- dittoDimPlot(spe, var = "Condition", reduction.use = "UMAP", size = 0.2) +
      scale_color_viridis(discrete = TRUE) +
      ggtitle("Condition on UMAP")
    
    # UMAP plot colored by cell id
    # p_cell_id <- dittoDimPlot(spe, var = "cell_id", reduction.use = "UMAP", size = 0.2) +
    #   ggtitle("UMAP Plot Colored by Cell ID")
    
    # Print the plots
    print(p1)
    print(p2)
    print(p)
    # print(p_cell_id)
    
    # Save the plots
    ggsave(file.path(configManager$config$output$dir, "UMAP_mouseID.png"), plot = p1, width = 7, height = 5)
    ggsave(file.path(configManager$config$output$dir, "UMAP_CD45.png"), plot = p2, width = 7, height = 5)
    ggsave(file.path(configManager$config$output$dir, "UMAP_Condition.png"), plot = p, width = 7, height = 5)
    # ggsave(file.path(configManager$config$output$dir, "UMAP_cell_id.png"), plot = p_cell_id, width = 7, height = 5)
  }
  
  return(spe)
}

if (interactive() || identical(environment(), globalenv())) {
  spe_processed <- runProcessSteinbockData()
} 