# core_visualization.R
# Core utility functions for visualization components

#' Save a ggplot object to a file
#'
#' @param plot ggplot object to save
#' @param filename Output filename
#' @param width Plot width in inches
#' @param height Plot height in inches
#' @param dpi Resolution in dots per inch
#' @return Invisibly returns the filename
save_plot <- function(plot, filename, width = 10, height = 8, dpi = 300) {
  # Create directory if it doesn't exist
  dir.create(dirname(filename), showWarnings = FALSE, recursive = TRUE)
  
  # Determine file format from extension
  ext <- tolower(tools::file_ext(filename))
  
  # Save plot
  if (ext == "pdf") {
    pdf(filename, width = width, height = height)
    print(plot)
    dev.off()
  } else if (ext %in% c("png", "jpg", "jpeg", "tiff")) {
    if (ext == "jpg") ext <- "jpeg"
    do.call(ext, list(filename = filename, width = width, height = height, 
                     units = "in", res = dpi))
    print(plot)
    dev.off()
  } else {
    warning(paste("Unsupported file format:", ext, "- saving as PDF"))
    pdf_filename <- paste0(tools::file_path_sans_ext(filename), ".pdf")
    pdf(pdf_filename, width = width, height = height)
    print(plot)
    dev.off()
    return(invisible(pdf_filename))
  }
  
  message(paste("Saved plot to", filename))
  return(invisible(filename))
}

#' Get default visualization configuration
#'
#' @return A list containing default visualization parameters
get_default_viz_config <- function() {
  list(
    dimensions = list(
      width = 10, 
      height = 8,
      dpi = 300
    ),
    color_palettes = list(
      markers = c("black", "blue", "cyan", "green", "yellow", "red"),
      phenotypes = rainbow(10),
      communities = viridis::viridis(10),
      default = RColorBrewer::brewer.pal(9, "Set1")
    ),
    point_size = 1.5,
    theme = ggplot2::theme_minimal()
  )
}

#' Combine multiple plots into a single figure
#'
#' @param plot_list List of ggplot objects
#' @param ncol Number of columns in the grid
#' @param nrow Number of rows in the grid
#' @param titles Optional vector of titles for each plot
#' @return A combined plot using patchwork
combine_plots <- function(plot_list, ncol = NULL, nrow = NULL, titles = NULL) {
  if (!requireNamespace("patchwork", quietly = TRUE)) {
    stop("Package 'patchwork' is required for combine_plots()")
  }
  
  # Add titles if provided
  if (!is.null(titles) && length(titles) == length(plot_list)) {
    for (i in seq_along(plot_list)) {
      plot_list[[i]] <- plot_list[[i]] + ggplot2::ggtitle(titles[i])
    }
  }
  
  # Combine plots
  combined <- patchwork::wrap_plots(plot_list, ncol = ncol, nrow = nrow)
  return(combined)
}

#' Safely extract a column from SpatialExperiment object
#'
#' @param spe SpatialExperiment object
#' @param column_name Name of the column to extract
#' @param default Default value if column doesn't exist
#' @return Vector of values from the column, or default if it doesn't exist
safe_get_column <- function(spe, column_name, default = NULL) {
  if (column_name %in% colnames(colData(spe))) {
    return(spe[[column_name]])
  } else {
    warning(paste("Column", column_name, "not found in SpatialExperiment object"))
    return(default)
  }
}

#' Scale values for better visualization
#'
#' @param x Numeric vector to scale
#' @param method Scaling method: "minmax", "robust", or "zscore"
#' @return Scaled numeric vector
scale_values <- function(x, method = c("robust", "minmax", "zscore")) {
  method <- match.arg(method)
  
  if (method == "minmax") {
    # Min-max scaling to [0,1]
    result <- (x - min(x, na.rm = TRUE)) / 
      (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
  } else if (method == "robust") {
    # Robust scaling using quantiles
    q1 <- quantile(x, 0.01, na.rm = TRUE)
    q99 <- quantile(x, 0.99, na.rm = TRUE)
    result <- (x - q1) / (q99 - q1)
    result[result < 0] <- 0
    result[result > 1] <- 1
  } else if (method == "zscore") {
    # Z-score standardization
    result <- (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
  }
  
  return(result)
} 