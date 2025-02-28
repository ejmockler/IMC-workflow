library(ggplot2)
library(dplyr)
library(tidyr)

# Define the cell markers for each cell type (should match the definitions in your validation code)
cell_markers <- list(
  macrophages = c("CD45", "CD11b", "CD206"),
  neutrophils  = c("CD45", "Ly6G"),
  endothelial  = c("CD31"),
  fibroblasts  = c("CD140b")
)

# This function visualizes the proportion of cells (per valid sample/day) that are positive for the specified cell type markers.
visualize_cell_type_proportions <- function(spe, imc_info) {
  # Check that the exprs assay exists in the SPE object
  if (!("exprs" %in% names(assays(spe)))) {
    stop("The SpatialExperiment object does not have an 'exprs' assay")
  }
  
  # Extract expression data from the 'exprs' assay
  expr_data <- assay(spe, "exprs")
  
  # Extract sample numbers from colData; assumes sample_id format "IMC_241218_Alun_XXXX"
  sample_ids <- colData(spe)$sample_id
  sample_num <- as.numeric(gsub(".*_(\\d+)$", "\\1", sample_ids))
  
  # Build lookup table from imc_info: for each sample number, retrieve the corresponding day
  sample_day_lookup <- imc_info %>%
    group_by(sample_num) %>%
    summarize(day = first(day), .groups = "drop")
  
  # Map each cell's sample number to its day using the lookup
  day_vec <- sample_day_lookup$day[match(sample_num, sample_day_lookup$sample_num)]
  
  # Build a cell-level info dataframe
  cell_info <- data.frame(
    cell_id = colnames(spe),
    sample_num = sample_num,
    day = day_vec,
    stringsAsFactors = FALSE
  )
  
  # Remove cells that do not have a valid day mapping (i.e. those not found in imc_info)
  valid_cells <- !is.na(cell_info$day)
  if (sum(valid_cells) == 0){
    stop("No cells have valid day information. Check your sample_id format or imc_info.")
  }
  cell_info <- cell_info[valid_cells, ]
  expr_data <- expr_data[, valid_cells, drop = FALSE]
  
  # Initialize a data frame to store positivity per cell for each cell type
  cell_positive <- data.frame(
    cell_id = cell_info$cell_id,
    sample_num = cell_info$sample_num,
    day = cell_info$day,
    stringsAsFactors = FALSE
  )
  
  # For each cell type, compute the mean expression of the required markers and use a 75th percentile threshold for positivity.
  for (cell_type in names(cell_markers)) {
    markers <- cell_markers[[cell_type]]
    # Only use markers that exist in the expression matrix
    available_markers <- markers[markers %in% rownames(expr_data)]
    if (length(available_markers) == 0) {
      warning(sprintf("No markers found in exprs for cell type: %s", cell_type))
      cell_positive[[cell_type]] <- NA
      next
    }
    # Compute per-cell mean expression for the available markers
    mean_expr <- colMeans(expr_data[available_markers, , drop = FALSE])
    # Compute threshold as the 75th percentile across cells (heuristic threshold)
    threshold <- quantile(mean_expr, 0.75, na.rm = TRUE)
    # Mark cell as positive if the mean expression exceeds the threshold
    cell_positive[[cell_type]] <- mean_expr > threshold
  }
  
  # Reshape the data to long format for plotting.
  proportions <- cell_positive %>%
    pivot_longer(cols = all_of(names(cell_markers)), names_to = "cell_type", values_to = "positive") %>%
    group_by(day, cell_type) %>%
    summarize(
      n_positive = sum(positive, na.rm = TRUE),
      total = n(),
      proportion = n_positive / total,
      .groups = 'drop'
    )
  
  # Modify the plotting section to create faceted bar plots with error bars
  p <- ggplot(proportions, aes(x = factor(day), y = proportion)) +
    geom_bar(stat = "identity", fill = "lightblue", alpha = 0.7) +
    facet_wrap(~cell_type, scales = "free_y", ncol = 2) +
    labs(title = "Cell Type Marker Positivity Over Time",
         x = "Day",
         y = "Proportion of Positive Cells") +
    theme_minimal() +
    theme(
      strip.text = element_text(size = 12, face = "bold"),
      axis.title = element_text(size = 11),
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      panel.grid.major.x = element_blank(),
      panel.border = element_rect(color = "grey80", fill = NA)
    )
  
  # Add percentage labels on top of the bars
  p <- p + geom_text(
    aes(label = sprintf("%.1f%%", proportion * 100)),
    vjust = -0.5,
    size = 3
  )
  
  # Adjust y-axis to show percentages
  p <- p + scale_y_continuous(
    labels = scales::percent,
    limits = function(x) c(0, max(x) * 1.1) # Add 10% padding for labels
  )
  
  print(p)
  return(proportions)
}

# Function to visualize a random IMC image with cell masks
visualize_random_image <- function(spe, images, masks) {
  # Get a random sample ID from the SPE object
  sample_ids <- unique(colData(spe)$sample_id)
  random_sample <- sample(sample_ids, 1)
  
  # Extract the corresponding image and mask
  img_idx <- which(names(images) == random_sample)
  
  if (length(img_idx) == 0) {
    stop("No matching image found for sample: ", random_sample)
  }
  
  # Select markers to visualize (ensure they exist in the data)
  available_markers <- channelNames(images[[1]])
  message("Available markers: ", paste(available_markers, collapse = ", "))
  
  # Select markers that exist in our data
  default_markers <- c("CD45", "CD11b", "CD31", "CD140b")
  markers_to_show <- intersect(default_markers, available_markers)
  
  if (length(markers_to_show) == 0) {
    # If none of our default markers are available, just take the first few
    markers_to_show <- head(available_markers, 4)
  }
  
  message("Visualizing markers: ", paste(markers_to_show, collapse = ", "))
  
  # Create a single-image CytoImageList with metadata
  cur_image <- images[img_idx]
  cur_mask <- masks[img_idx]
  
  # Add required metadata
  mcols(cur_image)$img_id <- random_sample
  
  # Create the plot
  message("Visualizing sample: ", random_sample)
  plotPixels(
    image = cur_image,
    mask = cur_mask,
    img_id = random_sample,
    colour_by = markers_to_show,
    bcg = list(q = c(0.01, 0.99)),  # Auto-contrast
    scale_bar = list(length = 50, label = "50 µm"),
    return_plot = TRUE,
    display = "all"  # Show all markers side by side
  )
}

# Example usage:
# After running your validation pipeline which produces validated_data, you can visualize the proportions:
#
proportions <- visualize_cell_type_proportions(validated_data$data$spe, validated_data$imc_info) 

# Example usage:
# After running your validation pipeline:
#
random_image <- visualize_random_image(validated_data$data$spe, 
                                     validated_data$data$images, 
                                     validated_data$data$masks) 