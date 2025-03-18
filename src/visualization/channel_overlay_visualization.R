# channel_overlay_visualization.R
# Functions for creating channel overlays and previewing single-cell images

# Load required libraries
library(cytomapper)
library(EBImage)
library(viridis)
library(ggplot2)
library(gridExtra)
library(cowplot)
library(dplyr)

#' Create composite overlay visualizations from all IMC channels
#'
#' @param images CytoImageList containing the IMC channel images
#' @param masks CytoImageList containing cell segmentation masks (optional)
#' @param spe SpatialExperiment object with cell data
#' @param output_dir Directory to save visualization outputs
#' @param channels_to_highlight Optional vector of channel names to highlight in separate panels
#' @param max_cells Maximum number of single cells to display (default: 25)
#' @param color_scheme Color scheme for visualization (default: "viridis")
#' @param width Plot width in inches (default: 12)
#' @param height Plot height in inches (default: 10)
#' @param dpi Resolution in dots per inch (default: 300)
#' @return Invisibly returns paths to saved files
create_channel_overlay_visualization <- function(
  images,
  masks = NULL,
  spe = NULL,
  output_dir = "results/visualizations/channel_overlays",
  channels_to_highlight = NULL,
  max_cells = 25,
  color_scheme = "viridis",
  width = 12,
  height = 10,
  dpi = 300
) {
  # Debug information
  cat("DEBUG: create_channel_overlay_visualization called\n")
  if (is.null(images)) {
    stop("Images CytoImageList is NULL")
  }
  
  # Get channel names with more robust error handling
  tryCatch({
    channel_names <- channelNames(images)
    if (is.null(channel_names) || length(channel_names) == 0) {
      # Handle case where channel names are missing
      warning("No channel names found in images, using numbered channels instead")
      if (length(dim(images[[1]])) == 3) {
        # Create numbered channel names based on third dimension
        n_channels <- dim(images[[1]])[3]
        channel_names <- paste0("Channel_", 1:n_channels)
        # Set these names in the image object
        cytomapper::channelNames(images) <- channel_names
      } else {
        stop("Cannot determine number of channels in images")
      }
    }
    cat("DEBUG: Number of channels:", length(channel_names), "\n")
    cat("DEBUG: Available channels:", paste(channel_names, collapse = ", "), "\n")
  }, error = function(e) {
    cat("DEBUG: Error getting channel names:", e$message, "\n")
    stop("Failed to get channel names from images: ", e$message)
  })
  
  # Create output directory
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  # If no channels to highlight are specified, use all channels
  if (is.null(channels_to_highlight)) {
    # By default, we'll use all available channels
    channels_to_highlight <- channel_names
  } else {
    # Validate that specified channels exist
    invalid_channels <- setdiff(channels_to_highlight, channel_names)
    if (length(invalid_channels) > 0) {
      warning(paste("The following channels do not exist:", 
                   paste(invalid_channels, collapse = ", ")))
      channels_to_highlight <- intersect(channels_to_highlight, channel_names)
    }
  }
  
  # Generate result file paths
  composite_path <- file.path(output_dir, "composite_overlay.pdf")
  heatmap_path <- file.path(output_dir, "channel_heatmap.pdf")
  single_cell_path <- file.path(output_dir, "single_cell_preview.pdf")
  
  # 1. Create composite overlay of all channels
  cat("DEBUG: Creating composite overlay visualization\n")
  tryCatch({
    # Select a subset of visually distinctive channels for RGB overlay
    # Typically DNA1, a membrane marker, and a functional marker work well
    rgb_channels <- c()
    
    # Common channel patterns to look for
    dna_pattern <- "DNA|Ir19[1-3]|Histone"
    membrane_pattern <- "CD45|HLA|Pan|CD3[1-9]|CD4[4-9]"
    functional_pattern <- "Ki67|CD38|pSTAT|FOXP3"
    
    # Find DNA marker
    dna_channel <- grep(dna_pattern, channel_names, value = TRUE)
    if (length(dna_channel) > 0) {
      dna_channel <- dna_channel[1]
      rgb_channels <- c(rgb_channels, dna_channel)
    }
    
    # Find membrane marker
    membrane_channel <- grep(membrane_pattern, channel_names, value = TRUE)
    if (length(membrane_channel) > 0) {
      membrane_channel <- membrane_channel[1]
      rgb_channels <- c(rgb_channels, membrane_channel)
    }
    
    # Find functional marker
    functional_channel <- grep(functional_pattern, channel_names, value = TRUE)
    if (length(functional_channel) > 0) {
      functional_channel <- functional_channel[1]
      rgb_channels <- c(rgb_channels, functional_channel)
    }
    
    # If we couldn't find good channels, just take the first 3
    if (length(rgb_channels) < 3 && length(channel_names) >= 3) {
      rgb_channels <- channel_names[1:3]
    }
    
    # Create RGB composite overlay
    pdf(composite_path, width = width, height = height)
    
    # For each image in the CytoImageList
    for (img_idx in seq_along(images)) {
      # Get image name
      img_name <- names(images)[img_idx]
      
      # Create RGB composite
      cat("DEBUG: Creating RGB composite for image:", img_name, "\n")
      
      # Create a plot for this image
      layout(matrix(c(1, 2, 3, 4), 2, 2))
      
      # Plot RGB overlay
      composite_img <- NULL
      if (length(rgb_channels) >= 3) {
        composite_img <- EBImage::rgbImage(
          red = normalize_image(images[[img_idx]][,,rgb_channels[1]]),
          green = normalize_image(images[[img_idx]][,,rgb_channels[2]]),
          blue = normalize_image(images[[img_idx]][,,rgb_channels[3]])
        )
        EBImage::display(composite_img, method = "raster", all = TRUE, 
                        title = paste("RGB Composite:", img_name, "-", 
                                     paste(rgb_channels, collapse = "/")))
      } else {
        # If we don't have 3 channels, just show the first channel
        EBImage::display(normalize_image(images[[img_idx]][,,1]), method = "raster", 
                        title = paste("Single Channel:", img_name, "-", channel_names[1]))
      }
      
      # Plot masks if available
      if (!is.null(masks) && length(masks) >= img_idx) {
        # Display the mask
        mask_img <- masks[[img_idx]]
        EBImage::display(normalize_image(mask_img), method = "raster", 
                        title = paste("Cell Masks:", img_name))
      }
      
      # Display two additional highlighted channels
      if (length(channel_names) > 3) {
        # Show additional important channels
        additional_channels <- setdiff(channel_names, rgb_channels)[1:2]
        for (i in seq_along(additional_channels)[1:2]) {
          if (i <= length(additional_channels)) {
            chan <- additional_channels[i]
            EBImage::display(normalize_image(images[[img_idx]][,,chan]), method = "raster", 
                           title = paste("Channel:", chan))
          }
        }
      }
    }
    dev.off()
    cat("DEBUG: Composite overlay saved to", composite_path, "\n")
  }, error = function(e) {
    cat("DEBUG: Error creating composite overlay:", e$message, "\n")
  })
  
  # 2. Create heatmap of all channels
  cat("DEBUG: Creating channel heatmap visualization\n")
  tryCatch({
    pdf(heatmap_path, width = width, height = height)
    
    # For each image in the CytoImageList
    for (img_idx in seq_along(images)) {
      # Get image name
      img_name <- names(images)[img_idx]
      cat("DEBUG: Creating heatmap for image:", img_name, "\n")
      
      # Determine grid layout based on number of channels
      n_channels <- length(channels_to_highlight)
      n_cols <- min(4, n_channels)
      n_rows <- ceiling(n_channels / n_cols)
      
      # Set up plotting grid
      par(mfrow = c(n_rows, n_cols))
      
      # Plot each channel as a heatmap
      for (chan in channels_to_highlight) {
        # Extract channel data
        chan_data <- images[[img_idx]][,,chan]
        
        # Normalize and display as heatmap
        chan_data_norm <- normalize_image(chan_data)
        image(chan_data_norm, col = viridis::viridis(100), 
              main = paste(chan, "-", img_name), axes = FALSE)
      }
    }
    dev.off()
    cat("DEBUG: Channel heatmap saved to", heatmap_path, "\n")
  }, error = function(e) {
    cat("DEBUG: Error creating channel heatmap:", e$message, "\n")
  })
  
  # 3. Create single-cell preview if we have masks and SPE
  if (!is.null(masks) && !is.null(spe)) {
    cat("DEBUG: Creating single-cell preview visualization\n")
    tryCatch({
      pdf(single_cell_path, width = width, height = height)
      
      # Extract single cells using the masks
      for (img_idx in seq_along(images)) {
        # Get image name
        img_name <- names(images)[img_idx]
        cat("DEBUG: Creating single-cell preview for image:", img_name, "\n")
        
        # Get cells from this image
        cells_in_img <- spe$sample_id == img_name
        
        if (sum(cells_in_img) > 0) {
          # Get cell IDs from this image
          cell_ids <- unique(masks[[img_idx]])
          cell_ids <- cell_ids[cell_ids > 0]  # Remove background (0)
          
          # Limit the number of cells to display
          n_cells <- min(length(cell_ids), max_cells)
          if (length(cell_ids) > max_cells) {
            set.seed(42)  # For reproducibility
            cell_ids <- sample(cell_ids, max_cells)
          }
          
          # Determine grid layout
          n_cols <- min(5, n_cells)
          n_rows <- ceiling(n_cells / n_cols)
          
          # Set up plotting grid
          par(mfrow = c(n_rows, n_cols))
          
          # Extract and display individual cells
          for (cell_id in cell_ids) {
            # Create mask for this specific cell
            cell_mask <- masks[[img_idx]] == cell_id
            
            # Select channels for RGB display (same as in composite)
            if (length(rgb_channels) >= 3) {
              # Create RGB composite for this cell
              cell_img <- EBImage::rgbImage(
                red = normalize_image(images[[img_idx]][,,rgb_channels[1]] * cell_mask),
                green = normalize_image(images[[img_idx]][,,rgb_channels[2]] * cell_mask),
                blue = normalize_image(images[[img_idx]][,,rgb_channels[3]] * cell_mask)
              )
            } else {
              # If we don't have 3 channels, just use the first channel
              cell_img <- normalize_image(images[[img_idx]][,,1] * cell_mask)
            }
            
            # Display the cell
            EBImage::display(cell_img, method = "raster", all = TRUE,
                           title = paste("Cell", cell_id))
          }
        } else {
          # No cells found for this image
          plot(1, type = "n", axes = FALSE, xlab = "", ylab = "")
          text(1, 1, paste("No cells found for image:", img_name))
        }
      }
      dev.off()
      cat("DEBUG: Single-cell preview saved to", single_cell_path, "\n")
    }, error = function(e) {
      cat("DEBUG: Error creating single-cell preview:", e$message, "\n")
    })
  } else {
    cat("DEBUG: Skipping single-cell preview (missing masks or SPE)\n")
  }
  
  # Return the paths to the generated files
  result_paths <- c(
    composite = composite_path,
    heatmap = heatmap_path,
    single_cell = single_cell_path
  )
  
  return(invisible(result_paths))
}

#' Normalize image data for visualization
#'
#' @param x Image data to normalize
#' @param percentile Percentile to use for maximum value (default: 0.99)
#' @return Normalized image data in range [0,1]
normalize_image <- function(x, percentile = 0.99) {
  x_min <- min(x, na.rm = TRUE)
  x_max <- quantile(x, percentile, na.rm = TRUE)
  
  # Clip values above max
  x[x > x_max] <- x_max
  
  # Rescale to [0,1]
  x_norm <- (x - x_min) / (x_max - x_min)
  x_norm[x_norm < 0] <- 0
  x_norm[x_norm > 1] <- 1
  
  return(x_norm)
} 