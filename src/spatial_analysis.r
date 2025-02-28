# Ensure necessary packages are installed
ensure_packages <- function(packages) {
  missing_packages <- packages[
    !vapply(packages, requireNamespace, FUN.VALUE = logical(1), quietly = TRUE)
  ]
  if (length(missing_packages) > 0) {
    install.packages(missing_packages, repos = "http://cran.us.r-project.org")
  }
  invisible(NULL)
}

required_packages <- c("SpatialExperiment", "ggplot2", "dplyr", "FNN", "grid")
ensure_packages(required_packages)

# Load libraries
library(SpatialExperiment)
library(ggplot2)
library(dplyr)
library(FNN)      # for k-nearest neighbors
library(grid)     # for rasterGrob and image plotting

#----------------------------------
# 1. Load Data Files
#----------------------------------

# Load the SpatialExperiment data (cells, marker expressions, spatial coordinates, etc.)
spe <- readRDS("/Users/noot/Documents/IMC/data/spe.rds")

# Load the corresponding tissue image(s) - can be a list or a single image object.
images <- readRDS("/Users/noot/Documents/IMC/data/images.rds")

# Load segmentation masks or tissue region annotations.
masks <- readRDS("/Users/noot/Documents/IMC/data/masks.rds")

#----------------------------------
# 2. Inspect and Validate the Data
#----------------------------------

# Inspect spatial coordinates stored in the spe object
coords <- spatialCoords(spe)
print("Sample of spatial coordinates:")
print(head(coords))

# Check for cell-type annotations in the SpatialExperiment object
if ("cell_type" %in% colnames(colData(spe))) {
  print("Cell type distribution:")
  print(table(spe$cell_type))
}

#----------------------------------
# 3. Neighborhood Analysis using k-Nearest Neighbors
#----------------------------------

# Convert spatial coordinates to a data frame for neighbor analysis
df_coords <- as.data.frame(coords)

# Define k (number of neighbors) - adjust as needed (here we choose k = 6)
k_neighbors <- 6

# Calculate indices of the k-nearest neighbors for each cell
neighbors <- get.knn(df_coords, k = k_neighbors)$nn.index
print("Neighbors for the first few cells:")
print(head(neighbors))

# (Optionally, you may later integrate these neighbor indices with cell-type annotations
#  to build an interaction matrix, for example summarizing contacts between macrophages, endothelial cells, etc.)

#----------------------------------
# 4. Visualization: Overlay Tissue Image, Masks, and Cell Coordinates
#----------------------------------

# Process the tissue image: convert to raster depending on its class / type
if (inherits(images, "CytoImageList")) {
  # Extract the underlying image array from the CytoImageList object
  tissue_image_array <- imageData(images)[[1]]
  
  # Check if the object is numeric, indicating a grayscale or intensity image
  if (is.numeric(tissue_image_array)) {
    # Normalize the image array values to the range [0, 1]
    tissue_image_norm <- (tissue_image_array - min(tissue_image_array)) /
      (max(tissue_image_array) - min(tissue_image_array))
    
    # Create a color matrix from the normalized image by converting each value to a gray color
    # Here we use the rgb() function to map the normalized intensity to grayscale
    tissue_color_matrix <- matrix(rgb(tissue_image_norm, tissue_image_norm, tissue_image_norm),
                                  nrow = nrow(tissue_image_norm), ncol = ncol(tissue_image_norm))
    
    # Finally, convert the color matrix to a raster object
    tissue_img <- as.raster(tissue_color_matrix)
  } else {
    # If tissue_image_array is not numeric, try to convert it directly.
    tissue_img <- as.raster(tissue_image_array)
  }
} else {
  # For images that are not CytoImageList objects, assume they can be directly coerced.
  tissue_img <- as.raster(images)
}

# Create a base ggplot object using the tissue image as the background
p <- ggplot() +
  annotation_custom(
    rasterGrob(tissue_img, width = unit(1, "npc"), height = unit(1, "npc")),
    -Inf, Inf, -Inf, Inf
  ) +
  ggtitle("Spatial Layout with Tissue Image, Masks, and Cell Annotations") +
  theme_void()

# Prepare cell coordinates data.frame
cell_df <- as.data.frame(spatialCoords(spe))
# Ensure coordinate column names are set correctly
if (!("X" %in% names(cell_df))) {
  names(cell_df)[1:2] <- c("X", "Y")
}
if ("cell_type" %in% colnames(colData(spe))) {
  cell_df$cell_type <- colData(spe)$cell_type
} else {
  cell_df$cell_type <- "NA"
}

# Overlay cell positions onto the plot, coloring by cell type if available.
p <- p + geom_point(
  data = cell_df,
  aes(x = X, y = Y, color = cell_type),
  size = 1.5,
  alpha = 0.8
)

# If masks are available, overlay them as dashed polygon boundaries.
if (!is.null(masks) && length(masks) > 0) {
  # This example assumes a single polygon mask in masks[[1]]
  mask_df <- as.data.frame(masks[[1]])
  if(all(c("x", "y") %in% colnames(mask_df))){
    p <- p + geom_polygon(
      data = mask_df,
      aes(x = x, y = y),
      fill = NA,
      color = "blue",
      linetype = "dashed",
      size = 1
    )
  } else {
    warning("The mask data does not have the expected 'x' and 'y' columns.")
  }
}

# Print the final plot to visualize the spatial context
print(p)

#----------------------------------
# 5. Next Steps for Analysis
#----------------------------------

# The following are the next steps to further realize our analysis:
# a) Integrate cell-type specific neighbor interactions:
#    - Link neighbor indices with cell types to generate interaction matrices.
#
# b) Explore spatial metrics:
#    - Implement Ripley's K or textural point pattern analyses to statistically evaluate clustering.
#
# c) Expand temporal analysis:
#    - If separate spe, images, and masks exist for Day 1, Day 3, and Day 7,
#      create a loop or function to process and compare across these time points.
#
# d) Adjust parameter settings:
#    - Tune the k value in k-nearest neighbors or consider using distance thresholds
#      based on the tissue's spatial resolution (µm/pixel) and known cell sizes.
#
# e) Prepare intermediate reports:
#    - Generate summary statistics, spatial heatmaps, and interactive plots for review with your collaborators.