# phenotype_visualization.R
# Functions for cell phenotype visualizations

# Load required libraries
library(ggplot2)
library(SpatialExperiment)

# Source core utilities
source("src/visualization/core_visualization.R")

#' Plot spatial distribution of cell phenotypes
#'
#' @param spe SpatialExperiment object
#' @param phenotype_column Column with phenotype information (default: "phenotype")
#' @param sample_id Sample ID to plot (if NULL, plots all samples)
#' @param img_id Column name for sample/image ID (default: "sample_id")
#' @param point_size Size of points (default: 1)
#' @param title Plot title (default: "Phenotype Spatial Distribution")
#' @return A ggplot object
plot_phenotype_spatial <- function(
  spe,
  phenotype_column = "phenotype",
  sample_id = NULL,
  img_id = "sample_id",
  point_size = 1,
  title = "Phenotype Spatial Distribution"
) {
  # Try alternative column names if default not found
  if (!phenotype_column %in% colnames(colData(spe))) {
    for (col in c("celltype", "cluster", "phenograph_corrected")) {
      if (col %in% colnames(colData(spe))) {
        phenotype_column <- col
        break
      }
    }
  }
  
  # Check if phenotype column exists
  if (!phenotype_column %in% colnames(colData(spe))) {
    stop("No suitable phenotype column found in SpatialExperiment object")
  }
  
  # Filter to specific sample if requested
  plot_spe <- spe
  if (!is.null(sample_id) && img_id %in% colnames(colData(spe))) {
    plot_spe <- spe[, spe[[img_id]] == sample_id]
    if (ncol(plot_spe) == 0) {
      stop(paste("No cells found for sample_id:", sample_id))
    }
  }
  
  # Create plot data
  plot_data <- data.frame(
    x = spatialCoords(plot_spe)[,1],
    y = spatialCoords(plot_spe)[,2],
    phenotype = as.factor(plot_spe[[phenotype_column]])
  )
  
  # If multiple samples, add sample information
  if (is.null(sample_id) && img_id %in% colnames(colData(plot_spe))) {
    plot_data$sample <- plot_spe[[img_id]]
    
    # Create plot with faceting
    p <- ggplot(plot_data, aes(x = x, y = y, color = phenotype)) +
      geom_point(size = point_size) +
      facet_wrap(~ sample, scales = "free") +
      theme_minimal() +
      labs(
        title = title,
        color = phenotype_column
      )
  } else {
    # Create plot without faceting
    p <- ggplot(plot_data, aes(x = x, y = y, color = phenotype)) +
      geom_point(size = point_size) +
      theme_minimal() +
      labs(
        title = title,
        color = phenotype_column
      )
  }
  
  # Add custom color scheme if available
  if ("color_vectors" %in% names(metadata(spe)) &&
      phenotype_column %in% names(metadata(spe)$color_vectors)) {
    p <- p + ggplot2::scale_color_manual(values = metadata(spe)$color_vectors[[phenotype_column]])
  }
  
  return(p)
}

#' Plot cell phenotype compositions
#'
#' @param spe SpatialExperiment object
#' @param phenotype_column Column with phenotype information (default: "phenotype")
#' @param group_by Optional column to group by
#' @param sort Whether to sort phenotypes by frequency (default: TRUE)
#' @param plot_type Type of plot: "bar" or "pie" (default: "bar")
#' @param title Plot title (default: "Phenotype Composition")
#' @return A ggplot object
plot_phenotype_composition <- function(
  spe,
  phenotype_column = "phenotype",
  group_by = NULL,
  sort = TRUE,
  plot_type = c("bar", "pie"),
  title = "Phenotype Composition"
) {
  plot_type <- match.arg(plot_type)
  
  # Try alternative column names if default not found
  if (!phenotype_column %in% colnames(colData(spe))) {
    for (col in c("celltype", "cluster", "phenograph_corrected")) {
      if (col %in% colnames(colData(spe))) {
        phenotype_column <- col
        break
      }
    }
  }
  
  # Check if phenotype column exists
  if (!phenotype_column %in% colnames(colData(spe))) {
    stop("No suitable phenotype column found in SpatialExperiment object")
  }
  
  # Calculate compositions
  if (is.null(group_by)) {
    # Overall compositions
    counts <- table(spe[[phenotype_column]])
    props <- prop.table(counts) * 100
    
    if (sort) {
      props <- sort(props, decreasing = TRUE)
    }
    
    plot_data <- data.frame(
      Phenotype = names(props),
      Proportion = as.numeric(props)
    )
    
    # Create bar plot
    if (plot_type == "bar") {
      p <- ggplot(plot_data, aes(x = Phenotype, y = Proportion, fill = Phenotype)) +
        geom_bar(stat = "identity") +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
        labs(
          title = title,
          x = phenotype_column,
          y = "Proportion (%)"
        )
    } else {
      # Create pie chart
      p <- ggplot(plot_data, aes(x = "", y = Proportion, fill = Phenotype)) +
        geom_bar(stat = "identity", width = 1) +
        coord_polar("y", start = 0) +
        theme_minimal() +
        theme(
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          panel.border = element_blank(),
          panel.grid = element_blank(),
          axis.ticks = element_blank(),
          axis.text.x = element_blank()
        ) +
        labs(title = title)
    }
    
  } else {
    # Group-specific compositions
    if (!group_by %in% colnames(colData(spe))) {
      stop(paste("Column", group_by, "not found in SpatialExperiment object"))
    }
    
    counts <- table(spe[[group_by]], spe[[phenotype_column]])
    props <- prop.table(counts, margin = 1) * 100
    
    plot_data <- as.data.frame.table(props)
    names(plot_data) <- c("Group", "Phenotype", "Proportion")
    
    # Create grouped bar plot
    p <- ggplot(plot_data, aes(x = Group, y = Proportion, fill = Phenotype)) +
      geom_bar(stat = "identity", position = position_stack()) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      labs(
        title = title,
        x = group_by,
        y = "Proportion (%)"
      )
  }
  
  # Add custom color scheme if available
  if ("color_vectors" %in% names(metadata(spe)) &&
      phenotype_column %in% names(metadata(spe)$color_vectors)) {
    p <- p + ggplot2::scale_fill_manual(values = metadata(spe)$color_vectors[[phenotype_column]])
  }
  
  return(p)
}

#' Plot neighborhood enrichment between phenotypes
#'
#' @param spe SpatialExperiment object with neighborhood information
#' @param phenotype_column Column with phenotype information (default: "phenotype")
#' @param neighborhood_name Name of the neighborhood measure (default: "neighborhood")
#' @param normalize Whether to normalize the interaction strengths (default: TRUE)
#' @param title Plot title (default: "Phenotype Neighborhood Enrichment")
#' @return A ggplot object
plot_phenotype_neighborhood <- function(
  spe,
  phenotype_column = "phenotype",
  neighborhood_name = "neighborhood",
  normalize = TRUE,
  title = "Phenotype Neighborhood Enrichment"
) {
  # Try alternative column names if default not found
  if (!phenotype_column %in% colnames(colData(spe))) {
    for (col in c("celltype", "cluster", "phenograph_corrected")) {
      if (col %in% colnames(colData(spe))) {
        phenotype_column <- col
        break
      }
    }
  }
  
  # Check if phenotype column exists
  if (!phenotype_column %in% colnames(colData(spe))) {
    stop("No suitable phenotype column found in SpatialExperiment object")
  }
  
  # Check if neighborhood data exists
  neighborhood_pattern <- paste0("^", neighborhood_name)
  if (!any(grepl(neighborhood_pattern, colPairNames(spe)))) {
    stop(paste("No neighborhood data found with name pattern:", neighborhood_pattern))
  }
  
  # Extract neighborhood data
  neighborhood_matrices <- colPairNames(spe)[grepl(neighborhood_pattern, colPairNames(spe))]
  
  # We'll use the first neighborhood matrix for now
  neighborhood_matrix <- neighborhood_matrices[1]
  
  # Get phenotype information
  phenotypes <- spe[[phenotype_column]]
  unique_phenotypes <- unique(phenotypes)
  
  # Create interaction count matrix
  interaction_counts <- matrix(0, 
                             nrow = length(unique_phenotypes), 
                             ncol = length(unique_phenotypes),
                             dimnames = list(unique_phenotypes, unique_phenotypes))
  
  # Extract interaction pairs
  pairs <- as.matrix(colPair(spe, neighborhood_matrix))
  
  # Count interactions between phenotypes
  for (i in seq_len(nrow(pairs))) {
    from_cell <- pairs[i, 1]
    to_cell <- pairs[i, 2]
    from_type <- phenotypes[from_cell]
    to_type <- phenotypes[to_cell]
    interaction_counts[from_type, to_type] <- interaction_counts[from_type, to_type] + 1
  }
  
  # Normalize if requested
  if (normalize) {
    # Calculate expected frequencies
    phenotype_counts <- table(phenotypes)
    expected <- outer(phenotype_counts, phenotype_counts) / sum(phenotype_counts)
    
    # Calculate interaction enrichment (observed / expected)
    interaction_counts <- interaction_counts / expected
    interaction_counts[is.na(interaction_counts)] <- 0
    interaction_counts[is.infinite(interaction_counts)] <- 0
  }
  
  # Convert to long format for ggplot
  plot_data <- reshape2::melt(interaction_counts, varnames = c("From", "To"), 
                            value.name = "Interaction")
  
  # Create heatmap
  p <- ggplot(plot_data, aes(x = From, y = To, fill = Interaction)) +
    geom_tile() +
    scale_fill_gradient2(
      low = "blue", mid = "white", high = "red",
      midpoint = if(normalize) 1 else median(plot_data$Interaction)
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(
      title = title,
      x = phenotype_column,
      y = phenotype_column,
      fill = if(normalize) "Enrichment" else "Count"
    )
  
  return(p)
}

#' Create all phenotype visualizations
#'
#' @param spe SpatialExperiment object
#' @param output_dir Directory to save plots to
#' @param config Visualization configuration
#' @return Invisibly returns the output directory
create_phenotype_visualizations <- function(spe, output_dir, config = NULL) {
  # Use default config if not provided
  if (is.null(config)) {
    config <- get_default_viz_config()
  }
  
  # Determine phenotype column
  phenotype_column <- NULL
  for (col in c("phenotype", "celltype", "cluster", "phenograph_corrected")) {
    if (col %in% colnames(colData(spe))) {
      phenotype_column <- col
      break
    }
  }
  
  if (is.null(phenotype_column)) {
    message("No suitable phenotype column found. Skipping phenotype visualizations.")
    return(invisible(output_dir))
  }
  
  # Create spatial distribution plot
  message("Creating phenotype spatial distribution plot...")
  p <- plot_phenotype_spatial(
    spe,
    phenotype_column = phenotype_column,
    point_size = config$point_size,
    title = paste(phenotype_column, "Spatial Distribution")
  )
  
  save_plot(
    p,
    file.path(output_dir, "phenotype_spatial.pdf"),
    width = config$dimensions$width,
    height = config$dimensions$height,
    dpi = config$dimensions$dpi
  )
  
  # Create composition bar plot
  message("Creating phenotype composition plot...")
  p <- plot_phenotype_composition(
    spe,
    phenotype_column = phenotype_column,
    plot_type = "bar",
    title = paste(phenotype_column, "Composition")
  )
  
  save_plot(
    p,
    file.path(output_dir, "phenotype_composition.pdf"),
    width = config$dimensions$width,
    height = config$dimensions$height,
    dpi = config$dimensions$dpi
  )
  
  # Create pie chart
  message("Creating phenotype pie chart...")
  p <- plot_phenotype_composition(
    spe,
    phenotype_column = phenotype_column,
    plot_type = "pie",
    title = paste(phenotype_column, "Composition")
  )
  
  save_plot(
    p,
    file.path(output_dir, "phenotype_pie.pdf"),
    width = config$dimensions$width,
    height = config$dimensions$height,
    dpi = config$dimensions$dpi
  )
  
  # Create neighborhood enrichment plot if available
  if (any(grepl("neighborhood", colPairNames(spe)))) {
    message("Creating phenotype neighborhood enrichment plot...")
    p <- plot_phenotype_neighborhood(
      spe,
      phenotype_column = phenotype_column,
      title = paste(phenotype_column, "Neighborhood Enrichment")
    )
    
    save_plot(
      p,
      file.path(output_dir, "phenotype_neighborhood.pdf"),
      width = config$dimensions$width,
      height = config$dimensions$height,
      dpi = config$dimensions$dpi
    )
  }
  
  # Create UMAP with phenotypes if available
  if (all(c("UMAP1", "UMAP2") %in% colnames(colData(spe)))) {
    message("Creating UMAP colored by phenotypes...")
    
    # Create plot data
    plot_data <- data.frame(
      UMAP1 = spe[["UMAP1"]],
      UMAP2 = spe[["UMAP2"]],
      phenotype = as.factor(spe[[phenotype_column]])
    )
    
    # Create UMAP plot
    p <- ggplot(plot_data, aes(x = UMAP1, y = UMAP2, color = phenotype)) +
      geom_point(size = config$point_size) +
      theme_minimal() +
      labs(
        title = paste("UMAP -", phenotype_column),
        color = phenotype_column
      )
    
    # Add custom color scheme if available
    if ("color_vectors" %in% names(metadata(spe)) &&
        phenotype_column %in% names(metadata(spe)$color_vectors)) {
      p <- p + ggplot2::scale_color_manual(values = metadata(spe)$color_vectors[[phenotype_column]])
    }
    
    save_plot(
      p,
      file.path(output_dir, "phenotype_umap.pdf"),
      width = config$dimensions$width,
      height = config$dimensions$height,
      dpi = config$dimensions$dpi
    )
  }
  
  return(invisible(output_dir))
} 