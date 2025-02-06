#' Factory for creating data loaders
#' @description Creates appropriate data loader instances based on data type
DataLoaderFactory <- R6::R6Class("DataLoaderFactory",
  public = list(
    #' Create a new loader instance
    #' @param data_type Type of data to load ("spe", "images", "masks", "panel", "imc")
    #' @param config Optional configuration parameters
    create_loader = function(data_type, config = NULL) {
      loader <- switch(data_type,
        "spe" = SpatialExperimentLoader$new(),
        "images" = ImageLoader$new(),
        "masks" = MaskLoader$new(),
        "panel" = PanelLoader$new(),
        "imc" = IMCLoader$new(),
        stop("Unknown data type: ", data_type)
      )
      
      if (!is.null(config)) {
        loader$configure(config)
      }
      
      loader
    },
    
    #' Load multiple data types at once
    #' @param types Vector of data types to load
    #' @param paths Named list of paths corresponding to data types
    load_multiple = function(types, paths) {
      loaded_data <- list()
      for (type in types) {
        loader <- self$create_loader(type)
        loaded_data[[type]] <- loader$load(paths[[type]])
      }
      loaded_data
    }
  )
)

#' Base loader class with enhanced validation
DataLoader <- R6::R6Class("DataLoader",
  public = list(
    config = NULL,
    
    configure = function(config) {
      self$config <- config
      invisible(self)
    },
    
    load = function(path) {
      private$validate_path(path)
      data <- private$do_load(path)
      private$validate_data(data)
      data
    }
  ),
  
  private = list(
    validate_path = function(path) {
      if (!file.exists(path)) {
        stop(sprintf("Path does not exist: %s", path))
      }
    },
    
    do_load = function(path) {
      stop("Abstract method: implement in subclass")
    },
    
    validate_data = function(data) {
      stop("Abstract method: implement in subclass")
    }
  )
)

#' Spatial experiment data loader
SpatialExperimentLoader <- R6::R6Class("SpatialExperimentLoader",
  inherit = DataLoader,
  public = list(
    load = function(path) {
      message("Loading SpatialExperiment (spe) ...")
      spe <- readRDS(path)
      self$validate(spe)
      spe
    },
    
    validate = function(spe) {
      message("Class of spe: ", class(spe))
      message("Dimensions of spatial coordinates:")
      print(dim(spatialCoords(spe)))
    }
  )
)

#' Image data loader
ImageLoader <- R6::R6Class("ImageLoader",
  inherit = DataLoader,
  public = list(
    load = function(path) {
      message("Loading images ...")
      images <- readRDS(path)
      self$validate(images)
      images
    },
    
    validate = function(images) {
      message("Class of images: ", class(images))
      message("Number of images: ", length(images))
      message("Channel Names in images:")
      print(channelNames(images))
    }
  )
)

#' Mask data loader
MaskLoader <- R6::R6Class("MaskLoader",
  inherit = DataLoader,
  public = list(
    load = function(path) {
      message("Loading masks ...")
      masks <- readRDS(path)
      self$validate(masks)
      masks
    },
    
    validate = function(masks) {
      if (is.null(masks)) {
        message("No masks available.")
      } else {
        message("Class of masks: ", class(masks))
        message("Number of masks: ", length(masks))
      }
    }
  )
)

#' Panel CSV data loader
PanelLoader <- R6::R6Class("PanelLoader",
  inherit = DataLoader,
  public = list(
    load = function(path) {
      message("Reading panel.csv ...")
      panel <- read.csv(path, stringsAsFactors = FALSE)
      self$validate(panel)
      panel
    },
    
    validate = function(panel) {
      message("Panel.csv structure:")
      str(panel)
    }
  )
)

#' IMC data loader for raw IMC text files
IMCLoader <- R6::R6Class("IMCLoader",
  inherit = DataLoader,
  private = list(
    do_load = function(path) {
      message("Loading IMC text files from: ", path)
      
      # List all IMC txt files in directory
      imc_files <- list.files(path, pattern = "\\.txt$", full.names = TRUE)
      if (length(imc_files) == 0) {
        stop("No IMC text files found in: ", path)
      }
      
      # Load each file
      imc_data <- lapply(imc_files, function(file) {
        # Extract metadata from filename
        metadata <- list(
          day = as.numeric(gsub(".*D([1-7])_.*", "\\1", basename(file))),
          mouse = as.numeric(gsub(".*_M([1-2])_.*", "\\1", basename(file))),
          roi = as.numeric(gsub(".*_ROI([0-9]+).*", "\\1", basename(file)))
        )
        
        # Read and parse IMC data
        data <- read.table(file, header = TRUE, sep = "\t", check.names = FALSE)
        
        list(
          metadata = metadata,
          data = data
        )
      })
      
      names(imc_data) <- basename(imc_files)
      imc_data
    },
    
    validate_data = function(data) {
      # Validate IMC data structure
      for (file_name in names(data)) {
        imc_file <- data[[file_name]]
        
        # Check metadata
        if (!all(c("day", "mouse", "roi") %in% names(imc_file$metadata))) {
          stop("Missing metadata in file: ", file_name)
        }
        
        # Check data structure
        if (ncol(imc_file$data) < 3) {  # Assuming at least x, y coordinates and one marker
          stop("Invalid data structure in file: ", file_name)
        }
      }
      
      message("IMC data validation complete")
      message("Number of files loaded: ", length(data))
    }
  )
)

# Similar implementations for additional data types can be added here... 