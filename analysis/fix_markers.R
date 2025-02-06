#----------------------------------
# Fix marker names in SPE object
#----------------------------------

fix_marker_names <- function(spe) {
    # Show current names
    message("Current marker names:")
    print(rownames(spe))
    
    # Fix CD11b name
    rownames(spe)[rownames(spe) == "CD11b 1"] <- "CD11b"
    
    # Verify the change
    message("\nUpdated marker names:")
    print(rownames(spe))
    
    # Also fix in assay names if present
    for(assay_name in assayNames(spe)) {
        rownames(assay(spe, assay_name))[rownames(assay(spe, assay_name)) == "CD11b 1"] <- "CD11b"
    }
    
    # Save the updated SPE object
    saveRDS(spe, "/Users/noot/Documents/IMC/data/spe.rds")
    
    message("\nSPE object updated and saved with corrected marker names.")
    return(spe)
}

# Execute the fix
spe <- readRDS("/Users/noot/Documents/IMC/data/spe.rds")
spe <- fix_marker_names(spe)