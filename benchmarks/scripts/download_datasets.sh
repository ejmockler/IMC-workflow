#!/bin/bash
#
# Download public IMC datasets for benchmarking
#
# Usage:
#   ./download_datasets.sh [dataset_name]
#
# Available datasets:
#   - bodenmiller_example: Example IMC dataset from Zenodo (small, tutorial)
#   - jackson_breast: Jackson et al. breast cancer data (if public)
#   - all: Download all available datasets
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$BENCHMARK_DIR/data"

# Create data directory
mkdir -p "$DATA_DIR"

# Function: Print colored message
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function: Download Bodenmiller example dataset
download_bodenmiller_example() {
    local dataset_name="bodenmiller_example"
    local output_dir="$DATA_DIR/$dataset_name"

    log_info "Downloading Bodenmiller example IMC dataset from Zenodo..."

    mkdir -p "$output_dir"
    cd "$output_dir"

    # Zenodo record: 5949116 (Example imaging mass cytometry raw data)
    local zenodo_record="5949116"
    local zenodo_url="https://zenodo.org/records/$zenodo_record/files"

    log_info "Fetching file list from Zenodo record $zenodo_record..."

    # Download dataset (adjust URLs based on actual Zenodo record structure)
    # Note: This is a placeholder - actual URLs need to be determined

    log_warn "Manual download required:"
    echo "1. Visit: https://zenodo.org/records/$zenodo_record"
    echo "2. Download all .txt files (IMC raw data)"
    echo "3. Place in: $output_dir/"
    echo "4. Create metadata.csv with ROI information"

    # Create placeholder metadata template
    cat > "$output_dir/metadata_template.csv" << EOF
roi_name,condition,timepoint,region,mouse_id
ROI_001,control,baseline,tissue,M1
ROI_002,treatment,day1,tissue,M1
# Add more ROIs as needed
EOF

    log_info "Created metadata template at: $output_dir/metadata_template.csv"
    log_warn "Manual configuration required. Edit template and rename to metadata.csv"

    cd "$SCRIPT_DIR"
}

# Function: Download high-resolution kidney dataset (2025 Nature Methods)
download_highres_kidney() {
    local dataset_name="highres_kidney_2025"
    local output_dir="$DATA_DIR/$dataset_name"

    log_info "Downloading high-resolution kidney IMC dataset (Nature Methods 2025)..."

    mkdir -p "$output_dir"
    cd "$output_dir"

    # Zenodo DOI: 10.5281/zenodo.17077712
    local zenodo_doi="10.5281/zenodo.17077712"

    log_warn "This dataset is very large (high-resolution subcellular IMC)"
    log_warn "Manual download recommended:"
    echo "1. Visit: https://doi.org/$zenodo_doi"
    echo "2. Download kidney tissue samples (.txt or .mcd format)"
    echo "3. Place in: $output_dir/"
    echo "4. Create metadata.csv"

    # Create metadata template
    cat > "$output_dir/metadata_template.csv" << EOF
roi_name,tissue_type,resolution,acquisition_date
kidney_roi_001,cortex,high_res,2024
kidney_roi_002,medulla,high_res,2024
# Add actual ROI names from dataset
EOF

    log_info "Created metadata template"

    cd "$SCRIPT_DIR"
}

# Function: Check if dataset already downloaded
check_dataset_exists() {
    local dataset_name=$1
    local dataset_dir="$DATA_DIR/$dataset_name"

    if [ -d "$dataset_dir" ] && [ "$(ls -A $dataset_dir)" ]; then
        log_warn "Dataset '$dataset_name' already exists at: $dataset_dir"
        read -p "Re-download? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Skipping download"
            return 1
        fi
    fi
    return 0
}

# Function: Validate downloaded dataset
validate_dataset() {
    local dataset_name=$1
    local dataset_dir="$DATA_DIR/$dataset_name"

    log_info "Validating dataset: $dataset_name"

    # Check for .txt files (IMC raw data)
    local txt_count=$(find "$dataset_dir" -name "*.txt" | wc -l)

    if [ $txt_count -eq 0 ]; then
        log_error "No .txt files found in $dataset_dir"
        log_warn "Dataset may not be downloaded or in wrong format"
        return 1
    fi

    log_info "Found $txt_count .txt files"

    # Check for metadata
    if [ -f "$dataset_dir/metadata.csv" ]; then
        log_info "Metadata found: metadata.csv"
    else
        log_warn "No metadata.csv found (metadata_template.csv created)"
    fi

    # Create dataset summary
    cat > "$dataset_dir/dataset_info.txt" << EOF
Dataset: $dataset_name
Location: $dataset_dir
Downloaded: $(date)
ROI count: $txt_count
Status: Manual configuration required (see metadata_template.csv)
EOF

    log_info "Dataset summary: $dataset_dir/dataset_info.txt"

    return 0
}

# Main function
main() {
    local dataset=$1

    if [ -z "$dataset" ]; then
        log_error "No dataset specified"
        echo "Usage: $0 [dataset_name]"
        echo ""
        echo "Available datasets:"
        echo "  bodenmiller_example  - Small example dataset from Zenodo"
        echo "  highres_kidney       - High-resolution kidney dataset (Nature Methods 2025)"
        echo "  all                  - Download all available datasets"
        exit 1
    fi

    log_info "Starting dataset download: $dataset"
    log_info "Target directory: $DATA_DIR"

    case $dataset in
        bodenmiller_example)
            if check_dataset_exists "bodenmiller_example"; then
                download_bodenmiller_example
                validate_dataset "bodenmiller_example"
            fi
            ;;

        highres_kidney)
            if check_dataset_exists "highres_kidney_2025"; then
                download_highres_kidney
                validate_dataset "highres_kidney_2025"
            fi
            ;;

        all)
            log_info "Downloading all available datasets..."
            for ds in bodenmiller_example highres_kidney; do
                if check_dataset_exists "$ds"; then
                    download_$ds
                    validate_dataset "$ds"
                fi
            done
            ;;

        *)
            log_error "Unknown dataset: $dataset"
            echo "Available: bodenmiller_example, highres_kidney, all"
            exit 1
            ;;
    esac

    log_info "Download process complete"
    log_info ""
    log_info "Next steps:"
    echo "1. Verify downloaded files in: $DATA_DIR/$dataset/"
    echo "2. Complete metadata.csv (rename from template)"
    echo "3. Run: jupyter notebook ../comparison_notebooks/01_data_preparation.ipynb"
}

# Run main
main "$@"
