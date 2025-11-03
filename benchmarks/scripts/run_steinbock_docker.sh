#!/bin/bash
#
# Run Steinbock pipeline on IMC data using Docker
#
# Usage:
#   ./run_steinbock_docker.sh <data_directory> [output_directory]
#
# Requirements:
#   - Docker installed and running
#   - IMC data in .txt format (or .mcd)
#   - Sufficient memory allocated to Docker (≥8GB recommended)
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check Docker
if ! command -v docker &> /dev/null; then
    log_error "Docker not found. Please install Docker first."
    exit 1
fi

if ! docker info &> /dev/null; then
    log_error "Docker daemon not running. Please start Docker."
    exit 1
fi

# Parse arguments
DATA_DIR=${1:?Usage: $0 <data_directory> [output_directory]}
OUTPUT_DIR=${2:-$(dirname "$DATA_DIR")/steinbock_outputs/$(basename "$DATA_DIR")}

# Resolve absolute paths
DATA_DIR=$(cd "$DATA_DIR" && pwd)
OUTPUT_DIR=$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)

log_info "Steinbock Pipeline Execution"
log_info "Data: $DATA_DIR"
log_info "Output: $OUTPUT_DIR"

# Steinbock Docker image
STEINBOCK_IMAGE="ghcr.io/bodenmillergroup/steinbock:0.16.1"

log_info "Pulling Steinbock Docker image: $STEINBOCK_IMAGE"
docker pull $STEINBOCK_IMAGE

# Create steinbock working directory structure
STEINBOCK_WORK="$OUTPUT_DIR/steinbock_workdir"
mkdir -p "$STEINBOCK_WORK"/{img,masks,intensities,regionprops,neighbors}

log_info "Setting up Steinbock workspace: $STEINBOCK_WORK"

# Copy data to steinbock img directory
log_info "Preparing input data..."
if [ -d "$DATA_DIR" ]; then
    # Copy .txt files to steinbock img directory
    find "$DATA_DIR" -name "*.txt" -exec cp {} "$STEINBOCK_WORK/img/" \;
    txt_count=$(ls "$STEINBOCK_WORK/img"/*.txt 2>/dev/null | wc -l)
    log_info "Copied $txt_count .txt files"

    if [ $txt_count -eq 0 ]; then
        log_error "No .txt files found in $DATA_DIR"
        exit 1
    fi
else
    log_error "Data directory not found: $DATA_DIR"
    exit 1
fi

# Record start time
START_TIME=$(date +%s)

# Run Steinbock preprocessing
log_info "Step 1/5: Preprocessing - creating panel..."
docker run --rm \
    -v "$STEINBOCK_WORK:/data" \
    $STEINBOCK_IMAGE \
    preprocess imc panel \
        --txt /data/img \
        -o /data/panel.csv \
        2>&1 | tee "$OUTPUT_DIR/steinbock_preprocess_panel.log"

log_info "Step 1.5/5: Preprocessing - converting images to TIFF..."
docker run --rm \
    -v "$STEINBOCK_WORK:/data" \
    $STEINBOCK_IMAGE \
    preprocess imc images \
        --txt /data/img \
        --panel /data/panel.csv \
        --imgout /data/img \
        2>&1 | tee "$OUTPUT_DIR/steinbock_preprocess_images.log"

# Run Steinbock segmentation (DeepCell/Mesmer)
log_info "Step 2/5: Cell segmentation (this may take a while)..."
docker run --rm \
    -v "$STEINBOCK_WORK:/data" \
    $STEINBOCK_IMAGE \
    segment deepcell \
        --app mesmer \
        --minmax \
        --pixelsize 1.0 \
        2>&1 | tee "$OUTPUT_DIR/steinbock_segment.log"

# Measure intensities
log_info "Step 3/5: Measuring cell intensities..."
docker run --rm \
    -v "$STEINBOCK_WORK:/data" \
    $STEINBOCK_IMAGE \
    measure intensities \
        --aggr mean \
        2>&1 | tee "$OUTPUT_DIR/steinbock_intensities.log"

# Measure region props
log_info "Step 4/5: Measuring region properties..."
docker run --rm \
    -v "$STEINBOCK_WORK:/data" \
    $STEINBOCK_IMAGE \
    measure regionprops \
        2>&1 | tee "$OUTPUT_DIR/steinbock_regionprops.log"

# Measure neighbors
log_info "Step 5/5: Measuring spatial neighborhoods..."
docker run --rm \
    -v "$STEINBOCK_WORK:/data" \
    $STEINBOCK_IMAGE \
    measure neighbors \
        --type expansion \
        --dmax 4 \
        2>&1 | tee "$OUTPUT_DIR/steinbock_neighbors.log"

# Record end time
END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))

log_info "Steinbock pipeline completed in ${RUNTIME}s"

# Create run metadata
cat > "$OUTPUT_DIR/run_metadata.json" << EOF
{
  "pipeline": "steinbock",
  "version": "0.16.1",
  "docker_image": "$STEINBOCK_IMAGE",
  "data_directory": "$DATA_DIR",
  "output_directory": "$OUTPUT_DIR",
  "runtime_seconds": $RUNTIME,
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "parameters": {
    "preprocessing": {
      "method": "imc_panel",
      "txt_dir": "/data/img"
    },
    "segmentation": {
      "method": "deepcell",
      "app": "mesmer",
      "minmax": true,
      "pixelsize": 1.0
    },
    "intensities": {
      "aggr": "mean"
    },
    "neighbors": {
      "type": "expansion",
      "dmax": 4
    }
  },
  "outputs": {
    "panel": "panel.csv",
    "masks": "masks/",
    "intensities": "intensities/",
    "regionprops": "regionprops/",
    "neighbors": "neighbors/"
  }
}
EOF

# Summarize results
log_info ""
log_info "=== Steinbock Results Summary ==="
log_info "Masks: $(ls "$STEINBOCK_WORK/masks"/*.tiff 2>/dev/null | wc -l) files"
log_info "Intensities: $(ls "$STEINBOCK_WORK/intensities"/*.csv 2>/dev/null | wc -l) files"
log_info "Neighbors: $(ls "$STEINBOCK_WORK/neighbors"/*.csv 2>/dev/null | wc -l) files"
log_info ""
log_info "Full results: $OUTPUT_DIR"
log_info "Metadata: $OUTPUT_DIR/run_metadata.json"
log_info ""
log_info "Next steps:"
echo "1. Review outputs in: $STEINBOCK_WORK/"
echo "2. Run our pipeline on same data"
echo "3. Compare results: jupyter notebook ../comparison_notebooks/04_quantitative_comparison.ipynb"
