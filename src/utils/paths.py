"""Config-driven path resolution for the IMC pipeline."""

import json
from pathlib import Path

# Project root is three levels up from this file: src/utils/paths.py -> project root
_PROJECT_ROOT = Path(__file__).parent.parent.parent


class ProjectPaths:
    """Computed path properties for all standard IMC pipeline locations."""

    def __init__(self, config_path=None):
        self._root = _PROJECT_ROOT
        self._config = self._load_config(config_path)

    def _load_config(self, config_path) -> dict:
        path = Path(config_path) if config_path else self._root / "config.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    # ---- Project root ----

    @property
    def base_dir(self) -> Path:
        return self._root

    # ---- Raw data ----

    @property
    def data_dir(self) -> Path:
        rel = self._config.get("data", {}).get("raw_data_dir", "data/241218_IMC_Alun")
        return self._root / rel

    @property
    def metadata_csv(self) -> Path:
        rel = self._config.get("data", {}).get("metadata_file",
                                                "data/241218_IMC_Alun/Metadata-Table 1.csv")
        return self._root / rel

    # ---- Analysis outputs ----

    @property
    def results_dir(self) -> Path:
        return self._root / "results"

    @property
    def biological_analysis_dir(self) -> Path:
        return self.results_dir / "biological_analysis"

    @property
    def annotations_dir(self) -> Path:
        return self.biological_analysis_dir / "cell_type_annotations"

    @property
    def differential_abundance_dir(self) -> Path:
        return self.biological_analysis_dir / "differential_abundance"

    @property
    def spatial_neighborhoods_dir(self) -> Path:
        return self.biological_analysis_dir / "spatial_neighborhoods"

    @property
    def figures_dir(self) -> Path:
        return self.results_dir / "figures"

    @property
    def power_analysis_dir(self) -> Path:
        return self.results_dir / "power_analysis"

    @property
    def benchmark_dir(self) -> Path:
        return self.results_dir / "benchmark"

    @property
    def roi_results_dir(self) -> Path:
        return self.results_dir / "roi_results"


# Module-level singleton for scripts that just need simple path access
_default: ProjectPaths = None


def _get_default() -> ProjectPaths:
    global _default
    if _default is None:
        _default = ProjectPaths()
    return _default


def get_paths(config_path=None) -> ProjectPaths:
    """Return a ProjectPaths instance, using the default singleton when no config_path given."""
    if config_path is None:
        return _get_default()
    return ProjectPaths(config_path)
