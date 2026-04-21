"""
Display-only configuration loader.

Separates viz concerns (colors, labels, plot layouts) from analysis config
(gating rules, thresholds, endpoint specs). Viz changes do NOT affect
analysis results or config_sha256 provenance.

Canonical file: viz.json at project root. Load via VizConfig.load().
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


def _find_project_root(start: Path) -> Path:
    """Walk up from start until we find a directory containing config.json."""
    for candidate in (start, *start.parents):
        if (candidate / 'config.json').exists():
            return candidate
    raise FileNotFoundError(
        f'Could not locate project root (no config.json found from {start})'
    )


class VizConfig:
    """Loaded viz.json with typed accessors for display knobs.

    Use case: notebooks and plotting utilities read this instead of reaching
    into config.json for colors/labels. Keeps analysis and viz concerns
    separable — a color tweak is not an analysis change.
    """

    def __init__(self, raw: Mapping[str, Any], project_root: Path) -> None:
        self.raw = dict(raw)
        self.project_root = project_root

    # ---- construction ----------------------------------------------------

    @classmethod
    def load(cls, path: Optional[Path] = None) -> 'VizConfig':
        """Load viz.json. If path is None, search upward from CWD for project root."""
        if path is None:
            root = _find_project_root(Path.cwd().resolve())
            path = root / 'viz.json'
        else:
            path = Path(path).resolve()
            root = _find_project_root(path.parent)
        if not path.exists():
            raise FileNotFoundError(f'viz.json not found at {path}')
        with open(path) as f:
            raw = json.load(f)
        return cls(raw=raw, project_root=root)

    # ---- cell-type display -----------------------------------------------

    @property
    def cell_type_display(self) -> Dict[str, Dict[str, str]]:
        return self.raw.get('cell_type_display', {})

    @property
    def cell_type_colors(self) -> Dict[str, str]:
        return {k: v['color'] for k, v in self.cell_type_display.items() if 'color' in v}

    @property
    def cell_type_labels(self) -> Dict[str, str]:
        return {k: v['label'] for k, v in self.cell_type_display.items() if 'label' in v}

    @property
    def cell_type_order(self) -> List[str]:
        """Cell type identifiers in viz.json declaration order — use as plot ordering."""
        return list(self.cell_type_display.keys())

    def ct_label(self, cell_type: str) -> str:
        """Human-readable label for a cell_type id; falls back to title-cased id."""
        return self.cell_type_labels.get(
            cell_type, cell_type.replace('_', ' ').title()
        )

    def ct_color(self, cell_type: str, default: str = '#888888') -> str:
        return self.cell_type_colors.get(cell_type, default)

    # ---- timepoint display -----------------------------------------------

    @property
    def timepoint_display(self) -> Dict[str, Any]:
        return self.raw.get('timepoint_display', {})

    @property
    def timepoint_order(self) -> List[str]:
        return list(self.timepoint_display.get('order', []))

    @property
    def timepoint_colors(self) -> Dict[str, str]:
        return dict(self.timepoint_display.get('colors', {}))

    # ---- channel-group display -------------------------------------------

    @property
    def channel_group_colormaps(self) -> Dict[str, str]:
        return dict(self.raw.get('channel_group_colormaps', {}))

    # ---- validation plots (multichannel viz pipeline) --------------------

    @property
    def validation_plots(self) -> Dict[str, Any]:
        """Config for validation-plot layout (primary_markers, always_include, layout)."""
        return dict(self.raw.get('validation_plots', {}))

    # ---- figure defaults -------------------------------------------------

    @property
    def figure_defaults(self) -> Dict[str, Any]:
        return dict(self.raw.get('figure_defaults', {}))

    def apply_rcparams(self) -> None:
        """Apply seaborn style + matplotlib rcParams from figure_defaults. Idempotent."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            return
        defaults = self.figure_defaults
        style = defaults.get('style')
        if style:
            sns.set_style(style)
        rc_updates = {}
        if 'dpi' in defaults:
            rc_updates['figure.dpi'] = defaults['dpi']
        if 'font_size' in defaults:
            rc_updates['font.size'] = defaults['font_size']
        if 'figsize' in defaults:
            rc_updates['figure.figsize'] = tuple(defaults['figsize'])
        if rc_updates:
            plt.rcParams.update(rc_updates)
