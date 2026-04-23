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


class VizConfigValidationError(ValueError):
    """Raised when viz.json disagrees with config.json on load-bearing keys.

    Examples that trigger this: renaming a cell type in config.json without
    updating viz.json, changing the analysis TIMEPOINT_ORDER without
    propagating to viz.json timepoint_display.order.
    """


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
    def load(
        cls,
        path: Optional[Path] = None,
        validate: bool = True,
    ) -> 'VizConfig':
        """Load viz.json. If path is None, search upward from CWD for project root.

        When ``validate=True`` (default), cross-checks viz.json against
        config.json to catch silent drift between the two files:
          - ``cell_type_display`` keys must match ``cell_type_annotation.cell_types``
          - ``timepoint_display.order`` must match ``TIMEPOINT_ORDER``
            from ``src.analysis.temporal_interface_analysis``

        Raises ``VizConfigValidationError`` on mismatch. Set ``validate=False``
        only in test fixtures that intentionally construct a partial VizConfig.
        """
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
        instance = cls(raw=raw, project_root=root)
        if validate:
            instance._validate_against_analysis_config()
        return instance

    # ---- cross-file validation -------------------------------------------

    def _validate_against_analysis_config(self) -> None:
        """Hard-fail on drift between viz.json and the analysis-side truths.

        Four invariants:
          1. Cell type roster in viz.json == cell type roster in config.json.
             A collaborator renaming a type in one file and not the other
             would otherwise produce plots with missing colors or mislabeled
             categories.
          2. timepoint_display.order == TIMEPOINT_ORDER (module constant).
             Display ordering must match the pre-registered analysis ordering
             for Family A/B/C comparisons.
          3. channel_group_colormaps keys ⊆ config.channels.channel_groups
             (plus 'default'). Catches stale colormap entries that survive
             a channel-group rename on the analysis side.
          4. validation_plots.primary_markers values ⊆ protein_channels, and
             .always_include ⊆ protein_channels. Catches markers deleted from
             the config panel but surviving in viz layout.
        """
        errors: List[str] = []

        # Load config.json once for all cross-checks.
        try:
            with open(self.project_root / 'config.json') as f:
                cfg = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            errors.append(f'cannot load config.json for cross-check: {e}')
            cfg = None

        # Invariant 1: cell type roster
        if cfg is not None:
            config_types = set(
                cfg.get('cell_type_annotation', {}).get('cell_types', {}).keys()
            )
            viz_types = set(self.cell_type_display.keys())
            missing_in_viz = config_types - viz_types
            extra_in_viz = viz_types - config_types
            if missing_in_viz:
                errors.append(
                    f'cell types in config.json but missing from viz.json '
                    f'cell_type_display: {sorted(missing_in_viz)}'
                )
            if extra_in_viz:
                errors.append(
                    f'cell types in viz.json cell_type_display but not in '
                    f'config.json: {sorted(extra_in_viz)}'
                )

        # Invariant 2: timepoint order. Import from the zero-dependency
        # constants module so loading viz.json doesn't drag in scipy/pandas
        # via src.analysis.__init__.
        try:
            from src.constants import TIMEPOINT_ORDER
            expected_order = list(TIMEPOINT_ORDER)
        except ImportError:
            expected_order = None  # tolerate missing constants module

        if expected_order is not None:
            viz_order = self.timepoint_order
            if viz_order != expected_order:
                errors.append(
                    f'timepoint_display.order {viz_order} does not match '
                    f'analysis-side TIMEPOINT_ORDER {expected_order}'
                )

        # Invariant 3: channel_group_colormaps keys vs config channel_groups.
        # 'default' is always permitted.
        if cfg is not None:
            channel_group_names = set(cfg.get('channel_groups', {}).keys())
            colormap_keys = set(self.channel_group_colormaps.keys()) - {'default'}
            extra_colormap = colormap_keys - channel_group_names
            if extra_colormap:
                errors.append(
                    f'channel_group_colormaps has keys not present in '
                    f'config.channel_groups (or \"default\"): '
                    f'{sorted(extra_colormap)}'
                )

        # Invariant 4: validation_plots markers vs protein_channels.
        if cfg is not None:
            protein_channels = set(cfg.get('channels', {}).get('protein_channels', []))
            vp = self.validation_plots
            primary = vp.get('primary_markers', {}) or {}
            primary_vals = set(
                v for v in primary.values()
                if isinstance(v, str) and not v.startswith('_')
            )
            bad_primary = primary_vals - protein_channels
            if bad_primary:
                errors.append(
                    f'validation_plots.primary_markers references markers not '
                    f'in config.channels.protein_channels: {sorted(bad_primary)}'
                )
            always_include = set(
                m for m in (vp.get('always_include') or [])
                if isinstance(m, str) and not m.startswith('_')
            )
            bad_always = always_include - protein_channels
            if bad_always:
                errors.append(
                    f'validation_plots.always_include references markers not '
                    f'in config.channels.protein_channels: {sorted(bad_always)}'
                )

        if errors:
            joined = '\n  - '.join(errors)
            raise VizConfigValidationError(
                f'viz.json is out of sync with analysis config:\n  - {joined}\n'
                f'Fix either viz.json or config.json so they agree, or load with '
                f'VizConfig.load(validate=False) in a test fixture.'
            )

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
