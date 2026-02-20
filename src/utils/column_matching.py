"""
Robust Column Matching for IMC Data

Provides intelligent column matching with regex patterns, alias support,
and fuzzy matching to handle variations in IMC column naming conventions.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from difflib import SequenceMatcher
import numpy as np


@dataclass
class ColumnMatch:
    """Result of column matching operation."""
    marker_name: str
    matched_column: str
    confidence: float
    match_type: str  # 'exact', 'regex', 'alias', 'fuzzy'
    alternatives: List[str]


@dataclass
class MatchingConfig:
    """Configuration for column matching."""
    # Pattern matching
    case_sensitive: bool = False
    fuzzy_threshold: float = 0.8
    
    # IMC-specific patterns
    require_parentheses: bool = True  # Expect "Marker(Mass)" format
    allow_partial_matches: bool = False
    
    # Alias handling
    use_aliases: bool = True
    custom_aliases: Dict[str, List[str]] = None
    
    # Recovery options
    suggest_alternatives: bool = True
    max_alternatives: int = 3


class ColumnMatcher:
    """
    Robust column matching for IMC data with comprehensive pattern support.
    
    Handles common IMC column naming variations:
    - CD206(163Dy) / CD206 (163Dy) / cd206_163dy
    - DNA1(191Ir) / DNA1 / dna1_191ir  
    - Background / background / BG
    """
    
    def __init__(self, config: MatchingConfig = None):
        """Initialize column matcher with configuration."""
        self.config = config or MatchingConfig()
        self.logger = logging.getLogger('ColumnMatcher')
        
        # Default aliases for common IMC markers
        self._default_aliases = {
            'DNA1': ['dna1', 'hoechst', 'nucleus', 'nuclei'],
            'DNA2': ['dna2', 'hoechst2', 'pi', 'propidium'],
            'CD206': ['cd206', 'mrc1', 'mannose_receptor'],
            'CD44': ['cd44', 'pgp1', 'hermes'],
            'CD68': ['cd68', 'macrosialin'],
            'CD31': ['cd31', 'pecam1', 'endothelial'],
            'αSMA': ['asma', 'alpha_sma', 'smooth_muscle_actin', 'acta2'],
            'Collagen': ['collagen', 'col1a1', 'col3a1'],
            'Background': ['background', 'bg', 'blank', 'control']
        }
        
        # Merge with custom aliases
        self.aliases = self._default_aliases.copy()
        if self.config.custom_aliases:
            self.aliases.update(self.config.custom_aliases)
        
        # Pre-compile regex patterns
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for efficiency."""
        self.patterns = {}
        
        # Standard IMC pattern: Marker(Mass) or Marker (Mass)
        imc_pattern = r'^{marker}\s*\([^)]+\)$'
        
        # Flexible pattern: marker name anywhere in column
        flexible_pattern = r'{marker}'
        
        # Underscore pattern: marker_mass or marker_other
        underscore_pattern = r'^{marker}(_[^_]*)?$'
        
        self.pattern_templates = {
            'imc_standard': imc_pattern,
            'flexible': flexible_pattern,
            'underscore': underscore_pattern
        }
    
    def find_best_match(
        self, 
        marker_name: str, 
        available_columns: List[str]
    ) -> Optional[ColumnMatch]:
        """
        Find the best matching column for a marker name.
        
        Args:
            marker_name: Target marker name to match
            available_columns: List of available column names
            
        Returns:
            Best column match or None if no suitable match found
        """
        self.logger.debug(f"Finding match for marker '{marker_name}' in {len(available_columns)} columns")
        
        all_matches = self.find_all_matches(marker_name, available_columns)
        
        if not all_matches:
            self.logger.warning(f"No matches found for marker '{marker_name}'")
            return None
        
        # Return highest confidence match
        best_match = max(all_matches, key=lambda m: m.confidence)
        
        self.logger.info(f"Best match for '{marker_name}': '{best_match.matched_column}' "
                        f"(confidence: {best_match.confidence:.2f}, type: {best_match.match_type})")
        
        return best_match
    
    def find_all_matches(
        self, 
        marker_name: str, 
        available_columns: List[str]
    ) -> List[ColumnMatch]:
        """
        Find all possible matches for a marker name.
        
        Args:
            marker_name: Target marker name to match
            available_columns: List of available column names
            
        Returns:
            List of all matches sorted by confidence
        """
        matches = []
        
        # Try exact match first
        exact_match = self._try_exact_match(marker_name, available_columns)
        if exact_match:
            matches.append(exact_match)
        
        # Try regex patterns
        regex_matches = self._try_regex_matches(marker_name, available_columns)
        matches.extend(regex_matches)
        
        # Try alias matching
        if self.config.use_aliases:
            alias_matches = self._try_alias_matches(marker_name, available_columns)
            matches.extend(alias_matches)
        
        # Try fuzzy matching as fallback
        fuzzy_matches = self._try_fuzzy_matches(marker_name, available_columns)
        matches.extend(fuzzy_matches)
        
        # Remove duplicates and sort by confidence
        unique_matches = self._deduplicate_matches(matches)
        unique_matches.sort(key=lambda m: m.confidence, reverse=True)
        
        return unique_matches
    
    def _try_exact_match(self, marker_name: str, columns: List[str]) -> Optional[ColumnMatch]:
        """Try exact string matching."""
        for col in columns:
            if self._compare_strings(marker_name, col):
                return ColumnMatch(
                    marker_name=marker_name,
                    matched_column=col,
                    confidence=1.0,
                    match_type='exact',
                    alternatives=[]
                )
        return None
    
    def _try_regex_matches(self, marker_name: str, columns: List[str]) -> List[ColumnMatch]:
        """Try regex pattern matching."""
        matches = []
        
        for pattern_name, pattern_template in self.pattern_templates.items():
            # Create pattern for this marker
            escaped_marker = re.escape(marker_name)
            pattern = pattern_template.format(marker=escaped_marker)
            
            # Compile with appropriate flags
            flags = 0 if self.config.case_sensitive else re.IGNORECASE
            regex = re.compile(pattern, flags)
            
            # Test against all columns
            for col in columns:
                if regex.match(col):
                    # Calculate confidence based on pattern specificity
                    confidence = self._calculate_regex_confidence(pattern_name, col, marker_name)
                    
                    matches.append(ColumnMatch(
                        marker_name=marker_name,
                        matched_column=col,
                        confidence=confidence,
                        match_type='regex',
                        alternatives=[]
                    ))
        
        return matches
    
    def _try_alias_matches(self, marker_name: str, columns: List[str]) -> List[ColumnMatch]:
        """Try matching using marker aliases."""
        matches = []
        
        # Get aliases for this marker
        marker_aliases = self.aliases.get(marker_name, [])
        
        # Also check if marker_name is an alias for something else
        for canonical_name, aliases in self.aliases.items():
            if self._compare_strings(marker_name, canonical_name) or \
               any(self._compare_strings(marker_name, alias) for alias in aliases):
                marker_aliases.extend(aliases)
                if canonical_name not in marker_aliases:
                    marker_aliases.append(canonical_name)
        
        # Try matching each alias
        for alias in marker_aliases:
            for col in columns:
                if self._test_alias_match(alias, col):
                    confidence = 0.85  # High confidence for alias matches
                    
                    matches.append(ColumnMatch(
                        marker_name=marker_name,
                        matched_column=col,
                        confidence=confidence,
                        match_type='alias',
                        alternatives=[]
                    ))
        
        return matches
    
    def _try_fuzzy_matches(self, marker_name: str, columns: List[str]) -> List[ColumnMatch]:
        """Try fuzzy string matching as fallback."""
        matches = []
        
        for col in columns:
            similarity = self._calculate_string_similarity(marker_name, col)
            
            if similarity >= self.config.fuzzy_threshold:
                matches.append(ColumnMatch(
                    marker_name=marker_name,
                    matched_column=col,
                    confidence=similarity * 0.7,  # Lower confidence for fuzzy matches
                    match_type='fuzzy',
                    alternatives=[]
                ))
        
        return matches
    
    def _compare_strings(self, str1: str, str2: str) -> bool:
        """Compare two strings respecting case sensitivity config."""
        if self.config.case_sensitive:
            return str1 == str2
        else:
            return str1.lower() == str2.lower()
    
    def _test_alias_match(self, alias: str, column: str) -> bool:
        """Test if an alias matches a column using flexible patterns."""
        # Try exact match
        if self._compare_strings(alias, column):
            return True
        
        # Try as substring (for columns like "CD206(163Dy)")
        if not self.config.case_sensitive:
            return alias.lower() in column.lower()
        else:
            return alias in column
    
    def _calculate_regex_confidence(self, pattern_name: str, column: str, marker_name: str) -> float:
        """Calculate confidence score for regex matches."""
        base_confidence = {
            'imc_standard': 0.95,  # Highest for standard IMC format
            'flexible': 0.75,      # Medium for flexible matching
            'underscore': 0.85     # High for underscore format
        }
        
        confidence = base_confidence.get(pattern_name, 0.5)
        
        # Boost confidence if marker name is prominent in column
        marker_ratio = len(marker_name) / len(column)
        if marker_ratio > 0.5:
            confidence = min(0.98, confidence + 0.1)
        
        return confidence
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings."""
        if not self.config.case_sensitive:
            str1 = str1.lower()
            str2 = str2.lower()
        
        return SequenceMatcher(None, str1, str2).ratio()
    
    def _deduplicate_matches(self, matches: List[ColumnMatch]) -> List[ColumnMatch]:
        """Remove duplicate matches, keeping highest confidence."""
        seen_columns = {}
        
        for match in matches:
            col = match.matched_column
            if col not in seen_columns or match.confidence > seen_columns[col].confidence:
                seen_columns[col] = match
        
        return list(seen_columns.values())
    
    def match_multiple_markers(
        self,
        marker_names: List[str],
        available_columns: List[str]
    ) -> Dict[str, Optional[ColumnMatch]]:
        """
        HIGH-PERFORMANCE: Match multiple markers using single-pass greedy assignment.
        
        Eliminates the O(n²×m²) prioritization bottleneck that was causing 60+ minute runtimes.
        Now runs in O(n×m) time where n=markers, m=columns.
        
        Args:
            marker_names: List of marker names to match
            available_columns: List of available column names
            
        Returns:
            Dictionary mapping marker_name -> best_match (or None)
        """
        self.logger.info(f"Matching {len(marker_names)} markers against {len(available_columns)} columns")
        
        # CRITICAL FIX: Build all candidates ONCE instead of repeatedly
        all_candidates = []
        
        # For each marker, get simple regex matches only (skip expensive fuzzy matching)
        for marker_name in marker_names:
            # Direct regex pattern match - O(m) per marker
            # Fixed pattern: Insert optional hyphens BEFORE digits to match Ki67 -> Ki-67
            # Pattern: Ki-?67[_-]?\d*... matches both Ki67_... AND Ki-67_...
            # But CD3[_-]?\d*... matches CD3_1841... but NOT CD38_...
            escaped_marker = re.escape(marker_name)
            # Insert optional hyphen before any digit in the marker name
            flexible_marker = re.sub(r'(\d)', r'-?\1', escaped_marker)
            pattern = re.compile(f'^{flexible_marker}[_-]?\\d*[_-]?\\([^)]+\\)', re.IGNORECASE)

            for col in available_columns:
                if pattern.match(col):
                    all_candidates.append(ColumnMatch(
                        marker_name=marker_name,
                        matched_column=col,
                        confidence=0.95,  # High confidence for IMC standard format
                        match_type='regex',
                        alternatives=[]
                    ))
        
        # Sort by confidence (all 0.95 here, but keeps extensibility)
        all_candidates.sort(key=lambda m: m.confidence, reverse=True)
        
        # Greedy assignment in single pass
        results = {}
        used_columns = set()
        
        for candidate in all_candidates:
            # Skip if marker already assigned or column already used
            if candidate.marker_name in results or candidate.matched_column in used_columns:
                continue
                
            # Assign the match
            results[candidate.marker_name] = candidate
            used_columns.add(candidate.matched_column)
            self.logger.info(f"Best match for '{candidate.marker_name}': '{candidate.matched_column}' "
                           f"(confidence: {candidate.confidence:.2f}, type: {candidate.match_type})")
        
        # Mark unmatched markers as None
        for marker_name in marker_names:
            if marker_name not in results:
                results[marker_name] = None
                self.logger.warning(f"No match found for marker '{marker_name}'")
        
        return results
    
    def generate_matching_report(
        self, 
        matching_results: Dict[str, Optional[ColumnMatch]],
        available_columns: List[str]
    ) -> Dict[str, any]:
        """Generate comprehensive matching report."""
        successful_matches = {k: v for k, v in matching_results.items() if v is not None}
        failed_matches = [k for k, v in matching_results.items() if v is None]
        
        # Confidence distribution
        confidences = [match.confidence for match in successful_matches.values()]
        
        # Match type distribution
        match_types = {}
        for match in successful_matches.values():
            match_types[match.match_type] = match_types.get(match.match_type, 0) + 1
        
        # Unused columns (potential missing markers)
        used_columns = {match.matched_column for match in successful_matches.values()}
        unused_columns = [col for col in available_columns if col not in used_columns]
        
        report = {
            'total_markers': len(matching_results),
            'successful_matches': len(successful_matches),
            'failed_matches': len(failed_matches),
            'success_rate': len(successful_matches) / len(matching_results) if matching_results else 0,
            
            'confidence_stats': {
                'mean': np.mean(confidences) if confidences else 0,
                'min': np.min(confidences) if confidences else 0,
                'max': np.max(confidences) if confidences else 0
            },
            
            'match_type_distribution': match_types,
            'failed_marker_names': failed_matches,
            'unused_columns': unused_columns[:10],  # Limit to avoid spam
            
            'detailed_matches': {
                name: {
                    'column': match.matched_column,
                    'confidence': match.confidence,
                    'type': match.match_type
                } for name, match in successful_matches.items()
            }
        }
        
        return report


# Convenience functions
def create_column_matcher(
    case_sensitive: bool = False,
    fuzzy_threshold: float = 0.8,
    custom_aliases: Dict[str, List[str]] = None
) -> ColumnMatcher:
    """Create column matcher with common settings."""
    config = MatchingConfig(
        case_sensitive=case_sensitive,
        fuzzy_threshold=fuzzy_threshold,
        custom_aliases=custom_aliases
    )
    return ColumnMatcher(config)


def match_imc_columns(
    marker_names: List[str], 
    available_columns: List[str],
    **kwargs
) -> Dict[str, Optional[str]]:
    """
    Simple interface for IMC column matching.
    
    Args:
        marker_names: List of marker names to match
        available_columns: List of available column names
        **kwargs: Additional configuration options
        
    Returns:
        Dictionary mapping marker_name -> matched_column (or None)
    """
    matcher = create_column_matcher(**kwargs)
    results = matcher.match_multiple_markers(marker_names, available_columns)
    
    # Convert to simple string mapping
    return {
        name: match.matched_column if match else None 
        for name, match in results.items()
    }