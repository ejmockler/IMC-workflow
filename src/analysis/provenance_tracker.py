"""
Intelligent Provenance Tracker - KISS Implementation

Captures reasoning behind scientific decisions, not computational steps.
Automatically generates methods sections from decision logs.
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum


class DecisionType(Enum):
    """Types of decisions worth tracking for reproducibility."""
    PARAMETER_CHOICE = "parameter_choice"
    QUALITY_GATE = "quality_gate"
    DATA_TRANSFORMATION = "data_transformation"
    METHOD_SELECTION = "method_selection"
    THRESHOLD_SETTING = "threshold_setting"
    VALIDATION_OUTCOME = "validation_outcome"


class DecisionSeverity(Enum):
    """Impact level of decisions on final results."""
    CRITICAL = "critical"    # Changes analysis fundamentally
    IMPORTANT = "important"  # Affects interpretation
    ROUTINE = "routine"      # Standard procedure


@dataclass
class DataLineage:
    """Track input → processing → output with checksums."""
    input_source: str
    input_checksum: str
    processing_step: str
    output_description: str
    output_checksum: Optional[str] = None


@dataclass
class ProvenanceDecision:
    """Single decision with scientific reasoning."""
    decision_id: str
    timestamp: str
    decision_type: DecisionType
    severity: DecisionSeverity
    
    # Core decision data
    parameter_name: str
    parameter_value: Any
    reasoning: str
    
    # Context
    alternatives_considered: List[Any] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    references: List[str] = field(default_factory=list)
    
    # Quality metrics if applicable
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    quality_outcome: Optional[str] = None
    
    # Technical metadata
    execution_context: Dict[str, Any] = field(default_factory=dict)


class ProvenanceTracker:
    """
    Intelligent provenance tracker for IMC analysis.
    
    Captures WHY decisions were made, not just WHAT was computed.
    Generates human-readable methods sections automatically.
    """
    
    def __init__(self, analysis_id: str, output_dir: Optional[str] = None):
        """
        Initialize provenance tracker.
        
        Args:
            analysis_id: Unique identifier for this analysis run
            output_dir: Directory to save provenance records
        """
        self.analysis_id = analysis_id
        self.start_time = datetime.now()
        self.decisions: List[ProvenanceDecision] = []
        self.data_lineage: List[DataLineage] = []
        
        # Output management
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "provenance"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.provenance_file = self.output_dir / f"provenance_{analysis_id}.json"
        
        # Decision counter for unique IDs
        self._decision_counter = 0
        
        # Analysis metadata
        self.metadata = {
            "analysis_id": analysis_id,
            "start_time": self.start_time.isoformat(),
            "tracker_version": "1.0",
            "purpose": "Capture scientific decision-making for reproducible analysis"
        }
    
    def _generate_decision_id(self) -> str:
        """Generate unique decision ID."""
        self._decision_counter += 1
        return f"{self.analysis_id}_decision_{self._decision_counter:03d}"
    
    def _compute_checksum(self, data: Any) -> str:
        """Compute SHA-256 checksum for data integrity."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def log_parameter_decision(
        self,
        parameter_name: str,
        parameter_value: Any,
        reasoning: str,
        alternatives_considered: Optional[List[Any]] = None,
        evidence: Optional[Dict[str, Any]] = None,
        severity: DecisionSeverity = DecisionSeverity.IMPORTANT,
        references: Optional[List[str]] = None
    ) -> str:
        """
        Log a parameter choice with scientific reasoning.
        
        Args:
            parameter_name: Name of the parameter being set
            parameter_value: Chosen value
            reasoning: Scientific rationale for this choice
            alternatives_considered: Other values considered
            evidence: Supporting evidence (metrics, literature, etc.)
            severity: Impact level of this decision
            references: Literature or documentation references
            
        Returns:
            Decision ID for cross-referencing
        """
        decision = ProvenanceDecision(
            decision_id=self._generate_decision_id(),
            timestamp=datetime.now().isoformat(),
            decision_type=DecisionType.PARAMETER_CHOICE,
            severity=severity,
            parameter_name=parameter_name,
            parameter_value=parameter_value,
            reasoning=reasoning,
            alternatives_considered=alternatives_considered or [],
            evidence=evidence or {},
            references=references or [],
            execution_context={
                "analysis_id": self.analysis_id,
                "method": "log_parameter_decision"
            }
        )
        
        self.decisions.append(decision)
        return decision.decision_id
    
    def log_quality_gate(
        self,
        gate_name: str,
        measured_values: Dict[str, Any],
        outcome: str,
        threshold_value: Optional[Any] = None,
        reasoning: Optional[str] = None
    ) -> str:
        """
        Log quality control decisions.
        
        Args:
            gate_name: Name of quality gate
            measured_values: Actual measured metrics
            outcome: PASS/FAIL/WARNING
            threshold_value: Threshold used for decision
            reasoning: Rationale for threshold choice
            
        Returns:
            Decision ID
        """
        decision = ProvenanceDecision(
            decision_id=self._generate_decision_id(),
            timestamp=datetime.now().isoformat(),
            decision_type=DecisionType.QUALITY_GATE,
            severity=DecisionSeverity.CRITICAL,
            parameter_name=gate_name,
            parameter_value=threshold_value,
            reasoning=reasoning or f"Quality gate assessment for {gate_name}",
            quality_metrics=measured_values,
            quality_outcome=outcome,
            execution_context={
                "analysis_id": self.analysis_id,
                "method": "log_quality_gate"
            }
        )
        
        self.decisions.append(decision)
        return decision.decision_id
    
    def log_data_transformation(
        self,
        transformation_name: str,
        parameters: Dict[str, Any],
        reasoning: str,
        input_description: Optional[str] = None,
        output_description: Optional[str] = None
    ) -> str:
        """
        Log data transformation decisions.
        
        Args:
            transformation_name: Name of transformation applied
            parameters: Transformation parameters
            reasoning: Why this transformation was chosen
            input_description: Description of input data
            output_description: Description of output data
            
        Returns:
            Decision ID
        """
        decision = ProvenanceDecision(
            decision_id=self._generate_decision_id(),
            timestamp=datetime.now().isoformat(),
            decision_type=DecisionType.DATA_TRANSFORMATION,
            severity=DecisionSeverity.IMPORTANT,
            parameter_name=transformation_name,
            parameter_value=parameters,
            reasoning=reasoning,
            execution_context={
                "analysis_id": self.analysis_id,
                "method": "log_data_transformation",
                "input_description": input_description,
                "output_description": output_description
            }
        )
        
        self.decisions.append(decision)
        return decision.decision_id
    
    def log_method_selection(
        self,
        method_name: str,
        chosen_method: str,
        alternatives: List[str],
        reasoning: str,
        performance_comparison: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Log method selection decisions.
        
        Args:
            method_name: Category of method (e.g., "clustering", "segmentation")
            chosen_method: Selected method
            alternatives: Methods considered but not chosen
            reasoning: Rationale for selection
            performance_comparison: Comparative performance metrics
            
        Returns:
            Decision ID
        """
        decision = ProvenanceDecision(
            decision_id=self._generate_decision_id(),
            timestamp=datetime.now().isoformat(),
            decision_type=DecisionType.METHOD_SELECTION,
            severity=DecisionSeverity.CRITICAL,
            parameter_name=method_name,
            parameter_value=chosen_method,
            reasoning=reasoning,
            alternatives_considered=alternatives,
            evidence=performance_comparison or {},
            execution_context={
                "analysis_id": self.analysis_id,
                "method": "log_method_selection"
            }
        )
        
        self.decisions.append(decision)
        return decision.decision_id
    
    def log_threshold_setting(
        self,
        threshold_name: str,
        threshold_value: Union[float, int],
        reasoning: str,
        data_distribution: Optional[Dict[str, float]] = None,
        validation_metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Log threshold-setting decisions.
        
        Args:
            threshold_name: Name of threshold parameter
            threshold_value: Chosen threshold value
            reasoning: Scientific justification
            data_distribution: Distribution statistics that informed decision
            validation_metrics: Validation metrics for threshold performance
            
        Returns:
            Decision ID
        """
        decision = ProvenanceDecision(
            decision_id=self._generate_decision_id(),
            timestamp=datetime.now().isoformat(),
            decision_type=DecisionType.THRESHOLD_SETTING,
            severity=DecisionSeverity.IMPORTANT,
            parameter_name=threshold_name,
            parameter_value=threshold_value,
            reasoning=reasoning,
            evidence={
                "data_distribution": data_distribution or {},
                "validation_metrics": validation_metrics or {}
            },
            execution_context={
                "analysis_id": self.analysis_id,
                "method": "log_threshold_setting"
            }
        )
        
        self.decisions.append(decision)
        return decision.decision_id
    
    def log_validation_outcome(
        self,
        validation_name: str,
        outcome: str,
        metrics: Dict[str, float],
        interpretation: str,
        action_taken: Optional[str] = None
    ) -> str:
        """
        Log validation results and interpretations.
        
        Args:
            validation_name: Name of validation check
            outcome: Validation outcome (PASS/FAIL/WARNING)
            metrics: Validation metrics
            interpretation: Scientific interpretation of results
            action_taken: Action taken based on validation
            
        Returns:
            Decision ID
        """
        decision = ProvenanceDecision(
            decision_id=self._generate_decision_id(),
            timestamp=datetime.now().isoformat(),
            decision_type=DecisionType.VALIDATION_OUTCOME,
            severity=DecisionSeverity.IMPORTANT,
            parameter_name=validation_name,
            parameter_value=outcome,
            reasoning=interpretation,
            quality_metrics=metrics,
            quality_outcome=outcome,
            execution_context={
                "analysis_id": self.analysis_id,
                "method": "log_validation_outcome",
                "action_taken": action_taken
            }
        )
        
        self.decisions.append(decision)
        return decision.decision_id
    
    def track_data_lineage(
        self,
        input_source: str,
        processing_step: str,
        output_description: str,
        input_data: Optional[Any] = None,
        output_data: Optional[Any] = None
    ) -> None:
        """
        Track data lineage with checksums for integrity.
        
        Args:
            input_source: Description of input data source
            processing_step: Description of processing applied
            output_description: Description of output data
            input_data: Input data for checksum (optional)
            output_data: Output data for checksum (optional)
        """
        lineage = DataLineage(
            input_source=input_source,
            input_checksum=self._compute_checksum(input_data) if input_data is not None else "not_computed",
            processing_step=processing_step,
            output_description=output_description,
            output_checksum=self._compute_checksum(output_data) if output_data is not None else "not_computed"
        )
        
        self.data_lineage.append(lineage)
    
    def generate_methods_section(
        self,
        include_parameter_tables: bool = True,
        include_quality_metrics: bool = True,
        group_by_analysis_stage: bool = True
    ) -> str:
        """
        Auto-generate methods section from logged decisions.
        
        Args:
            include_parameter_tables: Include tables of parameter choices
            include_quality_metrics: Include quality control metrics
            group_by_analysis_stage: Group decisions by analysis stage
            
        Returns:
            Formatted methods section text
        """
        methods_text = []
        
        # Header
        methods_text.append("# Methods")
        methods_text.append(f"Analysis ID: {self.analysis_id}")
        methods_text.append(f"Analysis Date: {self.start_time.strftime('%Y-%m-%d')}")
        methods_text.append("")
        
        # Group decisions by type or stage
        if group_by_analysis_stage:
            methods_text.extend(self._generate_methods_by_stage())
        else:
            methods_text.extend(self._generate_methods_by_type())
        
        # Parameter summary table
        if include_parameter_tables:
            methods_text.append("\n## Parameter Summary")
            methods_text.extend(self._generate_parameter_table())
        
        # Quality control summary
        if include_quality_metrics:
            methods_text.append("\n## Quality Control")
            methods_text.extend(self._generate_quality_summary())
        
        # Data lineage
        if self.data_lineage:
            methods_text.append("\n## Data Processing Pipeline")
            methods_text.extend(self._generate_lineage_summary())
        
        return "\n".join(methods_text)
    
    def _generate_methods_by_stage(self) -> List[str]:
        """Generate methods text grouped by analysis stage."""
        # Define analysis stages
        stage_order = [
            DecisionType.DATA_TRANSFORMATION,
            DecisionType.QUALITY_GATE,
            DecisionType.METHOD_SELECTION,
            DecisionType.PARAMETER_CHOICE,
            DecisionType.THRESHOLD_SETTING,
            DecisionType.VALIDATION_OUTCOME
        ]
        
        stage_names = {
            DecisionType.DATA_TRANSFORMATION: "Data Processing",
            DecisionType.QUALITY_GATE: "Quality Control",
            DecisionType.METHOD_SELECTION: "Method Selection",
            DecisionType.PARAMETER_CHOICE: "Parameter Optimization",
            DecisionType.THRESHOLD_SETTING: "Threshold Determination",
            DecisionType.VALIDATION_OUTCOME: "Validation Results"
        }
        
        methods_text = []
        
        for stage in stage_order:
            stage_decisions = [d for d in self.decisions if d.decision_type == stage]
            if not stage_decisions:
                continue
            
            methods_text.append(f"\n## {stage_names[stage]}")
            
            for decision in stage_decisions:
                methods_text.extend(self._format_decision_for_methods(decision))
        
        return methods_text
    
    def _generate_methods_by_type(self) -> List[str]:
        """Generate methods text grouped by decision type."""
        methods_text = []
        
        # Group by decision type
        by_type = {}
        for decision in self.decisions:
            if decision.decision_type not in by_type:
                by_type[decision.decision_type] = []
            by_type[decision.decision_type].append(decision)
        
        for decision_type, decisions in by_type.items():
            methods_text.append(f"\n## {decision_type.value.replace('_', ' ').title()}")
            
            for decision in decisions:
                methods_text.extend(self._format_decision_for_methods(decision))
        
        return methods_text
    
    def _format_decision_for_methods(self, decision: ProvenanceDecision) -> List[str]:
        """Format a single decision for methods section."""
        text = []
        
        # Main decision
        text.append(f"\n**{decision.parameter_name}**: {decision.parameter_value}")
        text.append(f"{decision.reasoning}")
        
        # Alternatives considered
        if decision.alternatives_considered:
            alt_text = ", ".join(str(alt) for alt in decision.alternatives_considered)
            text.append(f"Alternatives considered: {alt_text}")
        
        # Evidence/metrics
        if decision.evidence:
            evidence_items = []
            for key, value in decision.evidence.items():
                if isinstance(value, dict):
                    evidence_items.append(f"{key}: {json.dumps(value, default=str)}")
                else:
                    evidence_items.append(f"{key}: {value}")
            if evidence_items:
                text.append(f"Supporting evidence: {'; '.join(evidence_items)}")
        
        # Quality metrics
        if decision.quality_metrics:
            metrics_text = []
            for metric, value in decision.quality_metrics.items():
                if isinstance(value, float):
                    metrics_text.append(f"{metric}={value:.3f}")
                else:
                    metrics_text.append(f"{metric}={value}")
            text.append(f"Quality metrics: {', '.join(metrics_text)}")
        
        # References
        if decision.references:
            text.append(f"References: {'; '.join(decision.references)}")
        
        return text
    
    def _generate_parameter_table(self) -> List[str]:
        """Generate parameter summary table."""
        parameter_decisions = [d for d in self.decisions if d.decision_type == DecisionType.PARAMETER_CHOICE]
        
        if not parameter_decisions:
            return ["No parameter decisions logged."]
        
        text = []
        text.append("| Parameter | Value | Reasoning |")
        text.append("|-----------|-------|-----------|")
        
        for decision in parameter_decisions:
            value_str = str(decision.parameter_value)
            if len(value_str) > 50:
                value_str = value_str[:47] + "..."
            
            reasoning_str = decision.reasoning
            if len(reasoning_str) > 80:
                reasoning_str = reasoning_str[:77] + "..."
            
            text.append(f"| {decision.parameter_name} | {value_str} | {reasoning_str} |")
        
        return text
    
    def _generate_quality_summary(self) -> List[str]:
        """Generate quality control summary."""
        quality_decisions = [d for d in self.decisions if d.decision_type == DecisionType.QUALITY_GATE]
        
        if not quality_decisions:
            return ["No quality gates executed."]
        
        text = []
        text.append("| Quality Gate | Outcome | Key Metrics |")
        text.append("|--------------|---------|-------------|")
        
        for decision in quality_decisions:
            metrics_summary = ""
            if decision.quality_metrics:
                key_metrics = list(decision.quality_metrics.keys())[:3]  # First 3 metrics
                metrics_summary = ", ".join(f"{k}={decision.quality_metrics[k]}" for k in key_metrics)
                if len(decision.quality_metrics) > 3:
                    metrics_summary += f" (+{len(decision.quality_metrics)-3} more)"
            
            text.append(f"| {decision.parameter_name} | {decision.quality_outcome} | {metrics_summary} |")
        
        return text
    
    def _generate_lineage_summary(self) -> List[str]:
        """Generate data lineage summary."""
        text = []
        text.append("| Input | Processing | Output | Checksum |")
        text.append("|-------|------------|--------|----------|")
        
        for lineage in self.data_lineage:
            text.append(f"| {lineage.input_source} | {lineage.processing_step} | {lineage.output_description} | {lineage.output_checksum} |")
        
        return text
    
    def save_provenance_record(self) -> str:
        """
        Save complete provenance record to JSON file.
        
        Returns:
            Path to saved provenance file
        """
        # Update metadata
        self.metadata["end_time"] = datetime.now().isoformat()
        self.metadata["total_decisions"] = len(self.decisions)
        self.metadata["total_lineage_records"] = len(self.data_lineage)
        
        # Compile complete record
        record = {
            "metadata": self.metadata,
            "decisions": [asdict(decision) for decision in self.decisions],
            "data_lineage": [asdict(lineage) for lineage in self.data_lineage],
            "methods_section": self.generate_methods_section()
        }
        
        # Save to file
        with open(self.provenance_file, 'w') as f:
            json.dump(record, f, indent=2, default=str)
        
        return str(self.provenance_file)
    
    def load_provenance_record(self, file_path: str) -> None:
        """
        Load provenance record from JSON file.
        
        Args:
            file_path: Path to provenance JSON file
        """
        with open(file_path, 'r') as f:
            record = json.load(f)
        
        self.metadata = record["metadata"]
        
        # Reconstruct decisions
        self.decisions = []
        for decision_dict in record["decisions"]:
            # Convert enum values back
            decision_dict["decision_type"] = DecisionType(decision_dict["decision_type"])
            decision_dict["severity"] = DecisionSeverity(decision_dict["severity"])
            
            decision = ProvenanceDecision(**decision_dict)
            self.decisions.append(decision)
        
        # Reconstruct lineage
        self.data_lineage = []
        for lineage_dict in record["data_lineage"]:
            lineage = DataLineage(**lineage_dict)
            self.data_lineage.append(lineage)
    
    def get_decision_summary(self) -> Dict[str, Any]:
        """Get summary statistics of logged decisions."""
        return {
            "total_decisions": len(self.decisions),
            "decisions_by_type": {
                dt.value: len([d for d in self.decisions if d.decision_type == dt])
                for dt in DecisionType
            },
            "decisions_by_severity": {
                severity.value: len([d for d in self.decisions if d.severity == severity])
                for severity in DecisionSeverity
            },
            "quality_gates": {
                "total": len([d for d in self.decisions if d.decision_type == DecisionType.QUALITY_GATE]),
                "passed": len([d for d in self.decisions if d.decision_type == DecisionType.QUALITY_GATE and d.quality_outcome == "PASS"]),
                "failed": len([d for d in self.decisions if d.decision_type == DecisionType.QUALITY_GATE and d.quality_outcome == "FAIL"])
            },
            "data_lineage_records": len(self.data_lineage)
        }


# Convenience functions for quick integration
def create_tracker(analysis_id: str, output_dir: Optional[str] = None) -> ProvenanceTracker:
    """Create a new provenance tracker instance."""
    return ProvenanceTracker(analysis_id, output_dir)


def load_tracker(provenance_file: str) -> ProvenanceTracker:
    """Load existing provenance tracker from file."""
    tracker = ProvenanceTracker("loaded", None)
    tracker.load_provenance_record(provenance_file)
    return tracker


# Example usage for documentation
def _example_usage():
    """Example showing how to use the provenance tracker."""
    # Create tracker
    tracker = ProvenanceTracker("kidney_analysis_001")
    
    # Log parameter decisions
    tracker.log_parameter_decision(
        parameter_name="slic_compactness",
        parameter_value=0.1,
        reasoning="Kidney tubule morphology requires tight segments to preserve biological boundaries",
        alternatives_considered=[0.05, 0.2, 0.5],
        evidence={"tubule_boundary_preservation": 0.92, "oversegmentation_risk": 0.15}
    )
    
    # Log quality gate
    tracker.log_quality_gate(
        gate_name="dna_signal_threshold",
        measured_values={"median_signal": 245, "background_ratio": 12.3},
        outcome="PASS",
        threshold_value=200,
        reasoning="Threshold ensures tissue detection while excluding imaging artifacts"
    )
    
    # Log data transformation
    tracker.log_data_transformation(
        transformation_name="arcsinh_transform",
        parameters={"cofactor": 5},
        reasoning="Arcsinh transformation compresses dynamic range while preserving gradients for clustering"
    )
    
    # Track data lineage
    tracker.track_data_lineage(
        input_source="IMC_ROI_001.txt",
        processing_step="arcsinh_transform -> slic_segmentation -> clustering",
        output_description="Clustered superpixel features"
    )
    
    # Generate methods section
    methods_text = tracker.generate_methods_section()
    print(methods_text)
    
    # Save provenance record
    provenance_file = tracker.save_provenance_record()
    print(f"Provenance saved to: {provenance_file}")


if __name__ == "__main__":
    _example_usage()