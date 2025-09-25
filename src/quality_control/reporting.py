"""
Quality Control Reporting and Visualization

Provides essential quality reporting with actionable insights and trend visualization
without overengineering complex dashboards.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import logging

from .statistical_monitoring import QualityMonitor, QualityMetrics
from .quality_gates import QualityGateEngine


class QualityReporter:
    """
    Quality reporting and visualization for IMC analysis.
    
    Focuses on actionable insights and clear trend visualization
    without complex dashboard overengineering.
    """
    
    def __init__(self, output_dir: str = "quality_reports"):
        """Initialize quality reporter."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger('QualityReporter')
    
    def generate_quality_report(
        self,
        quality_monitor: QualityMonitor,
        gate_engine: QualityGateEngine,
        report_name: str = "quality_report"
    ) -> Dict[str, str]:
        """
        Generate comprehensive quality report.
        
        Returns:
            Dictionary of generated file paths
        """
        self.logger.info(f"Generating quality report: {report_name}")
        
        generated_files = {}
        
        # 1. Summary statistics report
        summary_file = self._generate_summary_report(quality_monitor, gate_engine, report_name)
        generated_files['summary'] = str(summary_file)
        
        # 2. Quality trends visualization
        if len(quality_monitor.quality_history) > 1:
            trends_file = self._generate_trends_plot(quality_monitor, report_name)
            generated_files['trends'] = str(trends_file)
        
        # 3. Batch comparison visualization
        if len(quality_monitor.quality_history) > 0:
            batch_file = self._generate_batch_comparison_plot(quality_monitor, report_name)
            generated_files['batch_comparison'] = str(batch_file)
        
        # 4. Quality distribution plot
        if len(quality_monitor.quality_history) > 5:
            distribution_file = self._generate_quality_distribution_plot(quality_monitor, report_name)
            generated_files['distribution'] = str(distribution_file)
        
        # 5. Actionable alerts summary
        alerts_file = self._generate_alerts_summary(quality_monitor, gate_engine, report_name)
        generated_files['alerts'] = str(alerts_file)
        
        self.logger.info(f"Generated {len(generated_files)} quality report files")
        return generated_files
    
    def _generate_summary_report(
        self,
        quality_monitor: QualityMonitor,
        gate_engine: QualityGateEngine,
        report_name: str
    ) -> Path:
        """Generate text-based summary report."""
        summary_file = self.output_dir / f"{report_name}_summary.txt"
        
        # Get comprehensive summaries
        quality_summary = quality_monitor.generate_quality_summary()
        decision_summary = gate_engine.get_decision_summary()
        
        with open(summary_file, 'w') as f:
            f.write("IMC Analysis Quality Control Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 20 + "\n")
            if quality_summary.get('status') != 'no_data':
                stats = quality_summary['overall_quality_stats']
                f.write(f"Total ROIs analyzed: {quality_summary['total_rois']}\n")
                f.write(f"Unique batches: {quality_summary['unique_batches']}\n")
                f.write(f"Mean quality score: {stats['mean']:.3f}\n")
                f.write(f"Quality range: {stats['min']:.3f} - {stats['max']:.3f}\n")
                f.write(f"Quality std deviation: {stats['std']:.3f}\n\n")
            
            # Quality distribution
            f.write("QUALITY DISTRIBUTION\n")
            f.write("-" * 20 + "\n")
            if 'quality_distribution' in quality_summary:
                dist = quality_summary['quality_distribution']
                f.write(f"Excellent (>0.85): {dist['excellent_(>0.85)']} ROIs\n")
                f.write(f"Good (0.7-0.85): {dist['good_(0.7-0.85)']} ROIs\n")
                f.write(f"Fair (0.5-0.7): {dist['fair_(0.5-0.7)']} ROIs\n")
                f.write(f"Poor (<0.5): {dist['poor_(<0.5)']} ROIs\n\n")
            
            # Quality gate decisions
            f.write("QUALITY GATE DECISIONS\n")
            f.write("-" * 24 + "\n")
            if decision_summary.get('status') != 'no_decisions':
                dist = decision_summary['decision_distribution']
                f.write(f"Pass rate: {decision_summary['pass_rate']:.1%}\n")
                f.write(f"Passed: {dist.get('pass', 0)} ROIs\n")
                f.write(f"Warnings: {dist.get('warn', 0)} ROIs\n")
                f.write(f"Failed: {dist.get('fail', 0)} ROIs\n")
                f.write(f"Aborted: {dist.get('abort', 0)} ROIs\n\n")
            
            # Batch consistency
            f.write("BATCH CONSISTENCY\n")
            f.write("-" * 17 + "\n")
            if 'batch_consistency' in quality_summary:
                for batch_info in quality_summary['batch_consistency']:
                    f.write(f"Batch {batch_info['batch_id']}: {batch_info['status']} ")
                    f.write(f"({batch_info['n_rois']} ROIs, CV={batch_info['cv_quality']:.2f})\n")
                f.write("\n")
            
            # Trend analysis
            f.write("TREND ANALYSIS\n")
            f.write("-" * 14 + "\n")
            if 'trend_analysis' in quality_summary:
                trend = quality_summary['trend_analysis']
                if trend.get('status') != 'insufficient_data':
                    f.write(f"Recent trend: {trend['status']}\n")
                    f.write(f"Trend slope: {trend['slope']:.4f}\n")
                    f.write(f"Recent mean quality: {trend['recent_mean']:.3f}\n\n")
            
            # Recent decisions
            f.write("RECENT DECISIONS\n")
            f.write("-" * 16 + "\n")
            if 'recent_decisions' in decision_summary:
                for roi_id, decision, reason in decision_summary['recent_decisions'][-5:]:
                    f.write(f"{roi_id}: {decision.upper()} - {reason[:80]}...\n")
        
        return summary_file
    
    def _generate_trends_plot(self, quality_monitor: QualityMonitor, report_name: str) -> Path:
        """Generate quality trends visualization."""
        trends_file = self.output_dir / f"{report_name}_trends.png"
        
        # Extract data for plotting
        history = quality_monitor.quality_history
        roi_indices = list(range(len(history)))
        overall_qualities = [qm.overall_quality() for qm in history]
        coordinate_qualities = [qm.coordinate_quality for qm in history]
        ion_count_qualities = [qm.ion_count_quality for qm in history]
        biological_qualities = [qm.biological_quality for qm in history]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Overall quality trend
        ax1.plot(roi_indices, overall_qualities, 'b-', linewidth=2, label='Overall Quality')
        ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Minimum Threshold')
        ax1.axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='Warning Threshold')
        ax1.set_ylabel('Quality Score')
        ax1.set_title('Quality Trends Over ROIs')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Component quality trends
        ax2.plot(roi_indices, coordinate_qualities, 'g-', alpha=0.7, label='Coordinate Quality')
        ax2.plot(roi_indices, ion_count_qualities, 'r-', alpha=0.7, label='Ion Count Quality')
        ax2.plot(roi_indices, biological_qualities, 'm-', alpha=0.7, label='Biological Quality')
        ax2.set_xlabel('ROI Index')
        ax2.set_ylabel('Quality Score')
        ax2.set_title('Component Quality Trends')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(trends_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return trends_file
    
    def _generate_batch_comparison_plot(self, quality_monitor: QualityMonitor, report_name: str) -> Path:
        """Generate batch comparison visualization."""
        batch_file = self.output_dir / f"{report_name}_batches.png"
        
        # Group data by batch
        batch_data = {}
        for qm in quality_monitor.quality_history:
            if qm.batch_id not in batch_data:
                batch_data[qm.batch_id] = []
            batch_data[qm.batch_id].append(qm.overall_quality())
        
        if len(batch_data) < 2:
            # Create simple histogram if only one batch
            plt.figure(figsize=(10, 6))
            all_qualities = [qm.overall_quality() for qm in quality_monitor.quality_history]
            plt.hist(all_qualities, bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('Quality Score')
            plt.ylabel('Frequency')
            plt.title('Quality Score Distribution')
            plt.axvline(x=0.5, color='r', linestyle='--', label='Minimum Threshold')
            plt.axvline(x=0.7, color='orange', linestyle='--', label='Warning Threshold')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            # Box plot comparison across batches
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Box plot
            batch_names = list(batch_data.keys())
            batch_qualities = list(batch_data.values())
            
            box_plot = ax1.boxplot(batch_qualities, labels=batch_names, patch_artist=True)
            ax1.set_ylabel('Quality Score')
            ax1.set_title('Quality Distribution by Batch')
            ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Minimum')
            ax1.axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='Warning')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Color boxes by mean quality
            for patch, qualities in zip(box_plot['boxes'], batch_qualities):
                mean_qual = np.mean(qualities)
                if mean_qual < 0.5:
                    patch.set_facecolor('lightcoral')
                elif mean_qual < 0.7:
                    patch.set_facecolor('khaki')
                else:
                    patch.set_facecolor('lightgreen')
            
            # Batch statistics
            batch_stats = []
            for batch_id, qualities in batch_data.items():
                batch_stats.append({
                    'Batch': batch_id,
                    'N_ROIs': len(qualities),
                    'Mean': np.mean(qualities),
                    'Std': np.std(qualities),
                    'CV': np.std(qualities) / np.mean(qualities) if np.mean(qualities) > 0 else 0
                })
            
            # Table of batch statistics
            ax2.axis('tight')
            ax2.axis('off')
            table_data = [[stat['Batch'], stat['N_ROIs'], f"{stat['Mean']:.3f}", 
                          f"{stat['Std']:.3f}", f"{stat['CV']:.3f}"] for stat in batch_stats]
            table = ax2.table(cellText=table_data,
                             colLabels=['Batch', 'N ROIs', 'Mean', 'Std', 'CV'],
                             cellLoc='center',
                             loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            ax2.set_title('Batch Statistics')
        
        plt.tight_layout()
        plt.savefig(batch_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return batch_file
    
    def _generate_quality_distribution_plot(self, quality_monitor: QualityMonitor, report_name: str) -> Path:
        """Generate quality distribution analysis."""
        dist_file = self.output_dir / f"{report_name}_distribution.png"
        
        history = quality_monitor.quality_history
        
        # Create subplot grid
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Overall quality histogram
        overall_qualities = [qm.overall_quality() for qm in history]
        ax1.hist(overall_qualities, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
        ax1.axvline(x=0.5, color='r', linestyle='--', linewidth=2, label='Minimum')
        ax1.axvline(x=0.7, color='orange', linestyle='--', linewidth=2, label='Warning')
        ax1.axvline(x=np.mean(overall_qualities), color='green', linestyle='-', linewidth=2, label='Mean')
        ax1.set_xlabel('Overall Quality Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Overall Quality Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Component quality comparison
        component_data = {
            'Coordinate': [qm.coordinate_quality for qm in history],
            'Ion Count': [qm.ion_count_quality for qm in history],
            'Biological': [qm.biological_quality for qm in history]
        }
        
        positions = [1, 2, 3]
        violin_parts = ax2.violinplot([component_data['Coordinate'], 
                                      component_data['Ion Count'],
                                      component_data['Biological']], 
                                     positions=positions)
        ax2.set_xticks(positions)
        ax2.set_xticklabels(['Coordinate', 'Ion Count', 'Biological'])
        ax2.set_ylabel('Quality Score')
        ax2.set_title('Component Quality Distributions')
        ax2.grid(True, alpha=0.3)
        
        # 3. Quality vs technical metrics
        n_pixels = [qm.n_pixels for qm in history if qm.n_pixels > 0]
        qualities_with_pixels = [qm.overall_quality() for qm in history if qm.n_pixels > 0]
        
        if n_pixels:
            ax3.scatter(n_pixels, qualities_with_pixels, alpha=0.6, color='purple')
            ax3.set_xlabel('Number of Pixels')
            ax3.set_ylabel('Quality Score')
            ax3.set_title('Quality vs Dataset Size')
            ax3.grid(True, alpha=0.3)
        
        # 4. Quality over time (ROI order)
        roi_order = list(range(len(overall_qualities)))
        ax4.scatter(roi_order, overall_qualities, alpha=0.6, color='red', s=30)
        
        # Add trend line
        if len(overall_qualities) > 2:
            z = np.polyfit(roi_order, overall_qualities, 1)
            p = np.poly1d(z)
            ax4.plot(roi_order, p(roi_order), "r--", alpha=0.8, linewidth=2)
        
        ax4.set_xlabel('ROI Processing Order')
        ax4.set_ylabel('Quality Score')
        ax4.set_title('Quality Trends Over Time')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(dist_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return dist_file
    
    def _generate_alerts_summary(
        self,
        quality_monitor: QualityMonitor,
        gate_engine: QualityGateEngine,
        report_name: str
    ) -> Path:
        """Generate actionable alerts summary."""
        alerts_file = self.output_dir / f"{report_name}_alerts.json"
        
        # Get current status
        quality_summary = quality_monitor.generate_quality_summary()
        decision_summary = gate_engine.get_decision_summary()
        
        alerts = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'overall_status': 'healthy',
            'critical_alerts': [],
            'warnings': [],
            'recommendations': [],
            'summary_stats': quality_summary.get('overall_quality_stats', {}),
            'decision_stats': decision_summary
        }
        
        # Check for critical issues
        if quality_summary.get('status') != 'no_data':
            stats = quality_summary['overall_quality_stats']
            
            # Low overall quality
            if stats['mean'] < 0.5:
                alerts['critical_alerts'].append({
                    'type': 'low_quality',
                    'message': f"Mean quality critically low: {stats['mean']:.3f}",
                    'action': 'Review data preprocessing and acquisition parameters'
                })
                alerts['overall_status'] = 'critical'
            
            # High quality variation
            if stats['std'] > 0.3:
                alerts['warnings'].append({
                    'type': 'high_variation',
                    'message': f"High quality variation: std={stats['std']:.3f}",
                    'action': 'Check for batch effects or systematic issues'
                })
            
            # Poor quality distribution
            if 'quality_distribution' in quality_summary:
                dist = quality_summary['quality_distribution']
                poor_fraction = dist['poor_(<0.5)'] / quality_summary['total_rois']
                if poor_fraction > 0.2:
                    alerts['warnings'].append({
                        'type': 'many_poor_rois',
                        'message': f"{poor_fraction:.1%} of ROIs have poor quality",
                        'action': 'Review ROI selection criteria and preprocessing'
                    })
        
        # Check gate decisions
        if decision_summary.get('status') != 'no_decisions':
            fail_rate = decision_summary['decision_distribution'].get('fail', 0) / decision_summary['total_decisions']
            if fail_rate > 0.3:
                alerts['warnings'].append({
                    'type': 'high_failure_rate',
                    'message': f"High ROI failure rate: {fail_rate:.1%}",
                    'action': 'Adjust quality thresholds or improve data quality'
                })
        
        # Generate recommendations
        recommendations = []
        
        if alerts['critical_alerts']:
            recommendations.append("URGENT: Address critical quality issues before continuing analysis")
        
        if len(alerts['warnings']) > 2:
            recommendations.append("Multiple quality concerns detected - consider systematic review")
        
        # Batch-specific recommendations
        if 'batch_consistency' in quality_summary:
            poor_batches = [b for b in quality_summary['batch_consistency'] if b['status'] == 'poor']
            if poor_batches:
                batch_ids = [b['batch_id'] for b in poor_batches]
                recommendations.append(f"Review batch consistency for: {', '.join(batch_ids)}")
        
        # Trend-based recommendations
        if 'trend_analysis' in quality_summary:
            trend = quality_summary['trend_analysis']
            if trend.get('status') == 'declining':
                recommendations.append("Quality declining over time - check for systematic drift")
        
        alerts['recommendations'] = recommendations
        
        # Update overall status
        if alerts['critical_alerts']:
            alerts['overall_status'] = 'critical'
        elif alerts['warnings']:
            alerts['overall_status'] = 'warning'
        
        # Save alerts
        with open(alerts_file, 'w') as f:
            json.dump(alerts, f, indent=2)
        
        return alerts_file


# Convenience function
def generate_quality_reports(
    quality_monitor: QualityMonitor,
    gate_engine: QualityGateEngine,
    output_dir: str = "quality_reports",
    report_name: str = "imc_quality"
) -> Dict[str, str]:
    """
    Generate all quality control reports.
    
    Returns:
        Dictionary of generated file paths
    """
    reporter = QualityReporter(output_dir)
    return reporter.generate_quality_report(quality_monitor, gate_engine, report_name)