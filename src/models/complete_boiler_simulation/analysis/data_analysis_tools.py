#!/usr/bin/env python3
"""
Boiler Data Analysis Tools - ASCII Fixed Version

Enhanced analysis tools for annual boiler operation data with comprehensive
reporting capabilities and ASCII-safe output.

Key Features:
- Comprehensive efficiency analysis
- Fouling progression tracking  
- Soot blowing effectiveness evaluation
- Economic optimization analysis
- ML-ready feature extraction
- ASCII-safe logging and output

Author: Enhanced Boiler Modeling System
Version: 8.1 - ASCII Compatibility Fix
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime

# Set up logging for analysis tools
logger = logging.getLogger(__name__)

# Calculate project root and create output directories
project_root = Path(__file__).parent.parent.parent.parent.parent
output_dir = project_root / "outputs" / "analysis"
output_dir.mkdir(parents=True, exist_ok=True)

figures_dir = project_root / "outputs" / "figures"  
figures_dir.mkdir(parents=True, exist_ok=True)


class BoilerDataAnalyzer:
    """Enhanced boiler data analyzer with comprehensive reporting capabilities."""
    
    def __init__(self, data: pd.DataFrame):
        """Initialize analyzer with boiler operation dataset."""
        self.data = data.copy()
        self.logger = logging.getLogger(f"{__name__}.BoilerDataAnalyzer")
        
        # Validate required columns
        self._validate_dataset()
        
        # Convert timestamp if needed
        if 'timestamp' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        
        self.logger.info(f"Analyzer initialized with {len(self.data)} records")
    
    def _validate_dataset(self):
        """Validate that dataset contains required columns."""
        required_columns = ['system_efficiency', 'load_factor']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            raise ValueError(f"Dataset missing required columns: {missing_columns}")
    
    def generate_comprehensive_report(self, save_plots: bool = True) -> Dict:
        """Generate comprehensive analysis report."""
        
        self.logger.info("Starting comprehensive analysis report generation")
        
        report = {
            'dataset_summary': self._analyze_dataset_summary(),
            'efficiency_analysis': self._analyze_efficiency_patterns(),
            'load_factor_analysis': self._analyze_load_patterns(),
            'fouling_analysis': self._analyze_fouling_progression(),
            'soot_blowing_analysis': self._analyze_soot_blowing_effectiveness(),
            'optimization_opportunities': self._identify_optimization_opportunities(),
            'ml_features': self._extract_ml_features(),
            'report_metadata': {
                'generated_at': datetime.now(),
                'dataset_records': len(self.data),
                'analysis_version': '8.1'
            }
        }
        
        if save_plots:
            self._generate_analysis_plots()
        
        # Save report summary
        self._save_report_summary(report)
        
        self.logger.info("Comprehensive analysis report completed")
        return report
    
    def _analyze_dataset_summary(self) -> Dict:
        """Analyze basic dataset characteristics."""
        
        summary = {
            'total_records': len(self.data),
            'date_range': {
                'start': self.data['timestamp'].min() if 'timestamp' in self.data.columns else None,
                'end': self.data['timestamp'].max() if 'timestamp' in self.data.columns else None
            },
            'data_quality': {
                'missing_values': self.data.isnull().sum().to_dict(),
                'data_completeness': (1 - self.data.isnull().sum() / len(self.data)).to_dict()
            }
        }
        
        return summary
    
    def _analyze_efficiency_patterns(self) -> Dict:
        """Analyze system efficiency patterns and trends."""
        
        if 'system_efficiency' not in self.data.columns:
            return {'error': 'system_efficiency column not found'}
        
        efficiency = self.data['system_efficiency']
        
        analysis = {
            'statistics': {
                'mean': float(efficiency.mean()),
                'std': float(efficiency.std()),
                'min': float(efficiency.min()),
                'max': float(efficiency.max()),
                'variation_coefficient': float(efficiency.std() / efficiency.mean())
            },
            'performance_assessment': self._assess_efficiency_performance(efficiency),
            'trends': self._analyze_efficiency_trends()
        }
        
        return analysis
    
    def _analyze_load_patterns(self) -> Dict:
        """Analyze load factor patterns and operational characteristics."""
        
        if 'load_factor' not in self.data.columns:
            return {'error': 'load_factor column not found'}
        
        load_factor = self.data['load_factor']
        
        analysis = {
            'statistics': {
                'mean': float(load_factor.mean()),
                'std': float(load_factor.std()),
                'min': float(load_factor.min()),
                'max': float(load_factor.max()),
                'operating_range': f"{load_factor.min():.1%} - {load_factor.max():.1%}"
            },
            'operational_patterns': self._analyze_operational_patterns()
        }
        
        return analysis
    
    def _analyze_fouling_progression(self) -> Dict:
        """Analyze fouling progression and accumulation patterns."""
        
        # Look for fouling-related columns
        fouling_columns = [col for col in self.data.columns if 'fouling' in col.lower()]
        
        if not fouling_columns:
            return {'message': 'No fouling data available for analysis'}
        
        analysis = {
            'fouling_progression': self._calculate_fouling_trends(fouling_columns),
            'section_comparison': self._compare_fouling_by_section(fouling_columns)
        }
        
        return analysis
    
    def _analyze_soot_blowing_effectiveness(self) -> Dict:
        """Analyze soot blowing effectiveness and optimization opportunities."""
        
        # Look for soot blowing related columns
        sb_columns = [col for col in self.data.columns if 'soot' in col.lower() or 'cleaning' in col.lower()]
        
        if not sb_columns:
            return {'message': 'No soot blowing data available for analysis'}
        
        analysis = {
            'cleaning_frequency': self._analyze_cleaning_frequency(),
            'effectiveness_metrics': self._calculate_cleaning_effectiveness()
        }
        
        return analysis
    
    def _identify_optimization_opportunities(self) -> Dict:
        """Identify potential optimization opportunities."""
        
        opportunities = []
        
        # Efficiency optimization
        if 'system_efficiency' in self.data.columns:
            efficiency = self.data['system_efficiency']
            if efficiency.std() > 0.02:  # High variation
                opportunities.append({
                    'area': 'efficiency_stability',
                    'description': 'High efficiency variation detected - optimize control systems',
                    'potential_impact': 'Medium',
                    'variation': f"{efficiency.std():.2%}"
                })
        
        # Load factor optimization
        if 'load_factor' in self.data.columns:
            load_factor = self.data['load_factor']
            if load_factor.mean() < 0.80:  # Low average load
                opportunities.append({
                    'area': 'load_optimization',
                    'description': 'Average load factor below optimal range',
                    'potential_impact': 'High',
                    'current_average': f"{load_factor.mean():.1%}"
                })
        
        return {'opportunities': opportunities}
    
    def _extract_ml_features(self) -> Dict:
        """Extract features suitable for ML model development."""
        
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        
        features = {
            'feature_columns': list(numeric_columns),
            'feature_statistics': {},
            'correlation_matrix': self.data[numeric_columns].corr().to_dict() if len(numeric_columns) > 1 else {},
            'ml_recommendations': {
                'target_variables': ['system_efficiency', 'energy_balance_error_pct'],
                'feature_categories': {
                    'operational': [col for col in numeric_columns if any(keyword in col.lower() 
                                   for keyword in ['load', 'temp', 'flow', 'pressure'])],
                    'performance': [col for col in numeric_columns if any(keyword in col.lower() 
                                   for keyword in ['efficiency', 'error', 'loss'])],
                    'maintenance': [col for col in numeric_columns if any(keyword in col.lower() 
                                   for keyword in ['fouling', 'soot', 'cleaning'])]
                }
            }
        }
        
        # Calculate feature statistics
        for col in numeric_columns:
            features['feature_statistics'][col] = {
                'mean': float(self.data[col].mean()),
                'std': float(self.data[col].std()),
                'range': [float(self.data[col].min()), float(self.data[col].max())]
            }
        
        return features
    
    def _assess_efficiency_performance(self, efficiency: pd.Series) -> Dict:
        """Assess efficiency performance against industry standards."""
        
        assessment = {
            'grade': 'Unknown',
            'comments': []
        }
        
        avg_efficiency = efficiency.mean()
        std_efficiency = efficiency.std()
        
        # Performance grading
        if avg_efficiency >= 0.85 and std_efficiency <= 0.02:
            assessment['grade'] = 'Excellent'
            assessment['comments'].append('High efficiency with good stability')
        elif avg_efficiency >= 0.80 and std_efficiency <= 0.03:
            assessment['grade'] = 'Good' 
            assessment['comments'].append('Acceptable efficiency performance')
        elif avg_efficiency >= 0.75:
            assessment['grade'] = 'Fair'
            assessment['comments'].append('Efficiency within acceptable range but has improvement potential')
        else:
            assessment['grade'] = 'Poor'
            assessment['comments'].append('Efficiency below industry standards')
        
        # Variation assessment
        if std_efficiency > 0.03:
            assessment['comments'].append('High efficiency variation - control system optimization needed')
        
        return assessment
    
    def _analyze_efficiency_trends(self) -> Dict:
        """Analyze efficiency trends over time."""
        
        if 'timestamp' not in self.data.columns or 'system_efficiency' not in self.data.columns:
            return {'message': 'Insufficient data for trend analysis'}
        
        # Simple trend analysis
        data_with_time = self.data.set_index('timestamp')['system_efficiency'].resample('D').mean()
        
        # Calculate basic trend
        if len(data_with_time) > 1:
            trend_slope = np.polyfit(range(len(data_with_time)), data_with_time.values, 1)[0]
            
            return {
                'daily_average_trend': 'improving' if trend_slope > 0.0001 else 'declining' if trend_slope < -0.0001 else 'stable',
                'trend_magnitude': abs(trend_slope) * 365,  # Annual change rate
                'data_points': len(data_with_time)
            }
        
        return {'message': 'Insufficient data points for trend analysis'}
    
    def _analyze_operational_patterns(self) -> Dict:
        """Analyze operational load patterns."""
        
        patterns = {}
        
        if 'timestamp' in self.data.columns and 'load_factor' in self.data.columns:
            # Daily patterns
            self.data['hour'] = self.data['timestamp'].dt.hour
            hourly_load = self.data.groupby('hour')['load_factor'].mean()
            
            patterns['hourly_variation'] = {
                'peak_hour': int(hourly_load.idxmax()),
                'low_hour': int(hourly_load.idxmin()),
                'variation_range': f"{hourly_load.min():.1%} - {hourly_load.max():.1%}"
            }
        
        return patterns
    
    def _calculate_fouling_trends(self, fouling_columns: List[str]) -> Dict:
        """Calculate fouling accumulation trends."""
        
        trends = {}
        for col in fouling_columns[:5]:  # Limit to first 5 columns
            if col in self.data.columns:
                values = self.data[col].dropna()
                if len(values) > 1:
                    trends[col] = {
                        'initial': float(values.iloc[0]),
                        'final': float(values.iloc[-1]),
                        'change': float(values.iloc[-1] - values.iloc[0]),
                        'trend': 'increasing' if values.iloc[-1] > values.iloc[0] else 'stable'
                    }
        
        return trends
    
    def _compare_fouling_by_section(self, fouling_columns: List[str]) -> Dict:
        """Compare fouling levels across different boiler sections."""
        
        comparison = {}
        for col in fouling_columns[:5]:  # Limit analysis
            if col in self.data.columns:
                comparison[col] = {
                    'mean_fouling': float(self.data[col].mean()),
                    'max_fouling': float(self.data[col].max())
                }
        
        return comparison
    
    def _analyze_cleaning_frequency(self) -> Dict:
        """Analyze cleaning frequency patterns."""
        
        # Look for cleaning indicators
        cleaning_cols = [col for col in self.data.columns if 'clean' in col.lower() or 'soot' in col.lower()]
        
        if not cleaning_cols:
            return {'message': 'No cleaning data available'}
        
        frequency = {}
        for col in cleaning_cols[:3]:  # Limit analysis
            if col in self.data.columns:
                if self.data[col].dtype in ['bool', 'int64']:
                    cleaning_events = self.data[col].sum()
                    total_hours = len(self.data)
                    frequency[col] = {
                        'events': int(cleaning_events) if not pd.isna(cleaning_events) else 0,
                        'frequency_pct': float(cleaning_events / total_hours * 100) if total_hours > 0 else 0
                    }
        
        return frequency
    
    def _calculate_cleaning_effectiveness(self) -> Dict:
        """Calculate cleaning effectiveness metrics."""
        
        effectiveness = {
            'message': 'Cleaning effectiveness analysis requires before/after efficiency data'
        }
        
        # This would require more sophisticated analysis with before/after comparisons
        # For now, return placeholder
        
        return effectiveness
    
    def _generate_analysis_plots(self):
        """Generate analysis plots and save to files."""
        
        try:
            # Efficiency distribution plot
            if 'system_efficiency' in self.data.columns:
                plt.figure(figsize=(10, 6))
                plt.hist(self.data['system_efficiency'], bins=30, alpha=0.7, edgecolor='black')
                plt.title('System Efficiency Distribution')
                plt.xlabel('System Efficiency')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                plt.savefig(figures_dir / 'efficiency_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # Load factor time series (if timestamp available)
            if 'timestamp' in self.data.columns and 'load_factor' in self.data.columns:
                plt.figure(figsize=(12, 6))
                plt.plot(self.data['timestamp'], self.data['load_factor'], alpha=0.7)
                plt.title('Load Factor Over Time')
                plt.xlabel('Time')
                plt.ylabel('Load Factor')
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.savefig(figures_dir / 'load_factor_timeseries.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            self.logger.info("Analysis plots generated successfully")
            
        except Exception as e:
            self.logger.warning(f"Plot generation failed: {e}")
    
    def _save_report_summary(self, report: Dict):
        """Save analysis report summary to file."""
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = output_dir / f"analysis_report_{timestamp}.txt"
            
            with open(report_file, 'w') as f:
                f.write("BOILER DATA ANALYSIS REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated: {report['report_metadata']['generated_at']}\n")
                f.write(f"Dataset Records: {report['report_metadata']['dataset_records']:,}\n\n")
                
                # Dataset summary
                if 'dataset_summary' in report:
                    f.write("DATASET SUMMARY:\n")
                    f.write(f"  Total Records: {report['dataset_summary']['total_records']:,}\n")
                    
                # Efficiency analysis
                if 'efficiency_analysis' in report and 'statistics' in report['efficiency_analysis']:
                    stats = report['efficiency_analysis']['statistics']
                    f.write("\nEFFICIENCY ANALYSIS:\n")
                    f.write(f"  Mean Efficiency: {stats['mean']:.1%}\n")
                    f.write(f"  Efficiency Range: {stats['min']:.1%} - {stats['max']:.1%}\n")
                    f.write(f"  Standard Deviation: {stats['std']:.2%}\n")
                    
                    if 'performance_assessment' in report['efficiency_analysis']:
                        assessment = report['efficiency_analysis']['performance_assessment']
                        f.write(f"  Performance Grade: {assessment['grade']}\n")
                
                # Optimization opportunities
                if 'optimization_opportunities' in report:
                    opps = report['optimization_opportunities'].get('opportunities', [])
                    if opps:
                        f.write(f"\nOPTIMIZATION OPPORTUNITIES:\n")
                        for opp in opps:
                            f.write(f"  - {opp['area']}: {opp['description']}\n")
            
            self.logger.info(f"Analysis report saved: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save analysis report: {e}")


def main():
    """Demonstration of BoilerDataAnalyzer capabilities."""
    
    print("*" * 25)
    print("BOILER DATA ANALYSIS DEMONSTRATION")
    print("*" * 25)
    
    # This would normally load real data
    print("\n>> To use this analyzer:")
    print("   1. Run the AnnualBoilerSimulator to generate data")
    print("   2. Load the CSV file into a pandas DataFrame")
    print("   3. Create BoilerDataAnalyzer instance with the data")
    print("   4. Call generate_comprehensive_report()")
    
    print("\n>> Example usage:")
    print("   # Load annual data")
    print("   df = pd.read_csv('massachusetts_boiler_annual_20241212_143045.csv')")
    print("   ")
    print("   # Create analyzer")
    print("   analyzer = BoilerDataAnalyzer(df)")
    print("   ")
    print("   # Generate comprehensive report")
    print("   report = analyzer.generate_comprehensive_report(save_plots=True)")
    print("   ")
    print("   # Access specific analyses")
    print("   fouling_analysis = report['fouling_analysis']")
    print("   optimization_opportunities = report['optimization_opportunities']")


if __name__ == "__main__":
    main()
