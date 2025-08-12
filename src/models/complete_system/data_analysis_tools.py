#!/usr/bin/env python3
"""
Annual Boiler Data Analysis Tools

This module provides comprehensive analysis tools for the generated annual boiler
operation dataset, including trend analysis, fouling progression, and performance
optimization insights.

Classes:
    BoilerDataAnalyzer: Main analysis class for annual operation data
    FoulingAnalyzer: Specialized fouling progression analysis
    PerformanceOptimizer: Performance optimization recommendations

Dependencies:
    - pandas: Data analysis
    - numpy: Numerical calculations
    - matplotlib: Plotting
    - seaborn: Statistical visualization
    - scipy: Statistical analysis

Author: Enhanced Boiler Modeling System
Version: 6.0 - Annual Data Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")


class BoilerDataAnalyzer:
    """Comprehensive analysis tools for annual boiler operation data."""
    
    def __init__(self, data: pd.DataFrame):
        """Initialize analyzer with annual operation data."""
        self.data = data.copy()
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        self.data = self.data.sort_values('timestamp').reset_index(drop=True)
        
        # Identify fouling columns
        self.fouling_columns = [col for col in self.data.columns if 'fouling' in col and 'avg' in col]
        self.section_names = list(set([col.split('_fouling_')[0] for col in self.fouling_columns]))
        
        # Update to use hours instead of days for cleaning intervals
        self.hours_since_cleaning_columns = [col for col in self.data.columns if 'hours_since_cleaning' in col]
        
        print(f"📊 Boiler Data Analyzer initialized")
        print(f"   Data period: {self.data['timestamp'].min()} to {self.data['timestamp'].max()}")
        print(f"   Total records: {len(self.data):,}")
        print(f"   Sections tracked: {len(self.section_names)}")
        
    def generate_comprehensive_report(self, save_plots: bool = True) -> Dict:
        """Generate comprehensive annual operation report."""
        print(f"\n📈 Generating comprehensive annual analysis report...")
        
        report = {}
        
        # 1. Overall Performance Summary
        report['performance_summary'] = self._analyze_overall_performance()
        
        # 2. Seasonal Analysis
        report['seasonal_analysis'] = self._analyze_seasonal_patterns()
        
        # 3. Fouling Progression Analysis
        report['fouling_analysis'] = self._analyze_fouling_progression()
        
        # 4. Soot Blowing Effectiveness
        report['soot_blowing_analysis'] = self._analyze_soot_blowing_effectiveness()
        
        # 5. Coal Quality Impact
        report['coal_quality_analysis'] = self._analyze_coal_quality_impact()
        
        # 6. Ambient Conditions Impact
        report['ambient_impact_analysis'] = self._analyze_ambient_impact()
        
        # 7. Efficiency Optimization Opportunities
        report['optimization_opportunities'] = self._identify_optimization_opportunities()
        
        # Generate plots if requested
        if save_plots:
            self._generate_analysis_plots()
        
        # Print summary
        self._print_report_summary(report)
        
        return report
    
    def _analyze_overall_performance(self) -> Dict:
        """Analyze overall system performance metrics."""
        performance_metrics = {
            'system_efficiency': self.data['system_efficiency'],
            'final_steam_temp_F': self.data['final_steam_temp_F'],
            'stack_temp_F': self.data['stack_temp_F'],
            'total_nox_lb_hr': self.data['total_nox_lb_hr'],
            'load_factor': self.data['load_factor']
        }
        
        summary = {}
        for metric, values in performance_metrics.items():
            summary[metric] = {
                'mean': values.mean(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'median': values.median(),
                'p5': values.quantile(0.05),
                'p95': values.quantile(0.95)
            }
        
        # Calculate availability and reliability metrics
        total_hours = len(self.data) * 4  # 4-hour intervals
        summary['operational_stats'] = {
            'total_operating_hours': total_hours,
            'average_load_factor': self.data['load_factor'].mean(),
            'load_factor_variability': self.data['load_factor'].std(),
            'high_load_hours': (self.data['load_factor'] > 0.85).sum() * 4,
            'low_load_hours': (self.data['load_factor'] < 0.60).sum() * 4,
            'solution_convergence_rate': self.data['solution_converged'].mean() if 'solution_converged' in self.data.columns else 1.0
        }
        
        return summary
    
    def _analyze_seasonal_patterns(self) -> Dict:
        """Analyze seasonal performance patterns."""
        seasonal_data = self.data.groupby('season').agg({
            'system_efficiency': ['mean', 'std'],
            'stack_temp_F': ['mean', 'std'],
            'total_nox_lb_hr': ['mean', 'std'],
            'ambient_temp_F': ['mean', 'min', 'max'],
            'load_factor': ['mean', 'std'],
            'coal_rate_lb_hr': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        seasonal_data.columns = [f"{col[0]}_{col[1]}" for col in seasonal_data.columns]
        
        # Calculate seasonal efficiency variations
        efficiency_by_season = self.data.groupby('season')['system_efficiency'].mean()
        max_seasonal_efficiency = efficiency_by_season.max()
        min_seasonal_efficiency = efficiency_by_season.min()
        seasonal_efficiency_range = max_seasonal_efficiency - min_seasonal_efficiency
        
        return {
            'seasonal_statistics': seasonal_data.to_dict(),
            'efficiency_range': seasonal_efficiency_range,
            'best_season': efficiency_by_season.idxmax(),
            'worst_season': efficiency_by_season.idxmin(),
            'winter_performance': efficiency_by_season.get('winter', 0),
            'summer_performance': efficiency_by_season.get('summer', 0)
        }
    
    def _analyze_fouling_progression(self) -> Dict:
        """Analyze fouling buildup and cleaning cycles."""
        fouling_analysis = {}
        
        for section in self.section_names:
            fouling_col = f"{section}_fouling_gas_avg"
            cleaning_col = f"{section}_hours_since_cleaning"
            
            if fouling_col in self.data.columns and cleaning_col in self.data.columns:
                # Fouling statistics
                fouling_data = self.data[fouling_col]
                cleaning_data = self.data[cleaning_col]
                
                # Calculate fouling accumulation rate
                fouling_rate = self._calculate_fouling_rate(fouling_data, cleaning_data)
                
                # Cleaning effectiveness
                cleaning_effectiveness = self._calculate_cleaning_effectiveness(
                    fouling_data, cleaning_data
                )
                
                fouling_analysis[section] = {
                    'avg_fouling': fouling_data.mean(),
                    'max_fouling': fouling_data.max(),
                    'min_fouling': fouling_data.min(),
                    'fouling_variability': fouling_data.std(),
                    'avg_hours_between_cleaning': cleaning_data.mean(),
                    'max_hours_between_cleaning': cleaning_data.max(),
                    'fouling_accumulation_rate': fouling_rate,
                    'cleaning_effectiveness': cleaning_effectiveness,
                    'cleaning_frequency_per_day': 24 / cleaning_data.mean() if cleaning_data.mean() > 0 else 0
                }
        
        return fouling_analysis
    
    def _calculate_fouling_rate(self, fouling_data: pd.Series, cleaning_data: pd.Series) -> float:
        """Calculate average fouling accumulation rate per hour."""
        # Find cleaning events (when hours_since_cleaning resets to 0 or low value)
        cleaning_events = cleaning_data.diff() < -2  # Significant decrease indicates cleaning
        
        if cleaning_events.sum() < 2:
            return 0.0
        
        # Calculate fouling rate between cleaning events
        rates = []
        for i in range(len(cleaning_events)):
            if cleaning_events.iloc[i] and i > 5:  # Found cleaning event with enough history
                # Look back to find previous cleaning or start
                start_idx = max(0, i - 20)  # Look back up to 20 records (80 hours)
                prev_cleaning_mask = cleaning_events.iloc[start_idx:i]
                prev_cleaning = prev_cleaning_mask[::-1].idxmax() if prev_cleaning_mask.any() else start_idx
                
                # Calculate rate over this period
                fouling_change = fouling_data.iloc[i-1] - fouling_data.iloc[prev_cleaning]
                time_change = cleaning_data.iloc[i-1]  # Hours since last cleaning
                
                if time_change > 0:
                    rates.append(fouling_change / time_change)
        
        return np.mean(rates) if rates else 0.0
    
    def _calculate_cleaning_effectiveness(self, fouling_data: pd.Series, cleaning_data: pd.Series) -> float:
        """Calculate average cleaning effectiveness."""
        # Find cleaning events (hours since cleaning resets)
        cleaning_events = cleaning_data.diff() < -2
        
        effectiveness_values = []
        for i in range(1, len(cleaning_events)):
            if cleaning_events.iloc[i]:
                # Fouling before and after cleaning
                before_cleaning = fouling_data.iloc[i-1]
                after_cleaning = fouling_data.iloc[i]
                
                if before_cleaning > 0:
                    effectiveness = (before_cleaning - after_cleaning) / before_cleaning
                    effectiveness_values.append(max(0, min(1, effectiveness)))
        
        return np.mean(effectiveness_values) if effectiveness_values else 0.0
    
    def _analyze_soot_blowing_effectiveness(self) -> Dict:
        """Analyze soot blowing impact on performance."""
        # Compare performance before and after soot blowing events
        soot_blowing_events = self.data['soot_blowing_active'] == True
        
        if soot_blowing_events.sum() < 10:
            return {'insufficient_data': True}
        
        # Performance metrics before and after soot blowing
        before_indices = []
        after_indices = []
        
        for i in range(1, len(self.data)-1):
            if soot_blowing_events.iloc[i]:
                if i > 0:
                    before_indices.append(i-1)
                if i < len(self.data)-1:
                    after_indices.append(i+1)
        
        if len(before_indices) < 5 or len(after_indices) < 5:
            return {'insufficient_data': True}
        
        before_data = self.data.iloc[before_indices]
        after_data = self.data.iloc[after_indices]
        
        effectiveness_analysis = {}
        metrics = ['system_efficiency', 'stack_temp_F', 'final_steam_temp_F']
        
        for metric in metrics:
            before_mean = before_data[metric].mean()
            after_mean = after_data[metric].mean()
            improvement = after_mean - before_mean
            
            effectiveness_analysis[metric] = {
                'before_cleaning': before_mean,
                'after_cleaning': after_mean,
                'improvement': improvement,
                'percent_improvement': (improvement / before_mean * 100) if before_mean != 0 else 0
            }
        
        # Calculate overall cleaning benefit
        efficiency_improvement = effectiveness_analysis['system_efficiency']['improvement']
        effectiveness_analysis['overall_benefit_score'] = efficiency_improvement * 1000  # Scale for readability
        
        return effectiveness_analysis
    
    def _analyze_coal_quality_impact(self) -> Dict:
        """Analyze impact of different coal qualities on performance."""
        coal_quality_analysis = {}
        
        for quality in self.data['coal_quality'].unique():
            quality_data = self.data[self.data['coal_quality'] == quality]
            
            coal_quality_analysis[quality] = {
                'records_count': len(quality_data),
                'percentage_of_year': len(quality_data) / len(self.data) * 100,
                'avg_efficiency': quality_data['system_efficiency'].mean(),
                'avg_nox_emissions': quality_data['total_nox_lb_hr'].mean(),
                'avg_stack_temp': quality_data['stack_temp_F'].mean(),
                'avg_load_factor': quality_data['load_factor'].mean(),
                'efficiency_std': quality_data['system_efficiency'].std()
            }
        
        # Rank coal qualities by performance
        qualities = list(coal_quality_analysis.keys())
        efficiency_ranking = sorted(qualities, 
                                  key=lambda x: coal_quality_analysis[x]['avg_efficiency'], 
                                  reverse=True)
        
        return {
            'by_quality': coal_quality_analysis,
            'efficiency_ranking': efficiency_ranking,
            'best_quality': efficiency_ranking[0] if efficiency_ranking else None,
            'worst_quality': efficiency_ranking[-1] if efficiency_ranking else None
        }
    
    def _analyze_ambient_impact(self) -> Dict:
        """Analyze impact of ambient conditions on performance."""
        # Bin ambient temperature for analysis
        self.data['temp_bin'] = pd.cut(self.data['ambient_temp_F'], 
                                      bins=5, labels=['Very Cold', 'Cold', 'Moderate', 'Warm', 'Hot'])
        
        temp_impact = self.data.groupby('temp_bin').agg({
            'system_efficiency': ['mean', 'std'],
            'stack_temp_F': ['mean', 'std'],
            'air_flow_scfh': ['mean', 'std'],
            'load_factor': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        temp_impact.columns = [f"{col[0]}_{col[1]}" for col in temp_impact.columns]
        
        # Correlation analysis
        correlations = {
            'efficiency_vs_temp': self.data['system_efficiency'].corr(self.data['ambient_temp_F']),
            'efficiency_vs_humidity': self.data['system_efficiency'].corr(self.data['ambient_humidity_pct']),
            'nox_vs_temp': self.data['total_nox_lb_hr'].corr(self.data['ambient_temp_F']),
            'stack_temp_vs_ambient': self.data['stack_temp_F'].corr(self.data['ambient_temp_F'])
        }
        
        return {
            'temperature_impact': temp_impact.to_dict(),
            'correlations': correlations,
            'optimal_temp_range': self._find_optimal_temperature_range()
        }
    
    def _find_optimal_temperature_range(self) -> Dict:
        """Find ambient temperature range with best performance."""
        # Group by temperature ranges and find best efficiency
        temp_ranges = [(i, i+10) for i in range(int(self.data['ambient_temp_F'].min()), 
                                               int(self.data['ambient_temp_F'].max()), 10)]
        
        best_efficiency = 0
        optimal_range = None
        
        for temp_min, temp_max in temp_ranges:
            mask = (self.data['ambient_temp_F'] >= temp_min) & (self.data['ambient_temp_F'] < temp_max)
            range_data = self.data[mask]
            
            if len(range_data) > 10:  # Minimum data points for reliability
                avg_efficiency = range_data['system_efficiency'].mean()
                if avg_efficiency > best_efficiency:
                    best_efficiency = avg_efficiency
                    optimal_range = (temp_min, temp_max)
        
        return {
            'optimal_temp_range_F': optimal_range,
            'optimal_efficiency': best_efficiency,
            'data_points_in_range': len(self.data[
                (self.data['ambient_temp_F'] >= optimal_range[0]) & 
                (self.data['ambient_temp_F'] < optimal_range[1])
            ]) if optimal_range else 0
        }
    
    def _identify_optimization_opportunities(self) -> Dict:
        """Identify key optimization opportunities."""
        opportunities = {}
        
        # 1. Soot blowing optimization
        max_fouling_section = None
        max_fouling_value = 0
        
        for section in self.section_names:
            fouling_col = f"{section}_fouling_gas_avg"
            if fouling_col in self.data.columns:
                avg_fouling = self.data[fouling_col].mean()
                if avg_fouling > max_fouling_value:
                    max_fouling_value = avg_fouling
                    max_fouling_section = section
        
        opportunities['soot_blowing'] = {
            'worst_fouling_section': max_fouling_section,
            'worst_fouling_value': max_fouling_value,
            'potential_improvement': 'Increase cleaning frequency for this section'
        }
        
        # 2. Load optimization
        efficiency_by_load = self.data.groupby(pd.cut(self.data['load_factor'], bins=5))['system_efficiency'].mean()
        optimal_load_bin = efficiency_by_load.idxmax()
        
        opportunities['load_optimization'] = {
            'optimal_load_range': str(optimal_load_bin),
            'optimal_efficiency': efficiency_by_load.max(),
            'current_avg_load': self.data['load_factor'].mean(),
            'recommendation': 'Operate more frequently in optimal load range'
        }
        
        # 3. Coal quality optimization
        coal_analysis = self._analyze_coal_quality_impact()
        best_coal = coal_analysis['best_quality']
        worst_coal = coal_analysis['worst_quality']
        
        if best_coal and worst_coal:
            efficiency_gain = (coal_analysis['by_quality'][best_coal]['avg_efficiency'] - 
                             coal_analysis['by_quality'][worst_coal]['avg_efficiency'])
            
            opportunities['coal_quality'] = {
                'best_coal_quality': best_coal,
                'worst_coal_quality': worst_coal,
                'potential_efficiency_gain': efficiency_gain,
                'recommendation': f'Increase usage of {best_coal}, reduce {worst_coal}'
            }
        
        # 4. Seasonal adjustments
        seasonal_analysis = self._analyze_seasonal_patterns()
        
        opportunities['seasonal_optimization'] = {
            'best_season': seasonal_analysis['best_season'],
            'worst_season': seasonal_analysis['worst_season'],
            'seasonal_range': seasonal_analysis['efficiency_range'],
            'recommendation': 'Adjust maintenance schedule based on seasonal patterns'
        }
        
        return opportunities
    
    def _generate_analysis_plots(self):
        """Generate comprehensive analysis plots."""
        print(f"📊 Generating analysis plots...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Overall performance trends
        plt.subplot(4, 3, 1)
        self.data.set_index('timestamp')['system_efficiency'].resample('D').mean().plot()
        plt.title('Daily System Efficiency Trend')
        plt.ylabel('Efficiency')
        plt.xticks(rotation=45)
        
        # 2. Load factor distribution
        plt.subplot(4, 3, 2)
        plt.hist(self.data['load_factor'], bins=30, alpha=0.7, edgecolor='black')
        plt.title('Load Factor Distribution')
        plt.xlabel('Load Factor')
        plt.ylabel('Frequency')
        
        # 3. Seasonal efficiency comparison
        plt.subplot(4, 3, 3)
        seasonal_eff = self.data.groupby('season')['system_efficiency'].mean()
        seasonal_eff.plot(kind='bar')
        plt.title('Average Efficiency by Season')
        plt.ylabel('Efficiency')
        plt.xticks(rotation=45)
        
        # 4. Coal quality impact
        plt.subplot(4, 3, 4)
        coal_eff = self.data.groupby('coal_quality')['system_efficiency'].mean()
        coal_eff.plot(kind='bar', color='orange')
        plt.title('Efficiency by Coal Quality')
        plt.ylabel('Efficiency')
        plt.xticks(rotation=45)
        
        # 5. Ambient temperature vs efficiency
        plt.subplot(4, 3, 5)
        plt.scatter(self.data['ambient_temp_F'], self.data['system_efficiency'], alpha=0.3)
        plt.title('Efficiency vs Ambient Temperature')
        plt.xlabel('Ambient Temperature (°F)')
        plt.ylabel('System Efficiency')
        
        # 6. NOx emissions over time
        plt.subplot(4, 3, 6)
        self.data.set_index('timestamp')['total_nox_lb_hr'].resample('W').mean().plot(color='red')
        plt.title('Weekly NOx Emissions Trend')
        plt.ylabel('NOx (lb/hr)')
        plt.xticks(rotation=45)
        
        # 7. Fouling progression for worst section
        plt.subplot(4, 3, 7)
        if self.section_names:
            worst_section = self.section_names[0]  # Use first section as example
            fouling_col = f"{worst_section}_fouling_gas_avg"
            if fouling_col in self.data.columns:
                self.data.set_index('timestamp')[fouling_col].resample('D').mean().plot(color='brown')
                plt.title(f'Fouling Progression - {worst_section}')
                plt.ylabel('Fouling Factor')
                plt.xticks(rotation=45)
        
        # 8. Stack temperature vs ambient
        plt.subplot(4, 3, 8)
        plt.scatter(self.data['ambient_temp_F'], self.data['stack_temp_F'], alpha=0.3, color='green')
        plt.title('Stack vs Ambient Temperature')
        plt.xlabel('Ambient Temperature (°F)')
        plt.ylabel('Stack Temperature (°F)')
        
        # 9. Efficiency distribution by season
        plt.subplot(4, 3, 9)
        for season in self.data['season'].unique():
            season_data = self.data[self.data['season'] == season]['system_efficiency']
            plt.hist(season_data, alpha=0.5, label=season, bins=20)
        plt.title('Efficiency Distribution by Season')
        plt.xlabel('System Efficiency')
        plt.ylabel('Frequency')
        plt.legend()
        
        # 10. Load factor vs efficiency
        plt.subplot(4, 3, 10)
        plt.scatter(self.data['load_factor'], self.data['system_efficiency'], alpha=0.3, color='purple')
        plt.title('Efficiency vs Load Factor')
        plt.xlabel('Load Factor')
        plt.ylabel('System Efficiency')
        
        # 11. Monthly performance summary
        plt.subplot(4, 3, 11)
        monthly_eff = self.data.groupby('month')['system_efficiency'].mean()
        monthly_eff.plot(kind='line', marker='o', color='navy')
        plt.title('Monthly Average Efficiency')
        plt.xlabel('Month')
        plt.ylabel('Efficiency')
        
        # 12. Soot blowing effectiveness
        plt.subplot(4, 3, 12)
        soot_blowing_data = self.data.groupby('soot_blowing_active')['system_efficiency'].mean()
        soot_blowing_data.plot(kind='bar', color=['lightblue', 'darkblue'])
        plt.title('Efficiency: Normal vs Soot Blowing')
        plt.ylabel('Average Efficiency')
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig('annual_boiler_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Analysis plots saved as 'annual_boiler_analysis.png'")
    
    def _print_report_summary(self, report: Dict):
        """Print comprehensive report summary."""
        print(f"\n" + "="*80)
        print("ANNUAL BOILER OPERATION ANALYSIS REPORT")
        print("="*80)
        
        # Performance Summary
        perf = report['performance_summary']
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   Average Efficiency: {perf['system_efficiency']['mean']:.1%} ± {perf['system_efficiency']['std']:.1%}")
        print(f"   Average Load Factor: {perf['load_factor']['mean']:.1%}")
        print(f"   Operating Hours: {perf['operational_stats']['total_operating_hours']:,}")
        print(f"   High Load Hours (>85%): {perf['operational_stats']['high_load_hours']:,}")
        
        # Seasonal Analysis
        seasonal = report['seasonal_analysis']
        print(f"\n🌡️ SEASONAL PERFORMANCE:")
        print(f"   Best Season: {seasonal['best_season']} (Efficiency: {seasonal.get('winter_performance', 0):.1%})")
        print(f"   Worst Season: {seasonal['worst_season']}")
        print(f"   Seasonal Efficiency Range: {seasonal['efficiency_range']:.2%}")
        
        # Coal Quality Impact
        coal = report['coal_quality_analysis']
        print(f"\n⚡ COAL QUALITY IMPACT:")
        print(f"   Best Coal Quality: {coal['best_quality']}")
        print(f"   Worst Coal Quality: {coal['worst_quality']}")
        if coal['best_quality'] and coal['worst_quality']:
            best_eff = coal['by_quality'][coal['best_quality']]['avg_efficiency']
            worst_eff = coal['by_quality'][coal['worst_quality']]['avg_efficiency']
            print(f"   Quality Impact: {(best_eff - worst_eff):.2%} efficiency difference")
        
        # Optimization Opportunities
        opt = report['optimization_opportunities']
        print(f"\n🎯 KEY OPTIMIZATION OPPORTUNITIES:")
        
        if 'soot_blowing' in opt:
            print(f"   1. Soot Blowing: Focus on {opt['soot_blowing']['worst_fouling_section']}")
        
        if 'coal_quality' in opt:
            print(f"   2. Coal Quality: Increase {opt['coal_quality']['best_coal_quality']} usage")
            print(f"      Potential gain: {opt['coal_quality']['potential_efficiency_gain']:.2%}")
        
        if 'load_optimization' in opt:
            print(f"   3. Load Optimization: Target {opt['load_optimization']['optimal_load_range']}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   • Optimize soot blowing frequency for high-fouling sections")
        print(f"   • Negotiate coal contracts to increase high-quality fuel percentage")
        print(f"   • Adjust operating loads to maximize efficiency")
        print(f"   • Plan major maintenance during lowest-demand seasons")


def main():
    """Demonstrate the analysis tools with sample data."""
    print("📊" * 25)
    print("BOILER DATA ANALYSIS DEMONSTRATION")
    print("📊" * 25)
    
    # This would normally load real data
    print("\n📁 To use this analyzer:")
    print("   1. Run the AnnualBoilerSimulator to generate data")
    print("   2. Load the CSV file into a pandas DataFrame")
    print("   3. Create BoilerDataAnalyzer instance with the data")
    print("   4. Call generate_comprehensive_report()")
    
    print("\n💻 Example usage:")
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