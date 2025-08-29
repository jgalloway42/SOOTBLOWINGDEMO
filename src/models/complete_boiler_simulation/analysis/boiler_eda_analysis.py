"""
Comprehensive Boiler EDA Analysis Functions

This module contains reusable analysis functions extracted from the boiler fouling 
EDA notebooks for consistent and comprehensive analysis across different datasets.

Functions:
- Operational performance analysis
- Soot blowing effectiveness analysis  
- Fouling pattern analysis
- Cleaning schedule optimization
- Coal quality impact analysis
- Seasonal pattern analysis
- Feature correlation analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

def analyze_operational_performance(data, config):
    """Analyze key operational performance metrics"""
    
    print("🔧 OPERATIONAL PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    # System efficiency analysis
    if 'system_efficiency' in data.columns:
        efficiency = data['system_efficiency']
        print(f"\n⚡ System Efficiency:")
        print(f"   Range: {efficiency.min():.1%} to {efficiency.max():.1%}")
        print(f"   Mean: {efficiency.mean():.1%}")
        print(f"   Std Dev: {efficiency.std():.3f}")
        print(f"   Coefficient of Variation: {efficiency.std()/efficiency.mean():.1%}")
        
        # Efficiency over time
        time_blocks = 12
        block_size = len(data) // time_blocks
        initial_eff = data['system_efficiency'].iloc[:block_size].mean()
        final_eff = data['system_efficiency'].iloc[-block_size:].mean()
        degradation = final_eff - initial_eff
        
        print(f"   Annual degradation: {degradation:+.3f} ({degradation/initial_eff:+.1%})")
    
    # Load factor analysis
    if 'load_factor' in data.columns:
        load_factor = data['load_factor']
        print(f"\n📊 Load Factor:")
        print(f"   Range: {load_factor.min():.1%} to {load_factor.max():.1%}")
        print(f"   Mean: {load_factor.mean():.1%}")
        print(f"   Within 60-105%: {((load_factor >= 0.6) & (load_factor <= 1.05)).mean():.1%}")
    
    # Stack temperature analysis
    if 'stack_temp_F' in data.columns:
        stack_temp = data['stack_temp_F']
        print(f"\n🌡️ Stack Temperature:")
        print(f"   Range: {stack_temp.min():.1f}°F to {stack_temp.max():.1f}°F")
        print(f"   Mean: {stack_temp.mean():.1f}°F")
        print(f"   Std Dev: {stack_temp.std():.1f}°F")
        
        # Temperature increase over time
        initial_temp = data['stack_temp_F'].iloc[:1000].mean()
        final_temp = data['stack_temp_F'].iloc[-1000:].mean()
        temp_increase = final_temp - initial_temp
        print(f"   Annual increase: {temp_increase:+.1f}°F")


def analyze_soot_blowing_patterns(data):
    """Analyze soot blowing activity and effectiveness"""
    
    print(f"\n🧹 SOOT BLOWING ANALYSIS")
    print("=" * 50)
    
    # Find soot blowing columns
    soot_cols = [col for col in data.columns if 'soot' in col.lower() and 'blowing' in col.lower()]
    
    if not soot_cols:
        print("   ❌ No soot blowing columns found")
        return {}
    
    # Overall soot blowing activity
    if 'soot_blowing_active' in data.columns:
        soot_events = data['soot_blowing_active'].sum()
        soot_frequency = data['soot_blowing_active'].mean()
        
        print(f"\n🔧 Overall Soot Blowing Activity:")
        print(f"   Total events: {soot_events:,}")
        print(f"   Frequency: {soot_frequency:.1%}")
        print(f"   Hours per cleaning: {1/soot_frequency:.1f} hours" if soot_frequency > 0 else "   No cleaning events")
    
    # Section-specific soot blowing
    section_cols = [col for col in soot_cols if col != 'soot_blowing_active']
    if section_cols:
        print(f"\n🏭 Section-Specific Soot Blowing:")
        for col in section_cols[:10]:  # Show first 10 sections
            events = data[col].sum()
            frequency = data[col].mean()
            if events > 0:
                section_name = col.replace('_soot_blowing_active', '').replace('_', ' ').title()
                print(f"   {section_name}: {events} events ({frequency:.1%} frequency)")
    
    # Efficiency during soot blowing
    if 'soot_blowing_active' in data.columns and 'system_efficiency' in data.columns:
        soot_impact = data.groupby('soot_blowing_active')['system_efficiency'].agg(['mean', 'std', 'count'])
        
        print(f"\n⚡ Efficiency During Soot Blowing:")
        for active in [False, True]:
            if active in soot_impact.index:
                status = "During Soot Blowing" if active else "Normal Operation"
                mean_eff = soot_impact.loc[active, 'mean']
                std_eff = soot_impact.loc[active, 'std']
                count = soot_impact.loc[active, 'count']
                print(f"   • {status}: {mean_eff:.2%} efficiency (±{std_eff:.3f}, n={count})")
        
        # Statistical comparison
        if True in soot_impact.index and False in soot_impact.index:
            normal_eff = data[~data['soot_blowing_active']]['system_efficiency']
            soot_eff = data[data['soot_blowing_active']]['system_efficiency']
            
            if len(soot_eff) > 0:
                diff = soot_eff.mean() - normal_eff.mean()
                print(f"   • Efficiency difference: {diff:+.3f} ({diff/normal_eff.mean():+.1%})")
    
    # Effectiveness analysis
    if 'avg_cleaning_effectiveness' in data.columns:
        effectiveness = data['avg_cleaning_effectiveness']
        active_effectiveness = effectiveness[effectiveness > 0]
        if len(active_effectiveness) > 0:
            print(f"\n✨ Cleaning Effectiveness:")
            print(f"   Range: {active_effectiveness.min():.1f}% to {active_effectiveness.max():.1f}%")
            print(f"   Mean: {active_effectiveness.mean():.1f}%")
            print(f"   Target (80-95%): {((active_effectiveness >= 80) & (active_effectiveness <= 95)).mean():.1%}")
    
    return {
        'soot_events': soot_events if 'soot_blowing_active' in data.columns else 0,
        'soot_frequency': soot_frequency if 'soot_blowing_active' in data.columns else 0,
        'section_counts': len(section_cols)
    }


def analyze_fouling_patterns(data, config):
    """Analyze fouling patterns across boiler sections"""
    
    print(f"\n🔧 FOULING DYNAMICS ANALYSIS")
    print("=" * 50)
    
    # Find fouling columns
    fouling_cols = [col for col in data.columns if 'fouling' in col.lower()]
    
    if not fouling_cols:
        print("   ❌ No fouling columns found")
        return {}
    
    print(f"Found {len(fouling_cols)} fouling-related columns")
    
    # Fouling progression analysis
    print(f"\n📈 FOULING ACCUMULATION PATTERNS:")
    
    # Sample key time points
    sample_hours = [0, 1000, 2000, 4000, 6000, 8000]
    available_hours = [h for h in sample_hours if h < len(data)]
    
    print(f"\nFouling progression at key time points:")
    print(f"{'Hour':<8} {'Main Section':<12} {'Section 2':<12} {'Section 3':<12}")
    print(f"{'-'*8} {'-'*12} {'-'*12} {'-'*12}")
    
    for hour in available_hours:
        val1 = data[fouling_cols[0]].iloc[hour] if len(fouling_cols) > 0 else 0
        val2 = data[fouling_cols[1]].iloc[hour] if len(fouling_cols) > 1 else 0
        val3 = data[fouling_cols[2]].iloc[hour] if len(fouling_cols) > 2 else 0
        print(f"{hour:<8} {val1:<12.4f} {val2:<12.4f} {val3:<12.4f}")
    
    # Statistical analysis of fouling factors
    fouling_stats = {}
    for col in fouling_cols[:5]:  # First 5 sections
        values = data[col]
        fouling_stats[col] = {
            'range': (values.min(), values.max()),
            'mean': values.mean(),
            'std': values.std(),
            'growth': values.iloc[-100:].mean() - values.iloc[:100].mean()
        }
        
        print(f"\n{col}:")
        print(f"  Range: {values.min():.4f} to {values.max():.4f}")
        print(f"  Mean: {values.mean():.4f}")
        print(f"  Growth over time: {fouling_stats[col]['growth']:+.4f}")
    
    return fouling_stats


def analyze_cleaning_effectiveness(data, config):
    """Analyze cleaning effectiveness and fouling reduction"""
    
    print(f"\n🧹 CLEANING EFFECTIVENESS VALIDATION")
    print("=" * 50)
    
    fouling_cols = [col for col in data.columns if 'fouling' in col.lower()]
    
    if 'soot_blowing_active' in data.columns and fouling_cols:
        cleaning_effectiveness = {}
        
        # Analyze fouling before and after cleaning events
        for fouling_col in fouling_cols[:3]:  # First 3 sections for analysis
            section_name = fouling_col.replace('_fouling_factor', '').replace('_', ' ').title()
            section_cleaning_col = f"{fouling_col.replace('_fouling_factor', '')}_soot_blowing_active"
            
            if section_cleaning_col in data.columns:
                # Find cleaning events
                cleaning_events = data[data[section_cleaning_col] == True]
                
                if len(cleaning_events) > 10:  # Need sufficient events
                    effectiveness_values = []
                    
                    for idx in cleaning_events.index[::50]:  # Sample every 50th event
                        if idx > 24 and idx < len(data) - 24:  # Need before/after data
                            before_fouling = data[fouling_col].iloc[idx-24:idx].mean()
                            after_fouling = data[fouling_col].iloc[idx+1:idx+25].mean()
                            
                            if before_fouling > after_fouling:
                                reduction = (before_fouling - after_fouling) / before_fouling * 100
                                effectiveness_values.append(reduction)
                    
                    if effectiveness_values:
                        effectiveness = np.mean(effectiveness_values)
                        cleaning_effectiveness[section_name] = effectiveness
                        
                        if effectiveness > 80:
                            status = "✅ Excellent"
                        elif effectiveness > 60:
                            status = "⚠️ Good"
                        else:
                            status = "❌ Poor"
                        
                        print(f"   {section_name}: {effectiveness:.1f}% fouling reduction {status}")
        
        if cleaning_effectiveness:
            avg_effectiveness = np.mean(list(cleaning_effectiveness.values()))
            print(f"\n📊 Overall cleaning effectiveness: {avg_effectiveness:.1f}%")
            
            if avg_effectiveness > 80:
                print("   ✅ Excellent cleaning performance - maintain current schedule")
            elif avg_effectiveness > 60:
                print("   ⚠️ Good cleaning performance - minor optimizations possible")
            else:
                print("   ❌ Poor cleaning performance - schedule optimization needed")
        
        return cleaning_effectiveness
    else:
        print("   ❌ Insufficient soot blowing data for effectiveness validation")
        return {}


def analyze_cleaning_schedule_optimization(data, config):
    """Analyze current cleaning schedules and identify optimization opportunities"""
    
    print(f"\n📅 CLEANING SCHEDULE OPTIMIZATION ANALYSIS")
    print("=" * 50)
    
    fouling_cols = [col for col in data.columns if 'fouling' in col.lower()]
    
    if not fouling_cols:
        print("   ❌ No fouling data available for schedule analysis")
        return {}
    
    optimization_results = {}
    
    for fouling_col in fouling_cols[:3]:  # Analyze first 3 sections
        section_name = fouling_col.replace('_fouling_factor', '').replace('_', ' ').title()
        
        # Calculate cleaning effectiveness (fouling reduction after cleaning)
        fouling_values = data[fouling_col]
        rolling_min = fouling_values.rolling(window=48, min_periods=1).min()  # 48h window
        rolling_max = fouling_values.rolling(window=48, min_periods=1).max()  # 48h window
        
        # Potential cleaning opportunities (high fouling periods)
        high_fouling_threshold = fouling_values.quantile(0.8)
        high_fouling_periods = fouling_values > high_fouling_threshold
        
        # Current cleaning frequency analysis
        section_cleaning_col = f"{fouling_col.replace('_fouling_factor', '')}_soot_blowing_active"
        
        if section_cleaning_col in data.columns:
            cleaning_events = data[section_cleaning_col].sum()
            cleaning_frequency = data[section_cleaning_col].mean()
            
            # Calculate time between cleanings
            cleaning_indices = data[data[section_cleaning_col] == True].index
            if len(cleaning_indices) > 1:
                time_between_cleanings = np.diff(cleaning_indices).mean()
                
                optimization_results[section_name] = {
                    'current_frequency': cleaning_frequency,
                    'cleaning_events': cleaning_events,
                    'avg_time_between': time_between_cleanings,
                    'high_fouling_periods': high_fouling_periods.sum(),
                    'optimization_potential': 'Medium' if time_between_cleanings > 168 else 'Low'  # > 7 days
                }
                
                print(f"\n🔧 {section_name}:")
                print(f"   Current cleaning frequency: {cleaning_frequency:.1%}")
                print(f"   Average time between cleanings: {time_between_cleanings:.1f} hours")
                print(f"   High fouling periods: {high_fouling_periods.sum()} hours")
                
                # Optimization recommendations
                if time_between_cleanings > 168:  # More than 7 days
                    print(f"   💡 Recommendation: Consider increasing cleaning frequency")
                elif time_between_cleanings < 48:  # Less than 2 days
                    print(f"   💡 Recommendation: Current frequency may be excessive")
                else:
                    print(f"   ✅ Recommendation: Current schedule appears optimal")
    
    return optimization_results


def analyze_coal_quality_impact(data):
    """Analyze impact of coal quality on performance and fouling"""
    
    print(f"\n🏭 COAL QUALITY IMPACT ANALYSIS")
    print("=" * 50)
    
    if 'coal_quality' not in data.columns:
        print("   ❌ No coal quality data available")
        return {}
    
    coal_analysis = {}
    coal_types = data['coal_quality'].unique()
    
    print(f"Found {len(coal_types)} coal types: {', '.join(coal_types)}")
    
    for coal_type in coal_types:
        mask = data['coal_quality'] == coal_type
        subset = data[mask]
        
        if len(subset) > 100:  # Need sufficient data
            # Efficiency impact
            if 'system_efficiency' in data.columns:
                avg_efficiency = subset['system_efficiency'].mean()
                
                # Fouling impact
                fouling_cols = [col for col in data.columns if 'fouling' in col.lower()]
                if fouling_cols:
                    avg_fouling = subset[fouling_cols[0]].mean()
                    
                    coal_analysis[coal_type] = {
                        'efficiency': avg_efficiency,
                        'fouling_rate': avg_fouling,
                        'records': len(subset)
                    }
                    
                    print(f"\n🔥 {coal_type.title()}:")
                    print(f"   Average efficiency: {avg_efficiency:.1%}")
                    print(f"   Average fouling factor: {avg_fouling:.3f}")
                    print(f"   Data records: {len(subset):,}")
    
    return coal_analysis


def analyze_seasonal_patterns(data):
    """Analyze seasonal and temporal patterns"""
    
    print(f"\n🌍 SEASONAL PATTERN ANALYSIS")
    print("=" * 50)
    
    if 'timestamp' not in data.columns:
        print("   ❌ No timestamp data available")
        return {}
    
    # Ensure timestamp is datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['month'] = data['timestamp'].dt.month
    data['season'] = data['timestamp'].dt.month.map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    seasonal_analysis = {}
    
    # Seasonal efficiency patterns
    if 'system_efficiency' in data.columns:
        seasonal_eff = data.groupby('season')['system_efficiency'].agg(['mean', 'std', 'count'])
        
        print(f"\n⚡ Seasonal Efficiency Patterns:")
        for season in ['Spring', 'Summer', 'Fall', 'Winter']:
            if season in seasonal_eff.index:
                mean_eff = seasonal_eff.loc[season, 'mean']
                std_eff = seasonal_eff.loc[season, 'std']
                count = seasonal_eff.loc[season, 'count']
                print(f"   {season}: {mean_eff:.1%} ± {std_eff:.3f} (n={count})")
        
        seasonal_analysis['efficiency'] = seasonal_eff.to_dict('index')
    
    # Monthly load patterns
    if 'load_factor' in data.columns:
        monthly_load = data.groupby('month')['load_factor'].mean()
        
        print(f"\n📊 Monthly Load Factor Patterns:")
        for month in range(1, 13):
            if month in monthly_load.index:
                load = monthly_load[month]
                month_name = pd.Timestamp(f'2024-{month:02d}-01').strftime('%B')
                print(f"   {month_name}: {load:.1%}")
        
        seasonal_analysis['load_patterns'] = monthly_load.to_dict()
    
    return seasonal_analysis


def create_comprehensive_visualizations(data, config):
    """Create comprehensive visualization dashboard"""
    
    print(f"\n📈 GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("=" * 50)
    
    # Create dashboard with multiple subplots
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Comprehensive Boiler Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # Plot 1: Efficiency over time
    ax1 = axes[0, 0]
    if 'system_efficiency' in data.columns:
        time_hours = range(len(data))
        ax1.plot(time_hours[::100], data['system_efficiency'].iloc[::100], alpha=0.7, linewidth=1)
        ax1.set_title('System Efficiency Over Time')
        ax1.set_xlabel('Hours')
        ax1.set_ylabel('Efficiency')
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Fouling progression
    ax2 = axes[0, 1]
    fouling_cols = [col for col in data.columns if 'fouling' in col.lower()]
    if fouling_cols:
        ax2.plot(time_hours[::100], data[fouling_cols[0]].iloc[::100], 
                color='orange', alpha=0.7, linewidth=1)
        ax2.set_title('Fouling Accumulation')
        ax2.set_xlabel('Hours')
        ax2.set_ylabel('Fouling Factor')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Efficiency vs Fouling scatter
    ax3 = axes[0, 2]
    if fouling_cols and 'system_efficiency' in data.columns:
        ax3.scatter(data[fouling_cols[0]].iloc[::50], data['system_efficiency'].iloc[::50], 
                   alpha=0.5, s=2)
        ax3.set_title('Efficiency vs Fouling')
        ax3.set_xlabel('Fouling Factor')
        ax3.set_ylabel('System Efficiency')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Load factor distribution
    ax4 = axes[1, 0]
    if 'load_factor' in data.columns:
        ax4.hist(data['load_factor'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.axvline(0.6, color='red', linestyle='--', alpha=0.7, label='Min (60%)')
        ax4.axvline(1.05, color='red', linestyle='--', alpha=0.7, label='Max (105%)')
        ax4.set_title('Load Factor Distribution')
        ax4.set_xlabel('Load Factor')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Soot blowing effectiveness
    ax5 = axes[1, 1]
    if 'soot_blowing_active' in data.columns and 'system_efficiency' in data.columns:
        soot_eff = data.groupby('soot_blowing_active')['system_efficiency'].mean()
        if len(soot_eff) == 2:
            labels = ['Normal Operation', 'During Soot Blowing']
            values = [soot_eff[False], soot_eff[True]]
            ax5.bar(labels, values, alpha=0.7, color=['blue', 'orange'])
            ax5.set_title('Efficiency During Soot Blowing')
            ax5.set_ylabel('System Efficiency')
            ax5.grid(True, alpha=0.3)
    
    # Plot 6: Stack temperature progression
    ax6 = axes[1, 2]
    if 'stack_temp_F' in data.columns:
        ax6.plot(time_hours[::100], data['stack_temp_F'].iloc[::100], 
                color='red', alpha=0.7, linewidth=1)
        ax6.set_title('Stack Temperature Over Time')
        ax6.set_xlabel('Hours')
        ax6.set_ylabel('Stack Temperature (°F)')
        ax6.grid(True, alpha=0.3)
    
    # Plot 7: Coal quality impact
    ax7 = axes[2, 0]
    if 'coal_quality' in data.columns and 'system_efficiency' in data.columns:
        coal_eff = data.groupby('coal_quality')['system_efficiency'].mean()
        if len(coal_eff) > 1:
            ax7.bar(coal_eff.index, coal_eff.values, alpha=0.7, color='green')
            ax7.set_title('Efficiency by Coal Quality')
            ax7.set_xlabel('Coal Quality')
            ax7.set_ylabel('System Efficiency')
            ax7.tick_params(axis='x', rotation=45)
            ax7.grid(True, alpha=0.3)
    
    # Plot 8: Monthly efficiency patterns
    ax8 = axes[2, 1]
    if 'timestamp' in data.columns and 'system_efficiency' in data.columns:
        data['month'] = pd.to_datetime(data['timestamp']).dt.month
        monthly_eff = data.groupby('month')['system_efficiency'].mean()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax8.plot(monthly_eff.index, monthly_eff.values, 'o-', alpha=0.7)
        ax8.set_title('Monthly Efficiency Patterns')
        ax8.set_xlabel('Month')
        ax8.set_ylabel('System Efficiency')
        ax8.set_xticks(range(1, 13))
        ax8.set_xticklabels([month_names[i-1] for i in range(1, 13)], rotation=45)
        ax8.grid(True, alpha=0.3)
    
    # Plot 9: Correlation heatmap (key variables)
    ax9 = axes[2, 2]
    key_cols = ['system_efficiency', 'load_factor', 'stack_temp_F']
    if fouling_cols:
        key_cols.append(fouling_cols[0])
    
    available_cols = [col for col in key_cols if col in data.columns]
    if len(available_cols) > 2:
        corr_matrix = data[available_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                   square=True, ax=ax9, cbar_kws={'shrink': 0.8})
        ax9.set_title('Key Variable Correlations')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Comprehensive visualization dashboard generated")
    
    return fig


def run_comprehensive_analysis(data, config=None):
    """Run complete comprehensive analysis suite"""
    
    if config is None:
        config = {
            'efficiency_target': (0.75, 0.88),
            'fouling_threshold': 1.5,
            'cleaning_effectiveness_target': 80
        }
    
    print("🚀 COMPREHENSIVE BOILER ANALYSIS SUITE")
    print("=" * 60)
    print(f"Dataset: {len(data)} records, {len(data.columns)} features")
    print()
    
    # Run all analysis components
    results = {}
    
    # 1. Operational Performance
    results['operational'] = analyze_operational_performance(data, config)
    
    # 2. Soot Blowing Analysis
    results['soot_blowing'] = analyze_soot_blowing_patterns(data)
    
    # 3. Fouling Patterns
    results['fouling'] = analyze_fouling_patterns(data, config)
    
    # 4. Cleaning Effectiveness
    results['cleaning_effectiveness'] = analyze_cleaning_effectiveness(data, config)
    
    # 5. Schedule Optimization
    results['schedule_optimization'] = analyze_cleaning_schedule_optimization(data, config)
    
    # 6. Coal Quality Impact
    results['coal_quality'] = analyze_coal_quality_impact(data)
    
    # 7. Seasonal Patterns
    results['seasonal'] = analyze_seasonal_patterns(data)
    
    # 8. Comprehensive Visualizations
    results['visualizations'] = create_comprehensive_visualizations(data, config)
    
    print("\n" + "=" * 60)
    print("✅ COMPREHENSIVE ANALYSIS COMPLETE")
    print("=" * 60)
    
    return results


def generate_optimization_recommendations(analysis_results, data):
    """Generate actionable optimization recommendations"""
    
    print("\n💡 OPTIMIZATION RECOMMENDATIONS")
    print("=" * 50)
    
    recommendations = []
    
    # Soot blowing optimization
    if 'soot_blowing' in analysis_results:
        soot_freq = analysis_results['soot_blowing'].get('soot_frequency', 0)
        if soot_freq < 0.02:  # Less than 2%
            recommendations.append("⚠️ Low soot blowing frequency - consider increasing cleaning schedule")
        elif soot_freq > 0.10:  # More than 10%
            recommendations.append("⚠️ High soot blowing frequency - investigate root causes of fouling")
        else:
            recommendations.append("✅ Optimal soot blowing frequency - maintain current schedule")
    
    # Efficiency optimization
    if 'system_efficiency' in data.columns:
        avg_eff = data['system_efficiency'].mean()
        if avg_eff < 0.80:
            recommendations.append("❌ Low efficiency - urgent optimization needed")
        elif avg_eff < 0.85:
            recommendations.append("⚠️ Good efficiency - optimization opportunities exist")
        else:
            recommendations.append("✅ Excellent efficiency - maintain current operations")
    
    # Cleaning effectiveness
    if 'cleaning_effectiveness' in analysis_results:
        effectiveness = analysis_results['cleaning_effectiveness']
        if effectiveness:
            avg_effectiveness = np.mean(list(effectiveness.values()))
            if avg_effectiveness < 60:
                recommendations.append("❌ Poor cleaning effectiveness - review cleaning procedures")
            elif avg_effectiveness < 80:
                recommendations.append("⚠️ Moderate cleaning effectiveness - fine-tune cleaning schedule")
            else:
                recommendations.append("✅ Excellent cleaning effectiveness - maintain current approach")
    
    print("\n📋 ACTIONABLE RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    return recommendations