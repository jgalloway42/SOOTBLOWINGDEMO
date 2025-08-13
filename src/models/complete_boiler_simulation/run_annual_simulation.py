#!/usr/bin/env python3
"""
Complete Annual Simulation Runner

This script runs the complete annual boiler simulation for Massachusetts,
generates the dataset, and performs comprehensive analysis.

Run this script to generate a full year of boiler operation data including:
- Variable load operation (45-100% capacity)
- Massachusetts seasonal weather patterns  
- Coal quality variations (4 different grades)
- Scheduled soot blowing cycles for all sections
- Complete fouling factors for all tube sections
- All temperatures (gas/water in/out for each section)
- All flow rates (coal, air, steam, flue gas)
- Complete stack gas analysis (NOx, O2, efficiency)
- System performance metrics

Usage:
    python run_annual_simulation.py

Author: Enhanced Boiler Modeling System
Version: 6.0 - Complete Annual Simulation
"""

import sys
import os
import traceback
import pandas as pd
from datetime import datetime

# Import our simulation modules
try:
    from annual_boiler_simulator import AnnualBoilerSimulator
    from data_analysis_tools import BoilerDataAnalyzer
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure all required modules are available:")
    print("- annual_boiler_simulator.py")
    print("- data_analysis_tools.py") 
    print("- boiler_system.py")
    print("- coal_combustion_models.py")
    print("- thermodynamic_properties.py")
    print("- fouling_and_soot_blowing.py")
    print("- heat_transfer_calculations.py")
    print("- analysis_and_visualization.py")
    print("- ml_dataset_generator.py")
    sys.exit(1)


def run_complete_simulation():
    """Run the complete annual simulation and analysis."""
    print("🏭" * 30)
    print("MASSACHUSETTS BOILER ANNUAL OPERATION SIMULATION")
    print("Complete Year of Realistic Operation Data Generation")
    print("🏭" * 30)
    
    try:
        # Step 1: Initialize the simulator
        print("\n🔧 STEP 1: Initializing Annual Boiler Simulator...")
        simulator = AnnualBoilerSimulator(start_date="2024-01-01")
        
        # Step 2: Generate annual operation data
        print("\n⚙️ STEP 2: Generating Annual Operation Data...")
        print("This will simulate a full year of operation with:")
        print("   • 24/7 continuous operation")
        print("   • Data recorded every hour (8,760 data points)")
        print("   • Variable load operation (45-100% capacity)")
        print("   • Massachusetts seasonal weather patterns")
        print("   • Multiple coal quality grades")
        print("   • Scheduled soot blowing cycles")
        print("   • Individual section cleaning tracking")
        print("   • Complete fouling tracking")
        print("   • Full stack gas analysis (CO, CO2, H2O, SO2, NOx, O2)")
        
        # Generate the dataset
        annual_data = simulator.generate_annual_data(
            hours_per_day=24,        # Continuous operation
            save_interval_hours=1    # Record every hour
        )
        
        # Step 3: Save the dataset
        print("\n💾 STEP 3: Saving Dataset...")
        filename = simulator.save_annual_data(annual_data)
        
        # Step 4: Analyze the data
        print("\n📊 STEP 4: Performing Comprehensive Analysis...")
        analyzer = BoilerDataAnalyzer(annual_data)
        
        # Generate comprehensive report
        analysis_report = analyzer.generate_comprehensive_report(save_plots=True)
        
        # Step 5: Export detailed results
        print("\n📄 STEP 5: Exporting Detailed Results...")
        
        # Save analysis report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"boiler_analysis_report_{timestamp}.txt"
        
        with open(report_filename, 'w') as f:
            f.write("MASSACHUSETTS BOILER ANNUAL ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Data Period: {annual_data['timestamp'].min()} to {annual_data['timestamp'].max()}\n")
            f.write(f"Total Records: {len(annual_data):,}\n\n")
            
            # Write key findings
            f.write("KEY PERFORMANCE METRICS:\n")
            f.write("-" * 30 + "\n")
            perf = analysis_report['performance_summary']
            f.write(f"Average System Efficiency: {perf['system_efficiency']['mean']:.1%}\n")
            f.write(f"Efficiency Range: {perf['system_efficiency']['min']:.1%} to {perf['system_efficiency']['max']:.1%}\n")
            f.write(f"Average Load Factor: {perf['load_factor']['mean']:.1%}\n")
            f.write(f"Average Steam Temperature: {perf['final_steam_temp_F']['mean']:.0f}°F\n")
            f.write(f"Average Stack Temperature: {perf['stack_temp_F']['mean']:.0f}°F\n")
            f.write(f"Average NOx Emissions: {perf['total_nox_lb_hr']['mean']:.1f} lb/hr\n")
            if 'co_ppm' in perf and perf['co_ppm'] is not None:
                f.write(f"Average CO Emissions: {perf['co_ppm']['mean']:.0f} ppm\n")
            if 'co2_pct' in perf and perf['co2_pct'] is not None:
                f.write(f"Average CO2 Emissions: {perf['co2_pct']['mean']:.1f}%\n")
            if 'so2_ppm' in perf and perf['so2_ppm'] is not None:
                f.write(f"Average SO2 Emissions: {perf['so2_ppm']['mean']:.0f} ppm\n")
            f.write("\n")
            
            # Seasonal analysis
            f.write("SEASONAL PERFORMANCE:\n")
            f.write("-" * 20 + "\n")
            seasonal = analysis_report['seasonal_analysis']
            f.write(f"Best performing season: {seasonal['best_season']}\n")
            f.write(f"Worst performing season: {seasonal['worst_season']}\n")
            f.write(f"Seasonal efficiency variation: {seasonal['efficiency_range']:.2%}\n\n")
            
            # Coal quality impact
            f.write("COAL QUALITY ANALYSIS:\n")
            f.write("-" * 22 + "\n")
            coal = analysis_report['coal_quality_analysis']
            for quality, data in coal['by_quality'].items():
                f.write(f"{quality}: {data['avg_efficiency']:.1%} efficiency, {data['percentage_of_year']:.1f}% of year\n")
            f.write(f"\nBest coal quality: {coal['best_quality']}\n")
            f.write(f"Worst coal quality: {coal['worst_quality']}\n\n")
            
            # Optimization opportunities
            f.write("OPTIMIZATION OPPORTUNITIES:\n")
            f.write("-" * 28 + "\n")
            opt = analysis_report['optimization_opportunities']
            if 'soot_blowing' in opt:
                f.write(f"• Soot Blowing: Focus on {opt['soot_blowing']['worst_fouling_section']}\n")
            if 'coal_quality' in opt:
                f.write(f"• Coal Quality: Increase {opt['coal_quality']['best_coal_quality']} usage\n")
                f.write(f"  Potential efficiency gain: {opt['coal_quality']['potential_efficiency_gain']:.2%}\n")
            if 'load_optimization' in opt:
                f.write(f"• Load Optimization: Target range {opt['load_optimization']['optimal_load_range']}\n")
        
        print(f"📄 Analysis report saved: {report_filename}")
        
        # Step 6: Generate summary statistics
        print("\n📈 STEP 6: Final Summary...")
        print_final_summary(annual_data, analysis_report, filename)
        
        return {
            'dataset': annual_data,
            'analysis': analysis_report,
            'data_filename': filename,
            'report_filename': report_filename
        }
        
    except Exception as e:
        print(f"\n❌ SIMULATION FAILED: {e}")
        traceback.print_exc()
        return None


def print_final_summary(data: pd.DataFrame, analysis: dict, filename: str):
    """Print comprehensive final summary."""
    print("\n" + "🎉" * 50)
    print("ANNUAL SIMULATION COMPLETE - DATASET READY FOR ANALYSIS!")
    print("🎉" * 50)
    
    print(f"\n📊 DATASET SUMMARY:")
    print(f"   📁 File: {filename}")
    print(f"   📅 Period: {data['timestamp'].min()} to {data['timestamp'].max()}")
    print(f"   📈 Records: {len(data):,} data points")
    print(f"   📋 Columns: {len(data.columns)} variables")
    print(f"   💾 Size: {data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    
    print(f"\n🏭 OPERATION CHARACTERISTICS:")
    print(f"   ⚡ Load Range: {data['load_factor'].min():.1%} to {data['load_factor'].max():.1%}")
    print(f"   ⚡ Average Load: {data['load_factor'].mean():.1%}")
    print(f"   🌡️ Ambient Range: {data['ambient_temp_F'].min():.0f}°F to {data['ambient_temp_F'].max():.0f}°F")
    print(f"   ⛽ Coal Qualities: {data['coal_quality'].nunique()} different grades")
    print(f"   🧹 Soot Blowing Events: {data['soot_blowing_active'].sum()} cleaning cycles")
    
    print(f"\n📊 PERFORMANCE METRICS:")
    perf = analysis['performance_summary']
    print(f"   🎯 Average Efficiency: {perf['system_efficiency']['mean']:.1%}")
    print(f"   🎯 Efficiency Range: {perf['system_efficiency']['min']:.1%} - {perf['system_efficiency']['max']:.1%}")
    print(f"   🌡️ Average Steam Temp: {perf['final_steam_temp_F']['mean']:.0f}°F")
    print(f"   🌡️ Average Stack Temp: {perf['stack_temp_F']['mean']:.0f}°F")
    print(f"   💨 Average NOx: {perf['total_nox_lb_hr']['mean']:.1f} lb/hr")
    if 'co_ppm' in data.columns:
        print(f"   💨 Average CO: {data['co_ppm'].mean():.0f} ppm")
    if 'co2_pct' in data.columns:
        print(f"   💨 Average CO2: {data['co2_pct'].mean():.1f}%")
    if 'so2_ppm' in data.columns:
        print(f"   💨 Average SO2: {data['so2_ppm'].mean():.0f} ppm")
    
    print(f"\n🔬 INCLUDED DATA CATEGORIES:")
    print(f"   ✅ Variable load operation (45-100% capacity)")
    print(f"   ✅ Massachusetts seasonal weather patterns")
    print(f"   ✅ Multiple coal quality variations")
    print(f"   ✅ Scheduled soot blowing for all sections:")
    
    # Show soot blowing schedule
    sections = [col.split('_fouling_')[0] for col in data.columns if '_fouling_gas_avg' in col]
    for section in sorted(set(sections)):
        hours_col = f"{section}_hours_since_cleaning"
        if hours_col in data.columns:
            avg_cycle = data[hours_col].mean()
            times_per_day = 24 / avg_cycle if avg_cycle > 0 else 0
            print(f"      • {section}: ~{avg_cycle:.0f} hour cycles ({times_per_day:.1f}x/day)")
    
    print(f"   ✅ Complete fouling factors for all tube sections")
    print(f"   ✅ All temperatures (gas/water in/out for each section)")
    print(f"   ✅ All flow rates (coal, air, steam, flue gas)")
    print(f"   ✅ Complete stack gas analysis (NOx, CO, CO2, H2O, SO2, O2, efficiency)")
    print(f"   ✅ System performance metrics")
    
    print(f"\n🎯 DATA APPLICATIONS:")
    print(f"   • Predictive maintenance optimization")
    print(f"   • Soot blowing schedule optimization")
    print(f"   • Coal quality impact analysis")
    print(f"   • Seasonal performance modeling")
    print(f"   • Emissions compliance monitoring")
    print(f"   • Energy efficiency improvements")
    print(f"   • Machine learning model training")
    print(f"   • Digital twin development")
    
    print(f"\n📋 COLUMN CATEGORIES ({len(data.columns)} total):")
    
    # Categorize columns
    categories = {
        'Time & Conditions': [col for col in data.columns if any(x in col for x in ['timestamp', 'year', 'month', 'day', 'hour', 'season', 'ambient'])],
        'Operating Parameters': [col for col in data.columns if any(x in col for x in ['load_factor', 'coal_rate', 'air_flow', 'fuel_input'])],
        'Coal Properties': [col for col in data.columns if col.startswith('coal_')],
        'Combustion Results': [col for col in data.columns if any(x in col for x in ['nox', 'excess_o2', 'combustion_efficiency', 'flame_temp', 'co_ppm', 'co2_pct', 'h2o_pct', 'so2_ppm'])],
        'System Performance': [col for col in data.columns if any(x in col for x in ['system_efficiency', 'steam_temp', 'stack_temp', 'heat_absorbed', 'steam_production'])],
        'Fouling Factors': [col for col in data.columns if 'fouling' in col],
        'Section Temperatures': [col for col in data.columns if any(x in col for x in ['_gas_temp_', '_water_temp_'])],
        'Heat Transfer': [col for col in data.columns if any(x in col for x in ['heat_transfer', 'overall_U'])],
        'Soot Blowing': [col for col in data.columns if any(x in col for x in ['soot_blowing', 'days_since_cleaning'])]
    }
    
    for category, cols in categories.items():
        if cols:
            print(f"   📂 {category}: {len(cols)} columns")
    
    print(f"\n🚀 READY FOR ANALYSIS!")
    print(f"   The dataset is now ready for comprehensive analysis including:")
    print(f"   • Load the CSV file into your preferred analysis tool")
    print(f"   • Use pandas, R, or other data science platforms")
    print(f"   • Build predictive models for maintenance optimization")
    print(f"   • Analyze fouling patterns and cleaning effectiveness")
    print(f"   • Study seasonal and operational impacts on performance")


def show_data_sample(data: pd.DataFrame):
    """Show a sample of the generated data."""
    print(f"\n📋 DATA SAMPLE (First 5 records):")
    print("=" * 120)
    
    # Select key columns for display
    key_columns = [
        'timestamp', 'load_factor', 'ambient_temp_F', 'coal_quality',
        'system_efficiency', 'final_steam_temp_F', 'stack_temp_F', 
        'total_nox_lb_hr', 'soot_blowing_active'
    ]
    
    # Add stack gas components if available
    if 'co_ppm' in data.columns:
        key_columns.append('co_ppm')
    if 'co2_pct' in data.columns:
        key_columns.append('co2_pct')
    
    # Add a fouling column and section cleaning indicator if available
    fouling_cols = [col for col in data.columns if 'fouling_gas_avg' in col]
    if fouling_cols:
        key_columns.append(fouling_cols[0])
    
    section_cleaning_cols = [col for col in data.columns if 'soot_blowing_active' in col and col != 'soot_blowing_active']
    if section_cleaning_cols:
        key_columns.append(section_cleaning_cols[0])
    
    sample_data = data[key_columns].head()
    print(sample_data.to_string(index=False))
    print("=" * 120)


if __name__ == "__main__":
    """Run the complete annual simulation when executed as main script."""
    
    print("Starting Massachusetts Boiler Annual Simulation...")
    print("This will generate a comprehensive year of operation data.")
    print("\nEstimated runtime: 10-30 minutes depending on system performance")
    
    response = input("\nProceed with simulation? (y/n): ").strip().lower()
    
    if response in ['y', 'yes']:
        # Run the complete simulation
        results = run_complete_simulation()
        
        if results:
            print(f"\n✅ SIMULATION SUCCESSFUL!")
            
            # Show a sample of the data
            show_data_sample(results['dataset'])
            
            print(f"\n📁 Files Generated:")
            print(f"   • Data: {results['data_filename']}")
            print(f"   • Report: {results['report_filename']}")
            print(f"   • Plots: annual_boiler_analysis.png")
            
            print(f"\n🎯 Next Steps:")
            print(f"   1. Load the CSV file for detailed analysis")
            print(f"   2. Use the data for machine learning model training")
            print(f"   3. Develop predictive maintenance strategies")
            print(f"   4. Optimize soot blowing schedules")
            print(f"   5. Analyze coal quality procurement strategies")
            
        else:
            print(f"\n❌ SIMULATION FAILED!")
            print(f"Check the error messages above for troubleshooting.")
            
    else:
        print("Simulation cancelled.")
        
    print(f"\n🏁 Program complete.")