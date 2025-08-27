#!/usr/bin/env python3
"""
Main Demonstration and Example Usage

This module demonstrates the complete enhanced boiler system with coal combustion
integration, soot blowing simulation, and ML dataset generation.

Functions:
    demonstrate_combustion_fouling_integration: Main integration demo
    demonstrate_soot_blowing_simulation: Soot blowing capabilities
    demonstrate_ml_dataset_generation: ML dataset creation
    validate_system_performance: System validation tests

Dependencies:
    - All other modules in the enhanced boiler system
    - datetime: Timestamp generation
    - numpy: Numerical calculations

Author: Enhanced Boiler Modeling System
Version: 5.0 - Complete Integration Demo
"""

import datetime
import numpy as np

from coal_combustion_models import CoalCombustionModel, SootProductionModel, CombustionFoulingIntegrator
from thermodynamic_properties import PropertyCalculator
from fouling_and_soot_blowing import SootBlowingSimulator
from heat_transfer_calculations import EnhancedBoilerTubeSection
from boiler_system import EnhancedCompleteBoilerSystem
from ml_dataset_generator import MLDatasetGenerator
from analysis_and_visualization import SystemAnalyzer, Visualizer


def demonstrate_combustion_fouling_integration():
    """Demonstrate the integrated combustion-fouling-ML system."""
    print("=" * 100)
    print("COMBUSTION-FOULING INTEGRATION FOR ML DATASET GENERATION")
    print("Advanced Soot Blowing Optimization with Coal Combustion Modeling")
    print("=" * 100)
    
    # Initialize boiler system
    print("\n🔧 Initializing Enhanced Boiler System...")
    boiler = EnhancedCompleteBoilerSystem(
        fuel_input=100e6,
        flue_gas_mass_flow=84000,
        furnace_exit_temp=2200.,
        base_fouling_multiplier=1.0
    )
    
    # Sample coal properties
    coal_properties = {
        'carbon': 72.0,
        'volatile_matter': 28.0,
        'fixed_carbon': 55.0,
        'sulfur': 1.2,
        'ash': 8.5
    }
    
    ultimate_analysis = {
        'C': 72.0, 'H': 5.0, 'O': 10.0, 'N': 1.5, 'S': 1.2, 'Ash': 8.5, 'Moisture': 1.8
    }
    
    # Test different combustion scenarios
    scenarios = [
        {
            'name': 'Optimal Combustion',
            'coal_rate': 8500, 'air_scfh': 900000, 'nox_eff': 0.35,
            'description': 'Clean combustion with optimal air/fuel ratio'
        },
        {
            'name': 'Rich Combustion (High Soot)',
            'coal_rate': 8500, 'air_scfh': 750000, 'nox_eff': 0.45,
            'description': 'Fuel-rich conditions leading to high soot production'
        },
        {
            'name': 'Lean Combustion (Low Soot)',
            'coal_rate': 8500, 'air_scfh': 1100000, 'nox_eff': 0.25,
            'description': 'Excess air conditions with reduced soot formation'
        },
        {
            'name': 'High Load Operation',
            'coal_rate': 10200, 'air_scfh': 1080000, 'nox_eff': 0.40,
            'description': 'High load with increased thermal stress'
        },
        {
            'name': 'Low Quality Coal',
            'coal_rate': 9000, 'air_scfh': 950000, 'nox_eff': 0.50,
            'description': 'Poor coal quality increasing soot production'
        }
    ]
    
    print(f"\n📊 Testing {len(scenarios)} Combustion Scenarios...")
    
    # Initialize fouling integrator
    fouling_integrator = CombustionFoulingIntegrator()
    
    results_comparison = []
    
    for i, scenario in enumerate(scenarios):
        print(f"\n{'-'*60}")
        print(f"SCENARIO {i+1}: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"{'-'*60}")
        
        # Create combustion model
        combustion_model = CoalCombustionModel(
            ultimate_analysis=ultimate_analysis,
            coal_lb_per_hr=scenario['coal_rate'],
            air_scfh=scenario['air_scfh'],
            NOx_eff=scenario['nox_eff']
        )
        
        # Calculate combustion
        combustion_model.calculate()
        
        # Calculate soot production
        soot_model = SootProductionModel()
        soot_data = soot_model.calculate_soot_production(combustion_model, coal_properties)
        
        # Calculate section fouling rates
        fouling_rates = fouling_integrator.calculate_section_fouling_rates(
            combustion_model, coal_properties, boiler
        )
        
        # Display results
        print(f"Combustion Results:")
        print(f"  Thermal NOx: {combustion_model.NO_thermal_lb_per_hr:.2f} lb/hr")
        print(f"  Fuel NOx: {combustion_model.NO_fuel_lb_per_hr:.2f} lb/hr")
        print(f"  Total NOx: {combustion_model.NO_total_lb_per_hr:.2f} lb/hr")
        print(f"  Excess O2: {combustion_model.dry_O2_pct:.1f}%")
        print(f"  Combustion Efficiency: {combustion_model.combustion_efficiency:.3f}")
        print(f"  Flame Temperature: {combustion_model.flame_temp_F:.0f}°F")
        
        print(f"\nSoot Production:")
        print(f"  Mass Rate: {soot_data.mass_production_rate:.4f} lb/hr")
        print(f"  Particle Size: {soot_data.particle_size_microns:.2f} μm")
        print(f"  Carbon Content: {soot_data.carbon_content:.1%}")
        print(f"  Deposition Tendency: {soot_data.deposition_tendency:.3f}")
        
        print(f"\nWorst Fouling Rates (Economizer Secondary):")
        econ_rates = fouling_rates.get('economizer_secondary', {'gas': [0], 'water': [0]})
        print(f"  Max Gas-side Rate: {max(econ_rates['gas']):.2e} hr-ft²-°F/Btu per hour")
        print(f"  Avg Gas-side Rate: {np.mean(econ_rates['gas']):.2e} hr-ft²-°F/Btu per hour")
        
        # Store for comparison
        results_comparison.append({
            'name': scenario['name'],
            'thermal_nox': combustion_model.NO_thermal_lb_per_hr,
            'fuel_nox': combustion_model.NO_fuel_lb_per_hr,
            'total_nox': combustion_model.NO_total_lb_per_hr,
            'excess_o2': combustion_model.dry_O2_pct,
            'combustion_eff': combustion_model.combustion_efficiency,
            'soot_rate': soot_data.mass_production_rate,
            'deposition': soot_data.deposition_tendency,
            'max_fouling_rate': max(econ_rates['gas']) if econ_rates['gas'] else 0
        })
    
    # Print comparison table
    print(f"\n" + "=" * 120)
    print("COMBUSTION SCENARIO COMPARISON")
    print("=" * 120)
    
    print(f"{'Scenario':<25} {'Thermal NOx':<12} {'Fuel NOx':<10} {'Excess O2':<10} {'Comb Eff':<10} {'Soot Rate':<12} {'Fouling Rate':<15}")
    print(f"{'Name':<25} {'(lb/hr)':<12} {'(lb/hr)':<10} {'(%)':<10} {'(--)':<10} {'(lb/hr)':<12} {'(hr-ft²-°F/Btu/hr)':<15}")
    print("-" * 120)
    
    for result in results_comparison:
        print(f"{result['name']:<25} {result['thermal_nox']:<12.2f} {result['fuel_nox']:<10.2f} "
              f"{result['excess_o2']:<10.1f} {result['combustion_eff']:<10.3f} "
              f"{result['soot_rate']:<12.4f} {result['max_fouling_rate']:<15.2e}")
    
    return results_comparison


def demonstrate_soot_blowing_simulation():
    """Comprehensive demonstration of soot blowing simulation capabilities."""
    print("=" * 100)
    print("SOOT BLOWING SIMULATION DEMONSTRATION")
    print("100 MMBtu/hr Boiler with Individual Segment Fouling Control")
    print("=" * 100)
    
    # Initialize system with moderate fouling
    print("\n🔧 Initializing System with Moderate Fouling...")
    boiler = EnhancedCompleteBoilerSystem(
        fuel_input=100e6,
        flue_gas_mass_flow=84000,
        furnace_exit_temp=3000,
        base_fouling_multiplier=1.5  # Moderate fouling
    )
    
    analyzer = SystemAnalyzer(boiler)
    
    # Solve baseline case
    print("\n🔄 Solving Baseline Case (Before Soot Blowing)...")
    baseline_results = boiler.solve_enhanced_system(max_iterations=10, tolerance=5.0)
    baseline_perf = boiler.system_performance.copy()
    
    print(f"Baseline Performance:")
    print(f"  Efficiency: {baseline_perf['system_efficiency']:.1%}")
    print(f"  Steam Temperature: {baseline_perf['final_steam_temperature']:.1f}°F")
    print(f"  Stack Temperature: {baseline_perf['stack_temperature']:.1f}°F")
    
    # Simulate progressive fouling buildup on economizer
    print(f"\n🕒 Simulating 720 Hours of Operation (30 days)...")
    economizer_section = boiler.sections['economizer_primary']
    economizer_section.simulate_fouling_buildup(720, fouling_rate_per_hour=0.002)
    
    # Solve with increased fouling
    fouled_results = boiler.solve_enhanced_system(max_iterations=10, tolerance=5.0)
    fouled_perf = boiler.system_performance.copy()
    
    print(f"After Fouling Buildup:")
    print(f"  Efficiency: {fouled_perf['system_efficiency']:.1%} (Δ{(fouled_perf['system_efficiency'] - baseline_perf['system_efficiency'])/baseline_perf['system_efficiency']*100:+.1f}%)")
    print(f"  Steam Temperature: {fouled_perf['final_steam_temperature']:.1f}°F (Δ{fouled_perf['final_steam_temperature'] - baseline_perf['final_steam_temperature']:+.1f}°F)")
    print(f"  Stack Temperature: {fouled_perf['stack_temperature']:.1f}°F (Δ{fouled_perf['stack_temperature'] - baseline_perf['stack_temperature']:+.1f}°F)")
    
    # Demonstrate targeted soot blowing
    print(f"\n💨 Applying Targeted Soot Blowing...")
    
    # Get current fouling state
    current_fouling = economizer_section.get_current_fouling_arrays()
    print(f"Current fouling levels in economizer:")
    for i, (gas_foul, water_foul) in enumerate(zip(current_fouling['gas'], current_fouling['water'])):
        print(f"  Segment {i}: Gas={gas_foul:.5f}, Water={water_foul:.5f}")
    
    # Apply soot blowing to worst segments (assuming segments 3-6 are dirtiest)
    dirty_segments = [3, 4, 5, 6]
    economizer_section.apply_soot_blowing(dirty_segments, cleaning_effectiveness=0.8)
    
    # Show fouling after cleaning
    cleaned_fouling = economizer_section.get_current_fouling_arrays()
    print(f"\nFouling after soot blowing (segments {dirty_segments}):")
    for i, (gas_foul, water_foul) in enumerate(zip(cleaned_fouling['gas'], cleaned_fouling['water'])):
        change_marker = " ✓" if i in dirty_segments else ""
        print(f"  Segment {i}: Gas={gas_foul:.5f}, Water={water_foul:.5f}{change_marker}")
    
    # Solve after soot blowing
    print(f"\n🔄 Solving After Soot Blowing...")
    cleaned_results = boiler.solve_enhanced_system(max_iterations=10, tolerance=5.0)
    cleaned_perf = boiler.system_performance.copy()
    
    print(f"After Soot Blowing:")
    print(f"  Efficiency: {cleaned_perf['system_efficiency']:.1%} (Δ{(cleaned_perf['system_efficiency'] - fouled_perf['system_efficiency'])/fouled_perf['system_efficiency']*100:+.1f}%)")
    print(f"  Steam Temperature: {cleaned_perf['final_steam_temperature']:.1f}°F (Δ{cleaned_perf['final_steam_temperature'] - fouled_perf['final_steam_temperature']:+.1f}°F)")
    print(f"  Stack Temperature: {cleaned_perf['stack_temperature']:.1f}°F (Δ{cleaned_perf['stack_temperature'] - fouled_perf['stack_temperature']:+.1f}°F)")
    
    # Performance comparison table
    print(f"\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    
    print(f"{'Condition':<20} {'Efficiency':<10} {'Steam T (°F)':<12} {'Stack T (°F)':<12} {'Heat Absorbed':<15}")
    print("-" * 75)
    print(f"{'Baseline':<20} {baseline_perf['system_efficiency']:<10.1%} {baseline_perf['final_steam_temperature']:<12.1f} "
          f"{baseline_perf['stack_temperature']:<12.1f} {baseline_perf['total_heat_absorbed']/1e6:<15.1f}")
    print(f"{'After Fouling':<20} {fouled_perf['system_efficiency']:<10.1%} {fouled_perf['final_steam_temperature']:<12.1f} "
          f"{fouled_perf['stack_temperature']:<12.1f} {fouled_perf['total_heat_absorbed']/1e6:<15.1f}")
    print(f"{'After Cleaning':<20} {cleaned_perf['system_efficiency']:<10.1%} {cleaned_perf['final_steam_temperature']:<12.1f} "
          f"{cleaned_perf['stack_temperature']:<12.1f} {cleaned_perf['total_heat_absorbed']/1e6:<15.1f}")
    
    # Calculate cleaning effectiveness
    fouling_loss = (baseline_perf['system_efficiency'] - fouled_perf['system_efficiency']) / baseline_perf['system_efficiency']
    cleaning_recovery = (cleaned_perf['system_efficiency'] - fouled_perf['system_efficiency']) / baseline_perf['system_efficiency']
    cleaning_effectiveness = cleaning_recovery / fouling_loss if fouling_loss > 0 else 0
    
    print(f"\nCLEANING EFFECTIVENESS ANALYSIS:")
    print(f"  Fouling Impact: {fouling_loss * 100:.2f}% efficiency loss")
    print(f"  Cleaning Recovery: {cleaning_recovery * 100:.2f}% efficiency recovered")
    print(f"  Cleaning Effectiveness: {cleaning_effectiveness * 100:.1f}% of fouling impact recovered")
    
    return {
        'baseline': baseline_perf,
        'fouled': fouled_perf,
        'cleaned': cleaned_perf,
        'cleaning_effectiveness': cleaning_effectiveness
    }


def demonstrate_ml_dataset_generation():
    """Demonstrate ML dataset generation for soot blowing optimization."""
    print(f"\n" + "=" * 100)
    print("ML DATASET GENERATION DEMONSTRATION")
    print("=" * 100)
    
    print(f"\n🤖 Generating Sample ML Dataset...")
    
    # Initialize boiler system
    boiler = EnhancedCompleteBoilerSystem(
        fuel_input=100e6,
        flue_gas_mass_flow=84000,
        furnace_exit_temp=3000,
        base_fouling_multiplier=1.0
    )
    
    # Generate smaller dataset for demonstration
    ml_generator = MLDatasetGenerator(boiler)
    sample_dataset = ml_generator.generate_comprehensive_dataset(num_scenarios=100)
    
    print(f"\n📊 Dataset Summary:")
    print(f"  Total Records: {len(sample_dataset)}")
    print(f"  Features: {len([col for col in sample_dataset.columns if col not in ['efficiency_gain', 'heat_transfer_gain', 'fuel_savings_per_hour', 'cleaning_cost', 'cleaning_time', 'payback_hours', 'roi_24hr', 'performance_score']])}")
    print(f"  Target Variables: 8")
    
    # Show feature importance analysis
    print(f"\n📈 Sample Feature Analysis:")
    
    # Key combustion features
    combustion_features = ['thermal_nox_lb_hr', 'fuel_nox_lb_hr', 'excess_o2_pct', 'combustion_efficiency']
    print(f"\nCombustion Features Statistics:")
    for feature in combustion_features:
        if feature in sample_dataset.columns:
            values = sample_dataset[feature]
            print(f"  {feature}: {values.mean():.3f} ± {values.std():.3f} (range: {values.min():.3f} to {values.max():.3f})")
    
    # Target variable analysis
    target_features = ['efficiency_gain', 'payback_hours', 'roi_24hr', 'performance_score']
    print(f"\nTarget Variables Statistics:")
    for target in target_features:
        if target in sample_dataset.columns:
            values = sample_dataset[target]
            print(f"  {target}: {values.mean():.3f} ± {values.std():.3f} (range: {values.min():.3f} to {values.max():.3f})")
    
    # Show correlation with soot production factors
    print(f"\n🔗 Soot Production Correlations:")
    soot_indicators = ['thermal_nox_lb_hr', 'excess_o2_pct', 'combustion_efficiency']
    fouling_indicators = [col for col in sample_dataset.columns if 'fouling_' in col and '_avg_gas' in col]
    
    if fouling_indicators:
        fouling_avg = sample_dataset[fouling_indicators].mean(axis=1)
        for indicator in soot_indicators:
            if indicator in sample_dataset.columns:
                correlation = sample_dataset[indicator].corr(fouling_avg)
                print(f"  {indicator} vs Avg Fouling: {correlation:.3f}")
    
    # Export sample dataset
    output_filename = f"soot_blowing_optimization_dataset_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    sample_dataset.to_csv(output_filename, index=False)
    print(f"\n💾 Sample dataset exported to: {output_filename}")
    
    return sample_dataset


def demonstrate_system_analysis():
    """Demonstrate comprehensive system analysis capabilities."""
    print(f"\n" + "=" * 100)
    print("COMPREHENSIVE SYSTEM ANALYSIS DEMONSTRATION")
    print("=" * 100)
    
    # Initialize and solve system
    boiler = EnhancedCompleteBoilerSystem(
        fuel_input=100e6,
        flue_gas_mass_flow=84000,
        furnace_exit_temp=3000,
        base_fouling_multiplier=1.2
    )
    
    print("\n🔄 Solving System for Analysis...")
    results = boiler.solve_enhanced_system(max_iterations=12, tolerance=4.0)
    
    # Initialize analysis tools
    analyzer = SystemAnalyzer(boiler)
    visualizer = Visualizer(boiler)
    
    # Comprehensive analysis
    print("\n📊 Performing Comprehensive Analysis...")
    analyzer.print_comprehensive_summary()
    analyzer.analyze_soot_blowing_effectiveness()
    
    # Economic analysis
    economic_metrics = analyzer.calculate_economic_metrics(
        fuel_cost_per_mmbtu=5.5,  # $/MMBtu
        electricity_cost_per_kwh=0.08  # $/kWh
    )
    
    print(f"\n💰 ECONOMIC ANALYSIS:")
    print(f"  Annual Fuel Cost: ${economic_metrics['annual_fuel_cost']:,.0f}")
    print(f"  Annual Steam Value: ${economic_metrics['annual_steam_value']:,.0f}")
    print(f"  Efficiency Savings: ${economic_metrics['efficiency_savings']:,.0f}")
    print(f"  Cost per MMBtu Steam: ${economic_metrics['cost_per_mmbtu_steam']:.2f}")
    
    # Export detailed results
    analyzer.export_detailed_results("comprehensive_boiler_analysis.txt")
    
    # Generate visualizations
    print(f"\n📈 Generating Visualizations...")
    try:
        visualizer.plot_comprehensive_profiles()
        visualizer.plot_fouling_analysis()
        
        # Section comparison
        key_sections = ['economizer_primary', 'superheater_secondary', 'generating_bank']
        visualizer.plot_section_comparison(key_sections)
        
    except Exception as e:
        print(f"Visualization error (may need display): {e}")
    
    return {
        'system_performance': boiler.system_performance,
        'economic_metrics': economic_metrics,
        'section_results': results
    }


def validate_system_performance():
    """Validate system performance against expected values."""
    print(f"\n" + "=" * 100)
    print("SYSTEM PERFORMANCE VALIDATION")
    print("=" * 100)
    
    validation_results = []
    
    # Test 1: Clean system performance
    print("\n🧪 Test 1: Clean System Performance")
    boiler_clean = EnhancedCompleteBoilerSystem(
        fuel_input=100e6,
        flue_gas_mass_flow=84000,
        furnace_exit_temp=3000,
        base_fouling_multiplier=0.5  # Very clean
    )
    
    try:
        results_clean = boiler_clean.solve_enhanced_system(max_iterations=10, tolerance=5.0)
        perf_clean = boiler_clean.system_performance
        
        # Expected ranges for clean system
        efficiency_ok = 0.82 <= perf_clean['system_efficiency'] <= 0.88
        steam_temp_ok = 680 <= perf_clean['final_steam_temperature'] <= 720
        
        validation_results.append({
            'test': 'Clean System',
            'efficiency': perf_clean['system_efficiency'],
            'steam_temp': perf_clean['final_steam_temperature'],
            'efficiency_ok': efficiency_ok,
            'steam_temp_ok': steam_temp_ok,
            'overall_pass': efficiency_ok and steam_temp_ok
        })
        
        print(f"  Efficiency: {perf_clean['system_efficiency']:.1%} ({'✓' if efficiency_ok else '✗'})")
        print(f"  Steam Temp: {perf_clean['final_steam_temperature']:.1f}°F ({'✓' if steam_temp_ok else '✗'})")
        
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        validation_results.append({'test': 'Clean System', 'overall_pass': False, 'error': str(e)})
    
    # Test 2: Fouled system performance
    print("\n🧪 Test 2: Fouled System Performance")
    boiler_fouled = EnhancedCompleteBoilerSystem(
        fuel_input=100e6,
        flue_gas_mass_flow=84000,
        furnace_exit_temp=3000,
        base_fouling_multiplier=2.5  # Heavy fouling
    )
    
    try:
        results_fouled = boiler_fouled.solve_enhanced_system(max_iterations=10, tolerance=8.0)
        perf_fouled = boiler_fouled.system_performance
        
        # Expected degradation due to fouling
        efficiency_degraded = perf_fouled['system_efficiency'] < 0.82
        temp_impact = abs(perf_fouled['final_steam_temperature'] - 700) < 50
        
        validation_results.append({
            'test': 'Fouled System',
            'efficiency': perf_fouled['system_efficiency'],
            'steam_temp': perf_fouled['final_steam_temperature'],
            'efficiency_degraded': efficiency_degraded,
            'temp_reasonable': temp_impact,
            'overall_pass': efficiency_degraded and temp_impact
        })
        
        print(f"  Efficiency: {perf_fouled['system_efficiency']:.1%} ({'✓' if efficiency_degraded else '✗'})")
        print(f"  Steam Temp: {perf_fouled['final_steam_temperature']:.1f}°F ({'✓' if temp_impact else '✗'})")
        
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        validation_results.append({'test': 'Fouled System', 'overall_pass': False, 'error': str(e)})
    
    # Test 3: Property calculations
    print("\n🧪 Test 3: Property Calculations")
    try:
        prop_calc = PropertyCalculator()
        
        # Test steam properties
        steam_props = prop_calc.get_steam_properties(700, 600)  # 700°F, 600 psia
        steam_reasonable = (
            0.4 <= steam_props.cp <= 0.6 and  # Btu/lbm-°F
            0.5 <= steam_props.density <= 3.0 and  # lbm/ft³
            steam_props.phase == 'superheated_steam'
        )
        
        # Test gas properties
        gas_props = prop_calc.get_flue_gas_properties(1500)  # 1500°F
        gas_reasonable = (
            0.01 <= gas_props.density <= 0.1 and  # lbm/ft³
            0.2 <= gas_props.cp <= 0.3  # Btu/lbm-°F
        )
        
        validation_results.append({
            'test': 'Property Calculations',
            'steam_props_ok': steam_reasonable,
            'gas_props_ok': gas_reasonable,
            'overall_pass': steam_reasonable and gas_reasonable
        })
        
        print(f"  Steam Properties: {'✓' if steam_reasonable else '✗'}")
        print(f"  Gas Properties: {'✓' if gas_reasonable else '✗'}")
        
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        validation_results.append({'test': 'Property Calculations', 'overall_pass': False, 'error': str(e)})
    
    # Summary
    passed_tests = sum(1 for result in validation_results if result.get('overall_pass', False))
    total_tests = len(validation_results)
    
    print(f"\n📋 VALIDATION SUMMARY:")
    print(f"  Tests Passed: {passed_tests}/{total_tests}")
    print(f"  Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("  ✅ All validation tests PASSED")
    else:
        print("  ⚠️  Some validation tests FAILED")
    
    return validation_results


def main():
    """Main execution function demonstrating all capabilities."""
    print("🎉" * 50)
    print("ENHANCED BOILER SYSTEM WITH COAL COMBUSTION INTEGRATION")
    print("Complete Demonstration of Soot Blowing Optimization System")
    print("🎉" * 50)
    
    try:
        # 1. Combustion integration demonstration
        print("\n" + "🔥" * 50)
        combustion_results = demonstrate_combustion_fouling_integration()
        
        # 2. Soot blowing simulation
        print("\n" + "💨" * 50)
        soot_blowing_results = demonstrate_soot_blowing_simulation()
        
        # 3. ML dataset generation
        print("\n" + "🤖" * 50)
        ml_dataset = demonstrate_ml_dataset_generation()
        
        # 4. System analysis
        print("\n" + "📊" * 50)
        analysis_results = demonstrate_system_analysis()
        
        # 5. Validation tests
        print("\n" + "🧪" * 50)
        validation_results = validate_system_performance()
        
        # Final summary
        print(f"\n" + "🎉" * 50)
        print("🎉 COMPLETE SYSTEM DEMONSTRATION FINISHED! 🎉")
        print("🎉" * 50)
        
        print(f"\n✅ ACHIEVEMENTS:")
        print(f"  • Coal combustion modeling with soot production")
        print(f"  • Dynamic fouling linked to combustion conditions")
        print(f"  • Individual segment soot blowing simulation")
        print(f"  • ML dataset generation ({len(ml_dataset)} records)")
        print(f"  • Comprehensive system analysis and visualization")
        print(f"  • Performance validation ({sum(1 for r in validation_results if r.get('overall_pass', False))}/{len(validation_results)} tests passed)")
        
        print(f"\n🎯 ML MODEL TRAINING READY:")
        print(f"  • Dataset: {len(ml_dataset)} scenarios with {len(ml_dataset.columns)} features")
        print(f"  • Target variables: Efficiency gain, ROI, payback time")
        print(f"  • Cleaning strategies: 6 different approaches evaluated")
        print(f"  • Operating conditions: Full range of coal properties and loads")
        
        return {
            'combustion_results': combustion_results,
            'soot_blowing_results': soot_blowing_results,
            'ml_dataset': ml_dataset,
            'analysis_results': analysis_results,
            'validation_results': validation_results
        }
        
    except Exception as e:
        print(f"\n❌ DEMONSTRATION ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    """Execute the complete demonstration when run as main module."""
    results = main()
    
    if results:
        print(f"\n🏁 Demonstration completed successfully!")
        print(f"All modules tested and integrated for ML-based soot blowing optimization.")
    else:
        print(f"\n💥 Demonstration encountered errors. Check the output above.")
