#!/usr/bin/env python3
"""
Diagnostic script to test the fixed boiler system
Run this to verify stack temperatures are now realistic
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import the fixed boiler system
from boiler_system import EnhancedCompleteBoilerSystem

def test_single_operation():
    """Test a single boiler operation to check stack temperature."""
    print("="*60)
    print("TESTING SINGLE BOILER OPERATION")
    print("="*60)
    
    # Initialize with realistic parameters
    boiler = EnhancedCompleteBoilerSystem(
        fuel_input=100e6,           # 100 MMBtu/hr
        flue_gas_mass_flow=84000,   # lb/hr
        furnace_exit_temp=2200,      # Reduced from 3000°F
        base_fouling_multiplier=0.5  # Lower fouling for better heat transfer
    )
    
    print("\nBoiler Configuration:")
    print(f"  Fuel input: {boiler.fuel_input/1e6:.0f} MMBtu/hr")
    print(f"  Flue gas flow: {boiler.flue_gas_mass_flow:,.0f} lb/hr")
    print(f"  Furnace exit temp: {boiler.furnace_exit_temp:.0f}°F")
    print(f"  Fouling multiplier: {boiler.base_fouling_multiplier}")
    
    # Solve the system
    print("\nSolving boiler system...")
    results = boiler.solve_enhanced_system(max_iterations=30, tolerance=10.0)
    
    # Display results
    perf = boiler.system_performance
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    print(f"  System Efficiency: {perf['system_efficiency']:.1%}")
    print(f"  Stack Temperature: {perf['stack_temperature']:.0f}°F  {'✓ GOOD!' if 250 <= perf['stack_temperature'] <= 350 else '✗ BAD!'}")
    print(f"  Final Steam Temp: {perf['final_steam_temperature']:.0f}°F")
    print(f"  Heat Absorbed: {perf['total_heat_absorbed']/1e6:.1f} MMBtu/hr")
    print(f"  Iterations: {perf['iterations_to_converge']}")
    
    # Temperature profile through boiler
    print("\nTemperature Profile:")
    print("-"*40)
    for section_name, data in results.items():
        summary = data['summary']
        print(f"{section_name:20s}: Gas {summary['gas_temp_in']:4.0f} → {summary['gas_temp_out']:4.0f}°F  "
              f"(Δ={summary['gas_temp_in']-summary['gas_temp_out']:3.0f}°F)")
    
    return perf


def test_multiple_loads():
    """Test boiler at different load conditions."""
    print("\n" + "="*60)
    print("TESTING MULTIPLE LOAD CONDITIONS")
    print("="*60)
    
    load_factors = [0.5, 0.7, 0.85, 1.0]
    results = []
    
    for load in load_factors:
        fuel_input = 100e6 * load
        gas_flow = 84000 * load
        furnace_temp = 2000 + 200 * load  # Scale with load
        
        boiler = EnhancedCompleteBoilerSystem(
            fuel_input=fuel_input,
            flue_gas_mass_flow=gas_flow,
            furnace_exit_temp=furnace_temp,
            base_fouling_multiplier=0.5
        )
        
        boiler.solve_enhanced_system(max_iterations=25, tolerance=15.0)
        perf = boiler.system_performance
        
        results.append({
            'load': load,
            'efficiency': perf['system_efficiency'],
            'stack_temp': perf['stack_temperature'],
            'steam_temp': perf['final_steam_temperature']
        })
        
        print(f"Load {load:3.0%}: Eff={perf['system_efficiency']:5.1%}, "
              f"Stack={perf['stack_temperature']:3.0f}°F, "
              f"Steam={perf['final_steam_temperature']:3.0f}°F")
    
    return results


def test_fouling_impact():
    """Test impact of fouling on performance."""
    print("\n" + "="*60)
    print("TESTING FOULING IMPACT")
    print("="*60)
    
    fouling_multipliers = [0.2, 0.5, 1.0, 1.5, 2.0]
    results = []
    
    for fouling in fouling_multipliers:
        boiler = EnhancedCompleteBoilerSystem(
            fuel_input=100e6,
            flue_gas_mass_flow=84000,
            furnace_exit_temp=2200,
            base_fouling_multiplier=fouling
        )
        
        boiler.solve_enhanced_system(max_iterations=25, tolerance=15.0)
        perf = boiler.system_performance
        
        results.append({
            'fouling': fouling,
            'efficiency': perf['system_efficiency'],
            'stack_temp': perf['stack_temperature']
        })
        
        condition = "Clean" if fouling < 0.5 else "Normal" if fouling <= 1.0 else "Fouled"
        print(f"Fouling {fouling:3.1f}x ({condition:6s}): "
              f"Eff={perf['system_efficiency']:5.1%}, "
              f"Stack={perf['stack_temperature']:3.0f}°F")
    
    return results


def plot_diagnostics(load_results, fouling_results):
    """Create diagnostic plots."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Boiler System Diagnostic Results', fontsize=16, fontweight='bold')
    
    # Load impact on efficiency
    loads = [r['load'] for r in load_results]
    effs = [r['efficiency'] for r in load_results]
    ax1.plot(loads, effs, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Load Factor')
    ax1.set_ylabel('System Efficiency')
    ax1.set_title('Efficiency vs Load')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.7, 0.9])
    
    # Load impact on stack temp
    stack_temps = [r['stack_temp'] for r in load_results]
    ax2.plot(loads, stack_temps, 'r-s', linewidth=2, markersize=8)
    ax2.axhline(y=280, color='g', linestyle='--', label='Target')
    ax2.axhline(y=350, color='orange', linestyle='--', label='Max acceptable')
    ax2.set_xlabel('Load Factor')
    ax2.set_ylabel('Stack Temperature (°F)')
    ax2.set_title('Stack Temperature vs Load')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([200, 400])
    
    # Fouling impact on efficiency
    foulings = [r['fouling'] for r in fouling_results]
    fouling_effs = [r['efficiency'] for r in fouling_results]
    ax3.plot(foulings, fouling_effs, 'g-^', linewidth=2, markersize=8)
    ax3.set_xlabel('Fouling Multiplier')
    ax3.set_ylabel('System Efficiency')
    ax3.set_title('Impact of Fouling on Efficiency')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0.7, 0.9])
    
    # Fouling impact on stack temp
    fouling_stacks = [r['stack_temp'] for r in fouling_results]
    ax4.plot(foulings, fouling_stacks, 'm-d', linewidth=2, markersize=8)
    ax4.axhline(y=280, color='g', linestyle='--', label='Target')
    ax4.axhline(y=350, color='orange', linestyle='--', label='Max acceptable')
    ax4.set_xlabel('Fouling Multiplier')
    ax4.set_ylabel('Stack Temperature (°F)')
    ax4.set_title('Impact of Fouling on Stack Temperature')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_ylim([200, 400])
    
    plt.tight_layout()
    plt.savefig('boiler_diagnostics.png', dpi=150, bbox_inches='tight')
    print("\n✓ Diagnostic plots saved to 'boiler_diagnostics.png'")
    plt.show()


def validate_annual_data(csv_file=None):
    """Validate existing annual data if available."""
    if csv_file is None:
        # Try to find the most recent file
        import glob
        files = glob.glob('massachusetts_boiler_annual*.csv')
        if files:
            csv_file = max(files)  # Get most recent
        else:
            print("\nNo annual data file found to validate")
            return
    
    print("\n" + "="*60)
    print(f"VALIDATING ANNUAL DATA: {csv_file}")
    print("="*60)
    
    df = pd.read_csv(csv_file)
    
    # Check key metrics
    metrics = {
        'Stack Temperature': df['stack_temp_F'],
        'System Efficiency': df['system_efficiency'],
        'CO2 Percentage': df['co2_pct'] if 'co2_pct' in df else None,
        'Load Factor': df['load_factor']
    }
    
    print("\nData Statistics:")
    print("-"*40)
    for name, data in metrics.items():
        if data is not None:
            print(f"{name:20s}: Mean={data.mean():6.2f}, "
                  f"Min={data.min():6.2f}, Max={data.max():6.2f}")
    
    # Check for problems
    problems = []
    
    if df['stack_temp_F'].mean() > 400:
        problems.append("✗ Stack temperature too high (>400°F average)")
    else:
        print("\n✓ Stack temperature is realistic")
    
    if df['system_efficiency'].mean() < 0.7:
        problems.append("✗ System efficiency too low (<70% average)")
    else:
        print("✓ System efficiency is realistic")
    
    if df['system_efficiency'].max() > 0.95:
        problems.append("✗ Maximum efficiency unrealistic (>95%)")
    
    if problems:
        print("\nPROBLEMS FOUND:")
        for p in problems:
            print(f"  {p}")
    else:
        print("\n✓ All validation checks passed!")
    
    return df


def main():
    """Run all diagnostic tests."""
    print("🔧" * 30)
    print("BOILER SYSTEM DIAGNOSTIC SUITE")
    print("Testing fixes for realistic stack temperature")
    print("🔧" * 30)
    
    # Test 1: Single operation
    single_result = test_single_operation()
    
    # Test 2: Multiple loads
    load_results = test_multiple_loads()
    
    # Test 3: Fouling impact
    fouling_results = test_fouling_impact()
    
    # Create diagnostic plots
    try:
        plot_diagnostics(load_results, fouling_results)
    except Exception as e:
        print(f"\nCould not create plots: {e}")
    
    # Validate existing data if available
    validate_annual_data('massachusetts_boiler_annual_20250815_121034.csv')
    
    # Summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    if single_result['stack_temperature'] <= 350:
        print("✅ STACK TEMPERATURE FIXED!")
        print(f"   Achieved: {single_result['stack_temperature']:.0f}°F")
        print(f"   Target: 250-350°F")
    else:
        print("⚠️ Stack temperature still needs adjustment")
        print(f"   Current: {single_result['stack_temperature']:.0f}°F")
        print(f"   Target: 250-350°F")
    
    if single_result['system_efficiency'] >= 0.75:
        print("✅ SYSTEM EFFICIENCY REALISTIC!")
        print(f"   Achieved: {single_result['system_efficiency']:.1%}")
    else:
        print("⚠️ System efficiency needs improvement")
        print(f"   Current: {single_result['system_efficiency']:.1%}")
    
    print("\n💡 Next Steps:")
    print("1. Replace boiler_system.py with the fixed version")
    print("2. Update annual_boiler_simulator.py with the key changes")
    print("3. Re-run the annual simulation")
    print("4. Verify results with this diagnostic script")


if __name__ == "__main__":
    main()