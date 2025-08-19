#!/usr/bin/env python3
"""
Debug Script for Simulation Issues

This script helps identify what's going wrong with the simulation.
Run this to get detailed error information.
"""

import sys
import traceback
import pandas as pd
from datetime import datetime

def test_imports():
    """Test if all imports work."""
    print("🔍 TESTING IMPORTS...")
    
    try:
        from annual_boiler_simulator import AnnualBoilerSimulator
        print("   ✅ annual_boiler_simulator imported successfully")
    except Exception as e:
        print(f"   ❌ annual_boiler_simulator import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        from boiler_system import EnhancedCompleteBoilerSystem
        print("   ✅ boiler_system imported successfully")
    except Exception as e:
        print(f"   ❌ boiler_system import failed: {e}")
        return False
    
    try:
        from coal_combustion_models import CoalCombustionModel, CombustionFoulingIntegrator
        print("   ✅ coal_combustion_models imported successfully")
    except Exception as e:
        print(f"   ❌ coal_combustion_models import failed: {e}")
        return False
    
    try:
        from thermodynamic_properties import PropertyCalculator
        print("   ✅ thermodynamic_properties imported successfully")
    except Exception as e:
        print(f"   ❌ thermodynamic_properties import failed: {e}")
        return False
    
    try:
        from fouling_and_soot_blowing import SootBlowingSimulator
        print("   ✅ fouling_and_soot_blowing imported successfully")
    except Exception as e:
        print(f"   ❌ fouling_and_soot_blowing import failed: {e}")
        return False
    
    try:
        from analysis_and_visualization import SystemAnalyzer
        print("   ✅ analysis_and_visualization imported successfully")
    except Exception as e:
        print(f"   ❌ analysis_and_visualization import failed: {e}")
        return False
    
    return True

def test_initialization():
    """Test simulator initialization."""
    print("\n🔍 TESTING SIMULATOR INITIALIZATION...")
    
    try:
        from annual_boiler_simulator import AnnualBoilerSimulator
        simulator = AnnualBoilerSimulator(start_date="2024-01-01")
        print("   ✅ AnnualBoilerSimulator initialized successfully")
        return simulator
    except Exception as e:
        print(f"   ❌ Simulator initialization failed: {e}")
        traceback.print_exc()
        return None

def test_single_hour():
    """Test generating a single hour of data."""
    print("\n🔍 TESTING SINGLE HOUR GENERATION...")
    
    try:
        from annual_boiler_simulator import AnnualBoilerSimulator
        simulator = AnnualBoilerSimulator(start_date="2024-01-01")
        
        current_datetime = simulator.start_date
        print(f"   Testing datetime: {current_datetime}")
        
        # Step 1: Generate operating conditions
        print("   Step 1: Generating operating conditions...")
        operating_conditions = simulator._generate_hourly_conditions(current_datetime)
        print(f"   ✅ Operating conditions: Load={operating_conditions['load_factor']:.3f}")
        
        # Step 2: Check soot blowing
        print("   Step 2: Checking soot blowing schedule...")
        soot_blowing_actions = simulator._check_soot_blowing_schedule(current_datetime)
        print(f"   ✅ Soot blowing checked")
        
        # Step 3: Simulate boiler operation
        print("   Step 3: Simulating boiler operation...")
        operation_data = simulator._simulate_boiler_operation(
            current_datetime, operating_conditions, soot_blowing_actions
        )
        print(f"   ✅ Operation simulated: Stack={operation_data['stack_temp_F']:.1f}°F")
        
        return operation_data
        
    except Exception as e:
        print(f"   ❌ Single hour generation failed: {e}")
        traceback.print_exc()
        return None

def test_load_calculation():
    """Test the load factor calculation specifically."""
    print("\n🔍 TESTING LOAD FACTOR CALCULATION...")
    
    try:
        from annual_boiler_simulator import AnnualBoilerSimulator
        simulator = AnnualBoilerSimulator(start_date="2024-01-01")
        
        current_datetime = simulator.start_date
        hour = current_datetime.hour
        day_of_year = current_datetime.timetuple().tm_yday
        
        print(f"   Testing: {current_datetime}, hour={hour}, day_of_year={day_of_year}")
        
        # Test the new containerboard load calculation
        load_factor = simulator._calculate_load_factor_containerboard(current_datetime, hour, day_of_year)
        print(f"   ✅ Load factor calculated: {load_factor:.3f}")
        
        # Test multiple hours to see variation
        print("   Testing 24-hour variation:")
        for test_hour in range(0, 24, 4):
            test_datetime = current_datetime.replace(hour=test_hour)
            test_load = simulator._calculate_load_factor_containerboard(test_datetime, test_hour, day_of_year)
            print(f"     Hour {test_hour:2d}: {test_load:.3f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Load calculation failed: {e}")
        traceback.print_exc()
        return False

def test_boiler_system():
    """Test the underlying boiler system."""
    print("\n🔍 TESTING BOILER SYSTEM...")
    
    try:
        from boiler_system import EnhancedCompleteBoilerSystem
        
        boiler = EnhancedCompleteBoilerSystem(
            fuel_input=100e6,
            flue_gas_mass_flow=84000,
            furnace_exit_temp=2200,
            base_fouling_multiplier=0.5
        )
        print("   ✅ Boiler system initialized")
        
        # Test solving
        print("   Testing boiler system solve...")
        results = boiler.solve_enhanced_system(max_iterations=10, tolerance=15.0)
        print("   ✅ Boiler system solved")
        
        # Check performance
        perf = boiler.system_performance
        print(f"   Stack temperature: {perf.get('stack_temperature', 'Not found')}")
        print(f"   System efficiency: {perf.get('system_efficiency', 'Not found')}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Boiler system test failed: {e}")
        traceback.print_exc()
        return False

def run_mini_test():
    """Run a very small test (just 3 hours)."""
    print("\n🔍 RUNNING MINI TEST (3 hours)...")
    
    try:
        from annual_boiler_simulator import AnnualBoilerSimulator
        simulator = AnnualBoilerSimulator(start_date="2024-01-01")
        
        test_data = []
        current_date = simulator.start_date
        
        for hour in range(3):
            current_datetime = current_date + pd.Timedelta(hours=hour)
            print(f"   Processing hour {hour}: {current_datetime}")
            
            # Generate operating conditions
            operating_conditions = simulator._generate_hourly_conditions(current_datetime)
            print(f"     Load factor: {operating_conditions['load_factor']:.3f}")
            
            # Check soot blowing
            soot_blowing_actions = simulator._check_soot_blowing_schedule(current_datetime)
            
            # Simulate operation
            operation_data = simulator._simulate_boiler_operation(
                current_datetime, operating_conditions, soot_blowing_actions
            )
            
            test_data.append(operation_data)
            print(f"     Stack temp: {operation_data['stack_temp_F']:.1f}°F")
        
        # Analyze mini results
        df = pd.DataFrame(test_data)
        print(f"\n   ✅ Mini test completed!")
        print(f"   Stack temp range: {df['stack_temp_F'].min():.1f} - {df['stack_temp_F'].max():.1f}°F")
        print(f"   Stack temp std: {df['stack_temp_F'].std():.1f}°F")
        print(f"   Load range: {df['load_factor'].min():.3f} - {df['load_factor'].max():.3f}")
        
        return df
        
    except Exception as e:
        print(f"   ❌ Mini test failed: {e}")
        traceback.print_exc()
        return None

def main():
    """Run all diagnostic tests."""
    print("🔧" * 50)
    print("SIMULATION DEBUG SUITE")
    print("🔧" * 50)
    
    # Test 1: Imports
    if not test_imports():
        print("\n❌ Import test failed. Cannot continue.")
        return
    
    # Test 2: Initialization
    simulator = test_initialization()
    if simulator is None:
        print("\n❌ Initialization test failed. Cannot continue.")
        return
    
    # Test 3: Load calculation
    if not test_load_calculation():
        print("\n❌ Load calculation test failed.")
        return
    
    # Test 4: Boiler system
    if not test_boiler_system():
        print("\n❌ Boiler system test failed.")
        return
    
    # Test 5: Single hour
    single_result = test_single_hour()
    if single_result is None:
        print("\n❌ Single hour test failed.")
        return
    
    # Test 6: Mini test
    mini_result = run_mini_test()
    if mini_result is None:
        print("\n❌ Mini test failed.")
        return
    
    print("\n" + "✅" * 50)
    print("ALL DIAGNOSTIC TESTS PASSED!")
    print("✅" * 50)
    
    print(f"\nThe simulation appears to be working. The issue might be in the validation criteria.")
    print(f"Mini test results:")
    print(f"  Stack temp std dev: {mini_result['stack_temp_F'].std():.1f}°F")
    print(f"  Load factor range: {mini_result['load_factor'].min():.1%} - {mini_result['load_factor'].max():.1%}")
    
    return mini_result

if __name__ == "__main__":
    main()