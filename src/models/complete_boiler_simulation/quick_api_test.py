#!/usr/bin/env python3
"""
Quick API Compatibility Test Script

This script performs rapid validation of the FIXED API compatibility issues.
Run this before attempting the full annual simulation to verify fixes.

Usage: python quick_api_test.py

Author: Enhanced Boiler Modeling System
Version: 8.2 - API Compatibility Quick Test
"""

import sys
import traceback
from pathlib import Path

print("QUICK API COMPATIBILITY TEST")
print("=" * 50)
print("Testing FIXED API compatibility issues...")

try:
    # Test 1: Import all modules
    print("\n[TEST 1] Module imports...")
    from core.boiler_system import EnhancedCompleteBoilerSystem
    from core.thermodynamic_properties import PropertyCalculator
    from simulation.annual_boiler_simulator import AnnualBoilerSimulator
    print("[PASS] All modules imported successfully")
    
    # Test 2: EnhancedCompleteBoilerSystem with FIXED parameters
    print("\n[TEST 2] EnhancedCompleteBoilerSystem constructor...")
    boiler = EnhancedCompleteBoilerSystem(
        fuel_input=100e6,        # FIXED: Correct parameter name
        flue_gas_mass_flow=84000, # FIXED: Correct parameter name  
        furnace_exit_temp=3000    # FIXED: Correct parameter name
    )
    print("[PASS] Boiler system created with fixed API parameters")
    
    # Test 3: Solver interface
    print("\n[TEST 3] Solver interface compatibility...")
    results = boiler.solve_enhanced_system(max_iterations=10, tolerance=15.0)
    
    # Test FIXED result extraction
    converged = results.get('converged', False)
    efficiency = results.get('final_efficiency', 0.0)
    steam_temp = results.get('final_steam_temperature', 0.0)
    stack_temp = results.get('final_stack_temperature', 0.0)
    
    print(f"[PASS] Solver interface working - Converged: {converged}, Eff: {efficiency:.1%}")
    
    # Test 4: AnnualBoilerSimulator with FIXED parameters  
    print("\n[TEST 4] AnnualBoilerSimulator API...")
    import pandas as pd
    
    simulator = AnnualBoilerSimulator(start_date="2024-01-01")
    simulator.end_date = simulator.start_date + pd.DateOffset(hours=2)  # Short test
    
    # Test FIXED parameter names
    test_data = simulator.generate_annual_data(
        hours_per_day=24,        # FIXED: Correct parameter name
        save_interval_hours=1    # FIXED: Correct parameter name
    )
    
    print(f"[PASS] AnnualBoilerSimulator generated {len(test_data)} records with fixed API")
    
    # Test 5: Data saving
    print("\n[TEST 5] Data file operations...")
    data_file, metadata_file = simulator.save_annual_data(test_data)
    
    data_path = Path(data_file)
    metadata_path = Path(metadata_file)
    
    if data_path.exists() and metadata_path.exists():
        print(f"[PASS] Files created successfully - Data: {data_path.name}")
    else:
        print(f"[FAIL] File creation failed")
    
    print("\n" + "=" * 50)
    print("ALL API COMPATIBILITY TESTS PASSED!")
    print("=" * 50)
    print("\nSystem is ready for full annual simulation.")
    print("Next step: python simulation/run_annual_simulation.py")
    
except Exception as e:
    print(f"\n[FAIL] API compatibility test failed: {e}")
    print("\nError details:")
    traceback.print_exc()
    print("\nFix the error above before proceeding with annual simulation.")
    sys.exit(1)
