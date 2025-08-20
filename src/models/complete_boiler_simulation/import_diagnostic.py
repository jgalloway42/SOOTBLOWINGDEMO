#!/usr/bin/env python3
"""
Import Diagnostic Script - 1-Hour Fix Sprint
This script systematically tests all imports to identify exactly what's failing.
Creates a detailed log file for analysis.
"""

import sys
import traceback
import datetime
from pathlib import Path

# Create logs directory
log_dir = Path("logs/debug")
log_dir.mkdir(parents=True, exist_ok=True)

# Create log file
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"import_diagnostic_{timestamp}.log"

def log_and_print(message):
    """Write to both console and log file."""
    # Remove emojis for Windows compatibility
    clean_message = message.encode('ascii', 'ignore').decode('ascii')
    print(clean_message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(clean_message + '\n')

def test_single_import(module_name, description=""):
    """Test a single import and return success/failure."""
    log_and_print(f"\n{'='*60}")
    log_and_print(f"TESTING: {module_name}")
    if description:
        log_and_print(f"PURPOSE: {description}")
    log_and_print(f"{'='*60}")
    
    try:
        # Dynamic import
        exec(f"import {module_name}")
        log_and_print(f"SUCCESS: {module_name} imported successfully")
        return True
    except Exception as e:
        log_and_print(f"FAILED: {module_name}")
        log_and_print(f"ERROR: {str(e)}")
        log_and_print(f"TRACEBACK:")
        tb_lines = traceback.format_exc().split('\n')
        for line in tb_lines:
            log_and_print(f"  {line}")
        return False

def test_specific_class_import(import_statement, description=""):
    """Test importing a specific class or function."""
    log_and_print(f"\n{'='*60}")
    log_and_print(f"TESTING: {import_statement}")
    if description:
        log_and_print(f"PURPOSE: {description}")
    log_and_print(f"{'='*60}")
    
    try:
        exec(import_statement)
        log_and_print(f"SUCCESS: {import_statement}")
        return True
    except Exception as e:
        log_and_print(f"FAILED: {import_statement}")
        log_and_print(f"ERROR: {str(e)}")
        log_and_print(f"TRACEBACK:")
        tb_lines = traceback.format_exc().split('\n')
        for line in tb_lines:
            log_and_print(f"  {line}")
        return False

def test_basic_functionality():
    """Test if we can create basic objects."""
    log_and_print(f"\n{'='*60}")
    log_and_print(f"TESTING: BASIC FUNCTIONALITY")
    log_and_print(f"{'='*60}")
    
    try:
        from annual_boiler_simulator import AnnualBoilerSimulator
        simulator = AnnualBoilerSimulator()
        log_and_print(f"SUCCESS: AnnualBoilerSimulator created")
        
        # Try to access some basic attributes
        log_and_print(f"  Start date: {simulator.start_date}")
        log_and_print(f"  End date: {simulator.end_date}")
        log_and_print(f"  Coal quality profiles: {len(simulator.coal_quality_profiles)} types")
        
        return True
    except Exception as e:
        log_and_print(f"FAILED: AnnualBoilerSimulator creation")
        log_and_print(f"ERROR: {str(e)}")
        log_and_print(f"TRACEBACK:")
        tb_lines = traceback.format_exc().split('\n')
        for line in tb_lines:
            log_and_print(f"  {line}")
        return False

def main():
    """Run complete import diagnostic."""
    log_and_print(f"BOILER SIMULATION IMPORT DIAGNOSTIC")
    log_and_print(f"Started: {datetime.datetime.now()}")
    log_and_print(f"Python version: {sys.version}")
    log_and_print(f"Working directory: {Path.cwd()}")
    log_and_print(f"Log file: {log_file}")
    
    # Test results tracking
    results = {}
    
    # Phase 1: Test core module imports
    log_and_print(f"\nPHASE 1: TESTING CORE MODULE IMPORTS")
    log_and_print(f"{'='*60}")
    
    core_modules = [
        ("numpy", "Mathematical operations"),
        ("pandas", "Data handling"),
        ("datetime", "Time handling"),
        ("pathlib", "File paths"),
        ("logging", "Logging system")
    ]
    
    for module, desc in core_modules:
        results[f"core_{module}"] = test_single_import(module, desc)
    
    # Phase 2: Test thermodynamic libraries
    log_and_print(f"\nPHASE 2: TESTING THERMODYNAMIC LIBRARIES")
    log_and_print(f"{'='*60}")
    
    thermo_modules = [
        ("iapws", "IAPWS steam properties"),
        ("thermo", "Fluid properties")
    ]
    
    for module, desc in thermo_modules:
        results[f"thermo_{module}"] = test_single_import(module, desc)
    
    # Phase 3: Test project modules (bottom-up dependency order)
    log_and_print(f"\nPHASE 3: TESTING PROJECT MODULES (DEPENDENCY ORDER)")
    log_and_print(f"{'='*60}")
    
    project_modules = [
        ("thermodynamic_properties", "Property calculations"),
        ("fouling_and_soot_blowing", "Fouling and cleaning models"),
        ("heat_transfer_calculations", "Heat transfer calculations"),
        ("coal_combustion_models", "Combustion modeling"),
        ("boiler_system", "Complete boiler system"),
        ("annual_boiler_simulator", "Annual simulation"),
        ("data_analysis_tools", "Data analysis"),
        ("analysis_and_visualization", "Visualization tools")
    ]
    
    for module, desc in project_modules:
        results[f"project_{module}"] = test_single_import(module, desc)
    
    # Phase 4: Test specific class imports
    log_and_print(f"\nPHASE 4: TESTING SPECIFIC CLASS IMPORTS")
    log_and_print(f"{'='*60}")
    
    class_imports = [
        ("from thermodynamic_properties import PropertyCalculator", "Steam property calculator"),
        ("from boiler_system import EnhancedCompleteBoilerSystem", "Enhanced boiler system"),
        ("from annual_boiler_simulator import AnnualBoilerSimulator", "Annual simulator"),
        ("from data_analysis_tools import BoilerDataAnalyzer", "Data analyzer"),
        ("from fouling_and_soot_blowing import BoilerSection", "Boiler section"),
        ("from heat_transfer_calculations import HeatTransferCalculator", "Heat transfer calculator"),
        ("from coal_combustion_models import CoalCombustionModel", "Combustion model")
    ]
    
    for import_stmt, desc in class_imports:
        class_name = import_stmt.split()[-1]
        results[f"class_{class_name}"] = test_specific_class_import(import_stmt, desc)
    
    # Phase 5: Test basic functionality
    log_and_print(f"\nPHASE 5: TESTING BASIC FUNCTIONALITY")
    log_and_print(f"{'='*60}")
    
    results["functionality"] = test_basic_functionality()
    
    # Phase 6: Test run_annual_simulation.py import chain
    log_and_print(f"\nPHASE 6: TESTING RUN_ANNUAL_SIMULATION.PY")
    log_and_print(f"{'='*60}")
    
    try:
        exec("import run_annual_simulation")
        log_and_print(f"SUCCESS: run_annual_simulation.py imports work")
        results["run_script"] = True
    except Exception as e:
        log_and_print(f"FAILED: run_annual_simulation.py")
        log_and_print(f"ERROR: {str(e)}")
        log_and_print(f"TRACEBACK:")
        tb_lines = traceback.format_exc().split('\n')
        for line in tb_lines:
            log_and_print(f"  {line}")
        results["run_script"] = False
    
    # Summary
    log_and_print(f"\n{'='*60}")
    log_and_print(f"DIAGNOSTIC SUMMARY")
    log_and_print(f"{'='*60}")
    
    total_tests = len(results)
    successful_tests = sum(results.values())
    failed_tests = total_tests - successful_tests
    
    log_and_print(f"Total tests: {total_tests}")
    log_and_print(f"Successful: {successful_tests}")
    log_and_print(f"Failed: {failed_tests}")
    log_and_print(f"Success rate: {successful_tests/total_tests*100:.1f}%")
    
    log_and_print(f"\nFAILED TESTS:")
    for test_name, success in results.items():
        if not success:
            log_and_print(f"  X {test_name}")
    
    log_and_print(f"\nSUCCESSFUL TESTS:")
    for test_name, success in results.items():
        if success:
            log_and_print(f"  OK {test_name}")
    
    # Next steps
    log_and_print(f"\n{'='*60}")
    log_and_print(f"RECOMMENDED NEXT STEPS")
    log_and_print(f"{'='*60}")
    
    if failed_tests == 0:
        log_and_print(f"ALL TESTS PASSED! No import issues detected.")
        log_and_print(f"   The problem may be in the simulation logic, not imports.")
    elif not results.get("thermo_iapws", True):
        log_and_print(f"CRITICAL: IAPWS library missing")
        log_and_print(f"   Run: pip install iapws")
    elif not results.get("project_thermodynamic_properties", True):
        log_and_print(f"CRITICAL: thermodynamic_properties module failing")
        log_and_print(f"   This is likely the root cause of other failures")
    elif not results.get("project_boiler_system", True):
        log_and_print(f"CRITICAL: boiler_system module failing")
        log_and_print(f"   Check heat_transfer_calculations and fouling_and_soot_blowing dependencies")
    elif not results.get("class_BoilerDataAnalyzer", True):
        log_and_print(f"LIKELY CULPRIT: data_analysis_tools module")
        log_and_print(f"   This module likely expects old SystemPerformance interface")
    else:
        log_and_print(f"MIXED RESULTS: Focus on the first failing test above")
    
    log_and_print(f"\nCompleted: {datetime.datetime.now()}")
    log_and_print(f"Log saved to: {log_file}")
    
    print(f"\nQUICK REFERENCE:")
    print(f"   Log file: {log_file}")
    print(f"   Success rate: {successful_tests/total_tests*100:.1f}%")
    print(f"   Failed tests: {failed_tests}")
    
    return results

if __name__ == "__main__":
    results = main()
