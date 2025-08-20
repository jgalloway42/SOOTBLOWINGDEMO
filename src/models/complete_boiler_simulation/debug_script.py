#!/usr/bin/env python3
"""
Enhanced Debug Script - Interface Compatibility Testing

This script provides comprehensive testing and validation for the fixed boiler simulation:
- Tests the fixed solver interface compatibility
- Validates that the annual simulator works with new structure
- Checks for Unicode logging issues
- Provides detailed diagnostics and validation

CRITICAL TESTS:
- Solver return structure validation
- Annual simulator interface compatibility
- Unicode-safe logging verification
- Energy balance and convergence testing

Author: Enhanced Boiler Modeling System
Version: 8.1 - Interface Compatibility Testing
"""

import sys
import traceback
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Ensure logs directory exists
log_dir = Path("logs/debug")
log_dir.mkdir(parents=True, exist_ok=True)

# Configure logging for debug script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "debug_script.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def print_test_step(step: str, description: str):
    """Print a formatted test step."""
    print(f"\n[TEST] {step}: {description}")

def print_result(success: bool, message: str):
    """Print test result with clear formatting."""
    status = "[PASS]" if success else "[FAIL]"
    print(f"  {status} {message}")

def test_imports():
    """Test that all required modules can be imported."""
    print_test_step("1", "Testing module imports")
    
    import_results = {}
    
    # Test core modules
    modules_to_test = [
        ("numpy", "np"),
        ("pandas", "pd"),
        ("logging", None),
        ("datetime", None),
        ("pathlib", None)
    ]
    
    for module_name, alias in modules_to_test:
        try:
            if alias:
                exec(f"import {module_name} as {alias}")
            else:
                exec(f"import {module_name}")
            import_results[module_name] = True
            print_result(True, f"{module_name} imported successfully")
        except ImportError as e:
            import_results[module_name] = False
            print_result(False, f"{module_name} import failed: {e}")
    
    # Test project modules
    project_modules = [
        "boiler_system",
        "annual_boiler_simulator",
        "thermodynamic_properties",
        "fouling_and_soot_blowing",
        "heat_transfer_calculations"
    ]
    
    for module_name in project_modules:
        try:
            exec(f"import {module_name}")
            import_results[module_name] = True
            print_result(True, f"{module_name} imported successfully")
        except ImportError as e:
            import_results[module_name] = False
            print_result(False, f"{module_name} import failed: {e}")
    
    return import_results

def test_enhanced_boiler_system():
    """Test the enhanced boiler system with fixed interface."""
    print_test_step("2", "Testing Enhanced Boiler System")
    
    try:
        from boiler_system import EnhancedCompleteBoilerSystem
        
        # Initialize boiler system
        print("  Initializing enhanced boiler system...")
        boiler = EnhancedCompleteBoilerSystem(
            fuel_input=100e6,
            flue_gas_mass_flow=84000,
            furnace_exit_temp=3000,
            base_fouling_multiplier=1.0
        )
        print_result(True, "Enhanced boiler system initialized")
        
        # Test solver interface
        print("  Testing solver interface...")
        results = boiler.solve_enhanced_system(max_iterations=10, tolerance=10.0)
        
        # Validate return structure
        expected_keys = ['converged', 'system_performance', 'section_results', 
                        'solver_iterations', 'energy_balance_error', 
                        'final_stack_temperature', 'final_steam_temperature', 'final_efficiency']
        
        missing_keys = [key for key in expected_keys if key not in results]
        
        if missing_keys:
            print_result(False, f"Missing keys in solver results: {missing_keys}")
            return False
        else:
            print_result(True, "All expected keys present in solver results")
        
        # Validate result values
        converged = results['converged']
        efficiency = results['final_efficiency']
        stack_temp = results['final_stack_temperature']
        steam_temp = results['final_steam_temperature']
        
        print(f"    Converged: {converged}")
        print(f"    Efficiency: {efficiency:.1%}")
        print(f"    Stack Temperature: {stack_temp:.0f}°F")
        print(f"    Steam Temperature: {steam_temp:.0f}°F")
        
        # Validate reasonable values
        efficiency_ok = 0.70 <= efficiency <= 0.95
        stack_temp_ok = 200 <= stack_temp <= 500
        steam_temp_ok = 600 <= steam_temp <= 800
        
        print_result(efficiency_ok, f"Efficiency {efficiency:.1%} in reasonable range (70-95%)")
        print_result(stack_temp_ok, f"Stack temperature {stack_temp:.0f}°F in reasonable range (200-500°F)")
        print_result(steam_temp_ok, f"Steam temperature {steam_temp:.0f}°F in reasonable range (600-800°F)")
        
        return efficiency_ok and stack_temp_ok and steam_temp_ok
        
    except Exception as e:
        print_result(False, f"Enhanced boiler system test failed: {e}")
        traceback.print_exc()
        return False

def test_annual_simulator_interface():
    """Test the annual simulator with fixed interface."""
    print_test_step("3", "Testing Annual Simulator Interface")
    
    try:
        from annual_boiler_simulator import AnnualBoilerSimulator
        
        # Initialize simulator
        print("  Initializing annual simulator...")
        simulator = AnnualBoilerSimulator(start_date="2024-01-01")
        print_result(True, "Annual simulator initialized")
        
        # Test operating conditions generation
        print("  Testing operating conditions generation...")
        test_datetime = simulator.start_date
        operating_conditions = simulator._generate_hourly_conditions(test_datetime)
        
        required_conditions = ['load_factor', 'fuel_input_btu_hr', 'flue_gas_mass_flow', 
                             'furnace_exit_temp', 'ambient_temp_F', 'ambient_humidity_pct', 
                             'coal_quality', 'season']
        
        missing_conditions = [key for key in required_conditions if key not in operating_conditions]
        
        if missing_conditions:
            print_result(False, f"Missing operating conditions: {missing_conditions}")
            return False
        else:
            print_result(True, "All required operating conditions generated")
        
        print(f"    Load factor: {operating_conditions['load_factor']:.3f}")
        print(f"    Fuel input: {operating_conditions['fuel_input_btu_hr']/1e6:.1f} MMBtu/hr")
        print(f"    Coal quality: {operating_conditions['coal_quality']}")
        
        # Test soot blowing schedule
        print("  Testing soot blowing schedule...")
        soot_blowing_actions = simulator._check_soot_blowing_schedule(test_datetime)
        print_result(True, f"Soot blowing schedule checked for {len(soot_blowing_actions)} sections")
        
        # Test complete operation simulation
        print("  Testing complete operation simulation...")
        operation_data = simulator._simulate_boiler_operation(
            test_datetime, operating_conditions, soot_blowing_actions
        )
        
        # Validate operation data structure
        required_data_keys = ['timestamp', 'system_efficiency', 'stack_temp_F', 
                             'final_steam_temp_F', 'solution_converged', 'load_factor']
        
        missing_data_keys = [key for key in required_data_keys if key not in operation_data]
        
        if missing_data_keys:
            print_result(False, f"Missing operation data keys: {missing_data_keys}")
            return False
        else:
            print_result(True, "All required operation data keys present")
        
        # Validate operation results
        converged = operation_data['solution_converged']
        efficiency = operation_data['system_efficiency']
        stack_temp = operation_data['stack_temp_F']
        
        print(f"    Solution converged: {converged}")
        print(f"    System efficiency: {efficiency:.1%}")
        print(f"    Stack temperature: {stack_temp:.0f}°F")
        
        # Test should pass regardless of convergence, but values should be reasonable
        values_reasonable = (0.70 <= efficiency <= 0.95 and 200 <= stack_temp <= 500)
        print_result(values_reasonable, "Operation simulation returned reasonable values")
        
        return values_reasonable
        
    except Exception as e:
        print_result(False, f"Annual simulator interface test failed: {e}")
        traceback.print_exc()
        return False

def test_unicode_logging():
    """Test that logging works without Unicode errors."""
    print_test_step("4", "Testing Unicode-Safe Logging")
    
    try:
        # Test logging with various characters
        test_messages = [
            "Basic ASCII message",
            "Testing logging without Unicode check marks",
            "System efficiency: 85.5%",
            "Temperature: 280 degrees F",
            "[OK] Status message",
            "[FAIL] Error message",
            "Energy balance within acceptable limits"
        ]
        
        logger.info("Testing Unicode-safe logging messages...")
        
        for i, message in enumerate(test_messages):
            try:
                logger.info(f"Test message {i+1}: {message}")
                print_result(True, f"Message {i+1} logged successfully")
            except UnicodeEncodeError as e:
                print_result(False, f"Unicode error in message {i+1}: {e}")
                return False
        
        print_result(True, "All logging messages processed without Unicode errors")
        return True
        
    except Exception as e:
        print_result(False, f"Unicode logging test failed: {e}")
        return False

def test_load_variation():
    """Test that the system produces varying results under different loads."""
    print_test_step("5", "Testing Load-Dependent Variation")
    
    try:
        from boiler_system import EnhancedCompleteBoilerSystem
        
        # Test different load conditions
        load_conditions = [
            (50e6, "50% Load"),
            (75e6, "75% Load"),
            (100e6, "100% Load")
        ]
        
        results = []
        
        for fuel_input, description in load_conditions:
            print(f"  Testing {description}...")
            
            boiler = EnhancedCompleteBoilerSystem(
                fuel_input=fuel_input,
                flue_gas_mass_flow=int(84000 * fuel_input / 100e6),
                furnace_exit_temp=2900,
                base_fouling_multiplier=1.0
            )
            
            solve_results = boiler.solve_enhanced_system(max_iterations=15, tolerance=10.0)
            
            efficiency = solve_results['final_efficiency']
            stack_temp = solve_results['final_stack_temperature']
            converged = solve_results['converged']
            
            results.append({
                'load': fuel_input/1e6,
                'efficiency': efficiency,
                'stack_temp': stack_temp,
                'converged': converged
            })
            
            print(f"    {description}: Eff={efficiency:.1%}, Stack={stack_temp:.0f}°F, Conv={converged}")
        
        # Check for variation in results
        efficiencies = [r['efficiency'] for r in results]
        stack_temps = [r['stack_temp'] for r in results]
        
        efficiency_range = max(efficiencies) - min(efficiencies)
        stack_temp_range = max(stack_temps) - min(stack_temps)
        
        variation_ok = efficiency_range > 0.01 or stack_temp_range > 10  # Should see some variation
        
        print(f"    Efficiency range: {efficiency_range:.2%}")
        print(f"    Stack temperature range: {stack_temp_range:.0f}°F")
        
        print_result(variation_ok, f"System shows load-dependent variation")
        
        return variation_ok
        
    except Exception as e:
        print_result(False, f"Load variation test failed: {e}")
        traceback.print_exc()
        return False

def test_short_simulation():
    """Test a short simulation run to verify end-to-end functionality."""
    print_test_step("6", "Testing Short Simulation Run (24 hours)")
    
    try:
        from annual_boiler_simulator import AnnualBoilerSimulator
        
        # Create simulator for short run
        simulator = AnnualBoilerSimulator(start_date="2024-01-01")
        
        # Manually generate 24 hours of data
        print("  Generating 24 hours of simulation data...")
        simulation_data = []
        
        for hour in range(24):
            test_datetime = simulator.start_date + timedelta(hours=hour)
            
            try:
                # Generate conditions and simulate
                operating_conditions = simulator._generate_hourly_conditions(test_datetime)
                soot_blowing_actions = simulator._check_soot_blowing_schedule(test_datetime)
                operation_data = simulator._simulate_boiler_operation(
                    test_datetime, operating_conditions, soot_blowing_actions
                )
                
                simulation_data.append(operation_data)
                
            except Exception as e:
                print_result(False, f"Hour {hour} simulation failed: {e}")
                return False
        
        # Convert to DataFrame and analyze
        df = pd.DataFrame(simulation_data)
        
        print(f"    Generated {len(df)} hourly records")
        print(f"    Columns: {len(df.columns)}")
        
        # Check for data quality
        efficiency_mean = df['system_efficiency'].mean()
        efficiency_std = df['system_efficiency'].std()
        stack_temp_mean = df['stack_temp_F'].mean()
        stack_temp_std = df['stack_temp_F'].std()
        converged_count = df['solution_converged'].sum()
        
        print(f"    Average efficiency: {efficiency_mean:.1%}")
        print(f"    Efficiency std dev: {efficiency_std:.2%}")
        print(f"    Average stack temp: {stack_temp_mean:.0f}°F")
        print(f"    Stack temp std dev: {stack_temp_std:.1f}°F")
        print(f"    Solutions converged: {converged_count}/{len(df)}")
        
        # Validate results
        efficiency_ok = 0.70 <= efficiency_mean <= 0.95
        no_nulls = df.isnull().sum().sum() == 0
        reasonable_variation = efficiency_std > 0.001 or stack_temp_std > 1.0
        
        print_result(efficiency_ok, f"Average efficiency {efficiency_mean:.1%} in reasonable range")
        print_result(no_nulls, "No null values in simulation data")
        print_result(reasonable_variation, "Simulation shows reasonable variation")
        
        return efficiency_ok and no_nulls
        
    except Exception as e:
        print_result(False, f"Short simulation test failed: {e}")
        traceback.print_exc()
        return False

def test_solver_convergence():
    """Test solver convergence behavior and error handling."""
    print_test_step("7", "Testing Solver Convergence and Error Handling")
    
    try:
        from boiler_system import EnhancedCompleteBoilerSystem
        
        # Test normal convergence
        print("  Testing normal convergence...")
        boiler = EnhancedCompleteBoilerSystem(fuel_input=100e6)
        results = boiler.solve_enhanced_system(max_iterations=20, tolerance=5.0)
        
        normal_converged = results['converged']
        print_result(normal_converged, f"Normal conditions converged: {normal_converged}")
        
        # Test challenging convergence (tight tolerance)
        print("  Testing challenging convergence...")
        results_tight = boiler.solve_enhanced_system(max_iterations=10, tolerance=1.0)
        
        tight_result = 'converged' in results_tight  # Should at least return proper structure
        print_result(tight_result, "Tight tolerance returned proper structure")
        
        # Test extreme conditions
        print("  Testing extreme conditions...")
        extreme_boiler = EnhancedCompleteBoilerSystem(
            fuel_input=150e6,  # High load
            flue_gas_mass_flow=120000,
            furnace_exit_temp=3500,  # High temperature
            base_fouling_multiplier=2.0  # Heavy fouling
        )
        
        extreme_results = extreme_boiler.solve_enhanced_system(max_iterations=5, tolerance=20.0)
        extreme_structure_ok = 'converged' in extreme_results
        
        print_result(extreme_structure_ok, "Extreme conditions returned proper structure")
        
        # Check that all results have consistent structure
        all_results = [results, results_tight, extreme_results]
        expected_keys = ['converged', 'final_efficiency', 'final_stack_temperature']
        
        structure_consistent = True
        for i, result in enumerate(all_results):
            missing = [key for key in expected_keys if key not in result]
            if missing:
                print_result(False, f"Result {i+1} missing keys: {missing}")
                structure_consistent = False
        
        if structure_consistent:
            print_result(True, "All solver results have consistent structure")
        
        return structure_consistent
        
    except Exception as e:
        print_result(False, f"Solver convergence test failed: {e}")
        traceback.print_exc()
        return False

def run_validation_suite():
    """Run complete validation suite."""
    print_header("ENHANCED BOILER SIMULATION VALIDATION SUITE")
    print("Version 8.1 - Interface Compatibility Testing")
    print(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Track test results
    test_results = {}
    
    # Run all tests
    test_functions = [
        ("Module Imports", test_imports),
        ("Enhanced Boiler System", test_enhanced_boiler_system),
        ("Annual Simulator Interface", test_annual_simulator_interface),
        ("Unicode-Safe Logging", test_unicode_logging),
        ("Load-Dependent Variation", test_load_variation),
        ("Short Simulation Run", test_short_simulation),
        ("Solver Convergence", test_solver_convergence)
    ]
    
    for test_name, test_function in test_functions:
        try:
            result = test_function()
            test_results[test_name] = result
        except Exception as e:
            print_result(False, f"{test_name} test crashed: {e}")
            test_results[test_name] = False
    
    # Summary
    print_header("VALIDATION RESULTS SUMMARY")
    
    passed_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {status:4s} | {test_name}")
    
    print(f"\nOVERALL RESULT: {passed_tests}/{total_tests} tests passed")
    
    # Determine overall status
    critical_tests = [
        "Enhanced Boiler System",
        "Annual Simulator Interface", 
        "Unicode-Safe Logging"
    ]
    
    critical_passed = all(test_results.get(test, False) for test in critical_tests)
    
    if critical_passed and passed_tests >= total_tests - 1:  # Allow 1 non-critical failure
        print("[SUCCESS] Validation suite passed - System ready for use")
        overall_success = True
    else:
        print("[WARNING] Validation suite failed - Review failed tests")
        overall_success = False
    
    # Save validation report
    save_validation_report(test_results, overall_success)
    
    return overall_success

def save_validation_report(test_results: dict, overall_success: bool):
    """Save detailed validation report."""
    
    report_file = log_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(report_file, 'w') as f:
        f.write("ENHANCED BOILER SIMULATION VALIDATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Validation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"System Version: 8.1 - Interface Compatibility Fix\n")
        f.write(f"Overall Result: {'PASS' if overall_success else 'FAIL'}\n\n")
        
        f.write("TEST RESULTS:\n")
        f.write("-" * 30 + "\n")
        
        for test_name, result in test_results.items():
            status = "PASS" if result else "FAIL"
            f.write(f"{status:4s} | {test_name}\n")
        
        passed = sum(1 for r in test_results.values() if r)
        total = len(test_results)
        f.write(f"\nSummary: {passed}/{total} tests passed\n\n")
        
        if overall_success:
            f.write("CONCLUSION:\n")
            f.write("The enhanced boiler simulation system has passed validation.\n")
            f.write("The interface compatibility fixes have resolved the solver issues.\n")
            f.write("The system is ready for annual simulation runs.\n")
        else:
            f.write("ISSUES IDENTIFIED:\n")
            failed_tests = [name for name, result in test_results.items() if not result]
            for test in failed_tests:
                f.write(f"- {test}\n")
            f.write("\nRecommendation: Address failed tests before production use.\n")
    
    print(f"\nValidation report saved: {report_file}")

def quick_interface_test():
    """Quick test specifically for the interface fix."""
    print_header("QUICK INTERFACE COMPATIBILITY TEST")
    
    try:
        # Test the specific fix
        from boiler_system import EnhancedCompleteBoilerSystem
        from annual_boiler_simulator import AnnualBoilerSimulator
        
        print("Testing solver return structure...")
        boiler = EnhancedCompleteBoilerSystem(fuel_input=100e6)
        results = boiler.solve_enhanced_system(max_iterations=5, tolerance=15.0)
        
        # This should not raise KeyError anymore
        converged = results['converged']  # This was the failing line
        efficiency = results['final_efficiency']
        stack_temp = results['final_stack_temperature']
        
        print(f"[OK] Solver returned: converged={converged}, eff={efficiency:.1%}, stack={stack_temp:.0f}°F")
        
        print("Testing annual simulator interface...")
        simulator = AnnualBoilerSimulator()
        test_datetime = simulator.start_date
        operating_conditions = simulator._generate_hourly_conditions(test_datetime)
        soot_blowing_actions = simulator._check_soot_blowing_schedule(test_datetime)
        
        # This should not raise KeyError anymore
        operation_data = simulator._simulate_boiler_operation(
            test_datetime, operating_conditions, soot_blowing_actions
        )
        
        print(f"[OK] Annual simulator returned: eff={operation_data['system_efficiency']:.1%}")
        
        print("\n[SUCCESS] Interface compatibility fix verified!")
        return True
        
    except KeyError as e:
        print(f"[FAIL] KeyError still occurs: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] Other error: {e}")
        return False

if __name__ == "__main__":
    # Run quick test first
    quick_success = quick_interface_test()
    
    if quick_success:
        # Run full validation suite
        full_success = run_validation_suite()
        
        if full_success:
            print("\n" + "="*70)
            print("READY FOR ANNUAL SIMULATION")
            print("="*70)
            print("The enhanced boiler simulation system is ready for use.")
            print("You can now run annual simulations without KeyError failures.")
        else:
            print("\nSome non-critical tests failed. Review the validation report.")
    else:
        print("\nCritical interface issue still exists. Check the implementation.")
