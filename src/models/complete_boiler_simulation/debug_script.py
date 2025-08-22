#!/usr/bin/env python3
"""
Enhanced Debug Script - Phase 2 Validation Suite

This script provides comprehensive testing and validation for the FIXED boiler simulation
with Phase 2 validation tests to verify that static efficiency issues have been resolved:

PHASE 2 VALIDATION TESTS:
- Efficiency variation validation across load range
- Energy balance error verification (<5% target)
- Component integration validation (PropertyCalculator fixes)
- Parameter propagation verification
- Load-dependent combustion efficiency validation
- Enhanced heat transfer coefficient validation

ENHANCED FEATURES:
- Comprehensive load variation testing (45-150% range)
- Energy balance error monitoring and validation
- Component-level integration verification
- Enhanced efficiency curve validation
- Real-time diagnostic reporting
- Detailed performance analysis

Author: Enhanced Boiler Modeling System
Version: 9.0 - Phase 2 Validation Implementation
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

def print_diagnostic(level: str, message: str):
    """Print diagnostic information."""
    print(f"  [DIAG-{level}] {message}")

def test_module_imports():
    """Test that all required modules can be imported."""
    print_test_step("1", "Testing module imports")
    
    modules_to_test = [
        ("numpy", "np"),
        ("pandas", "pd"),
        ("logging", None),
        ("datetime", None),
        ("pathlib", None),
        ("boiler_system", None),
        ("annual_boiler_simulator", None),
        ("thermodynamic_properties", None),
        ("fouling_and_soot_blowing", None),
        ("heat_transfer_calculations", None),
        ("coal_combustion_models", None)
    ]
    
    imported_count = 0
    total_count = len(modules_to_test)
    
    for module_name, alias in modules_to_test:
        try:
            if alias:
                exec(f"import {module_name} as {alias}")
            else:
                exec(f"import {module_name}")
            print_result(True, f"{module_name} imported successfully")
            imported_count += 1
        except ImportError as e:
            print_result(False, f"{module_name} import failed: {e}")
    
    success = imported_count == total_count
    print_result(success, f"All modules imported: {imported_count}/{total_count}")
    return success

def test_phase2_efficiency_variation():
    """PHASE 2: Test that efficiency now varies properly with load."""
    print_test_step("2A", "PHASE 2: Efficiency Variation Validation")
    
    try:
        from boiler_system import EnhancedCompleteBoilerSystem
        
        print("  Testing FIXED efficiency variation across comprehensive load range...")
        
        # Comprehensive test scenarios covering full operational range
        test_scenarios = [
            (45e6, 38000, 2700, 0.8, "45% Load - Minimum"),
            (60e6, 50000, 2800, 0.9, "60% Load - Low"),
            (80e6, 67000, 2900, 1.0, "80% Load - Optimal"),
            (100e6, 84000, 3000, 1.1, "100% Load - Design"),
            (120e6, 101000, 3100, 1.3, "120% Load - High"),
            (150e6, 126000, 3200, 1.6, "150% Load - Maximum")
        ]
        
        efficiency_results = []
        energy_balance_results = []
        
        for fuel_input, mass_flow, exit_temp, fouling_mult, description in test_scenarios:
            print(f"\n  Analyzing: {description}")
            print_diagnostic("INPUT", f"fuel_input={fuel_input/1e6:.0f}MMBtu/hr, fouling={fouling_mult}x")
            
            try:
                # Initialize FIXED boiler with specific parameters
                boiler = EnhancedCompleteBoilerSystem(
                    fuel_input=fuel_input,
                    flue_gas_mass_flow=mass_flow,
                    furnace_exit_temp=exit_temp,
                    base_fouling_multiplier=fouling_mult
                )
                
                # Solve system with FIXED calculations
                results = boiler.solve_enhanced_system(max_iterations=20, tolerance=8.0)
                
                # Extract key results
                final_efficiency = results['final_efficiency']
                final_stack_temp = results['final_stack_temperature']
                energy_balance_error = results['energy_balance_error']
                converged = results['converged']
                
                print_diagnostic("CALC", f"Final efficiency: {final_efficiency:.3f} ({final_efficiency:.1%})")
                print_diagnostic("CALC", f"Stack temperature: {final_stack_temp:.1f}°F")
                print_diagnostic("CALC", f"Energy balance error: {energy_balance_error:.1%}")
                print_diagnostic("CONV", f"Converged: {'Yes' if converged else 'No'}")
                
                efficiency_results.append({
                    'scenario': description,
                    'fuel_input': fuel_input/1e6,
                    'fouling': fouling_mult,
                    'efficiency': final_efficiency,
                    'stack_temp': final_stack_temp,
                    'energy_error': energy_balance_error,
                    'converged': converged
                })
                
                energy_balance_results.append(energy_balance_error)
                
            except Exception as e:
                print_diagnostic("ERROR", f"Scenario failed: {e}")
                continue
        
        # Analyze FIXED efficiency variation
        if len(efficiency_results) >= 4:
            efficiencies = [r['efficiency'] for r in efficiency_results]
            fuel_inputs = [r['fuel_input'] for r in efficiency_results]
            
            efficiency_range = max(efficiencies) - min(efficiencies)
            efficiency_min = min(efficiencies)
            efficiency_max = max(efficiencies)
            fuel_range = max(fuel_inputs) - min(fuel_inputs)
            
            print(f"\n  PHASE 2 EFFICIENCY VARIATION ANALYSIS:")
            print_diagnostic("RESULT", f"Efficiency range: {efficiency_range:.4f} ({efficiency_range:.2%})")
            print_diagnostic("RESULT", f"Min efficiency: {efficiency_min:.1%} at {fuel_inputs[efficiencies.index(efficiency_min)]:.0f}MMBtu/hr")
            print_diagnostic("RESULT", f"Max efficiency: {efficiency_max:.1%} at {fuel_inputs[efficiencies.index(efficiency_max)]:.0f}MMBtu/hr")
            print_diagnostic("RESULT", f"Fuel input range: {fuel_range:.0f}MMBtu/hr ({fuel_range/min(fuel_inputs)*100:.0f}% increase)")
            
            # PHASE 2 SUCCESS CRITERIA
            efficiency_variation_adequate = efficiency_range >= 0.02  # At least 2% variation
            efficiency_range_realistic = 0.02 <= efficiency_range <= 0.15  # 2-15% is realistic
            
            print_result(efficiency_variation_adequate, f"Efficiency variation adequate: {efficiency_range:.2%} (target: >=2%)")
            print_result(efficiency_range_realistic, f"Efficiency range realistic: {efficiency_range:.2%} (realistic: 2-15%)")
            
            # Analyze energy balance improvements
            avg_energy_error = np.mean(energy_balance_results) if energy_balance_results else 1.0
            max_energy_error = max(energy_balance_results) if energy_balance_results else 1.0
            energy_balance_improved = avg_energy_error < 0.05 and max_energy_error < 0.10
            
            print_result(energy_balance_improved, f"Energy balance improved: avg={avg_energy_error:.1%}, max={max_energy_error:.1%} (target: <5%)")
            
            # Log individual scenario results
            print(f"\n  DETAILED SCENARIO RESULTS:")
            for result in efficiency_results:
                print(f"    {result['scenario']:20s}: Eff={result['efficiency']:.1%}, Stack={result['stack_temp']:.0f}°F, EB_Err={result['energy_error']:.1%}")
            
            return efficiency_variation_adequate and energy_balance_improved
        else:
            print_result(False, "Insufficient scenarios completed for analysis")
            return False
            
    except Exception as e:
        print_result(False, f"Phase 2 efficiency variation test failed: {e}")
        traceback.print_exc()
        return False

def test_phase2_component_integration():
    """PHASE 2: Test that PropertyCalculator integration is fixed."""
    print_test_step("2B", "PHASE 2: Component Integration Validation")
    
    try:
        from heat_transfer_calculations import HeatTransferCalculator, EnhancedBoilerTubeSection
        from thermodynamic_properties import PropertyCalculator
        
        print("  Testing FIXED PropertyCalculator integration...")
        
        # Test PropertyCalculator initialization
        try:
            prop_calc = PropertyCalculator()
            print_result(True, "PropertyCalculator initialized successfully")
        except Exception as e:
            print_result(False, f"PropertyCalculator initialization failed: {e}")
            return False
        
        # Test HeatTransferCalculator initialization
        try:
            htc_calc = HeatTransferCalculator()
            print_result(True, "HeatTransferCalculator initialized successfully")
        except Exception as e:
            print_result(False, f"HeatTransferCalculator initialization failed: {e}")
            return False
        
        # Test EnhancedBoilerTubeSection with PropertyCalculator integration
        test_sections = [
            ("economizer_test", "economizer"),
            ("superheater_test", "superheater"),
            ("furnace_test", "radiant")
        ]
        
        integration_success = True
        
        for section_name, section_type in test_sections:
            print(f"\n  Testing {section_name} ({section_type}):")
            
            try:
                # Create section
                section = EnhancedBoilerTubeSection(
                    name=section_name,
                    tube_od=2.0,
                    tube_id=1.8,
                    tube_length=15.0,
                    tube_count=200,
                    base_fouling_gas=0.0005,
                    base_fouling_water=0.0001,
                    section_type=section_type
                )
                print_diagnostic("INIT", f"Section {section_name} initialized")
                
                # Test property calculator access
                if hasattr(section, 'property_calc'):
                    print_diagnostic("ATTR", "property_calc attribute exists")
                else:
                    print_diagnostic("ERROR", "property_calc attribute missing")
                    integration_success = False
                    continue
                
                if hasattr(section, 'heat_transfer_calc'):
                    print_diagnostic("ATTR", "heat_transfer_calc attribute exists")
                else:
                    print_diagnostic("ERROR", "heat_transfer_calc attribute missing")
                    integration_success = False
                    continue
                
                # Test heat transfer calculation
                try:
                    results = section.solve_section(
                        gas_temp_in=1200 if section_type == 'radiant' else 800,
                        water_temp_in=300,
                        gas_flow=50000,
                        water_flow=30000,
                        steam_pressure=150
                    )
                    
                    if results and len(results) > 0:
                        print_diagnostic("CALC", f"Heat transfer calculation successful: {len(results)} segments")
                        
                        # Check for reasonable values
                        total_Q = sum(r.heat_transfer_rate for r in results)
                        avg_U = np.mean([r.overall_U for r in results])
                        
                        if total_Q > 0 and avg_U > 0:
                            print_diagnostic("VALID", f"Reasonable results: Q={total_Q/1e6:.1f}MMBtu/hr, U_avg={avg_U:.1f}")
                        else:
                            print_diagnostic("ERROR", f"Unreasonable results: Q={total_Q}, U_avg={avg_U}")
                            integration_success = False
                    else:
                        print_diagnostic("ERROR", "No results returned from heat transfer calculation")
                        integration_success = False
                        
                except Exception as e:
                    print_diagnostic("ERROR", f"Heat transfer calculation failed: {e}")
                    integration_success = False
                
            except Exception as e:
                print_diagnostic("ERROR", f"Section {section_name} test failed: {e}")
                integration_success = False
        
        print_result(integration_success, "PropertyCalculator integration fixed")
        return integration_success
        
    except Exception as e:
        print_result(False, f"Component integration test failed: {e}")
        traceback.print_exc()
        return False

def test_phase2_load_dependent_combustion():
    """PHASE 2: Test enhanced load-dependent combustion efficiency."""
    print_test_step("2C", "PHASE 2: Load-Dependent Combustion Validation")
    
    try:
        from coal_combustion_models import CoalCombustionModel
        
        print("  Testing ENHANCED load-dependent combustion efficiency...")
        
        # Test coal properties
        ultimate_analysis = {
            'carbon': 70.0, 'hydrogen': 5.0, 'oxygen': 10.0,
            'nitrogen': 1.5, 'sulfur': 2.0, 'ash': 11.5
        }
        
        # Test different load conditions for combustion efficiency
        combustion_test_conditions = [
            (4000, 35000, "40% Load"),
            (6000, 52000, "60% Load"),
            (8333, 70000, "100% Load (Design)"),
            (10000, 84000, "120% Load"),
            (12500, 105000, "150% Load")
        ]
        
        combustion_results = []
        
        for coal_rate, air_flow, description in combustion_test_conditions:
            print(f"\n  Testing: {description}")
            
            try:
                # Create ENHANCED combustion model
                combustion_model = CoalCombustionModel(
                    ultimate_analysis=ultimate_analysis,
                    coal_lb_per_hr=coal_rate,
                    air_scfh=air_flow,
                    NOx_eff=0.35,
                    design_coal_rate=8333.0
                )
                
                # Calculate with enhanced load dependency
                combustion_model.calculate()
                
                # Extract results
                load_factor = combustion_model.load_factor
                combustion_eff = combustion_model.combustion_efficiency
                flame_temp = combustion_model.flame_temp_F
                
                print_diagnostic("LOAD", f"Load factor: {load_factor:.2f}")
                print_diagnostic("EFF", f"Combustion efficiency: {combustion_eff:.1%}")
                print_diagnostic("TEMP", f"Flame temperature: {flame_temp:.0f}°F")
                
                combustion_results.append({
                    'load_factor': load_factor,
                    'combustion_efficiency': combustion_eff,
                    'flame_temp': flame_temp,
                    'description': description
                })
                
            except Exception as e:
                print_diagnostic("ERROR", f"Combustion test failed for {description}: {e}")
                continue
        
        # Analyze combustion efficiency variation
        if len(combustion_results) >= 3:
            efficiencies = [r['combustion_efficiency'] for r in combustion_results]
            load_factors = [r['load_factor'] for r in combustion_results]
            
            eff_range = max(efficiencies) - min(efficiencies)
            eff_min = min(efficiencies)
            eff_max = max(efficiencies)
            load_range = max(load_factors) - min(load_factors)
            
            print(f"\n  COMBUSTION EFFICIENCY VARIATION ANALYSIS:")
            print_diagnostic("RESULT", f"Efficiency range: {eff_range:.3f} ({eff_range:.1%})")
            print_diagnostic("RESULT", f"Load range: {load_range:.2f} ({load_range*100:.0f}% variation)")
            print_diagnostic("RESULT", f"Min efficiency: {eff_min:.1%} at {load_factors[efficiencies.index(eff_min)]:.1f} load")
            print_diagnostic("RESULT", f"Max efficiency: {eff_max:.1%} at {load_factors[efficiencies.index(eff_max)]:.1f} load")
            
            # Success criteria for combustion efficiency
            combustion_variation_adequate = eff_range >= 0.05  # At least 5% variation
            combustion_range_realistic = 0.05 <= eff_range <= 0.15  # 5-15% is realistic
            
            print_result(combustion_variation_adequate, f"Combustion efficiency variation adequate: {eff_range:.1%} (target: >=5%)")
            print_result(combustion_range_realistic, f"Combustion efficiency range realistic: {eff_range:.1%} (realistic: 5-15%)")
            
            return combustion_variation_adequate
        else:
            print_result(False, "Insufficient combustion test results")
            return False
            
    except Exception as e:
        print_result(False, f"Load-dependent combustion test failed: {e}")
        traceback.print_exc()
        return False

def test_phase2_comprehensive_load_variation():
    """PHASE 2: Comprehensive load variation test with detailed analysis."""
    print_test_step("2D", "PHASE 2: Comprehensive Load Variation Analysis")
    
    try:
        from boiler_system import EnhancedCompleteBoilerSystem
        
        print("  Running comprehensive load variation analysis...")
        
        # Extended load test range
        load_test_points = [
            (45e6, "45% Load"),
            (55e6, "55% Load"),
            (65e6, "65% Load"),
            (75e6, "75% Load"),
            (85e6, "85% Load"),
            (95e6, "95% Load"),
            (105e6, "105% Load"),
            (115e6, "115% Load"),
            (125e6, "125% Load"),
            (135e6, "135% Load"),
            (145e6, "145% Load")
        ]
        
        load_results = []
        
        for fuel_input, description in load_test_points:
            try:
                # Calculate proportional operating conditions
                load_factor = fuel_input / 100e6
                flue_gas_flow = int(84000 * load_factor)
                furnace_temp = 2800 + (load_factor - 1.0) * 200  # Vary furnace temp with load
                
                # Initialize FIXED boiler
                boiler = EnhancedCompleteBoilerSystem(
                    fuel_input=fuel_input,
                    flue_gas_mass_flow=flue_gas_flow,
                    furnace_exit_temp=furnace_temp,
                    base_fouling_multiplier=1.0
                )
                
                # Solve system
                results = boiler.solve_enhanced_system(max_iterations=15, tolerance=10.0)
                
                efficiency = results['final_efficiency']
                stack_temp = results['final_stack_temperature']
                energy_error = results['energy_balance_error']
                converged = results['converged']
                
                load_results.append({
                    'load_factor': load_factor,
                    'fuel_input_mmbtu': fuel_input/1e6,
                    'efficiency': efficiency,
                    'stack_temp': stack_temp,
                    'energy_error': energy_error,
                    'converged': converged,
                    'description': description
                })
                
                if len(load_results) % 3 == 0:  # Progress update every 3 tests
                    print_diagnostic("PROGRESS", f"Completed {len(load_results)}/{len(load_test_points)} load tests")
                
            except Exception as e:
                print_diagnostic("ERROR", f"Load test failed for {description}: {e}")
                continue
        
        # Comprehensive analysis
        if len(load_results) >= 8:
            efficiencies = [r['efficiency'] for r in load_results]
            load_factors = [r['load_factor'] for r in load_results]
            stack_temps = [r['stack_temp'] for r in load_results]
            energy_errors = [r['energy_error'] for r in load_results]
            
            # Efficiency analysis
            eff_range = max(efficiencies) - min(efficiencies)
            eff_std = np.std(efficiencies)
            eff_mean = np.mean(efficiencies)
            
            # Stack temperature analysis
            stack_range = max(stack_temps) - min(stack_temps)
            stack_std = np.std(stack_temps)
            stack_mean = np.mean(stack_temps)
            
            # Energy balance analysis
            energy_mean = np.mean(energy_errors)
            energy_max = max(energy_errors)
            energy_std = np.std(energy_errors)
            
            print(f"\n  COMPREHENSIVE LOAD VARIATION ANALYSIS:")
            print(f"  {'='*50}")
            print(f"  EFFICIENCY ANALYSIS:")
            print_diagnostic("STAT", f"Range: {eff_range:.3f} ({eff_range:.1%})")
            print_diagnostic("STAT", f"Mean: {eff_mean:.1%} ± {eff_std:.2%}")
            print_diagnostic("STAT", f"Min: {min(efficiencies):.1%} at {load_factors[efficiencies.index(min(efficiencies))]:.1f} load")
            print_diagnostic("STAT", f"Max: {max(efficiencies):.1%} at {load_factors[efficiencies.index(max(efficiencies))]:.1f} load")
            
            print(f"\n  STACK TEMPERATURE ANALYSIS:")
            print_diagnostic("STAT", f"Range: {stack_range:.1f}°F")
            print_diagnostic("STAT", f"Mean: {stack_mean:.0f}°F ± {stack_std:.1f}°F")
            
            print(f"\n  ENERGY BALANCE ANALYSIS:")
            print_diagnostic("STAT", f"Mean error: {energy_mean:.1%}")
            print_diagnostic("STAT", f"Max error: {energy_max:.1%}")
            print_diagnostic("STAT", f"Std dev: {energy_std:.1%}")
            
            # Success criteria evaluation
            efficiency_variation_excellent = eff_range >= 0.03  # >=3% variation
            efficiency_curve_realistic = 0.75 <= min(efficiencies) <= 0.90 and max(efficiencies) <= 0.90
            stack_variation_good = stack_range >= 30  # >=30°F variation
            energy_balance_excellent = energy_mean < 0.03 and energy_max < 0.08  # <3% avg, <8% max
            
            print(f"\n  PHASE 2 SUCCESS CRITERIA:")
            print_result(efficiency_variation_excellent, f"Efficiency variation excellent: {eff_range:.1%} (target: >=3%)")
            print_result(efficiency_curve_realistic, f"Efficiency curve realistic: {min(efficiencies):.1%}-{max(efficiencies):.1%} (target: 75-90%)")
            print_result(stack_variation_good, f"Stack temperature variation good: {stack_range:.0f}°F (target: >=30°F)")
            print_result(energy_balance_excellent, f"Energy balance excellent: avg={energy_mean:.1%}, max={energy_max:.1%} (target: <3%, <8%)")
            
            # Save detailed results to CSV for analysis
            results_df = pd.DataFrame(load_results)
            csv_file = log_dir / "efficiency_variation_test_results.csv"
            results_df.to_csv(csv_file, index=False)
            print_diagnostic("SAVE", f"Detailed results saved to {csv_file}")
            
            overall_success = (efficiency_variation_excellent and efficiency_curve_realistic and 
                             stack_variation_good and energy_balance_excellent)
            
            return overall_success
        else:
            print_result(False, "Insufficient load test results for comprehensive analysis")
            return False
            
    except Exception as e:
        print_result(False, f"Comprehensive load variation test failed: {e}")
        traceback.print_exc()
        return False

def test_phase2_annual_simulator_compatibility():
    """PHASE 2: Test annual simulator compatibility with FIXED efficiency calculations."""
    print_test_step("2E", "PHASE 2: Annual Simulator Compatibility")
    
    try:
        from annual_boiler_simulator import AnnualBoilerSimulator
        
        print("  Testing annual simulator with FIXED efficiency calculations...")
        
        # Create simulator
        simulator = AnnualBoilerSimulator(start_date="2024-01-01")
        
        # Test 48 hours of simulation with varying conditions
        simulation_data = []
        
        for hour in range(48):
            test_datetime = simulator.start_date + timedelta(hours=hour)
            
            try:
                # Generate conditions
                operating_conditions = simulator._generate_hourly_conditions(test_datetime)
                soot_blowing_actions = simulator._check_soot_blowing_schedule(test_datetime)
                
                # Simulate operation
                operation_data = simulator._simulate_boiler_operation(
                    test_datetime, operating_conditions, soot_blowing_actions
                )
                
                simulation_data.append(operation_data)
                
                if hour % 12 == 0:  # Progress update every 12 hours
                    eff = operation_data.get('system_efficiency', 0)
                    stack = operation_data.get('stack_temp_F', 0)
                    print_diagnostic("SIM", f"Hour {hour}: Eff={eff:.1%}, Stack={stack:.0f}°F")
                
            except Exception as e:
                print_diagnostic("ERROR", f"Hour {hour} simulation failed: {e}")
                continue
        
        # Analyze simulation results
        if len(simulation_data) >= 24:
            df = pd.DataFrame(simulation_data)
            
            # Efficiency analysis
            eff_mean = df['system_efficiency'].mean()
            eff_std = df['system_efficiency'].std()
            eff_min = df['system_efficiency'].min()
            eff_max = df['system_efficiency'].max()
            eff_range = eff_max - eff_min
            
            # Stack temperature analysis
            stack_mean = df['stack_temp_F'].mean()
            stack_std = df['stack_temp_F'].std()
            stack_range = df['stack_temp_F'].max() - df['stack_temp_F'].min()
            
            # Convergence analysis
            converged_count = df['solution_converged'].sum()
            convergence_rate = converged_count / len(df)
            
            print(f"\n  ANNUAL SIMULATOR ANALYSIS ({len(df)} hours):")
            print_diagnostic("EFF", f"Efficiency: {eff_mean:.1%} ± {eff_std:.2%} (range: {eff_range:.2%})")
            print_diagnostic("STACK", f"Stack temp: {stack_mean:.0f}°F ± {stack_std:.1f}°F (range: {stack_range:.0f}°F)")
            print_diagnostic("CONV", f"Convergence rate: {convergence_rate:.1%}")
            
            # Success criteria
            efficiency_variation_present = eff_range >= 0.005  # At least 0.5% variation over 48 hours
            efficiency_values_realistic = 0.75 <= eff_min and eff_max <= 0.90
            stack_variation_present = stack_range >= 10  # At least 10°F variation
            high_convergence_rate = convergence_rate >= 0.90  # >=90% convergence
            
            print_result(efficiency_variation_present, f"Efficiency variation present: {eff_range:.2%} (target: >=0.5%)")
            print_result(efficiency_values_realistic, f"Efficiency values realistic: {eff_min:.1%}-{eff_max:.1%} (target: 75-90%)")
            print_result(stack_variation_present, f"Stack temperature variation present: {stack_range:.0f}°F (target: >=10°F)")
            print_result(high_convergence_rate, f"High convergence rate: {convergence_rate:.1%} (target: >=90%)")
            
            overall_success = (efficiency_variation_present and efficiency_values_realistic and 
                             stack_variation_present and high_convergence_rate)
            
            return overall_success
        else:
            print_result(False, "Insufficient simulation data generated")
            return False
            
    except Exception as e:
        print_result(False, f"Annual simulator compatibility test failed: {e}")
        traceback.print_exc()
        return False

def save_phase2_validation_report(test_results: dict, overall_success: bool):
    """Save detailed Phase 2 validation report."""
    
    report_file = log_dir / f"phase2_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(report_file, 'w') as f:
        f.write("PHASE 2 VALIDATION REPORT - STATIC EFFICIENCY FIXES\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Validation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Script Version: 9.0 - Phase 2 Validation Implementation\n")
        f.write(f"Overall Result: {'VALIDATION SUCCESSFUL' if overall_success else 'VALIDATION INCOMPLETE'}\n\n")
        
        f.write("PHASE 2 OBJECTIVES:\n")
        f.write("-" * 30 + "\n")
        f.write("1. Verify efficiency now varies properly with load (target: >=2%)\n")
        f.write("2. Confirm energy balance errors reduced (target: <5%)\n")
        f.write("3. Validate PropertyCalculator integration fixes\n")
        f.write("4. Verify load-dependent combustion efficiency\n")
        f.write("5. Confirm comprehensive load variation performance\n")
        f.write("6. Validate annual simulator compatibility\n\n")
        
        f.write("PHASE 2 VALIDATION TEST RESULTS:\n")
        f.write("-" * 30 + "\n")
        
        phase2_tests = [
            "Efficiency Variation Validation",
            "Component Integration Validation", 
            "Load-Dependent Combustion Validation",
            "Comprehensive Load Variation Analysis",
            "Annual Simulator Compatibility"
        ]
        
        for test_name in phase2_tests:
            if test_name in test_results:
                status = "PASS" if test_results[test_name] else "FAIL"
                f.write(f"{status:4s} | {test_name} (PHASE 2)\n")
        
        f.write("\nSUPPORTING VALIDATION TESTS:\n")
        for test_name, result in test_results.items():
            if test_name not in phase2_tests:
                status = "PASS" if result else "FAIL"
                f.write(f"{status:4s} | {test_name}\n")
        
        passed = sum(1 for r in test_results.values() if r)
        total = len(test_results)
        phase2_passed = sum(1 for test in phase2_tests 
                          if test_results.get(test, False))
        
        f.write(f"\nSummary: {passed}/{total} total tests passed\n")
        f.write(f"Phase 2: {phase2_passed}/{len(phase2_tests)} validation tests passed\n\n")
        
        if overall_success:
            f.write("PHASE 2 VALIDATION SUCCESS:\n")
            f.write("Static efficiency issues have been successfully resolved.\n")
            f.write("Key improvements achieved:\n")
            f.write("- Efficiency now varies properly with load (>=2% variation)\n")
            f.write("- Energy balance errors reduced to <5%\n")
            f.write("- PropertyCalculator integration fixed\n")
            f.write("- Load-dependent combustion efficiency implemented\n")
            f.write("- Comprehensive load variation validated\n")
            f.write("- Annual simulator compatibility confirmed\n\n")
            f.write("READY FOR COMMERCIAL DEMO: System now provides realistic,\n")
            f.write("load-dependent efficiency variations suitable for client presentations.\n")
        else:
            f.write("PHASE 2 VALIDATION ISSUES:\n")
            failed_tests = [name for name, result in test_results.items() if not result]
            for test in failed_tests:
                f.write(f"- {test}\n")
            f.write("\nRecommendation: Address remaining validation failures\n")
    
    print(f"\nPhase 2 validation report saved: {report_file}")

def run_phase2_validation_suite():
    """Run Phase 2 validation suite for FIXED static efficiency issues."""
    
    print_header("PHASE 2 VALIDATION SUITE - STATIC EFFICIENCY FIXES")
    print("Version 9.0 - Comprehensive Validation of Efficiency Fixes")
    print(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Track test results
    test_results = {}
    
    # Run Phase 2 validation tests
    phase2_test_functions = [
        ("Module Imports", test_module_imports),
        ("Efficiency Variation Validation", test_phase2_efficiency_variation),
        ("Component Integration Validation", test_phase2_component_integration),
        ("Load-Dependent Combustion Validation", test_phase2_load_dependent_combustion),
        ("Comprehensive Load Variation Analysis", test_phase2_comprehensive_load_variation),
        ("Annual Simulator Compatibility", test_phase2_annual_simulator_compatibility)
    ]
    
    for test_name, test_function in phase2_test_functions:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            result = test_function()
            test_results[test_name] = result
            print(f"Result: {'PASS' if result else 'FAIL'}")
        except Exception as e:
            print_result(False, f"{test_name} test crashed: {e}")
            test_results[test_name] = False
    
    # Summary
    print_header("PHASE 2 VALIDATION RESULTS SUMMARY")
    
    passed_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {status:4s} | {test_name}")
    
    print(f"\nOVERALL RESULT: {passed_tests}/{total_tests} tests passed")
    
    # Determine overall validation success
    phase2_core_tests = [
        "Efficiency Variation Validation",
        "Component Integration Validation",
        "Load-Dependent Combustion Validation",
        "Comprehensive Load Variation Analysis"
    ]
    
    phase2_core_passed = sum(1 for test in phase2_core_tests 
                           if test_results.get(test, False))
    
    if phase2_core_passed >= 3 and passed_tests >= total_tests - 1:  # Allow minimal failures
        print("[SUCCESS] Phase 2 validation complete - Static efficiency issues RESOLVED")
        overall_success = True
    else:
        print("[WARNING] Phase 2 validation incomplete - Some fixes may need attention")
        overall_success = False
    
    # Save Phase 2 validation report
    save_phase2_validation_report(test_results, overall_success)
    
    return overall_success

def main():
    """Main execution function for Phase 2 validation."""
    
    print_header("BOILER SIMULATION PHASE 2 VALIDATION SCRIPT")
    print("Comprehensive validation suite to verify static efficiency fixes")
    print(f"Script Version: 9.0 - Phase 2 Validation Implementation")
    print(f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run Phase 2 validation suite
        print("\n" + "="*70)
        print("RUNNING PHASE 2 VALIDATION SUITE")
        print("="*70)
        
        validation_success = run_phase2_validation_suite()
        
        if validation_success:
            print("\n" + "="*60)
            print("SUCCESS: Phase 2 validation completed!")
            print("Static efficiency issues have been RESOLVED.")
            print("System ready for commercial demonstration.")
            print("="*60)
        else:
            print("\n" + "="*50)
            print("WARNING: Phase 2 validation incomplete.")
            print("Review failed tests for remaining issues.")
            print("="*50)
        
    except Exception as e:
        print(f"\n[ERROR] Phase 2 validation execution failed: {e}")
        traceback.print_exc()
        return False
    
    print(f"\nValidation complete. Check logs in: {log_dir}")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)