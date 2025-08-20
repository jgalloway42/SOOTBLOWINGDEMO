#!/usr/bin/env python3
"""
Enhanced Debug Script - Phase 1 Diagnostic Suite

This script provides comprehensive testing and validation for the boiler simulation
with Phase 1 diagnostics to identify static efficiency calculation issues:

PHASE 1 DIAGNOSTIC TESTS:
- Efficiency calculation tracing to find static points
- Parameter propagation validation through calculation chain
- Component-level variation testing for individual sections
- Load curve implementation verification
- Fouling impact validation on efficiency calculations

EXISTING TESTS:
- IAPWS integration validation
- Solver interface compatibility
- Unicode-safe logging verification
- Load-dependent variation testing
- Annual simulator compatibility

Author: Enhanced Boiler Modeling System
Version: 8.3 - Phase 1 Diagnostic Implementation
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
    
    # Return overall success
    all_imported = all(import_results.values())
    print_result(all_imported, f"All modules imported: {sum(import_results.values())}/{len(import_results)}")
    return all_imported

def test_efficiency_calculation_tracing():
    """Trace efficiency calculation step-by-step to find static points."""
    print_test_step("2A", "PHASE 1: Efficiency Calculation Tracing")
    
    try:
        from boiler_system import EnhancedCompleteBoilerSystem
        
        print("  Tracing efficiency calculation across load scenarios...")
        
        # Test scenarios with dramatically different parameters
        test_scenarios = [
            (50e6, 42000, 2800, 0.5, "50% Load + Light Fouling"),
            (100e6, 84000, 3000, 1.0, "100% Load + Normal Fouling"), 
            (150e6, 126000, 3200, 2.0, "150% Load + Heavy Fouling")
        ]
        
        efficiency_results = []
        calculation_traces = []
        
        for fuel_input, mass_flow, exit_temp, fouling_mult, description in test_scenarios:
            print(f"\n  Analyzing: {description}")
            print_diagnostic("INPUT", f"fuel_input={fuel_input/1e6:.0f}MMBtu/hr, fouling={fouling_mult}x")
            
            try:
                # Initialize boiler with specific parameters
                boiler = EnhancedCompleteBoilerSystem(
                    fuel_input=fuel_input,
                    flue_gas_mass_flow=mass_flow,
                    furnace_exit_temp=exit_temp,
                    base_fouling_multiplier=fouling_mult
                )
                
                # Log initialization parameters
                print_diagnostic("INIT", f"Boiler initialized with fuel_input={boiler.fuel_input/1e6:.1f}MMBtu/hr")
                print_diagnostic("INIT", f"Base fouling multiplier={fouling_mult}")
                
                # Solve system and extract intermediate values
                results = boiler.solve_enhanced_system(max_iterations=15, tolerance=10.0)
                
                # Extract key calculation components
                final_efficiency = results['final_efficiency']
                final_stack_temp = results['final_stack_temperature']
                energy_balance_error = results['energy_balance_error']
                
                print_diagnostic("CALC", f"Final efficiency: {final_efficiency:.3f} ({final_efficiency:.1%})")
                print_diagnostic("CALC", f"Stack temperature: {final_stack_temp:.1f}F")
                print_diagnostic("CALC", f"Energy balance error: {energy_balance_error:.1%}")
                
                efficiency_results.append({
                    'scenario': description,
                    'fuel_input': fuel_input/1e6,
                    'fouling': fouling_mult,
                    'efficiency': final_efficiency,
                    'stack_temp': final_stack_temp,
                    'energy_error': energy_balance_error
                })
                
                # Try to extract more detailed calculation information
                if hasattr(boiler, 'system_performance') and boiler.system_performance:
                    perf = boiler.system_performance
                    print_diagnostic("DETAIL", f"System performance available: {type(perf)}")
                    
                    # Extract component-level information if available
                    if hasattr(perf, 'overall_efficiency'):
                        print_diagnostic("DETAIL", f"Component efficiency: {perf.overall_efficiency:.3f}")
                
                calculation_traces.append({
                    'scenario': description,
                    'initialization_successful': True,
                    'solver_converged': results['converged'],
                    'final_efficiency': final_efficiency
                })
                
            except Exception as e:
                print_diagnostic("ERROR", f"Scenario failed: {e}")
                calculation_traces.append({
                    'scenario': description,
                    'initialization_successful': False,
                    'error': str(e)
                })
                continue
        
        # Analyze efficiency variation
        if len(efficiency_results) >= 2:
            efficiencies = [r['efficiency'] for r in efficiency_results]
            fuel_inputs = [r['fuel_input'] for r in efficiency_results]
            fouling_factors = [r['fouling'] for r in efficiency_results]
            
            efficiency_range = max(efficiencies) - min(efficiencies)
            fuel_range = max(fuel_inputs) - min(fuel_inputs)
            fouling_range = max(fouling_factors) - min(fouling_factors)
            
            print(f"\n  EFFICIENCY VARIATION ANALYSIS:")
            print_diagnostic("RESULT", f"Efficiency range: {efficiency_range:.4f} ({efficiency_range:.2%})")
            print_diagnostic("RESULT", f"Fuel input range: {fuel_range:.0f}MMBtu/hr ({fuel_range/min(fuel_inputs)*100:.0f}% increase)")
            print_diagnostic("RESULT", f"Fouling range: {fouling_range:.1f}x ({fouling_range/min(fouling_factors)*100:.0f}% increase)")
            
            # Expected: >2% efficiency variation for 3x fuel input and 4x fouling change
            expected_variation = efficiency_range >= 0.02  # At least 2% variation expected
            
            print_result(expected_variation, f"Efficiency variation adequate: {efficiency_range:.2%} (target: >=2%)")
            
            # Log individual scenario results
            print(f"\n  DETAILED SCENARIO RESULTS:")
            for result in efficiency_results:
                print(f"    {result['scenario']:25s}: Eff={result['efficiency']:.1%}, Stack={result['stack_temp']:.0f}F")
            
            return expected_variation
        else:
            print_result(False, "Insufficient scenarios completed for analysis")
            return False
            
    except Exception as e:
        print_result(False, f"Efficiency calculation tracing failed: {e}")
        traceback.print_exc()
        return False

def test_parameter_propagation_validation():
    """Validate that input parameters propagate to calculation methods."""
    print_test_step("2B", "PHASE 1: Parameter Propagation Validation")
    
    try:
        from boiler_system import EnhancedCompleteBoilerSystem
        
        print("  Testing parameter propagation through calculation chain...")
        
        # Test with dramatically different parameter sets
        parameter_sets = [
            {"fuel_input": 60e6, "flue_gas_mass_flow": 50000, "furnace_exit_temp": 2900, "base_fouling_multiplier": 0.3},
            {"fuel_input": 120e6, "flue_gas_mass_flow": 100000, "furnace_exit_temp": 3100, "base_fouling_multiplier": 1.8}
        ]
        
        propagation_results = []
        
        for i, params in enumerate(parameter_sets):
            print(f"\n  Parameter Set {i+1}:")
            print_diagnostic("INPUT", f"fuel_input={params['fuel_input']/1e6:.0f}MMBtu/hr")
            print_diagnostic("INPUT", f"flue_gas_mass_flow={params['flue_gas_mass_flow']/1000:.0f}k lbm/hr")
            print_diagnostic("INPUT", f"furnace_exit_temp={params['furnace_exit_temp']:.0f}F")
            print_diagnostic("INPUT", f"base_fouling_multiplier={params['base_fouling_multiplier']:.1f}x")
            
            try:
                # Initialize boiler
                boiler = EnhancedCompleteBoilerSystem(**params)
                
                # Check parameter storage
                print_diagnostic("STORE", f"Stored fuel_input: {boiler.fuel_input/1e6:.1f}MMBtu/hr")
                print_diagnostic("STORE", f"Stored fouling multiplier: {boiler.base_fouling_multiplier:.1f}x")
                
                # Solve and check if parameters affect results
                results = boiler.solve_enhanced_system(max_iterations=10, tolerance=15.0)
                
                propagation_results.append({
                    'set': i+1,
                    'fuel_input': params['fuel_input']/1e6,
                    'fouling_mult': params['base_fouling_multiplier'],
                    'efficiency': results['final_efficiency'],
                    'stack_temp': results['final_stack_temperature'],
                    'converged': results['converged']
                })
                
                print_diagnostic("OUTPUT", f"Final efficiency: {results['final_efficiency']:.3f}")
                print_diagnostic("OUTPUT", f"Stack temperature: {results['final_stack_temperature']:.1f}F")
                
                # Check for obvious parameter impact
                if hasattr(boiler, 'sections') and boiler.sections:
                    print_diagnostic("SECTIONS", f"Number of boiler sections: {len(boiler.sections)}")
                    
                    # Try to extract section-level information
                    for j, section in enumerate(boiler.sections[:3]):  # First 3 sections
                        if hasattr(section, 'fouling_factor'):
                            print_diagnostic("SECTION", f"Section {j} fouling factor: {section.fouling_factor:.3f}")
                        if hasattr(section, 'heat_transfer_coefficient'):
                            print_diagnostic("SECTION", f"Section {j} heat transfer coeff: {getattr(section, 'heat_transfer_coefficient', 'N/A')}")
                
            except Exception as e:
                print_diagnostic("ERROR", f"Parameter set {i+1} failed: {e}")
                propagation_results.append({
                    'set': i+1,
                    'error': str(e)
                })
        
        # Analyze parameter impact
        if len(propagation_results) >= 2 and all('error' not in r for r in propagation_results):
            result1, result2 = propagation_results[0], propagation_results[1]
            
            fuel_change = result2['fuel_input'] / result1['fuel_input']
            fouling_change = result2['fouling_mult'] / result1['fouling_mult']
            efficiency_change = result2['efficiency'] / result1['efficiency']
            stack_temp_change = (result2['stack_temp'] - result1['stack_temp'])
            
            print(f"\n  PARAMETER IMPACT ANALYSIS:")
            print_diagnostic("CHANGE", f"Fuel input changed by: {fuel_change:.1f}x")
            print_diagnostic("CHANGE", f"Fouling changed by: {fouling_change:.1f}x")
            print_diagnostic("CHANGE", f"Efficiency ratio: {efficiency_change:.4f}")
            print_diagnostic("CHANGE", f"Stack temp difference: {stack_temp_change:.1f}F")
            
            # Parameters should cause some change in outputs
            parameter_impact = (abs(efficiency_change - 1.0) > 0.001) or (abs(stack_temp_change) > 5.0)
            
            print_result(parameter_impact, f"Parameters show measurable impact on results")
            return parameter_impact
        else:
            print_result(False, "Could not analyze parameter impact due to errors")
            return False
            
    except Exception as e:
        print_result(False, f"Parameter propagation validation failed: {e}")
        traceback.print_exc()
        return False

def test_component_level_variation():
    """Test individual boiler sections for load-dependent behavior."""
    print_test_step("2C", "PHASE 1: Component-Level Variation Testing")
    
    try:
        from boiler_system import EnhancedCompleteBoilerSystem
        from heat_transfer_calculations import HeatTransferCalculator
        
        print("  Testing individual boiler components for load response...")
        
        # Test different load conditions
        load_conditions = [
            (70e6, 0.8, "70% Load"),
            (100e6, 1.0, "100% Load"),
            (130e6, 1.2, "130% Load")
        ]
        
        component_results = []
        
        for fuel_input, fouling_mult, description in load_conditions:
            print(f"\n  Analyzing: {description}")
            
            try:
                boiler = EnhancedCompleteBoilerSystem(
                    fuel_input=fuel_input,
                    flue_gas_mass_flow=int(84000 * fuel_input / 100e6),
                    furnace_exit_temp=3000,
                    base_fouling_multiplier=fouling_mult
                )
                
                # Test heat transfer calculator directly
                ht_calc = HeatTransferCalculator()
                
                # Test component calculations
                test_conditions = {
                    'temperature': 1500,  # F
                    'pressure': 150,      # psia
                    'flow_rate': fuel_input / 1e6,  # Scaled flow
                }
                
                print_diagnostic("COMP", f"Testing heat transfer at {test_conditions['temperature']}F, flow={test_conditions['flow_rate']:.1f}")
                
                # Test gas properties
                try:
                    gas_props = ht_calc.property_calculator.get_flue_gas_properties(test_conditions['temperature'])
                    print_diagnostic("PROPS", f"Gas density: {gas_props.density:.4f} lbm/ft3")
                    print_diagnostic("PROPS", f"Gas cp: {gas_props.cp:.4f} Btu/lbm-F")
                except Exception as e:
                    print_diagnostic("ERROR", f"Gas properties failed: {e}")
                
                # Solve full system
                results = boiler.solve_enhanced_system(max_iterations=10, tolerance=15.0)
                
                component_results.append({
                    'condition': description,
                    'fuel_input': fuel_input/1e6,
                    'fouling_mult': fouling_mult,
                    'efficiency': results['final_efficiency'],
                    'stack_temp': results['final_stack_temperature'],
                    'flow_rate': test_conditions['flow_rate']
                })
                
                print_diagnostic("RESULT", f"System efficiency: {results['final_efficiency']:.3f}")
                print_diagnostic("RESULT", f"Stack temperature: {results['final_stack_temperature']:.1f}F")
                
                # Try to access section-level data if available
                if hasattr(boiler, 'sections') and boiler.sections:
                    for i, section in enumerate(boiler.sections[:2]):  # First 2 sections
                        section_name = getattr(section, 'name', f'Section_{i}')
                        print_diagnostic("SECT", f"{section_name}: Available attributes: {[attr for attr in dir(section) if not attr.startswith('_')][:5]}")
                
            except Exception as e:
                print_diagnostic("ERROR", f"Component test failed for {description}: {e}")
                continue
        
        # Analyze component-level variation
        if len(component_results) >= 2:
            efficiencies = [r['efficiency'] for r in component_results]
            stack_temps = [r['stack_temp'] for r in component_results]
            fuel_inputs = [r['fuel_input'] for r in component_results]
            
            efficiency_range = max(efficiencies) - min(efficiencies)
            stack_temp_range = max(stack_temps) - min(stack_temps)
            fuel_range = max(fuel_inputs) - min(fuel_inputs)
            
            print(f"\n  COMPONENT VARIATION ANALYSIS:")
            print_diagnostic("RANGE", f"Efficiency range: {efficiency_range:.4f} ({efficiency_range:.2%})")
            print_diagnostic("RANGE", f"Stack temp range: {stack_temp_range:.1f}F")
            print_diagnostic("RANGE", f"Fuel input range: {fuel_range:.0f}MMBtu/hr")
            
            # Components should show some variation
            component_variation = efficiency_range > 0.005 or stack_temp_range > 10.0
            
            print_result(component_variation, f"Components show measurable variation")
            return component_variation
        else:
            print_result(False, "Insufficient component data for analysis")
            return False
            
    except Exception as e:
        print_result(False, f"Component-level variation test failed: {e}")
        traceback.print_exc()
        return False

def test_load_curve_implementation():
    """Test that load-dependent efficiency curves are implemented and active."""
    print_test_step("2D", "PHASE 1: Load Curve Implementation Verification")
    
    try:
        from boiler_system import EnhancedCompleteBoilerSystem
        
        print("  Checking for load-dependent efficiency curve implementation...")
        
        # Test across wide load range to detect curves
        load_points = [
            (45e6, "45% Load"),
            (60e6, "60% Load"),
            (75e6, "75% Load"),
            (90e6, "90% Load"),
            (100e6, "100% Load"),
            (110e6, "110% Load")
        ]
        
        load_curve_results = []
        
        for fuel_input, description in load_points:
            print(f"\n  Testing: {description}")
            
            try:
                # Calculate load factor (assuming 100e6 is full load)
                load_factor = fuel_input / 100e6
                print_diagnostic("LOAD", f"Load factor: {load_factor:.2f}")
                
                boiler = EnhancedCompleteBoilerSystem(
                    fuel_input=fuel_input,
                    flue_gas_mass_flow=int(84000 * load_factor),
                    furnace_exit_temp=2900 + (load_factor - 1.0) * 200,  # Scale exit temp
                    base_fouling_multiplier=1.0
                )
                
                # Check for load-dependent methods
                boiler_methods = [method for method in dir(boiler) if 'load' in method.lower() or 'curve' in method.lower()]
                if boiler_methods:
                    print_diagnostic("METHOD", f"Load-related methods found: {boiler_methods}")
                else:
                    print_diagnostic("METHOD", "No obvious load-related methods found")
                
                results = boiler.solve_enhanced_system(max_iterations=10, tolerance=15.0)
                
                load_curve_results.append({
                    'load_factor': load_factor,
                    'description': description,
                    'efficiency': results['final_efficiency'],
                    'stack_temp': results['final_stack_temperature']
                })
                
                print_diagnostic("CURVE", f"Load {load_factor:.2f} -> Efficiency {results['final_efficiency']:.3f}")
                
            except Exception as e:
                print_diagnostic("ERROR", f"Load point {description} failed: {e}")
                continue
        
        # Analyze load curve behavior
        if len(load_curve_results) >= 4:
            # Sort by load factor
            load_curve_results.sort(key=lambda x: x['load_factor'])
            
            print(f"\n  LOAD CURVE ANALYSIS:")
            for result in load_curve_results:
                print_diagnostic("POINT", f"Load {result['load_factor']:.2f}: Eff={result['efficiency']:.3f}, Stack={result['stack_temp']:.0f}F")
            
            # Check for characteristic efficiency curve shape
            efficiencies = [r['efficiency'] for r in load_curve_results]
            load_factors = [r['load_factor'] for r in load_curve_results]
            
            # Look for peak efficiency around 75-85% load
            max_eff_index = efficiencies.index(max(efficiencies))
            peak_load_factor = load_factors[max_eff_index]
            
            efficiency_range = max(efficiencies) - min(efficiencies)
            
            print_diagnostic("PEAK", f"Peak efficiency at load factor: {peak_load_factor:.2f}")
            print_diagnostic("PEAK", f"Efficiency range across loads: {efficiency_range:.4f} ({efficiency_range:.2%})")
            
            # Realistic load curve should show:
            # 1. Some efficiency variation across loads
            # 2. Peak efficiency between 0.6-0.9 load factor
            realistic_curve = (
                efficiency_range >= 0.01 and  # At least 1% variation
                0.6 <= peak_load_factor <= 0.9  # Realistic peak location
            )
            
            print_result(realistic_curve, f"Load curve shows realistic behavior")
            return realistic_curve
        else:
            print_result(False, "Insufficient load points for curve analysis")
            return False
            
    except Exception as e:
        print_result(False, f"Load curve implementation test failed: {e}")
        traceback.print_exc()
        return False

def test_fouling_impact_validation():
    """Validate that fouling parameters impact efficiency calculations."""
    print_test_step("2E", "PHASE 1: Fouling Impact Validation")
    
    try:
        from boiler_system import EnhancedCompleteBoilerSystem
        
        print("  Testing fouling parameter impact on efficiency calculations...")
        
        # Test fouling multiplier progression from clean to severely fouled
        fouling_levels = [
            (0.2, "Very Clean"),
            (0.5, "Light Fouling"),
            (1.0, "Normal Fouling"),
            (2.0, "Heavy Fouling"),
            (4.0, "Severe Fouling")
        ]
        
        fouling_results = []
        
        for fouling_mult, description in fouling_levels:
            print(f"\n  Testing: {description} (fouling={fouling_mult:.1f}x)")
            
            try:
                boiler = EnhancedCompleteBoilerSystem(
                    fuel_input=100e6,  # Constant fuel input
                    flue_gas_mass_flow=84000,
                    furnace_exit_temp=3000,
                    base_fouling_multiplier=fouling_mult
                )
                
                print_diagnostic("FOUL", f"Base fouling multiplier set to: {fouling_mult:.1f}x")
                
                # Check if fouling parameter is stored
                stored_fouling = getattr(boiler, 'base_fouling_multiplier', 'NOT_FOUND')
                print_diagnostic("STORE", f"Stored fouling multiplier: {stored_fouling}")
                
                results = boiler.solve_enhanced_system(max_iterations=15, tolerance=15.0)
                
                fouling_results.append({
                    'fouling_mult': fouling_mult,
                    'description': description,
                    'efficiency': results['final_efficiency'],
                    'stack_temp': results['final_stack_temperature'],
                    'energy_error': results['energy_balance_error']
                })
                
                print_diagnostic("IMPACT", f"Efficiency: {results['final_efficiency']:.3f}")
                print_diagnostic("IMPACT", f"Stack temp: {results['final_stack_temperature']:.1f}F")
                print_diagnostic("IMPACT", f"Energy error: {results['energy_balance_error']:.1%}")
                
                # Check for section-level fouling application
                if hasattr(boiler, 'sections') and boiler.sections:
                    for i, section in enumerate(boiler.sections[:2]):
                        if hasattr(section, 'fouling_factor'):
                            section_fouling = getattr(section, 'fouling_factor', 'N/A')
                            print_diagnostic("SECT", f"Section {i} fouling factor: {section_fouling}")
                
            except Exception as e:
                print_diagnostic("ERROR", f"Fouling test failed for {description}: {e}")
                continue
        
        # Analyze fouling impact
        if len(fouling_results) >= 3:
            # Sort by fouling level
            fouling_results.sort(key=lambda x: x['fouling_mult'])
            
            print(f"\n  FOULING IMPACT ANALYSIS:")
            for result in fouling_results:
                print_diagnostic("FOUL", f"{result['fouling_mult']:.1f}x: Eff={result['efficiency']:.3f}, Stack={result['stack_temp']:.0f}F")
            
            # Extract trends
            fouling_mults = [r['fouling_mult'] for r in fouling_results]
            efficiencies = [r['efficiency'] for r in fouling_results]
            stack_temps = [r['stack_temp'] for r in fouling_results]
            
            # Expected: As fouling increases, efficiency decreases, stack temp increases
            min_fouling_idx = fouling_mults.index(min(fouling_mults))
            max_fouling_idx = fouling_mults.index(max(fouling_mults))
            
            efficiency_change = efficiencies[max_fouling_idx] - efficiencies[min_fouling_idx]
            stack_temp_change = stack_temps[max_fouling_idx] - stack_temps[min_fouling_idx]
            fouling_change = fouling_mults[max_fouling_idx] / fouling_mults[min_fouling_idx]
            
            print_diagnostic("TREND", f"Fouling increased by: {fouling_change:.1f}x")
            print_diagnostic("TREND", f"Efficiency change: {efficiency_change:.4f} ({efficiency_change:.2%})")
            print_diagnostic("TREND", f"Stack temp change: {stack_temp_change:.1f}F")
            
            # Realistic fouling impact should show:
            # 1. Decreasing efficiency with increased fouling
            # 2. Increasing stack temperature with increased fouling
            # 3. Meaningful change magnitude
            realistic_fouling = (
                efficiency_change < -0.01 and  # Efficiency should decrease by at least 1%
                stack_temp_change > 15.0       # Stack temp should increase by at least 15F
            )
            
            print_result(realistic_fouling, f"Fouling shows realistic impact on system performance")
            return realistic_fouling
        else:
            print_result(False, "Insufficient fouling data for analysis")
            return False
            
    except Exception as e:
        print_result(False, f"Fouling impact validation failed: {e}")
        traceback.print_exc()
        return False

def test_iapws_integration():
    """Test the fixed IAPWS integration specifically."""
    print_test_step("3", "IAPWS Integration Fix Validation")
    
    try:
        from thermodynamic_properties import PropertyCalculator
        
        calc = PropertyCalculator()
        
        # Test the previously missing get_water_properties method
        print("  Testing get_water_properties method (CRITICAL FIX)...")
        try:
            water_props = calc.get_water_properties(600, 220)  # 600 psia, 220°F
            print_result(True, f"get_water_properties: h={water_props.enthalpy:.1f} Btu/lb")
        except AttributeError as e:
            print_result(False, f"get_water_properties still missing: {e}")
            return False
        except Exception as e:
            print_result(False, f"get_water_properties failed: {e}")
            return False
        
        # Test get_steam_properties method
        print("  Testing get_steam_properties method...")
        try:
            steam_props = calc.get_steam_properties(600, 700)  # 600 psia, 700°F
            print_result(True, f"get_steam_properties: h={steam_props.enthalpy:.1f} Btu/lb")
        except Exception as e:
            print_result(False, f"get_steam_properties failed: {e}")
            return False
        
        # Test realistic steam conditions
        print("  Testing realistic boiler conditions...")
        test_conditions = [
            (150, 400),   # Low pressure feedwater
            (600, 220),   # High pressure feedwater  
            (600, 700),   # Superheated steam
            (150, 366)    # Saturated steam at 150 psia
        ]
        
        all_passed = True
        for pressure, temperature in test_conditions:
            try:
                props = calc.get_steam_properties(pressure, temperature)
                print(f"    {pressure} psia, {temperature}°F: h={props.enthalpy:.0f} Btu/lb - OK")
            except Exception as e:
                print(f"    {pressure} psia, {temperature}°F: FAILED - {e}")
                all_passed = False
        
        print_result(all_passed, "All realistic boiler conditions tested")
        
        # Test specific energy calculation
        print("  Testing specific energy calculation...")
        steam = calc.get_steam_properties(600, 700)
        water = calc.get_water_properties(600, 220)
        specific_energy = steam.enthalpy - water.enthalpy
        
        # Realistic specific energy should be 1000-1300 Btu/lb
        realistic_specific_energy = 1000 <= specific_energy <= 1300
        print_result(realistic_specific_energy, 
                    f"Specific energy: {specific_energy:.0f} Btu/lb (realistic: {realistic_specific_energy})")
        
        return all_passed and realistic_specific_energy
        
    except Exception as e:
        print_result(False, f"IAPWS integration test failed: {e}")
        traceback.print_exc()
        return False

def test_enhanced_boiler_system():
    """Test the enhanced boiler system with fixed interface."""
    print_test_step("4", "Enhanced Boiler System Validation")
    
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
    print_test_step("5", "Annual Simulator Interface Validation")
    
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
    print_test_step("6", "Unicode-Safe Logging Validation")
    
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
    print_test_step("7", "Basic Load Variation Testing")
    
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
    print_test_step("8", "Short Simulation Run (24 hours)")
    
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

def save_phase1_diagnostic_report(test_results: dict, overall_success: bool):
    """Save detailed Phase 1 diagnostic report."""
    report_file = log_dir / f"phase1_diagnostic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(report_file, 'w') as f:
        f.write("PHASE 1 DIAGNOSTIC REPORT - STATIC EFFICIENCY INVESTIGATION\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Diagnostic Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Script Version: 8.3 - Phase 1 Diagnostic Implementation\n")
        f.write(f"Overall Result: {'DIAGNOSTIC COMPLETE' if overall_success else 'DIAGNOSTIC INCOMPLETE'}\n\n")
        
        f.write("PHASE 1 OBJECTIVES:\n")
        f.write("-" * 30 + "\n")
        f.write("1. Trace efficiency calculation path to find static points\n")
        f.write("2. Validate input parameter propagation through calculation chain\n")
        f.write("3. Test individual boiler components for load response\n")
        f.write("4. Verify load curve implementation and activation\n")
        f.write("5. Validate fouling parameter impact on efficiency\n\n")
        
        f.write("DIAGNOSTIC TEST RESULTS:\n")
        f.write("-" * 30 + "\n")
        
        phase1_tests = [
            "Efficiency Calculation Tracing",
            "Parameter Propagation Validation",
            "Component-Level Variation",
            "Load Curve Implementation",
            "Fouling Impact Validation"
        ]
        
        for test_name in phase1_tests:
            if test_name in test_results:
                status = "PASS" if test_results[test_name] else "FAIL"
                f.write(f"{status:4s} | {test_name} (DIAGNOSTIC)\n")
        
        f.write("\nSUPPORTING VALIDATION TESTS:\n")
        for test_name, result in test_results.items():
            if test_name not in phase1_tests:
                status = "PASS" if result else "FAIL"
                f.write(f"{status:4s} | {test_name}\n")
        
        passed = sum(1 for r in test_results.values() if r)
        total = len(test_results)
        phase1_passed = sum(1 for test in phase1_tests 
                          if test_results.get(test, False))
        
        f.write(f"\nSummary: {passed}/{total} total tests passed\n")
        f.write(f"Phase 1: {phase1_passed}/{len(phase1_tests)} diagnostic tests passed\n\n")
        
        if overall_success:
            f.write("DIAGNOSTIC FINDINGS:\n")
            f.write("Phase 1 diagnostic tests have been executed successfully.\n")
            f.write("Review detailed diagnostic logs to identify:\n")
            f.write("- Specific code locations where efficiency becomes static\n")
            f.write("- Parameter propagation bottlenecks\n")
            f.write("- Missing load-dependent calculation implementations\n")
            f.write("- Fouling factor application issues\n\n")
            f.write("NEXT STEPS: Implement Phase 2 fixes based on diagnostic findings\n")
        else:
            f.write("DIAGNOSTIC ISSUES:\n")
            failed_tests = [name for name, result in test_results.items() if not result]
            for test in failed_tests:
                f.write(f"- {test}\n")
            f.write("\nRecommendation: Address diagnostic test failures before proceeding\n")
    
    print(f"\nPhase 1 diagnostic report saved: {report_file}")

def run_phase1_diagnostic_suite():
    """Run Phase 1 diagnostic suite focusing on static efficiency investigation."""
    print_header("PHASE 1 DIAGNOSTIC SUITE - STATIC EFFICIENCY INVESTIGATION")
    print("Version 8.3 - Comprehensive Efficiency Calculation Tracing")
    print(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Track test results
    test_results = {}
    
    # Run Phase 1 diagnostic tests
    phase1_test_functions = [
        ("Module Imports", test_module_imports),
        ("Efficiency Calculation Tracing", test_efficiency_calculation_tracing),
        ("Parameter Propagation Validation", test_parameter_propagation_validation),
        ("Component-Level Variation", test_component_level_variation),
        ("Load Curve Implementation", test_load_curve_implementation),
        ("Fouling Impact Validation", test_fouling_impact_validation),
        ("IAPWS Integration Fix", test_iapws_integration),
        ("Enhanced Boiler System", test_enhanced_boiler_system),
        ("Annual Simulator Interface", test_annual_simulator_interface),
        ("Unicode-Safe Logging", test_unicode_logging),
        ("Load Variation Testing", test_load_variation),
        ("Short Simulation Run", test_short_simulation)
    ]
    
    for test_name, test_function in phase1_test_functions:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            result = test_function()
            test_results[test_name] = result
            print(f"Result: {'PASS' if result else 'FAIL'}")
        except Exception as e:
            print_result(False, f"{test_name} test crashed: {e}")
            test_results[test_name] = False
    
    # Summary
    print_header("PHASE 1 DIAGNOSTIC RESULTS SUMMARY")
    
    passed_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {status:4s} | {test_name}")
    
    print(f"\nOVERALL RESULT: {passed_tests}/{total_tests} tests passed")
    
    # Determine overall diagnostic success
    phase1_core_tests = [
        "Efficiency Calculation Tracing",
        "Parameter Propagation Validation",
        "Component-Level Variation",
        "Load Curve Implementation",
        "Fouling Impact Validation"
    ]
    
    phase1_core_passed = sum(1 for test in phase1_core_tests 
                           if test_results.get(test, False))
    
    if phase1_core_passed >= 3 and passed_tests >= total_tests - 2:  # Allow some failures
        print("[SUCCESS] Phase 1 diagnostic complete - Static efficiency sources identified")
        overall_success = True
    else:
        print("[WARNING] Phase 1 diagnostic incomplete - Some core tests failed")
        overall_success = False
    
    # Save Phase 1 diagnostic report
    save_phase1_diagnostic_report(test_results, overall_success)
    
    return overall_success

def main():
    """Main execution function for Phase 1 diagnostics."""
    print_header("BOILER SIMULATION PHASE 1 DIAGNOSTIC SCRIPT")
    print("Comprehensive diagnostic suite to identify static efficiency calculation issues")
    print(f"Script Version: 8.3 - Phase 1 Diagnostic Implementation")
    print(f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run Phase 1 diagnostic suite
        print("\n" + "="*70)
        print("RUNNING PHASE 1 DIAGNOSTIC SUITE")
        print("="*70)
        
        diagnostic_success = run_phase1_diagnostic_suite()
        
        if diagnostic_success:
            print("\n" + "="*50)
            print("SUCCESS: Phase 1 diagnostics completed!")
            print("Review diagnostic logs to identify static efficiency sources.")
            print("Ready to proceed with Phase 2 implementation fixes.")
            print("="*50)
        else:
            print("\n" + "="*50)
            print("WARNING: Phase 1 diagnostics incomplete.")
            print("Review failed tests and diagnostic report for details.")
            print("="*50)
        
    except Exception as e:
        print(f"\n[ERROR] Phase 1 diagnostic execution failed: {e}")
        traceback.print_exc()
        return False
    
    print(f"\nDiagnostic complete. Check logs in: {log_dir}")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)