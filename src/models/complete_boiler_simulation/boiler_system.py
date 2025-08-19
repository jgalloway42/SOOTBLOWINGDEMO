#!/usr/bin/env python3
"""
Enhanced Complete Boiler System - IAPWS Integration with Solver Improvements

This module provides a comprehensive boiler system simulation with:
- IAPWS-97 steam properties for accurate efficiency calculations
- Enhanced solver stability with proper convergence
- Comprehensive logging for troubleshooting
- Realistic energy balance and performance calculations

Classes:
    EnhancedCompleteBoilerSystem: Main boiler system with IAPWS integration

Key Features:
- Industry-standard steam property calculations
- Robust solver convergence with damping
- Complete energy balance validation
- Professional logging and diagnostics

Author: Enhanced Boiler Modeling System
Version: 7.0 - IAPWS Integration with Solver Stability
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

# Import enhanced property calculator with IAPWS
from thermodynamic_properties import PropertyCalculator, SteamProperties, GasProperties

# Import existing components
from heat_transfer_calculations import HeatTransferCalculator
from fouling_and_soot_blowing import BoilerSection

# Set up logging
log_dir = Path("logs/solver")
log_dir.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create file handler for solver logs
solver_log_file = log_dir / "solver_convergence.log"
file_handler = logging.FileHandler(solver_log_file)
file_handler.setLevel(logging.DEBUG)

# Create console handler for warnings/errors
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


@dataclass
class SystemPerformance:
    """Enhanced system performance metrics with IAPWS-based calculations."""
    total_heat_absorbed: float      # Btu/hr
    system_efficiency: float        # Fraction (0-1)
    final_steam_temperature: float  # °F
    final_steam_pressure: float     # psia
    steam_superheat: float          # °F
    stack_temperature: float        # °F
    steam_production: float         # lbm/hr
    feedwater_temperature: float    # °F
    steam_enthalpy: float          # Btu/lb
    feedwater_enthalpy: float      # Btu/lb
    specific_energy: float         # Btu/lb
    
    # Energy balance components
    fuel_energy_input: float       # Btu/hr
    steam_energy_output: float     # Btu/hr
    stack_losses: float            # Btu/hr
    radiation_losses: float        # Btu/hr
    other_losses: float            # Btu/hr
    energy_balance_error: float    # Fraction


class EnhancedCompleteBoilerSystem:
    """
    Complete boiler system with IAPWS integration and enhanced solver stability.
    
    This class provides industrial-grade boiler simulation with:
    - IAPWS-97 steam properties for accurate efficiency calculations
    - Robust solver convergence with proper damping
    - Comprehensive energy balance validation
    - Professional logging and error handling
    """
    
    def __init__(self, fuel_input: float, flue_gas_mass_flow: float, 
                 furnace_exit_temp: float, base_fouling_multiplier: float = 0.5):
        """Initialize enhanced boiler system with IAPWS property calculator."""
        
        # System parameters
        self.design_capacity = 100e6  # Btu/hr design capacity
        self.fuel_input = fuel_input
        self.flue_gas_mass_flow = flue_gas_mass_flow
        self.furnace_exit_temp = furnace_exit_temp
        self.base_fouling_multiplier = base_fouling_multiplier
        
        # Enhanced property calculator with IAPWS
        self.property_calc = PropertyCalculator()
        self.heat_transfer_calc = HeatTransferCalculator()
        
        # Enhanced steam cycle parameters
        self.feedwater_temp = 220.0      # °F - realistic for industrial
        self.steam_pressure = 600.0      # psia - typical industrial pressure
        self.target_steam_temp = 700.0   # °F - superheated steam target
        self.target_stack_temp = 280.0   # °F - realistic stack temperature
        
        # Enhanced boiler sections with proper sizing
        self.sections = self._initialize_enhanced_sections()
        
        # Solver parameters for stability
        self.max_iterations = 25
        self.convergence_tolerance = 10.0  # °F
        self.damping_factor = 0.6          # Damping for stability
        self.min_damping = 0.3             # Minimum damping
        
        # Results storage
        self.system_performance: Optional[SystemPerformance] = None
        self.section_results: Dict = {}
        self.solver_history: List[Dict] = []
        self.attemperator_flow = 0.0
        
        logger.info(f"Enhanced boiler system initialized:")
        logger.info(f"  Design capacity: {self.design_capacity/1e6:.0f} MMBtu/hr")
        logger.info(f"  Steam conditions: {self.target_steam_temp}°F, {self.steam_pressure} psia")
        logger.info(f"  IAPWS property calculator: {self.property_calc.iapws_available}")
    
    def _initialize_enhanced_sections(self) -> Dict[str, BoilerSection]:
        """Initialize boiler sections with enhanced sizing for 100 MMBtu/hr capacity."""
        
        # Enhanced section specifications for realistic heat transfer
        section_configs = {
            'furnace_walls': {
                'num_segments': 12,
                'tube_count': 120,           # Increased from 80
                'tube_length': 40.0,         # Increased from 35
                'tube_od': 3.0,              # inches
                'initial_gas_fouling': 0.003, # Reduced from 0.008
                'initial_water_fouling': 0.001
            },
            'generating_bank': {
                'num_segments': 10,
                'tube_count': 180,           # Increased from 120
                'tube_length': 25.0,         # Increased from 20
                'tube_od': 2.5,
                'initial_gas_fouling': 0.004,
                'initial_water_fouling': 0.002
            },
            'superheater_primary': {
                'num_segments': 8,
                'tube_count': 90,            # Increased from 60
                'tube_length': 20.0,         # Increased from 15
                'tube_od': 2.0,
                'initial_gas_fouling': 0.005,
                'initial_water_fouling': 0.001
            },
            'superheater_secondary': {
                'num_segments': 8,
                'tube_count': 90,            # Increased from 60
                'tube_length': 20.0,         # Increased from 15
                'tube_od': 2.0,
                'initial_gas_fouling': 0.005,
                'initial_water_fouling': 0.001
            },
            'economizer_primary': {
                'num_segments': 10,
                'tube_count': 150,           # Increased from 100
                'tube_length': 22.0,         # Increased from 18
                'tube_od': 2.0,
                'initial_gas_fouling': 0.006,
                'initial_water_fouling': 0.003
            },
            'economizer_secondary': {
                'num_segments': 10,
                'tube_count': 150,           # Increased from 100
                'tube_length': 22.0,         # Increased from 18
                'tube_od': 2.0,
                'initial_gas_fouling': 0.006,
                'initial_water_fouling': 0.003
            },
            'air_heater': {
                'num_segments': 15,
                'tube_count': 300,           # Increased from 200
                'tube_length': 18.0,         # Increased from 12
                'tube_od': 1.5,
                'initial_gas_fouling': 0.0003, # Reduced from 0.0008
                'initial_water_fouling': 0.0001
            }
        }
        
        sections = {}
        for name, config in section_configs.items():
            section = BoilerSection(
                name=name,
                num_segments=config['num_segments'],
                tube_count=config['tube_count'],
                tube_length=config['tube_length'],
                tube_od=config['tube_od']
            )
            
            # Set initial fouling factors
            section.set_initial_fouling(
                config['initial_gas_fouling'],
                config['initial_water_fouling']
            )
            
            sections[name] = section
            logger.debug(f"Initialized {name}: {config['tube_count']} tubes × {config['tube_length']}ft")
        
        return sections
    
    def update_operating_conditions(self, fuel_input: float, flue_gas_mass_flow: float,
                                  furnace_exit_temp: float):
        """Update operating conditions and log changes."""
        old_fuel = self.fuel_input
        self.fuel_input = fuel_input
        self.flue_gas_mass_flow = flue_gas_mass_flow
        self.furnace_exit_temp = furnace_exit_temp
        
        logger.debug(f"Operating conditions updated:")
        logger.debug(f"  Fuel input: {old_fuel/1e6:.1f} → {fuel_input/1e6:.1f} MMBtu/hr")
        logger.debug(f"  Furnace exit: {furnace_exit_temp:.0f}°F")
    
    def solve_enhanced_system(self, max_iterations: Optional[int] = None, 
                            tolerance: Optional[float] = None) -> Dict:
        """
        Solve complete boiler system with enhanced stability and IAPWS integration.
        
        Returns:
            Dictionary with solution results and convergence information
        """
        max_iter = max_iterations or self.max_iterations
        tol = tolerance or self.convergence_tolerance
        
        logger.info(f"Starting enhanced boiler system solve...")
        logger.info(f"  Max iterations: {max_iter}, Tolerance: {tol}°F")
        
        # Clear previous results
        self.solver_history = []
        
        # Initial conditions with better estimates
        current_stack_temp = self._estimate_initial_stack_temp()
        current_steam_temp = self.target_steam_temp
        current_efficiency = 0.82  # Realistic initial guess
        
        # Solver loop with enhanced stability
        converged = False
        damping = self.damping_factor
        
        for iteration in range(max_iter):
            try:
                # Calculate system performance with current conditions
                performance = self._calculate_system_performance(
                    current_stack_temp, current_steam_temp
                )
                
                # Calculate new estimates
                new_stack_temp, new_steam_temp, new_efficiency = self._calculate_new_estimates(
                    performance, current_stack_temp, current_steam_temp
                )
                
                # Apply damping for stability
                stack_correction = (new_stack_temp - current_stack_temp) * damping
                steam_correction = (new_steam_temp - current_steam_temp) * damping
                
                current_stack_temp += stack_correction
                current_steam_temp += steam_correction
                current_efficiency = new_efficiency
                
                # Enforce reasonable bounds
                current_stack_temp = max(200, min(400, current_stack_temp))
                current_steam_temp = max(500, min(800, current_steam_temp))
                
                # Log iteration progress
                iter_data = {
                    'iteration': iteration + 1,
                    'stack_temp': current_stack_temp,
                    'steam_temp': current_steam_temp,
                    'efficiency': current_efficiency,
                    'stack_correction': stack_correction,
                    'steam_correction': steam_correction,
                    'damping': damping
                }
                self.solver_history.append(iter_data)
                
                logger.debug(f"Iteration {iteration+1}: Stack={current_stack_temp:.0f}°F, "
                           f"Steam={current_steam_temp:.0f}°F, Eff={current_efficiency:.1%}")
                
                # Check convergence
                if abs(stack_correction) < tol and abs(steam_correction) < tol:
                    converged = True
                    logger.info(f"Solver converged in {iteration+1} iterations")
                    break
                
                # Adaptive damping - reduce if oscillating
                if iteration > 5:
                    recent_corrections = [h['stack_correction'] for h in self.solver_history[-3:]]
                    if self._is_oscillating(recent_corrections):
                        damping = max(self.min_damping, damping * 0.8)
                        logger.debug(f"Oscillation detected, reducing damping to {damping:.2f}")
                
            except Exception as e:
                logger.error(f"Solver iteration {iteration+1} failed: {e}")
                # Use fallback calculation
                current_stack_temp = self.target_stack_temp
                current_steam_temp = self.target_steam_temp
                break
        
        # Calculate final performance
        try:
            final_performance = self._calculate_system_performance(
                current_stack_temp, current_steam_temp
            )
            self.system_performance = final_performance
        except Exception as e:
            logger.error(f"Final performance calculation failed: {e}")
            final_performance = self._create_fallback_performance(
                current_stack_temp, current_steam_temp, current_efficiency
            )
            self.system_performance = final_performance
        
        # Log final results
        self._log_final_results(converged, final_performance)
        
        return {
            'converged': converged,
            'iterations': len(self.solver_history),
            'final_performance': final_performance,
            'solver_history': self.solver_history
        }
    
    def _estimate_initial_stack_temp(self) -> float:
        """Estimate realistic initial stack temperature."""
        # Based on furnace exit temp and typical cooling
        cooling_ratio = 0.1  # Stack temp is ~10% of furnace temp above ambient
        ambient_temp = 60.0   # Assume 60°F ambient
        return ambient_temp + (self.furnace_exit_temp - ambient_temp) * cooling_ratio
    
    def _calculate_system_performance(self, stack_temp: float, steam_temp: float) -> SystemPerformance:
        """Calculate complete system performance using IAPWS steam properties."""
        
        # Get IAPWS steam properties
        steam_props = self.property_calc.get_steam_properties(steam_temp, self.steam_pressure)
        feedwater_props = self.property_calc.get_steam_properties(self.feedwater_temp, self.steam_pressure)
        
        logger.debug(f"IAPWS properties: Steam h={steam_props.enthalpy:.1f}, Water h={feedwater_props.enthalpy:.1f}")
        
        # Calculate specific energy (CRITICAL for efficiency)
        specific_energy = steam_props.enthalpy - feedwater_props.enthalpy
        
        # Estimate steam production rate
        steam_production = self._estimate_steam_production()
        
        # Calculate energy flows
        fuel_energy_input = self.fuel_input
        steam_energy_output = steam_production * specific_energy
        stack_losses = self._calculate_stack_losses(stack_temp)
        radiation_losses = self._calculate_radiation_losses()
        other_losses = self._calculate_other_losses()
        
        # System efficiency using IAPWS-based calculation
        system_efficiency = steam_energy_output / fuel_energy_input
        
        # Energy balance check
        total_outputs = steam_energy_output + stack_losses + radiation_losses + other_losses
        energy_balance_error = abs(fuel_energy_input - total_outputs) / fuel_energy_input
        
        # Calculate superheat
        saturation_temp = steam_props.saturation_temp
        steam_superheat = steam_temp - saturation_temp
        
        # Total heat absorbed by steam
        total_heat_absorbed = steam_energy_output
        
        logger.debug(f"Energy balance: Input={fuel_energy_input/1e6:.1f}, "
                   f"Steam={steam_energy_output/1e6:.1f}, "
                   f"Stack={stack_losses/1e6:.1f}, "
                   f"Efficiency={system_efficiency:.1%}")
        
        return SystemPerformance(
            total_heat_absorbed=total_heat_absorbed,
            system_efficiency=system_efficiency,
            final_steam_temperature=steam_temp,
            final_steam_pressure=self.steam_pressure,
            steam_superheat=steam_superheat,
            stack_temperature=stack_temp,
            steam_production=steam_production,
            feedwater_temperature=self.feedwater_temp,
            steam_enthalpy=steam_props.enthalpy,
            feedwater_enthalpy=feedwater_props.enthalpy,
            specific_energy=specific_energy,
            fuel_energy_input=fuel_energy_input,
            steam_energy_output=steam_energy_output,
            stack_losses=stack_losses,
            radiation_losses=radiation_losses,
            other_losses=other_losses,
            energy_balance_error=energy_balance_error
        )
    
    def _calculate_new_estimates(self, performance: SystemPerformance, 
                               current_stack: float, current_steam: float) -> Tuple[float, float, float]:
        """Calculate new temperature estimates based on energy balance."""
        
        # Stack temperature correction based on energy balance
        if performance.energy_balance_error > 0.05:  # > 5% error
            # Adjust stack temperature to balance energy
            error_magnitude = performance.energy_balance_error
            stack_adjustment = error_magnitude * 50  # Proportional adjustment
            new_stack_temp = current_stack + stack_adjustment
        else:
            new_stack_temp = current_stack
        
        # Steam temperature adjustment for target superheat
        target_superheat = 100.0  # °F target superheat
        current_superheat = performance.steam_superheat
        superheat_error = target_superheat - current_superheat
        
        steam_adjustment = superheat_error * 0.5  # Proportional control
        new_steam_temp = current_steam + steam_adjustment
        
        return new_stack_temp, new_steam_temp, performance.system_efficiency
    
    def _estimate_steam_production(self) -> float:
        """Estimate steam production rate based on system capacity."""
        # Typical steam production for 100 MMBtu/hr boiler
        base_steam_rate = 68000  # lbm/hr at full load
        load_factor = self.fuel_input / self.design_capacity
        return base_steam_rate * load_factor
    
    def _calculate_stack_losses(self, stack_temp: float) -> float:
        """Calculate stack gas energy losses."""
        ambient_temp = 60.0  # °F
        flue_gas_cp = 0.25   # Btu/lbm-°F (typical for flue gas)
        
        stack_loss = self.flue_gas_mass_flow * flue_gas_cp * (stack_temp - ambient_temp)
        return max(0, stack_loss)
    
    def _calculate_radiation_losses(self) -> float:
        """Calculate radiation and convection losses from boiler surface."""
        # Typical radiation losses for industrial boiler: 2-4% of fuel input
        radiation_fraction = 0.03  # 3% typical
        return self.fuel_input * radiation_fraction
    
    def _calculate_other_losses(self) -> float:
        """Calculate other miscellaneous losses."""
        # Incomplete combustion, ash handling, etc.: ~1-2%
        other_fraction = 0.015  # 1.5% typical
        return self.fuel_input * other_fraction
    
    def _is_oscillating(self, corrections: List[float]) -> bool:
        """Check if solver is oscillating."""
        if len(corrections) < 3:
            return False
        
        # Check for sign changes indicating oscillation
        signs = [1 if x > 0 else -1 for x in corrections]
        sign_changes = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i-1])
        
        return sign_changes >= 2
    
    def _create_fallback_performance(self, stack_temp: float, steam_temp: float, 
                                   efficiency: float) -> SystemPerformance:
        """Create fallback performance when calculations fail."""
        logger.warning("Using fallback performance calculations")
        
        # Use correlations for fallback
        steam_production = self._estimate_steam_production()
        specific_energy = 1000.0  # Btu/lb - approximate
        
        steam_energy = steam_production * specific_energy
        fuel_energy = self.fuel_input
        stack_losses = fuel_energy * 0.10  # 10% to stack
        radiation_losses = fuel_energy * 0.03  # 3% radiation
        other_losses = fuel_energy * 0.02  # 2% other
        
        return SystemPerformance(
            total_heat_absorbed=steam_energy,
            system_efficiency=efficiency,
            final_steam_temperature=steam_temp,
            final_steam_pressure=self.steam_pressure,
            steam_superheat=100.0,
            stack_temperature=stack_temp,
            steam_production=steam_production,
            feedwater_temperature=self.feedwater_temp,
            steam_enthalpy=1200.0,
            feedwater_enthalpy=200.0,
            specific_energy=specific_energy,
            fuel_energy_input=fuel_energy,
            steam_energy_output=steam_energy,
            stack_losses=stack_losses,
            radiation_losses=radiation_losses,
            other_losses=other_losses,
            energy_balance_error=0.05
        )
    
    def _log_final_results(self, converged: bool, performance: SystemPerformance):
        """Log final solver results."""
        logger.info(f"Enhanced boiler system solve completed:")
        logger.info(f"  Converged: {converged}")
        logger.info(f"  System efficiency: {performance.system_efficiency:.1%}")
        logger.info(f"  Stack temperature: {performance.stack_temperature:.0f}°F")
        logger.info(f"  Steam conditions: {performance.final_steam_temperature:.0f}°F, {performance.steam_superheat:.0f}°F superheat")
        logger.info(f"  Energy balance error: {performance.energy_balance_error:.1%}")
        logger.info(f"  Specific energy: {performance.specific_energy:.0f} Btu/lb")
        
        # Log energy breakdown
        total_input = performance.fuel_energy_input
        logger.info(f"Energy breakdown:")
        logger.info(f"  Steam: {performance.steam_energy_output/total_input:.1%}")
        logger.info(f"  Stack: {performance.stack_losses/total_input:.1%}")
        logger.info(f"  Radiation: {performance.radiation_losses/total_input:.1%}")
        logger.info(f"  Other: {performance.other_losses/total_input:.1%}")
        
        # Warning for unrealistic results
        if performance.system_efficiency < 0.70 or performance.system_efficiency > 0.90:
            logger.warning(f"System efficiency {performance.system_efficiency:.1%} outside typical range (70-90%)")
        
        if performance.energy_balance_error > 0.05:
            logger.warning(f"Energy balance error {performance.energy_balance_error:.1%} exceeds 5%")


# Test function for the enhanced system
def test_enhanced_boiler_system():
    """Test the enhanced boiler system with IAPWS integration."""
    print("Testing Enhanced Boiler System with IAPWS Integration...")
    
    # Create enhanced boiler system
    boiler = EnhancedCompleteBoilerSystem(
        fuel_input=100e6,      # 100 MMBtu/hr
        flue_gas_mass_flow=84000,  # lbm/hr
        furnace_exit_temp=2800,    # °F
        base_fouling_multiplier=0.5
    )
    
    # Solve system
    print("\nSolving enhanced boiler system...")
    results = boiler.solve_enhanced_system()
    
    # Display results
    perf = boiler.system_performance
    print(f"\nResults:")
    print(f"  Converged: {results['converged']}")
    print(f"  Iterations: {results['iterations']}")
    print(f"  System Efficiency: {perf.system_efficiency:.1%}")
    print(f"  Stack Temperature: {perf.stack_temperature:.0f}°F")
    print(f"  Steam Temperature: {perf.final_steam_temperature:.0f}°F")
    print(f"  Steam Superheat: {perf.steam_superheat:.0f}°F")
    print(f"  Specific Energy: {perf.specific_energy:.0f} Btu/lb")
    print(f"  Energy Balance Error: {perf.energy_balance_error:.1%}")
    
    # IAPWS verification
    print(f"\nIAPWS Steam Properties:")
    print(f"  Steam Enthalpy: {perf.steam_enthalpy:.0f} Btu/lb")
    print(f"  Feedwater Enthalpy: {perf.feedwater_enthalpy:.0f} Btu/lb")
    print(f"  Steam Production: {perf.steam_production:.0f} lbm/hr")
    
    return boiler, results


if __name__ == "__main__":
    test_enhanced_boiler_system()