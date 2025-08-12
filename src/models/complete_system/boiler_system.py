#!/usr/bin/env python3
"""
Complete Enhanced Boiler System Model

This module contains the complete boiler system model with thermo library integration,
configurable operating conditions, and comprehensive heat transfer analysis.

Classes:
    EnhancedCompleteBoilerSystem: Main boiler system model

Dependencies:
    - thermodynamic_properties: Property calculations
    - heat_transfer_calculations: Heat transfer and tube sections
    - typing: Type hints

Author: Enhanced Boiler Modeling System
Version: 5.0 - Complete Integration
"""

from typing import Dict, Optional, Union

from thermodynamic_properties import PropertyCalculator
from heat_transfer_calculations import EnhancedBoilerTubeSection


class EnhancedCompleteBoilerSystem:
    """Complete enhanced boiler system model with thermo library integration."""
    
    def __init__(self, fuel_input: float = 100e6, flue_gas_mass_flow: float = 84000,
                 furnace_exit_temp: float = 3000, base_fouling_multiplier: float = 1.0):
        """
        Initialize enhanced boiler system with configurable parameters.
        
        Args:
            fuel_input: Fuel heat input rate (Btu/hr)
            flue_gas_mass_flow: Total flue gas mass flow rate (lbm/hr)
            furnace_exit_temp: Furnace exit gas temperature (°F)
            base_fouling_multiplier: Multiplier for all base fouling factors (1.0 = default)
        """
        self.design_capacity = 100e6  # Btu/hr
        
        # User-configurable parameters
        self.fuel_input = fuel_input  # Btu/hr
        self.flue_gas_mass_flow = flue_gas_mass_flow  # lbm/hr
        self.furnace_exit_temp = furnace_exit_temp  # °F
        self.base_fouling_multiplier = base_fouling_multiplier
        
        # Initialize sections with configurable fouling
        self.sections = self._initialize_enhanced_sections()
        
        # Other system operating parameters
        self.combustion_efficiency = 0.85
        self.feedwater_flow = 68000  # lbm/hr (scaled for 100 MMBtu/hr)
        self.steam_pressure = 600  # psia (typical for smaller boiler)
        self.attemperator_flow = 0  # lbm/hr
        
        # Temperature targets
        self.feedwater_temp = 220  # °F
        self.final_steam_temp_target = 700  # °F (realistic superheat ~215°F at 600 psia)
        self.stack_temp_target = 300  # °F (slightly higher for smaller system)
        
        # Initialize property calculator
        self.property_calc = PropertyCalculator()
        
        # Results storage
        self.section_results: Dict[str, Dict] = {}
        self.system_performance: Dict[str, float] = {}
    
    def update_operating_conditions(self, fuel_input: Optional[float] = None,
                                  flue_gas_mass_flow: Optional[float] = None,
                                  furnace_exit_temp: Optional[float] = None,
                                  base_fouling_multiplier: Optional[float] = None):
        """Update operating conditions and reinitialize sections if fouling changed."""
        fouling_changed = False
        
        if fuel_input is not None:
            self.fuel_input = fuel_input
        if flue_gas_mass_flow is not None:
            self.flue_gas_mass_flow = flue_gas_mass_flow
        if furnace_exit_temp is not None:
            self.furnace_exit_temp = furnace_exit_temp
        if base_fouling_multiplier is not None:
            if abs(base_fouling_multiplier - self.base_fouling_multiplier) > 0.001:
                fouling_changed = True
            self.base_fouling_multiplier = base_fouling_multiplier
        
        # Reinitialize sections if fouling factors changed
        if fouling_changed:
            self.sections = self._initialize_enhanced_sections()
            print(f"✓ Sections reinitialized with fouling multiplier: {self.base_fouling_multiplier}")
    
    def _initialize_enhanced_sections(self) -> Dict[str, EnhancedBoilerTubeSection]:
        """Initialize all boiler heat transfer sections sized for 100 MMBtu/hr with configurable fouling."""
        sections = {}
        
        # Base fouling factors (before multiplier)
        base_fouling_factors = {
            'furnace_walls': {'gas': 0.0005, 'water': 0.0003},
            'generating_bank': {'gas': 0.002, 'water': 0.001},
            'superheater_primary': {'gas': 0.0015, 'water': 0.0005},
            'superheater_secondary': {'gas': 0.002, 'water': 0.0005},
            'economizer_primary': {'gas': 0.003, 'water': 0.002},
            'economizer_secondary': {'gas': 0.004, 'water': 0.0025},
            'air_heater': {'gas': 0.006, 'water': 0.001}
        }
        
        # Furnace Wall Tubes (scaled down by ~5x from 500 MMBtu design)
        fouling = base_fouling_factors['furnace_walls']
        sections['furnace_walls'] = EnhancedBoilerTubeSection(
            name="Furnace Wall Tubes",
            tube_od=2.5/12, tube_id=2.0/12, tube_length=35, tube_count=80,
            base_fouling_gas=fouling['gas'] * self.base_fouling_multiplier,
            base_fouling_water=fouling['water'] * self.base_fouling_multiplier,
            section_type='radiant'
        )
        
        # Generating Bank (scaled down)
        fouling = base_fouling_factors['generating_bank']
        sections['generating_bank'] = EnhancedBoilerTubeSection(
            name="Generating Bank", 
            tube_od=2.0/12, tube_id=1.75/12, tube_length=20, tube_count=160,
            base_fouling_gas=fouling['gas'] * self.base_fouling_multiplier,
            base_fouling_water=fouling['water'] * self.base_fouling_multiplier,
            section_type='convective'
        )
        
        # Primary Superheater (scaled down)
        fouling = base_fouling_factors['superheater_primary']
        sections['superheater_primary'] = EnhancedBoilerTubeSection(
            name="Primary Superheater",
            tube_od=2.0/12, tube_id=1.75/12, tube_length=15, tube_count=60,
            base_fouling_gas=fouling['gas'] * self.base_fouling_multiplier,
            base_fouling_water=fouling['water'] * self.base_fouling_multiplier,
            section_type='superheater'
        )
        
        # Secondary Superheater (scaled down)
        fouling = base_fouling_factors['superheater_secondary']
        sections['superheater_secondary'] = EnhancedBoilerTubeSection(
            name="Secondary Superheater",
            tube_od=2.0/12, tube_id=1.75/12, tube_length=15, tube_count=50,
            base_fouling_gas=fouling['gas'] * self.base_fouling_multiplier,
            base_fouling_water=fouling['water'] * self.base_fouling_multiplier,
            section_type='superheater'
        )
        
        # Primary Economizer (scaled down)
        fouling = base_fouling_factors['economizer_primary']
        sections['economizer_primary'] = EnhancedBoilerTubeSection(
            name="Primary Economizer",
            tube_od=2.0/12, tube_id=1.75/12, tube_length=18, tube_count=100,
            base_fouling_gas=fouling['gas'] * self.base_fouling_multiplier,
            base_fouling_water=fouling['water'] * self.base_fouling_multiplier,
            section_type='economizer'
        )
        
        # Secondary Economizer (scaled down)
        fouling = base_fouling_factors['economizer_secondary']
        sections['economizer_secondary'] = EnhancedBoilerTubeSection(
            name="Secondary Economizer",
            tube_od=2.0/12, tube_id=1.75/12, tube_length=15, tube_count=80,
            base_fouling_gas=fouling['gas'] * self.base_fouling_multiplier,
            base_fouling_water=fouling['water'] * self.base_fouling_multiplier,
            section_type='economizer'
        )
        
        # Air Heater (scaled down)
        fouling = base_fouling_factors['air_heater']
        sections['air_heater'] = EnhancedBoilerTubeSection(
            name="Air Heater",
            tube_od=1.75/12, tube_id=1.5/12, tube_length=12, tube_count=200,
            base_fouling_gas=fouling['gas'] * self.base_fouling_multiplier,
            base_fouling_water=fouling['water'] * self.base_fouling_multiplier,
            section_type='convective'
        )
        
        return sections
    
    def calculate_attemperator_flow(self, steam_temp_before: float, 
                                  target_temp: float, steam_flow: float) -> float:
        """Calculate attemperator spray water flow for temperature control."""
        if steam_temp_before <= target_temp:
            return 0
        
        # Get properties for energy balance
        steam_props = self.property_calc.get_steam_properties_safe(steam_temp_before, self.steam_pressure)
        water_props = self.property_calc.get_steam_properties_safe(self.feedwater_temp, self.steam_pressure)
        
        # Energy balance calculation
        numerator = steam_flow * steam_props.cp * (steam_temp_before - target_temp)
        denominator = water_props.cp * (target_temp - self.feedwater_temp) + steam_props.cp * (target_temp - steam_temp_before)
        
        if denominator <= 0:
            return 0
        
        spray_flow = numerator / denominator
        return max(0, min(spray_flow, steam_flow * 0.1))  # Limit to 10%
    
    def solve_enhanced_system(self, max_iterations: int = 15, tolerance: float = 3.0) -> Dict:
        """Solve the complete enhanced boiler system with iterative convergence."""
        print(f"Solving enhanced boiler system with thermo library...")
        print(f"Target: Stack {self.stack_temp_target}°F, Steam {self.final_steam_temp_target}°F")
        
        # Flow distribution (scaled for 100 MMBtu/hr)
        main_steam_flow = self.feedwater_flow * 0.97
        flows = {
            'furnace_walls': self.feedwater_flow * 0.35,
            'generating_bank': self.feedwater_flow,
            'superheater_primary': main_steam_flow,
            'superheater_secondary': main_steam_flow,
            'economizer_primary': self.feedwater_flow,
            'economizer_secondary': self.feedwater_flow,
            'air_heater': 56000,  # Combustion air flow (scaled from 280k)
        }
        
        # Section order
        section_order = ['furnace_walls', 'generating_bank', 'superheater_primary', 
                        'superheater_secondary', 'economizer_primary', 
                        'economizer_secondary', 'air_heater']
        
        # Initial water temperatures
        water_temps = {
            'furnace_walls': 350,
            'generating_bank': 400,
            'superheater_primary': 532,
            'superheater_secondary': 650,
            'economizer_primary': 280,
            'economizer_secondary': self.feedwater_temp,
            'air_heater': 80,
        }
        
        # Iterative solution
        for iteration in range(max_iterations):
            current_gas_temp = self.furnace_exit_temp
            total_heat_absorbed = 0
            section_summaries = {}
            
            print(f"\nIteration {iteration + 1}:")
            
            for section_name in section_order:
                section = self.sections[section_name]
                water_flow = flows[section_name]
                water_temp_in = water_temps[section_name]
                
                print(f"  Solving {section_name}: Gas {current_gas_temp:.0f}°F, Water {water_temp_in:.0f}°F")
                
                try:
                    segment_results = section.solve_section(
                        current_gas_temp, water_temp_in, self.flue_gas_mass_flow, 
                        water_flow, self.steam_pressure
                    )
                    
                    summary = section.get_section_summary()
                    section_summaries[section_name] = summary
                    current_gas_temp = summary['gas_temp_out']
                    total_heat_absorbed += summary['total_heat_transfer']
                    
                    self.section_results[section_name] = {
                        'summary': summary,
                        'segments': segment_results
                    }
                    
                    print(f"    Q: {summary['total_heat_transfer']/1e6:.1f} MMBtu/hr, Gas out: {current_gas_temp:.0f}°F")
                    
                except Exception as e:
                    print(f"    Error: {e}")
                    raise RuntimeError(f"Failed to solve {section_name}: {e}")
            
            # Attemperator control
            final_steam_temp = section_summaries['superheater_secondary']['water_temp_out']
            temp_error = abs(final_steam_temp - self.final_steam_temp_target)
            
            if temp_error > tolerance:
                self.attemperator_flow = self.calculate_attemperator_flow(
                    final_steam_temp, self.final_steam_temp_target, main_steam_flow
                )
                
                if self.attemperator_flow > 0:
                    # Apply attemperator correction
                    steam_props = self.property_calc.get_steam_properties_safe(final_steam_temp, self.steam_pressure)
                    water_props = self.property_calc.get_steam_properties_safe(self.feedwater_temp, self.steam_pressure)
                    
                    adjusted_temp = ((final_steam_temp * main_steam_flow * steam_props.cp + 
                                    self.feedwater_temp * self.attemperator_flow * water_props.cp) / 
                                   (main_steam_flow * steam_props.cp + self.attemperator_flow * water_props.cp))
                    
                    section_summaries['superheater_secondary']['water_temp_out'] = adjusted_temp
                    final_steam_temp = adjusted_temp
                    
                    print(f"    Attemperator: {self.attemperator_flow:.0f} lbm/hr, Final temp: {final_steam_temp:.1f}°F")
            
            # Check convergence
            stack_temp = current_gas_temp
            efficiency = total_heat_absorbed / self.fuel_input
            stack_error = abs(stack_temp - self.stack_temp_target)
            
            print(f"  Stack: {stack_temp:.1f}°F, Steam: {final_steam_temp:.1f}°F, Efficiency: {efficiency:.1%}")
            
            if stack_error < tolerance and temp_error < tolerance:
                print(f"\n✓ Converged after {iteration + 1} iterations")
                break
            
            # Update temperatures with damping
            damping = 0.7
            for section_name in section_order:
                if section_name in section_summaries:
                    old_temp = water_temps.get(section_name, 200)
                    new_temp = section_summaries[section_name]['water_temp_out']
                    water_temps[section_name] = old_temp * (1 - damping) + new_temp * damping
        
        else:
            print(f"\n⚠ Did not converge within {max_iterations} iterations")
        
        # Store system performance
        self.system_performance = {
            'total_heat_absorbed': total_heat_absorbed,
            'system_efficiency': efficiency,
            'final_steam_temperature': final_steam_temp,
            'steam_superheat': final_steam_temp - 532,
            'stack_temperature': stack_temp,
            'attemperator_flow': self.attemperator_flow,
            'steam_production': main_steam_flow + self.attemperator_flow,
            'iterations_to_converge': iteration + 1
        }
        
        return self.section_results