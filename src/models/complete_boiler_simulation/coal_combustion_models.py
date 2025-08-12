#!/usr/bin/env python3
"""
CORRECTED: Coal Combustion and Soot Production Models

This module contains the coal combustion analysis and soot production modeling
classes with CORRECTED fouling distribution patterns that match real boiler physics.

MAJOR CORRECTION: Fouling integration now follows realistic physics:
- Maximum fouling in hot furnace zones (soot formation)
- Decreasing fouling toward stack (cooling gas, less adhesion)

Classes:
    CoalCombustionModel: Simplified coal combustion analysis
    SootProductionModel: Soot formation and deposition modeling
    SootProductionData: Dataclass for soot characteristics
    CombustionFoulingIntegrator: CORRECTED fouling integration

Dependencies:
    - numpy: Numerical calculations
    - dataclasses: For structured data
    - typing: Type hints

Author: Enhanced Boiler Modeling System
Version: 5.1 - CORRECTED Soot Deposition Physics
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class SootProductionData:
    """Soot production and characteristics data"""
    mass_production_rate: float  # lb/hr
    particle_size_microns: float  # μm
    carbon_content: float  # fraction
    ash_content: float  # fraction
    deposition_tendency: float  # 0-1 scale
    erosion_factor: float  # relative to baseline


class CoalCombustionModel:
    """Simplified coal combustion model for integration with boiler system"""
    
    def __init__(self, ultimate_analysis, coal_lb_per_hr, air_scfh, NOx_eff=0.35,
                 air_temp_F=75.0, air_RH_pct=55.0, atm_press_inHg=30.25):
        """Initialize coal combustion model."""
        self._ultimate_analysis = ultimate_analysis.copy()
        self._coal_lb_per_hr = coal_lb_per_hr
        self._air_scfh = air_scfh
        self._NOx_eff = NOx_eff
        self._air_temp_F = air_temp_F
        self._air_RH_pct = air_RH_pct
        self._atm_press_inHg = atm_press_inHg
        self._results = {}
        self._calculated = False
    
    def calculate(self, debug=False):
        """Simplified calculation for integration with boiler system"""
        # Simplified combustion calculations
        fuel_input = self._coal_lb_per_hr * 12000  # Approximate BTU/lb
        excess_air = max(0, (self._air_scfh / 10 / self._coal_lb_per_hr) - 8)  # Simplified
        
        self._results = {
            'total_flue_gas_lb_per_hr': self._coal_lb_per_hr * 12 + self._air_scfh * 0.075,
            'CO2_lb_per_hr': self._coal_lb_per_hr * 2.5,
            'NO_total_lb_per_hr': self._coal_lb_per_hr * 0.01 * self._NOx_eff,
            'NO_thermal_lb_per_hr': self._coal_lb_per_hr * 0.003,
            'NO_fuel_lb_per_hr': self._coal_lb_per_hr * 0.007 * self._NOx_eff,
            'dry_O2_pct': excess_air * 0.8,
            'combustion_efficiency': 0.98 - excess_air * 0.002,
            'heat_released_btu_per_hr': fuel_input * (0.98 - excess_air * 0.002),
            'flame_temp_F': 3200 - excess_air * 20
        }
        self._calculated = True
    
    @property
    def total_flue_gas_lb_per_hr(self):
        if not self._calculated: self.calculate()
        return self._results['total_flue_gas_lb_per_hr']
    
    @property
    def NO_total_lb_per_hr(self):
        if not self._calculated: self.calculate()
        return self._results['NO_total_lb_per_hr']
    
    @property
    def NO_thermal_lb_per_hr(self):
        if not self._calculated: self.calculate()
        return self._results['NO_thermal_lb_per_hr']
    
    @property
    def NO_fuel_lb_per_hr(self):
        if not self._calculated: self.calculate()
        return self._results['NO_fuel_lb_per_hr']
    
    @property
    def dry_O2_pct(self):
        if not self._calculated: self.calculate()
        return self._results['dry_O2_pct']
    
    @property
    def combustion_efficiency(self):
        if not self._calculated: self.calculate()
        return self._results['combustion_efficiency']
    
    @property
    def heat_released_btu_per_hr(self):
        if not self._calculated: self.calculate()
        return self._results['heat_released_btu_per_hr']
    
    @property
    def flame_temp_F(self):
        if not self._calculated: self.calculate()
        return self._results['flame_temp_F']


class SootProductionModel:
    """Model soot production based on combustion conditions."""
    
    def __init__(self):
        """Initialize soot production model with empirical correlations."""
        # Empirical constants based on literature and field data
        self.base_soot_rate = 0.001  # lb soot per lb coal (baseline)
        self.nox_soot_correlation = 0.8  # Higher NOx often means more soot
        self.efficiency_impact = 2.0  # Lower efficiency increases soot
        self.temperature_impact = 1.5  # Temperature effects on soot formation
        
    def calculate_soot_production(self, combustion_model: CoalCombustionModel,
                                coal_properties: Dict) -> SootProductionData:
        """Calculate soot production based on combustion conditions."""
        # Get combustion parameters
        thermal_nox = combustion_model.NO_thermal_lb_per_hr
        fuel_nox = combustion_model.NO_fuel_lb_per_hr
        total_nox = combustion_model.NO_total_lb_per_hr
        excess_o2 = combustion_model.dry_O2_pct
        combustion_eff = combustion_model.combustion_efficiency
        flame_temp = combustion_model.flame_temp_F
        coal_rate = combustion_model._coal_lb_per_hr
        
        # Calculate soot production factors
        nox_factor = self._calculate_nox_soot_factor(thermal_nox, fuel_nox, coal_rate)
        excess_air_factor = self._calculate_excess_air_factor(excess_o2)
        efficiency_factor = self._calculate_efficiency_factor(combustion_eff)
        temperature_factor = self._calculate_temperature_factor(flame_temp)
        coal_factor = self._calculate_coal_factor(coal_properties)
        
        # Base soot production rate
        base_production = self.base_soot_rate * coal_rate
        
        # Apply all factors
        actual_soot_rate = (base_production * nox_factor * excess_air_factor * 
                           efficiency_factor * temperature_factor * coal_factor)
        
        # Calculate soot characteristics
        particle_size = self._calculate_particle_size(flame_temp, excess_o2)
        carbon_content = self._calculate_carbon_content(combustion_eff, coal_properties)
        ash_content = 1.0 - carbon_content
        deposition_tendency = self._calculate_deposition_tendency(
            particle_size, flame_temp, excess_o2
        )
        erosion_factor = self._calculate_erosion_factor(particle_size, coal_properties)
        
        return SootProductionData(
            mass_production_rate=actual_soot_rate,
            particle_size_microns=particle_size,
            carbon_content=carbon_content,
            ash_content=ash_content,
            deposition_tendency=deposition_tendency,
            erosion_factor=erosion_factor
        )
    
    def _calculate_nox_soot_factor(self, thermal_nox: float, fuel_nox: float, 
                                  coal_rate: float) -> float:
        """Calculate soot factor based on NOx formation."""
        # Higher NOx formation often correlates with incomplete combustion zones
        nox_ppm = (thermal_nox + fuel_nox) / coal_rate * 1000  # Approximate ppm
        
        if nox_ppm < 100:
            return 0.5  # Very clean combustion, low soot
        elif nox_ppm < 300:
            return 1.0  # Normal combustion
        elif nox_ppm < 500:
            return 1.5  # Higher soot formation
        else:
            return 2.0  # Poor combustion, high soot
    
    def _calculate_excess_air_factor(self, excess_o2_pct: float) -> float:
        """Calculate soot factor based on excess air."""
        if excess_o2_pct < 1:
            return 3.0  # Very fuel-rich, high soot
        elif excess_o2_pct < 3:
            return 1.8  # Slightly rich, increased soot
        elif excess_o2_pct < 6:
            return 1.0  # Optimal air/fuel ratio
        elif excess_o2_pct < 10:
            return 0.8  # Slightly lean, reduced soot
        else:
            return 0.6  # Very lean, low soot but poor efficiency
    
    def _calculate_efficiency_factor(self, combustion_eff: float) -> float:
        """Calculate soot factor based on combustion efficiency."""
        # Lower efficiency correlates with incomplete combustion and more soot
        return 2.0 - combustion_eff  # Linear relationship
    
    def _calculate_temperature_factor(self, flame_temp_F: float) -> float:
        """Calculate soot factor based on flame temperature."""
        # Soot formation peaks at intermediate temperatures
        optimal_temp = 3000  # °F for minimal soot
        temp_deviation = abs(flame_temp_F - optimal_temp) / 500
        return 1.0 + temp_deviation * 0.5
    
    def _calculate_coal_factor(self, coal_properties: Dict) -> float:
        """Calculate soot factor based on coal properties."""
        # Higher volatile matter and lower fixed carbon increase soot tendency
        volatile_matter = coal_properties.get('volatile_matter', 30)  # %
        fixed_carbon = coal_properties.get('fixed_carbon', 50)  # %
        sulfur = coal_properties.get('sulfur', 1.0)  # %
        
        vm_factor = 1.0 + (volatile_matter - 30) * 0.02  # Higher VM increases soot
        fc_factor = 1.0 + (50 - fixed_carbon) * 0.01  # Lower FC increases soot
        s_factor = 1.0 + sulfur * 0.1  # Sulfur can affect soot formation
        
        return vm_factor * fc_factor * s_factor
    
    def _calculate_particle_size(self, flame_temp_F: float, excess_o2_pct: float) -> float:
        """Calculate average soot particle size."""
        base_size = 2.0  # μm
        temp_effect = (flame_temp_F - 2500) / 1000 * 0.5  # Temperature effect
        o2_effect = (excess_o2_pct - 3) * 0.1  # Oxygen effect
        return max(0.5, base_size + temp_effect + o2_effect)
    
    def _calculate_carbon_content(self, combustion_eff: float, 
                                 coal_properties: Dict) -> float:
        """Calculate carbon content of soot."""
        base_carbon = 0.85  # 85% carbon baseline
        eff_effect = (1.0 - combustion_eff) * 0.5  # Lower efficiency = more carbon
        coal_carbon = coal_properties.get('carbon', 70) / 100
        return min(0.95, base_carbon + eff_effect + coal_carbon * 0.1)
    
    def _calculate_deposition_tendency(self, particle_size: float, 
                                     flame_temp_F: float, excess_o2_pct: float) -> float:
        """Calculate tendency for soot to deposit on surfaces."""
        size_factor = 1.0 / (1.0 + particle_size)  # Smaller particles stick better
        temp_factor = (4000 - flame_temp_F) / 2000  # Lower temps increase deposition
        o2_factor = 1.0 / (1.0 + excess_o2_pct * 0.1)  # Lower O2 increases sticking
        
        return max(0.1, min(1.0, size_factor * temp_factor * o2_factor))
    
    def _calculate_erosion_factor(self, particle_size: float, 
                                 coal_properties: Dict) -> float:
        """Calculate erosion factor due to soot particles."""
        size_effect = particle_size / 5.0  # Larger particles cause more erosion
        ash_content = coal_properties.get('ash', 10) / 100
        ash_effect = ash_content * 2.0  # Ash increases erosion
        
        return max(0.5, 1.0 + size_effect + ash_effect)


class CombustionFoulingIntegrator:
    """CORRECTED: Integrate combustion conditions with realistic fouling buildup patterns."""
    
    def __init__(self):
        """Initialize the CORRECTED combustion-fouling integrator."""
        self.soot_model = SootProductionModel()
        
        # CORRECTED: Section-specific fouling characteristics following real physics
        self.section_fouling_factors = {
            'furnace_walls': {
                'base_deposition': 4.0,      # HIGHEST - soot formation zone
                'temperature_factor': 2.0,   # Very sensitive to high temperature
                'velocity_factor': 0.8,      # High velocity reduces some deposition
                'surface_factor': 1.5,       # Rough surfaces increase fouling
                'description': 'Maximum fouling - primary soot formation'
            },
            'generating_bank': {
                'base_deposition': 3.2,      # HIGH - still in hot zone
                'temperature_factor': 1.8,
                'velocity_factor': 1.0,
                'surface_factor': 1.2,
                'description': 'High fouling - hot soot deposition'
            },
            'superheater_primary': {
                'base_deposition': 2.5,      # HIGH - particles still hot and sticky
                'temperature_factor': 1.5,
                'velocity_factor': 1.2,
                'surface_factor': 1.0,
                'description': 'Moderate-high fouling - transitional zone'
            },
            'superheater_secondary': {
                'base_deposition': 2.0,      # MODERATE-HIGH
                'temperature_factor': 1.2,
                'velocity_factor': 1.1,
                'surface_factor': 0.9,
                'description': 'Moderate fouling - cooling particles'
            },
            'economizer_primary': {
                'base_deposition': 1.0,      # MODERATE - cooler, less sticky
                'temperature_factor': 0.8,   # Lower sensitivity to temperature
                'velocity_factor': 0.9,
                'surface_factor': 1.1,
                'description': 'Moderate fouling - reduced deposition'
            },
            'economizer_secondary': {
                'base_deposition': 0.6,      # LOW-MODERATE - much cooler
                'temperature_factor': 0.6,
                'velocity_factor': 0.8,
                'surface_factor': 1.2,
                'description': 'Low fouling - cool gas conditions'
            },
            'air_heater': {
                'base_deposition': 0.2,      # LOWEST - cold, minimal sticking
                'temperature_factor': 0.3,   # Very low sensitivity
                'velocity_factor': 0.7,
                'surface_factor': 1.3,
                'description': 'Minimal fouling - cold gas, low adhesion'
            }
        }
    
    def calculate_section_fouling_rates(self, combustion_model: CoalCombustionModel,
                                      coal_properties: Dict,
                                      boiler_system) -> Dict[str, Dict[str, list]]:
        """Calculate CORRECTED fouling rates for each section and segment."""
        # Calculate base soot production
        soot_data = self.soot_model.calculate_soot_production(
            combustion_model, coal_properties
        )
        
        section_fouling_rates = {}
        
        print(f"\n🔥 CORRECTED FOULING DISTRIBUTION:")
        print(f"{'Section':<20} {'Base Rate':<10} {'Description':<30}")
        print("-" * 65)
        
        for section_name, section in boiler_system.sections.items():
            section_factors = self.section_fouling_factors.get(section_name, {
                'base_deposition': 1.0,
                'temperature_factor': 1.0,
                'velocity_factor': 1.0,
                'surface_factor': 1.0,
                'description': 'Default moderate deposition'
            })
            
            print(f"{section_name:<20} {section_factors['base_deposition']:<10.1f} {section_factors['description']:<30}")
            
            # Calculate segment-specific fouling rates
            gas_fouling_rates = []
            water_fouling_rates = []
            
            for segment_id in range(section.num_segments):
                segment_position = segment_id / (section.num_segments - 1) if section.num_segments > 1 else 0
                
                # CORRECTED: Calculate realistic local conditions
                local_gas_temp, local_velocity = self._calculate_realistic_local_conditions(
                    section_name, segment_position
                )
                
                # Apply CORRECTED deposition model
                base_rate = section_factors['base_deposition']
                temp_effect = self._realistic_temperature_effect(local_gas_temp, section_factors['temperature_factor'])
                velocity_effect = self._velocity_fouling_effect(local_velocity)
                position_effect = 1.0 - 0.3 * segment_position  # CORRECTED: Decreases along path
                surface_effect = section_factors['surface_factor']
                
                # Calculate fouling rate for this segment
                segment_fouling_rate = (soot_data.mass_production_rate * 
                                      soot_data.deposition_tendency * 
                                      base_rate * temp_effect * velocity_effect * 
                                      position_effect * surface_effect)
                
                # Convert to fouling resistance units (hr-ft²-°F/Btu per hour)
                gas_fouling_rate = segment_fouling_rate * 1e-6  # Conversion factor
                water_fouling_rate = gas_fouling_rate * 0.25  # Water side typically less
                
                gas_fouling_rates.append(gas_fouling_rate)
                water_fouling_rates.append(water_fouling_rate)
            
            section_fouling_rates[section_name] = {
                'gas': gas_fouling_rates,
                'water': water_fouling_rates
            }
        
        return section_fouling_rates
    
    def _calculate_realistic_local_conditions(self, section_name: str, 
                                            segment_position: float) -> tuple:
        """CORRECTED: Calculate realistic local gas temperature and velocity."""
        # CORRECTED: Realistic temperature progression through boiler
        section_base_temps = {
            'furnace_walls': 2800,        # HOTTEST - furnace exit
            'generating_bank': 2200,      # Hot superheated steam generation
            'superheater_primary': 1800,  # Primary superheating
            'superheater_secondary': 1400, # Secondary superheating  
            'economizer_primary': 1000,   # Feed water heating
            'economizer_secondary': 600,  # Final feed water heating
            'air_heater': 350            # COLDEST - near stack
        }
        
        base_temp = section_base_temps.get(section_name, 1200)
        
        # Temperature decreases along each section as gas cools
        local_gas_temp = base_temp - (base_temp * 0.15 * segment_position)  # 15% drop maximum
        
        # Velocity typically increases as gas cools and density decreases
        base_velocity = 45  # fps
        local_velocity = base_velocity + 15 * segment_position
        
        return local_gas_temp, local_velocity
    
    def _realistic_temperature_effect(self, gas_temp_F: float, sensitivity: float) -> float:
        """CORRECTED: Calculate realistic temperature effect on soot deposition."""
        # CORRECTED: Higher temperatures INCREASE soot sticking (opposite of original)
        if gas_temp_F > 2500:
            return 2.0 * sensitivity  # Very high temp, maximum sticking
        elif gas_temp_F > 2000:
            return 1.6 * sensitivity  # High temp, high sticking
        elif gas_temp_F > 1500:
            return 1.2 * sensitivity  # Moderate temp, moderate sticking
        elif gas_temp_F > 1000:
            return 0.8 * sensitivity  # Lower temp, reduced sticking
        elif gas_temp_F > 500:
            return 0.5 * sensitivity  # Low temp, low sticking
        else:
            return 0.2 * sensitivity  # Very low temp, minimal sticking
    
    def _velocity_fouling_effect(self, velocity_fps: float) -> float:
        """Calculate velocity effect on fouling deposition."""
        # Higher velocity reduces deposition due to scouring
        if velocity_fps < 20:
            return 1.8  # Low velocity, high deposition
        elif velocity_fps < 40:
            return 1.2
        elif velocity_fps < 60:
            return 1.0  # Baseline
        elif velocity_fps < 80:
            return 0.8
        else:
            return 0.6  # High velocity, low deposition