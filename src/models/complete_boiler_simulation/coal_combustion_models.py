#!/usr/bin/env python3
"""
FIXED: Coal Combustion and Soot Production Models

This module contains the coal combustion analysis and soot production modeling
classes with ENHANCED load-dependent combustion efficiency that properly responds
to fuel input changes and operating conditions.

MAJOR ENHANCEMENTS IMPLEMENTED:
- FIXED load-dependent combustion efficiency calculations
- Enhanced excess air calculations based on actual fuel/air ratios
- Improved NOx formation modeling with load dependency
- Better flame temperature calculations with fuel input effects
- Enhanced soot production modeling with load-dependent behavior

Classes:
    CoalCombustionModel: ENHANCED coal combustion analysis with load dependency
    SootProductionModel: Enhanced soot formation and deposition modeling
    SootProductionData: Dataclass for soot characteristics
    CombustionFoulingIntegrator: Enhanced fouling integration

Author: Enhanced Boiler Modeling System
Version: 9.0 - ENHANCED Load-Dependent Combustion Efficiency
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
    """ENHANCED coal combustion model with proper load-dependent efficiency calculations"""
    
    def __init__(self, ultimate_analysis, coal_lb_per_hr, air_scfh, NOx_eff=0.35,
                 air_temp_F=75.0, air_RH_pct=55.0, atm_press_inHg=30.25,
                 design_coal_rate=8333.0):  # Design rate for 100 MMBtu/hr
        """Initialize ENHANCED coal combustion model with load dependency."""
        self._ultimate_analysis = ultimate_analysis.copy()
        self._coal_lb_per_hr = coal_lb_per_hr
        self._air_scfh = air_scfh
        self._NOx_eff = NOx_eff
        self._air_temp_F = air_temp_F
        self._air_RH_pct = air_RH_pct
        self._atm_press_inHg = atm_press_inHg
        self._design_coal_rate = design_coal_rate  # Reference for load calculations
        self._results = {}
        self._calculated = False
    
    def calculate(self, debug=False):
        """ENHANCED calculation with proper load-dependent behavior"""
        
        # Calculate load factor for enhanced calculations
        load_factor = self._coal_lb_per_hr / self._design_coal_rate
        
        # ENHANCED: Fuel input energy calculation
        coal_heating_value = 12000  # Btu/lb (typical bituminous)
        fuel_input_btu_hr = self._coal_lb_per_hr * coal_heating_value
        
        # ENHANCED: Proper excess air calculation based on fuel/air ratio
        # Stoichiometric air requirement (approximate for bituminous coal)
        stoich_air_scfh = self._coal_lb_per_hr * 10.5  # ~10.5 SCF air per lb coal
        excess_air_fraction = max(0, (self._air_scfh - stoich_air_scfh) / stoich_air_scfh)
        excess_o2_pct = excess_air_fraction * 21 * 0.8  # Approximate conversion
        
        # ENHANCED: Load-dependent combustion efficiency
        combustion_efficiency = self._calculate_load_dependent_combustion_efficiency(
            load_factor, excess_air_fraction
        )
        
        # ENHANCED: Load-dependent flame temperature
        flame_temp_F = self._calculate_load_dependent_flame_temperature(
            load_factor, excess_air_fraction
        )
        
        # ENHANCED: Load-dependent NOx formation
        NO_thermal_lb_per_hr, NO_fuel_lb_per_hr = self._calculate_load_dependent_NOx(
            load_factor, flame_temp_F, excess_o2_pct
        )
        
        # ENHANCED: Flue gas mass flow with load dependency
        total_flue_gas_lb_per_hr = self._calculate_load_dependent_flue_gas(
            load_factor, excess_air_fraction
        )
        
        # Store enhanced results
        self._results = {
            'load_factor': load_factor,
            'fuel_input_btu_hr': fuel_input_btu_hr,
            'excess_air_fraction': excess_air_fraction,
            'total_flue_gas_lb_per_hr': total_flue_gas_lb_per_hr,
            'CO2_lb_per_hr': self._coal_lb_per_hr * 2.8,  # Enhanced CO2 calculation
            'NO_total_lb_per_hr': NO_thermal_lb_per_hr + NO_fuel_lb_per_hr,
            'NO_thermal_lb_per_hr': NO_thermal_lb_per_hr,
            'NO_fuel_lb_per_hr': NO_fuel_lb_per_hr,
            'dry_O2_pct': excess_o2_pct,
            'combustion_efficiency': combustion_efficiency,
            'heat_released_btu_per_hr': fuel_input_btu_hr * combustion_efficiency,
            'flame_temp_F': flame_temp_F
        }
        self._calculated = True
        
        if debug:
            print(f"ENHANCED Combustion Calculation:")
            print(f"  Load Factor: {load_factor:.2f}")
            print(f"  Combustion Efficiency: {combustion_efficiency:.1%}")
            print(f"  Flame Temperature: {flame_temp_F:.0f}°F")
            print(f"  Excess Air: {excess_air_fraction:.1%}")
    
    def _calculate_load_dependent_combustion_efficiency(self, load_factor: float, 
                                                      excess_air_fraction: float) -> float:
        """ENHANCED: Calculate combustion efficiency with proper load dependency."""
        
        # Base efficiency curve - peak efficiency around 80% load
        if load_factor <= 0.4:
            # Poor efficiency at very low loads
            base_efficiency = 0.90 + (load_factor / 0.4) * 0.06  # 90% to 96%
        elif load_factor <= 0.8:
            # Rising efficiency to optimum
            progress = (load_factor - 0.4) / 0.4  # 0 to 1
            base_efficiency = 0.96 + progress * 0.02  # 96% to 98%
        elif load_factor <= 1.0:
            # Slight decline from peak
            progress = (load_factor - 0.8) / 0.2  # 0 to 1
            base_efficiency = 0.98 - progress * 0.005  # 98% to 97.5%
        else:
            # Decline above design capacity
            excess_load = load_factor - 1.0
            base_efficiency = 0.975 - excess_load * 0.02  # Steeper decline
        
        # Excess air penalty
        if excess_air_fraction < 0.15:
            # Too little air - incomplete combustion
            air_penalty = (0.15 - excess_air_fraction) * 0.1  # Up to 10% penalty
        elif excess_air_fraction > 0.5:
            # Too much air - heat loss
            air_penalty = (excess_air_fraction - 0.5) * 0.05  # 5% penalty per 50% excess
        else:
            air_penalty = 0.0
        
        # Load-specific combustion penalties
        if load_factor < 0.3:
            # Very poor atomization and mixing at extremely low loads
            load_penalty = (0.3 - load_factor) * 0.15  # Up to 15% penalty
        elif load_factor > 1.3:
            # Poor residence time at very high loads
            load_penalty = (load_factor - 1.3) * 0.08  # 8% penalty above 130%
        else:
            load_penalty = 0.0
        
        final_efficiency = base_efficiency - air_penalty - load_penalty
        return max(0.85, min(0.985, final_efficiency))  # Realistic bounds
    
    def _calculate_load_dependent_flame_temperature(self, load_factor: float, 
                                                  excess_air_fraction: float) -> float:
        """ENHANCED: Calculate flame temperature with proper load dependency."""
        
        # Base flame temperature
        base_flame_temp = 3100  # °F for optimal conditions
        
        # Load effects on flame temperature
        if load_factor <= 0.5:
            # Lower temperatures at low loads due to poor mixing
            load_adjustment = -200 * (0.5 - load_factor)  # Up to -100°F
        elif load_factor <= 1.0:
            # Rising temperature with load
            load_adjustment = (load_factor - 0.5) * 200  # Up to +100°F
        else:
            # Peak temperature, then slight decline due to reduced residence time
            load_adjustment = 100 - (load_factor - 1.0) * 150  # Decline above 100%
        
        # Excess air effects
        if excess_air_fraction < 0.1:
            # Very hot, fuel-rich conditions
            air_adjustment = 200 * (0.1 - excess_air_fraction)  # Up to +20°F
        else:
            # Cooling effect of excess air
            air_adjustment = -excess_air_fraction * 400  # -400°F per 100% excess air
        
        final_flame_temp = base_flame_temp + load_adjustment + air_adjustment
        return max(2500, min(3500, final_flame_temp))  # Realistic bounds
    
    def _calculate_load_dependent_NOx(self, load_factor: float, flame_temp_F: float, 
                                    excess_o2_pct: float) -> tuple:
        """ENHANCED: Calculate NOx formation with proper load dependency."""
        
        # Base NOx rates
        base_thermal_rate = 0.002  # lb NOx per lb coal
        base_fuel_rate = 0.005     # lb NOx per lb coal
        
        # Thermal NOx (highly temperature dependent)
        if flame_temp_F > 2800:
            thermal_multiplier = 1.0 + ((flame_temp_F - 2800) / 300) ** 2  # Exponential rise
        else:
            thermal_multiplier = 0.3 + (flame_temp_F - 2500) / 300 * 0.7  # Linear below 2800°F
        
        # Load effects on thermal NOx
        if load_factor > 1.0:
            # Higher temperatures at high loads increase thermal NOx
            load_thermal_multiplier = 1.0 + (load_factor - 1.0) * 0.5
        else:
            load_thermal_multiplier = 0.5 + load_factor * 0.5
        
        # Fuel NOx (depends on fuel nitrogen and combustion conditions)
        if excess_o2_pct < 2.0:
            # Fuel-rich conditions reduce fuel NOx
            fuel_multiplier = 0.6 + excess_o2_pct * 0.2
        else:
            # Excess oxygen increases fuel NOx
            fuel_multiplier = 1.0 + (excess_o2_pct - 2.0) * 0.1
        
        # Load effects on fuel NOx
        fuel_load_multiplier = 0.8 + load_factor * 0.4  # Better mixing at higher loads
        
        # Apply NOx efficiency factor
        thermal_nox = (self._coal_lb_per_hr * base_thermal_rate * 
                      thermal_multiplier * load_thermal_multiplier * self._NOx_eff)
        fuel_nox = (self._coal_lb_per_hr * base_fuel_rate * 
                   fuel_multiplier * fuel_load_multiplier * self._NOx_eff)
        
        return thermal_nox, fuel_nox
    
    def _calculate_load_dependent_flue_gas(self, load_factor: float, 
                                         excess_air_fraction: float) -> float:
        """ENHANCED: Calculate flue gas mass flow with proper load dependency."""
        
        # Base flue gas calculation
        base_flue_gas_per_coal = 12.5  # lb flue gas per lb coal (typical)
        
        # Load effects (better combustion efficiency at optimal loads)
        if load_factor <= 0.8:
            load_multiplier = 0.95 + load_factor * 0.0625  # 95% to 100%
        else:
            load_multiplier = 1.0 + (load_factor - 0.8) * 0.1  # Slight increase above 80%
        
        # Excess air effects
        air_addition = excess_air_fraction * 8.0  # Additional mass per excess air
        
        total_flue_gas = (self._coal_lb_per_hr * base_flue_gas_per_coal * 
                         load_multiplier + self._coal_lb_per_hr * air_addition)
        
        return total_flue_gas
    
    # Property accessors (unchanged interface)
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
    
    @property
    def load_factor(self):
        if not self._calculated: self.calculate()
        return self._results['load_factor']


class SootProductionModel:
    """Enhanced soot production model with improved load dependency."""
    
    def __init__(self):
        """Initialize enhanced soot production model with load-dependent correlations."""
        # Enhanced empirical constants
        self.base_soot_rate = 0.0008  # lb soot per lb coal (baseline)
        self.load_sensitivity = 1.5   # Load effect amplification
        self.nox_soot_correlation = 0.9  # Stronger NOx-soot correlation
        self.efficiency_impact = 2.5  # Enhanced efficiency impact
        self.temperature_impact = 1.8  # Enhanced temperature effects
        
    def calculate_soot_production(self, combustion_model: CoalCombustionModel,
                                coal_properties: Dict) -> SootProductionData:
        """Calculate enhanced soot production based on combustion conditions."""
        
        # Get enhanced combustion parameters
        load_factor = combustion_model.load_factor
        thermal_nox = combustion_model.NO_thermal_lb_per_hr
        fuel_nox = combustion_model.NO_fuel_lb_per_hr
        total_nox = combustion_model.NO_total_lb_per_hr
        excess_o2 = combustion_model.dry_O2_pct
        combustion_eff = combustion_model.combustion_efficiency
        flame_temp = combustion_model.flame_temp_F
        coal_rate = combustion_model._coal_lb_per_hr
        
        # Enhanced soot production factors with load dependency
        load_factor_effect = self._calculate_enhanced_load_factor(load_factor)
        nox_factor = self._calculate_enhanced_nox_soot_factor(thermal_nox, fuel_nox, coal_rate)
        excess_air_factor = self._calculate_enhanced_excess_air_factor(excess_o2)
        efficiency_factor = self._calculate_enhanced_efficiency_factor(combustion_eff)
        temperature_factor = self._calculate_enhanced_temperature_factor(flame_temp)
        coal_factor = self._calculate_enhanced_coal_factor(coal_properties)
        
        # Enhanced base soot production rate with load dependency
        base_production = self.base_soot_rate * coal_rate * load_factor_effect
        
        # Apply all enhanced factors
        actual_soot_rate = (base_production * nox_factor * excess_air_factor * 
                           efficiency_factor * temperature_factor * coal_factor)
        
        # Enhanced soot characteristics calculation
        particle_size = self._calculate_enhanced_particle_size(flame_temp, excess_o2, load_factor)
        carbon_content = self._calculate_enhanced_carbon_content(combustion_eff, coal_properties)
        ash_content = 1.0 - carbon_content
        deposition_tendency = self._calculate_enhanced_deposition_tendency(
            particle_size, flame_temp, excess_o2, load_factor
        )
        erosion_factor = self._calculate_enhanced_erosion_factor(
            particle_size, coal_properties, load_factor
        )
        
        return SootProductionData(
            mass_production_rate=actual_soot_rate,
            particle_size_microns=particle_size,
            carbon_content=carbon_content,
            ash_content=ash_content,
            deposition_tendency=deposition_tendency,
            erosion_factor=erosion_factor
        )
    
    def _calculate_enhanced_load_factor(self, load_factor: float) -> float:
        """Enhanced load factor effects on soot production."""
        if load_factor < 0.5:
            # Very high soot at low loads due to poor combustion
            return 1.5 + (0.5 - load_factor) * 2.0  # Up to 2.5x at very low loads
        elif load_factor <= 0.8:
            # Decreasing soot with increasing load
            return 1.5 - (load_factor - 0.5) * 1.0  # 1.5x to 1.2x
        elif load_factor <= 1.2:
            # Minimum soot around optimal load
            return 1.2 - (load_factor - 0.8) * 0.5  # 1.2x to 1.0x
        else:
            # Increasing soot above design due to poor residence time
            return 1.0 + (load_factor - 1.2) * 1.5  # Rising above 120% load
    
    def _calculate_enhanced_nox_soot_factor(self, thermal_nox: float, fuel_nox: float, 
                                          coal_rate: float) -> float:
        """Enhanced NOx-soot correlation with better sensitivity."""
        # Higher NOx formation often correlates with combustion conditions that produce soot
        total_nox_rate = (thermal_nox + fuel_nox) / coal_rate if coal_rate > 0 else 0
        nox_ppm_equivalent = total_nox_rate * 1000  # Approximate ppm
        
        if nox_ppm_equivalent < 5:
            return 0.4  # Very clean combustion, minimal soot
        elif nox_ppm_equivalent < 15:
            return 0.7  # Good combustion
        elif nox_ppm_equivalent < 25:
            return 1.0  # Normal combustion
        elif nox_ppm_equivalent < 40:
            return 1.6  # Higher soot formation
        else:
            return 2.5  # Poor combustion, high soot
    
    def _calculate_enhanced_excess_air_factor(self, excess_o2_pct: float) -> float:
        """Enhanced excess air effects on soot production."""
        if excess_o2_pct < 0.5:
            return 4.0  # Very fuel-rich, extremely high soot
        elif excess_o2_pct < 1.5:
            return 2.5  # Fuel-rich, high soot
        elif excess_o2_pct < 3.0:
            return 1.8  # Slightly rich, increased soot
        elif excess_o2_pct < 6.0:
            return 1.0  # Optimal air/fuel ratio
        elif excess_o2_pct < 10.0:
            return 0.7  # Slightly lean, reduced soot
        else:
            return 0.5  # Very lean, low soot but poor efficiency
    
    def _calculate_enhanced_efficiency_factor(self, combustion_eff: float) -> float:
        """Enhanced combustion efficiency effects on soot production."""
        # Stronger correlation between efficiency and soot
        return 3.0 - 2.0 * combustion_eff  # More dramatic effect
    
    def _calculate_enhanced_temperature_factor(self, flame_temp_F: float) -> float:
        """Enhanced flame temperature effects on soot formation."""
        # Soot formation has complex temperature dependency
        optimal_temp = 3000  # °F for minimal soot
        temp_deviation = abs(flame_temp_F - optimal_temp) / 400  # Increased sensitivity
        
        if flame_temp_F < 2600:
            # Very low temperature increases soot due to incomplete combustion
            return 1.0 + (2600 - flame_temp_F) / 200
        elif flame_temp_F > 3400:
            # Very high temperature can also increase soot formation
            return 1.0 + (flame_temp_F - 3400) / 300
        else:
            return 1.0 + temp_deviation * 0.8  # Enhanced temperature sensitivity
    
    def _calculate_enhanced_coal_factor(self, coal_properties: Dict) -> float:
        """Enhanced coal property effects on soot production."""
        # Enhanced coal property correlations
        volatile_matter = coal_properties.get('volatile_matter', 30)  # %
        fixed_carbon = coal_properties.get('fixed_carbon', 50)  # %
        sulfur = coal_properties.get('sulfur', 1.0)  # %
        ash = coal_properties.get('ash', 10)  # %
        
        vm_factor = 1.0 + (volatile_matter - 30) * 0.03  # Stronger VM effect
        fc_factor = 1.0 + (50 - fixed_carbon) * 0.015  # Enhanced FC effect
        s_factor = 1.0 + sulfur * 0.15  # Enhanced sulfur effect
        ash_factor = 1.0 + ash * 0.01  # Ash content effect
        
        return vm_factor * fc_factor * s_factor * ash_factor
    
    def _calculate_enhanced_particle_size(self, flame_temp_F: float, excess_o2_pct: float, 
                                        load_factor: float) -> float:
        """Enhanced particle size calculation with load dependency."""
        base_size = 2.2  # μm
        temp_effect = (flame_temp_F - 2800) / 800 * 0.8  # Enhanced temperature effect
        o2_effect = (excess_o2_pct - 3) * 0.15  # Enhanced oxygen effect
        load_effect = (1.0 - load_factor) * 0.5  # Larger particles at low loads
        return max(0.8, base_size + temp_effect + o2_effect + load_effect)
    
    def _calculate_enhanced_carbon_content(self, combustion_eff: float, 
                                         coal_properties: Dict) -> float:
        """Enhanced carbon content calculation."""
        base_carbon = 0.87  # Higher baseline carbon content
        eff_effect = (1.0 - combustion_eff) * 0.8  # Stronger efficiency effect
        coal_carbon = coal_properties.get('carbon', 70) / 100
        return min(0.98, base_carbon + eff_effect + coal_carbon * 0.08)
    
    def _calculate_enhanced_deposition_tendency(self, particle_size: float, 
                                              flame_temp_F: float, excess_o2_pct: float,
                                              load_factor: float) -> float:
        """Enhanced deposition tendency with load effects."""
        size_factor = 1.2 / (1.0 + particle_size * 0.8)  # Enhanced size effect
        temp_factor = (4200 - flame_temp_F) / 1800  # Enhanced temperature effect
        o2_factor = 1.2 / (1.0 + excess_o2_pct * 0.15)  # Enhanced oxygen effect
        load_factor_effect = 1.0 + (1.0 - load_factor) * 0.3  # Higher deposition at low loads
        
        tendency = size_factor * temp_factor * o2_factor * load_factor_effect
        return max(0.05, min(1.0, tendency))
    
    def _calculate_enhanced_erosion_factor(self, particle_size: float, 
                                         coal_properties: Dict, load_factor: float) -> float:
        """Enhanced erosion factor with load dependency."""
        size_effect = particle_size / 4.0  # Enhanced size effect
        ash_content = coal_properties.get('ash', 10) / 100
        ash_effect = ash_content * 2.5  # Enhanced ash effect
        load_effect = load_factor * 0.2  # Higher erosion at higher loads
        
        return max(0.3, 1.0 + size_effect + ash_effect + load_effect)


class CombustionFoulingIntegrator:
    """ENHANCED: Integrate combustion conditions with realistic fouling buildup patterns."""
    
    def __init__(self):
        """Initialize the ENHANCED combustion-fouling integrator."""
        self.section_fouling_sensitivity = {
            'furnace_walls': 2.5,        # Highest sensitivity
            'generating_bank': 2.0,      # High sensitivity
            'superheater_primary': 1.8,  # High-moderate sensitivity
            'superheater_secondary': 1.6, # Moderate sensitivity
            'economizer_primary': 1.4,   # Moderate sensitivity
            'economizer_secondary': 1.2, # Lower sensitivity
            'air_heater': 1.0           # Lowest sensitivity
        }
        
        self.temperature_fouling_correlation = {
            'furnace_walls': {'base_temp': 2500, 'sensitivity': 0.8},
            'generating_bank': {'base_temp': 2000, 'sensitivity': 0.7},
            'superheater_primary': {'base_temp': 1600, 'sensitivity': 0.6},
            'superheater_secondary': {'base_temp': 1400, 'sensitivity': 0.6},
            'economizer_primary': {'base_temp': 800, 'sensitivity': 0.5},
            'economizer_secondary': {'base_temp': 600, 'sensitivity': 0.4},
            'air_heater': {'base_temp': 400, 'sensitivity': 0.3}
        }
    
    def calculate_section_fouling_rates(self, combustion_model: CoalCombustionModel,
                                      coal_properties: Dict, boiler_sections: Dict) -> Dict:
        """ENHANCED: Calculate fouling rates for each boiler section with load dependency."""
        
        # Get enhanced combustion data
        load_factor = combustion_model.load_factor
        soot_model = SootProductionModel()
        soot_data = soot_model.calculate_soot_production(combustion_model, coal_properties)
        
        section_fouling_rates = {}
        
        for section_name, section in boiler_sections.items():
            if hasattr(section, 'num_segments'):
                # Enhanced fouling rate calculation
                base_fouling_rate = self._calculate_enhanced_base_fouling_rate(
                    section_name, soot_data, load_factor
                )
                
                # Calculate segment-specific rates
                gas_rates = []
                water_rates = []
                
                for segment_id in range(section.num_segments):
                    segment_position = segment_id / (section.num_segments - 1) if section.num_segments > 1 else 0
                    
                    # Enhanced position-based fouling with load effects
                    position_multiplier = self._calculate_enhanced_position_multiplier(
                        section_name, segment_position, load_factor
                    )
                    
                    # Temperature-based fouling enhancement
                    temp_multiplier = self._calculate_enhanced_temperature_multiplier(
                        section_name, segment_position, combustion_model.flame_temp_F
                    )
                    
                    # Load-based fouling enhancement
                    load_multiplier = self._calculate_enhanced_load_multiplier(
                        section_name, load_factor
                    )
                    
                    # Final segment fouling rates
                    segment_gas_rate = (base_fouling_rate * position_multiplier * 
                                      temp_multiplier * load_multiplier)
                    segment_water_rate = segment_gas_rate * 0.3  # Water side fouling
                    
                    gas_rates.append(segment_gas_rate)
                    water_rates.append(segment_water_rate)
                
                section_fouling_rates[section_name] = {
                    'gas': gas_rates,
                    'water': water_rates,
                    'base_rate': base_fouling_rate,
                    'load_factor': load_factor
                }
        
        return section_fouling_rates
    
    def _calculate_enhanced_base_fouling_rate(self, section_name: str, 
                                            soot_data: SootProductionData,
                                            load_factor: float) -> float:
        """Enhanced base fouling rate calculation."""
        
        # Section sensitivity
        sensitivity = self.section_fouling_sensitivity.get(section_name, 1.0)
        
        # Base rate from soot production
        base_rate = soot_data.mass_production_rate * soot_data.deposition_tendency * 0.001
        
        # Enhanced load effects
        if load_factor < 0.6:
            load_effect = 1.5 + (0.6 - load_factor) * 2.0  # Higher fouling at low loads
        elif load_factor > 1.1:
            load_effect = 1.0 + (load_factor - 1.1) * 1.5  # Higher fouling at high loads
        else:
            load_effect = 1.0  # Minimal fouling at optimal loads
        
        return base_rate * sensitivity * load_effect
    
    def _calculate_enhanced_position_multiplier(self, section_name: str, 
                                              segment_position: float,
                                              load_factor: float) -> float:
        """Enhanced position-based fouling multiplier."""
        
        # Base position effects
        if 'furnace' in section_name or section_name == 'generating_bank':
            # Higher fouling at furnace exit (position 1.0)
            base_multiplier = 0.6 + segment_position * 0.8  # 0.6 to 1.4
        elif 'superheater' in section_name:
            # Peak fouling in middle sections
            if segment_position < 0.5:
                base_multiplier = 0.8 + segment_position * 0.6  # 0.8 to 1.1
            else:
                base_multiplier = 1.1 - (segment_position - 0.5) * 0.4  # 1.1 to 0.9
        elif 'economizer' in section_name:
            # Decreasing fouling toward stack
            base_multiplier = 1.2 - segment_position * 0.5  # 1.2 to 0.7
        else:  # air_heater
            # Low, uniform fouling
            base_multiplier = 0.5 + segment_position * 0.2  # 0.5 to 0.7
        
        # Load effects on position sensitivity
        load_effect = 1.0 + abs(load_factor - 0.8) * 0.3  # More variation at off-design loads
        
        return base_multiplier * load_effect
    
    def _calculate_enhanced_temperature_multiplier(self, section_name: str,
                                                 segment_position: float,
                                                 flame_temp_F: float) -> float:
        """Enhanced temperature-based fouling multiplier."""
        
        temp_config = self.temperature_fouling_correlation.get(section_name, 
                                                              {'base_temp': 1000, 'sensitivity': 0.5})
        
        # Estimate section temperature based on flame temperature and position
        section_temp = flame_temp_F * 0.3 + temp_config['base_temp'] * (1 - segment_position * 0.5)
        
        # Temperature effect on fouling (higher temps generally increase fouling to a point)
        if section_temp > 2000:
            temp_multiplier = 1.0 + (section_temp - 2000) / 1000 * temp_config['sensitivity']
        elif section_temp > 1000:
            temp_multiplier = 0.7 + (section_temp - 1000) / 1000 * 0.3
        else:
            temp_multiplier = 0.5 + section_temp / 1000 * 0.2
        
        return max(0.3, min(2.0, temp_multiplier))
    
    def _calculate_enhanced_load_multiplier(self, section_name: str, load_factor: float) -> float:
        """Enhanced load-based fouling multiplier."""
        
        # Section-specific load effects
        if 'furnace' in section_name:
            # Furnace sensitive to load changes
            if load_factor < 0.7:
                return 1.3 + (0.7 - load_factor) * 2.0  # Poor combustion at low load
            elif load_factor > 1.1:
                return 1.0 + (load_factor - 1.1) * 1.8  # Poor residence time at high load
            else:
                return 1.0
        elif 'superheater' in section_name:
            # Superheaters moderately sensitive
            return 1.0 + abs(load_factor - 0.85) * 0.8
        elif 'economizer' in section_name:
            # Economizers less sensitive but still affected
            return 1.0 + abs(load_factor - 0.8) * 0.5
        else:  # air_heater
            # Air heaters least sensitive
            return 1.0 + abs(load_factor - 0.8) * 0.3


# Test function for ENHANCED combustion model
def test_enhanced_combustion_model():
    """Test the ENHANCED combustion model with proper load dependency."""
    
    print("Testing ENHANCED Coal Combustion Model...")
    
    # Test coal properties
    ultimate_analysis = {
        'carbon': 70.0, 'hydrogen': 5.0, 'oxygen': 10.0,
        'nitrogen': 1.5, 'sulfur': 2.0, 'ash': 11.5
    }
    
    coal_properties = {
        'volatile_matter': 35.0, 'fixed_carbon': 53.5, 
        'sulfur': 2.0, 'ash': 11.5, 'carbon': 70.0
    }
    
    # Test different load conditions
    test_conditions = [
        (4000, 35000, "40% Load"),
        (6000, 52000, "60% Load"),
        (8333, 70000, "100% Load (Design)"),
        (10000, 84000, "120% Load"),
        (12500, 105000, "150% Load")
    ]
    
    results = []
    
    for coal_rate, air_flow, description in test_conditions:
        print(f"\n{description}:")
        print("-" * 40)
        
        # Create combustion model
        combustion_model = CoalCombustionModel(
            ultimate_analysis=ultimate_analysis,
            coal_lb_per_hr=coal_rate,
            air_scfh=air_flow,
            NOx_eff=0.35,
            design_coal_rate=8333.0  # 100 MMBtu/hr design point
        )
        
        # Calculate with debug output
        combustion_model.calculate(debug=True)
        
        # Get results
        load_factor = combustion_model.load_factor
        combustion_eff = combustion_model.combustion_efficiency
        flame_temp = combustion_model.flame_temp_F
        excess_air = combustion_model.dry_O2_pct
        
        print(f"  Load Factor: {load_factor:.2f}")
        print(f"  Combustion Efficiency: {combustion_eff:.1%}")
        print(f"  Flame Temperature: {flame_temp:.0f}°F")
        print(f"  Excess O2: {excess_air:.1f}%")
        
        results.append({
            'load': load_factor,
            'efficiency': combustion_eff,
            'flame_temp': flame_temp,
            'excess_air': excess_air
        })
    
    # Analyze variation
    if len(results) >= 2:
        efficiencies = [r['efficiency'] for r in results]
        eff_range = max(efficiencies) - min(efficiencies)
        eff_min = min(efficiencies)
        eff_max = max(efficiencies)
        
        print(f"\n{'='*50}")
        print("COMBUSTION EFFICIENCY VARIATION ANALYSIS:")
        print(f"  Efficiency Range: {eff_min:.1%} to {eff_max:.1%}")
        print(f"  Total Variation: {eff_range:.2%} ({eff_range/eff_min*100:.1f}% relative)")
        print(f"  ENHANCED: {'YES' if eff_range >= 0.05 else 'NO'} (target: >=5%)")
        print(f"{'='*50}")
    
    print(f"\n[OK] ENHANCED combustion model testing completed")
    return results


if __name__ == "__main__":
    test_enhanced_combustion_model()