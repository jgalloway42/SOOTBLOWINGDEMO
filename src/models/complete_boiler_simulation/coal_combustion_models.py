#!/usr/bin/env python3
"""
PHASE 3 FIXES: Enhanced Coal Combustion Models - Improved Load Variation

This module contains CRITICAL FIXES for Phase 3 combustion efficiency variation:
- ENHANCED load-dependent combustion efficiency (2.7% → ≥5% variation)
- STRENGTHENED excess air effects on combustion performance
- IMPROVED part-load and high-load combustion penalties
- ENHANCED air/fuel ratio impacts on efficiency
- Realistic combustion efficiency ranges across all operating conditions

CRITICAL FIXES IMPLEMENTED:
- Strengthened load sensitivity factors for combustion efficiency
- Enhanced excess air penalty calculations
- Improved part-load combustion degradation effects
- Added high-load combustion efficiency decline
- Realistic combustion efficiency bounds (88-98% range)

Author: Enhanced Boiler Modeling System
Version: 10.0 - PHASE 3 COMBUSTION EFFICIENCY ENHANCEMENT
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import math

class CoalCombustionModel:
    """PHASE 3 ENHANCED: Coal combustion model with improved load-dependent efficiency variation."""
    
    def __init__(self, ultimate_analysis: Dict[str, float], coal_lb_per_hr: float,
                 air_scfh: float, NOx_eff: float = 0.35, air_temp_F: float = 80.0,
                 air_RH_pct: float = 60.0, atm_press_inHg: float = 29.92,
                 design_coal_rate: float = 8333.0):
        """Initialize PHASE 3 ENHANCED combustion model with stronger load dependency."""
        
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
        """PHASE 3 ENHANCED calculation with stronger load-dependent behavior"""
        
        # Calculate load factor for enhanced calculations
        load_factor = self._coal_lb_per_hr / self._design_coal_rate
        
        # ENHANCED: Fuel input energy calculation
        coal_heating_value = 12000  # Btu/lb (typical bituminous)
        fuel_input_btu_hr = self._coal_lb_per_hr * coal_heating_value
        
        # PHASE 3 ENHANCED: Stronger excess air calculation
        stoich_air_scfh = self._coal_lb_per_hr * 10.5  # ~10.5 SCF air per lb coal
        excess_air_fraction = max(0, (self._air_scfh - stoich_air_scfh) / stoich_air_scfh)
        excess_o2_pct = excess_air_fraction * 21 * 0.8  # Approximate conversion
        
        # PHASE 3 CRITICAL FIX: Enhanced load-dependent combustion efficiency with stronger variation
        combustion_efficiency = self._calculate_enhanced_load_dependent_combustion_efficiency(
            load_factor, excess_air_fraction
        )
        
        # ENHANCED: Load-dependent flame temperature
        flame_temp_F = self._calculate_enhanced_flame_temperature(
            load_factor, excess_air_fraction
        )
        
        # ENHANCED: Load-dependent NOx formation
        NO_thermal_lb_per_hr, NO_fuel_lb_per_hr = self._calculate_enhanced_NOx(
            load_factor, flame_temp_F, excess_o2_pct
        )
        
        # ENHANCED: Flue gas mass flow with load dependency
        total_flue_gas_lb_per_hr = self._calculate_enhanced_flue_gas(
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
            print(f"PHASE 3 ENHANCED Combustion Calculation:")
            print(f"  Load Factor: {load_factor:.2f}")
            print(f"  Combustion Efficiency: {combustion_efficiency:.1%}")
            print(f"  Flame Temperature: {flame_temp_F:.0f}°F")
            print(f"  Excess Air: {excess_air_fraction:.1%}")
    
    def _calculate_enhanced_load_dependent_combustion_efficiency(self, load_factor: float, 
                                                              excess_air_fraction: float) -> float:
        """PHASE 3 CRITICAL FIX: Enhanced combustion efficiency with STRONGER load dependency."""
        
        # PHASE 3 FIX: Enhanced base combustion efficiency with stronger load curve
        if load_factor <= 0.3:
            # Very poor combustion at extremely low loads
            base_efficiency = 0.88 + (load_factor / 0.3) * 0.04  # 88% to 92%
        elif load_factor <= 0.6:
            # Improving combustion efficiency
            progress = (load_factor - 0.3) / 0.3  # 0 to 1
            base_efficiency = 0.92 + progress * 0.04  # 92% to 96%
        elif load_factor <= 1.0:
            # Peak combustion efficiency range
            progress = (load_factor - 0.6) / 0.4  # 0 to 1
            base_efficiency = 0.96 + progress * 0.02  # 96% to 98%
        elif load_factor <= 1.3:
            # Declining efficiency above design
            progress = (load_factor - 1.0) / 0.3  # 0 to 1
            base_efficiency = 0.98 - progress * 0.03  # 98% to 95%
        else:
            # Poor efficiency at extreme overload
            excess_load = load_factor - 1.3
            base_efficiency = 0.95 - excess_load * 0.05  # Steep decline
        
        # PHASE 3 ENHANCED: Stronger excess air penalty
        if excess_air_fraction < 0.15:  # Too little excess air
            excess_air_penalty = (0.15 - excess_air_fraction) * 0.08  # Stronger penalty
        elif excess_air_fraction > 0.35:  # Too much excess air
            excess_air_penalty = (excess_air_fraction - 0.35) * 0.06  # Stronger penalty
        else:
            excess_air_penalty = 0.0
        
        # PHASE 3 ENHANCED: Stronger load-specific combustion penalties
        load_penalties = 0.0
        
        # Part-load combustion degradation
        if load_factor < 0.5:
            load_penalties += (0.5 - load_factor) * 0.06  # Increased from 0.04
        
        # High-load combustion problems
        if load_factor > 1.2:
            load_penalties += (load_factor - 1.2) * 0.04  # Increased penalty
        
        # PHASE 3 ENHANCED: Air distribution effects at extreme loads
        air_distribution_penalty = 0.0
        if load_factor < 0.4:
            air_distribution_penalty = (0.4 - load_factor) * 0.03  # Poor air mixing at low load
        elif load_factor > 1.4:
            air_distribution_penalty = (load_factor - 1.4) * 0.05  # Poor air distribution at high load
        
        # Calculate final combustion efficiency
        final_efficiency = base_efficiency - excess_air_penalty - load_penalties - air_distribution_penalty
        
        # PHASE 3 FIX: Realistic bounds with wider variation range
        return max(0.88, min(0.98, final_efficiency))  # 88-98% range for stronger variation
    
    def _calculate_enhanced_flame_temperature(self, load_factor: float, 
                                            excess_air_fraction: float) -> float:
        """PHASE 3 ENHANCED: Calculate flame temperature with stronger load effects."""
        
        # Base flame temperature
        base_flame_temp = 3200  # °F
        
        # PHASE 3 ENHANCED: Stronger load effects on flame temperature
        if load_factor <= 0.8:
            load_adjustment = (load_factor - 0.8) * 300  # Up to -240°F at low load
        else:
            load_adjustment = (load_factor - 0.8) * 150  # +30°F at 20% overload
        
        # Enhanced excess air cooling effect
        excess_air_cooling = excess_air_fraction * 400  # Stronger cooling effect
        
        # Coal quality effects (enhanced)
        carbon_content = self._ultimate_analysis.get('carbon', 70) / 100
        coal_quality_effect = (carbon_content - 0.7) * 200  # ±60°F variation
        
        flame_temp = base_flame_temp + load_adjustment - excess_air_cooling + coal_quality_effect
        
        return max(2800, min(3600, flame_temp))  # Realistic bounds
    
    def _calculate_enhanced_NOx(self, load_factor: float, flame_temp_F: float, 
                              excess_o2_pct: float) -> Tuple[float, float]:
        """PHASE 3 ENHANCED: Calculate NOx formation with stronger load dependency."""
        
        # Enhanced thermal NOx (temperature dependent)
        if flame_temp_F > 3000:
            thermal_factor = math.exp((flame_temp_F - 3000) / 400)  # Stronger temperature effect
        else:
            thermal_factor = 0.5
        
        # Enhanced load effects on NOx
        load_nox_factor = 0.8 + load_factor * 0.4  # 0.8 to 1.2 range
        
        NO_thermal_lb_per_hr = (self._coal_lb_per_hr * 0.002 * thermal_factor * 
                               load_nox_factor * (1 + excess_o2_pct * 0.1))
        
        # Enhanced fuel NOx (nitrogen content dependent)
        nitrogen_content = self._ultimate_analysis.get('nitrogen', 1.5) / 100
        fuel_nox_factor = 1.0 + (load_factor - 1.0) * 0.3  # Enhanced load effect
        
        NO_fuel_lb_per_hr = (self._coal_lb_per_hr * nitrogen_content * 0.4 * 
                             fuel_nox_factor * (1 - self._NOx_eff))
        
        return NO_thermal_lb_per_hr, NO_fuel_lb_per_hr
    
    def _calculate_enhanced_flue_gas(self, load_factor: float, 
                                   excess_air_fraction: float) -> float:
        """PHASE 3 ENHANCED: Calculate flue gas mass flow with load dependency."""
        
        # Base flue gas calculation
        combustion_air_lb_per_hr = self._air_scfh * 0.075  # Standard air density
        
        # Enhanced load effects on flue gas flow
        load_flow_factor = 0.9 + load_factor * 0.2  # 0.9 to 1.1 range
        
        total_flue_gas = (self._coal_lb_per_hr + combustion_air_lb_per_hr) * load_flow_factor
        
        return total_flue_gas
    
    # Property accessors
    @property
    def load_factor(self) -> float:
        return self._results.get('load_factor', 0.0) if self._calculated else 0.0
    
    @property
    def combustion_efficiency(self) -> float:
        return self._results.get('combustion_efficiency', 0.0) if self._calculated else 0.0
    
    @property
    def flame_temp_F(self) -> float:
        return self._results.get('flame_temp_F', 0.0) if self._calculated else 0.0
    
    @property
    def dry_O2_pct(self) -> float:
        return self._results.get('dry_O2_pct', 0.0) if self._calculated else 0.0
    
    @property
    def heat_released_btu_per_hr(self) -> float:
        return self._results.get('heat_released_btu_per_hr', 0.0) if self._calculated else 0.0


class EnhancedAshCharacteristics:
    """PHASE 3 ENHANCED: Coal ash characteristics with stronger load dependency."""
    
    def __init__(self, coal_properties: Dict[str, float]):
        """Initialize enhanced ash characteristics calculator."""
        self.coal_properties = coal_properties
    
    def calculate_enhanced_characteristics(self, combustion_eff: float, 
                                         flame_temp_F: float, excess_o2_pct: float,
                                         load_factor: float) -> Dict[str, float]:
        """PHASE 3 ENHANCED: Calculate ash characteristics with stronger load effects."""
        
        # Enhanced particle size calculation
        particle_size = self._calculate_enhanced_particle_size(
            flame_temp_F, excess_o2_pct, load_factor
        )
        
        # Enhanced carbon content
        carbon_content = self._calculate_enhanced_carbon_content(
            combustion_eff, self.coal_properties
        )
        
        # Enhanced deposition tendency
        deposition_tendency = self._calculate_enhanced_deposition_tendency(
            particle_size, flame_temp_F, excess_o2_pct, load_factor
        )
        
        # Enhanced erosion factor
        erosion_factor = self._calculate_enhanced_erosion_factor(
            particle_size, self.coal_properties, load_factor
        )
        
        return {
            'particle_size_microns': particle_size,
            'carbon_content_fraction': carbon_content,
            'deposition_tendency': deposition_tendency,
            'erosion_factor': erosion_factor,
            'ash_fusion_temp_F': 2100 + (flame_temp_F - 3200) * 0.3,
            'silica_content': self.coal_properties.get('ash', 10) * 0.45,
            'alumina_content': self.coal_properties.get('ash', 10) * 0.25
        }
    
    def _calculate_enhanced_particle_size(self, flame_temp_F: float, excess_o2_pct: float,
                                        load_factor: float) -> float:
        """PHASE 3 ENHANCED: Calculate particle size with stronger load effects."""
        base_size = 2.5  # μm
        temp_effect = (flame_temp_F - 3200) / 800 * 1.2  # Enhanced temperature effect
        o2_effect = (excess_o2_pct - 3) * 0.2  # Enhanced oxygen effect
        load_effect = (1.0 - load_factor) * 0.8  # Stronger load effect - larger particles at low loads
        return max(1.0, base_size + temp_effect + o2_effect + load_effect)
    
    def _calculate_enhanced_carbon_content(self, combustion_eff: float, 
                                         coal_properties: Dict) -> float:
        """PHASE 3 ENHANCED: Calculate carbon content with stronger efficiency effects."""
        base_carbon = 0.85  # Base carbon content
        eff_effect = (1.0 - combustion_eff) * 1.2  # Stronger efficiency effect
        coal_carbon = coal_properties.get('carbon', 70) / 100
        return min(0.98, base_carbon + eff_effect + coal_carbon * 0.1)
    
    def _calculate_enhanced_deposition_tendency(self, particle_size: float, 
                                              flame_temp_F: float, excess_o2_pct: float,
                                              load_factor: float) -> float:
        """PHASE 3 ENHANCED: Calculate deposition tendency with stronger load effects."""
        size_factor = 1.5 / (1.0 + particle_size * 1.0)  # Enhanced size effect
        temp_factor = (4400 - flame_temp_F) / 2000  # Enhanced temperature effect
        o2_factor = 1.3 / (1.0 + excess_o2_pct * 0.2)  # Enhanced oxygen effect
        load_factor_effect = 1.0 + (1.0 - load_factor) * 0.5  # Stronger deposition at low loads
        
        tendency = size_factor * temp_factor * o2_factor * load_factor_effect
        return max(0.1, min(1.0, tendency))
    
    def _calculate_enhanced_erosion_factor(self, particle_size: float, 
                                         coal_properties: Dict, load_factor: float) -> float:
        """PHASE 3 ENHANCED: Calculate erosion factor with stronger load dependency."""
        size_effect = particle_size / 3.5  # Enhanced size effect
        ash_content = coal_properties.get('ash', 10) / 100
        ash_effect = ash_content * 3.0  # Enhanced ash effect
        load_effect = load_factor * 0.3  # Stronger erosion at higher loads
        
        return max(0.2, 1.0 + size_effect + ash_effect + load_effect)


# Test function for PHASE 3 ENHANCED combustion
def test_phase3_enhanced_combustion():
    """Test the PHASE 3 ENHANCED combustion model with stronger load variation."""
    print("Testing PHASE 3 ENHANCED Coal Combustion Model...")
    
    # Test coal properties
    ultimate_analysis = {
        'carbon': 70.0, 'hydrogen': 5.0, 'oxygen': 10.0,
        'nitrogen': 1.5, 'sulfur': 2.0, 'ash': 11.5
    }
    
    coal_properties = {
        'volatile_matter': 35.0, 'fixed_carbon': 53.5, 
        'sulfur': 2.0, 'ash': 11.5, 'carbon': 70.0
    }
    
    # PHASE 3 TEST: Wide range of load conditions for stronger variation
    test_conditions = [
        (2500, 26000, "30% Load - Very Low"),
        (4000, 35000, "48% Load - Low"),
        (6000, 52000, "72% Load - Medium"),
        (8333, 70000, "100% Load - Design"),
        (10000, 84000, "120% Load - High"),
        (12500, 105000, "150% Load - Very High")
    ]
    
    results = []
    
    for coal_rate, air_flow, description in test_conditions:
        print(f"\n{description}:")
        print("-" * 50)
        
        # Create PHASE 3 ENHANCED combustion model
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
            'excess_air': excess_air,
            'description': description
        })
    
    # PHASE 3 ANALYSIS: Check for enhanced variation
    if len(results) >= 4:
        efficiencies = [r['efficiency'] for r in results]
        eff_range = max(efficiencies) - min(efficiencies)
        eff_min = min(efficiencies)
        eff_max = max(efficiencies)
        
        print(f"\n{'='*60}")
        print("PHASE 3 ENHANCED COMBUSTION EFFICIENCY VARIATION ANALYSIS:")
        print(f"{'='*60}")
        print(f"  Efficiency Range: {eff_min:.1%} to {eff_max:.1%}")
        print(f"  Total Variation: {eff_range:.2%} ({eff_range/eff_min*100:.1f}% relative)")
        print(f"  PHASE 3 TARGET ACHIEVED: {'YES' if eff_range >= 0.05 else 'NO'} (target: ≥5%)")
        
        # Detailed load response analysis
        print(f"\nDetailed Load Response:")
        for r in results:
            print(f"  {r['load']:.1f} load: {r['efficiency']:.1%} efficiency")
        
        # Success criteria
        target_achieved = eff_range >= 0.05
        realistic_range = 0.88 <= eff_min <= 0.95 and 0.92 <= eff_max <= 0.98
        proper_curve = efficiencies[2] > efficiencies[0] and efficiencies[3] >= efficiencies[2]  # Efficiency improves then peaks
        
        print(f"\n✓ Target Variation Achieved (≥5%): {target_achieved}")
        print(f"✓ Realistic Efficiency Range: {realistic_range}")
        print(f"✓ Proper Load Response Curve: {proper_curve}")
        
        overall_success = target_achieved and realistic_range and proper_curve
        print(f"\n🎯 PHASE 3 COMBUSTION ENHANCEMENT SUCCESSFUL: {overall_success}")
        
        print(f"{'='*60}")
    
    print(f"\n[OK] PHASE 3 ENHANCED combustion model testing completed")
    return results


if __name__ == "__main__":
    test_phase3_enhanced_combustion()