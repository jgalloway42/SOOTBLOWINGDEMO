#!/usr/bin/env python3
"""
PHASE 3 FIXES: Enhanced Coal Combustion Models - Realistic Load Range

This module contains CRITICAL FIXES for Phase 3 combustion efficiency variation:
- REALISTIC load range focus (60-105% instead of extreme ranges)
- ENHANCED combustion efficiency variation (2.7% -> >=5% variation)
- STRENGTHENED excess air effects on combustion performance
- IMPROVED load-dependent combustion penalties within realistic range
- ENHANCED air/fuel ratio impacts on efficiency

CRITICAL FIXES IMPLEMENTED:
- Updated load dependency curves for 60-105% range
- Strengthened load sensitivity factors for combustion efficiency
- Enhanced excess air penalty calculations
- Improved realistic combustion efficiency bounds (88-98% range)
- Removed unrealistic extreme load logic

Author: Enhanced Boiler Modeling System
Version: 10.1 - REALISTIC LOAD RANGE AND COMBUSTION ENHANCEMENT
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import math

class CoalCombustionModel:
    """PHASE 3 ENHANCED: Coal combustion model with realistic load range and improved variation."""
    
    def __init__(self, ultimate_analysis: Dict[str, float], coal_lb_per_hr: float,
                 air_scfh: float, NOx_eff: float = 0.35, air_temp_F: float = 80.0,
                 air_RH_pct: float = 60.0, atm_press_inHg: float = 29.92,
                 design_coal_rate: float = 8333.0):
        """Initialize PHASE 3 ENHANCED combustion model with realistic load focus."""
        
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
        
        # Validate realistic operating range
        load_factor = coal_lb_per_hr / design_coal_rate
        if load_factor < 0.55:
            print(f"WARNING: Coal rate {load_factor:.1%} below realistic minimum (55%)")
        elif load_factor > 1.10:
            print(f"WARNING: Coal rate {load_factor:.1%} above realistic maximum (110%)")
    
    def calculate(self, debug=False):
        """PHASE 3 ENHANCED calculation focused on realistic load behavior."""
        
        # Calculate load factor for enhanced calculations
        load_factor = self._coal_lb_per_hr / self._design_coal_rate
        
        # Enhanced fuel input energy calculation
        coal_heating_value = 12000  # Btu/lb (typical bituminous)
        fuel_input_btu_hr = self._coal_lb_per_hr * coal_heating_value
        
        # PHASE 3 ENHANCED: Stronger excess air calculation
        stoich_air_scfh = self._coal_lb_per_hr * 10.5  # ~10.5 SCF air per lb coal
        excess_air_fraction = max(0, (self._air_scfh - stoich_air_scfh) / stoich_air_scfh)
        excess_o2_pct = excess_air_fraction * 21 * 0.8  # Approximate conversion
        
        # PHASE 3 CRITICAL FIX: Enhanced combustion efficiency with REALISTIC load focus
        combustion_efficiency = self._calculate_realistic_load_combustion_efficiency(
            load_factor, excess_air_fraction
        )
        
        # Enhanced flame temperature with realistic load effects
        flame_temp_F = self._calculate_realistic_flame_temperature(
            load_factor, excess_air_fraction
        )
        
        # Enhanced NOx formation with realistic load dependency
        NO_thermal_lb_per_hr, NO_fuel_lb_per_hr = self._calculate_realistic_NOx(
            load_factor, flame_temp_F, excess_o2_pct
        )
        
        # Enhanced flue gas mass flow with realistic load dependency
        total_flue_gas_lb_per_hr = self._calculate_realistic_flue_gas(
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
            print(f"PHASE 3 REALISTIC LOAD Combustion Calculation:")
            print(f"  Load Factor: {load_factor:.2f} (Realistic Range: 60-105%)")
            print(f"  Combustion Efficiency: {combustion_efficiency:.1%}")
            print(f"  Flame Temperature: {flame_temp_F:.0f}F")
            print(f"  Excess Air: {excess_air_fraction:.1%}")
    
    def _calculate_realistic_load_combustion_efficiency(self, load_factor: float, 
                                                       excess_air_fraction: float) -> float:
        """PHASE 3 CRITICAL FIX: Combustion efficiency with REALISTIC load dependency (60-105%)."""
        
        # REALISTIC LOAD RANGE: Enhanced base combustion efficiency curve
        if load_factor <= 0.60:
            # Poor combustion at minimum sustainable load
            base_efficiency = 0.88 + (load_factor / 0.60) * 0.04  # 88% to 92%
        elif load_factor <= 0.75:
            # Improving combustion efficiency (low-normal range)
            progress = (load_factor - 0.60) / 0.15  # 0 to 1
            base_efficiency = 0.92 + progress * 0.03  # 92% to 95%
        elif load_factor <= 0.95:
            # Peak combustion efficiency range (normal operations)
            progress = (load_factor - 0.75) / 0.20  # 0 to 1
            base_efficiency = 0.95 + progress * 0.03  # 95% to 98%
        elif load_factor <= 1.05:
            # Brief operation above design (realistic peak)
            progress = (load_factor - 0.95) / 0.10  # 0 to 1
            base_efficiency = 0.98 - progress * 0.02  # 98% to 96%
        else:
            # Brief peaks above 105% (rare, limited duration)
            excess_load = min(load_factor - 1.05, 0.05)  # Cap consideration at 110%
            base_efficiency = 0.96 - excess_load * 8.0  # Decline to ~94%
        
        # PHASE 3 ENHANCED: Stronger excess air penalty
        excess_air_penalty = 0.0
        if excess_air_fraction < 0.15:  # Too little excess air
            excess_air_penalty = (0.15 - excess_air_fraction) * 0.08  # Stronger penalty
        elif excess_air_fraction > 0.35:  # Too much excess air
            excess_air_penalty = (excess_air_fraction - 0.35) * 0.06  # Stronger penalty
        
        # PHASE 3 ENHANCED: Realistic load-specific combustion penalties
        load_penalties = 0.0
        
        # Part-load combustion degradation (realistic range)
        if load_factor < 0.65:
            load_penalties += (0.65 - load_factor) * 0.06  # Poor mixing at low loads
        
        # Peak load combustion issues (above design)
        if load_factor > 1.02:
            load_penalties += (load_factor - 1.02) * 0.04  # Reduced residence time
        
        # PHASE 3 ENHANCED: Air distribution effects at realistic extremes
        air_distribution_penalty = 0.0
        if load_factor < 0.62:
            # Poor air distribution at very low sustainable loads
            air_distribution_penalty = (0.62 - load_factor) * 0.03
        elif load_factor > 1.03:
            # Air distribution challenges at sustained high loads
            air_distribution_penalty = (load_factor - 1.03) * 0.05
        
        # Calculate final combustion efficiency
        final_efficiency = (base_efficiency - excess_air_penalty - 
                          load_penalties - air_distribution_penalty)
        
        # REALISTIC BOUNDS: Focus on achievable industrial range
        return max(0.88, min(0.98, final_efficiency))  # 88-98% realistic range
    
    def _calculate_realistic_flame_temperature(self, load_factor: float, 
                                             excess_air_fraction: float) -> float:
        """PHASE 3 ENHANCED: Flame temperature with realistic load effects (60-105%)."""
        
        # Base flame temperature for typical bituminous coal
        base_flame_temp = 3000  # F
        
        # REALISTIC LOAD DEPENDENCY: Focused on 60-105% range
        load_adjustment = 0.0
        if load_factor <= 0.70:
            # Lower flame temps at low loads (poor mixing)
            load_adjustment = -150 + (load_factor / 0.70) * 100  # -150F to -50F
        elif load_factor <= 0.95:
            # Optimal flame temperature range
            progress = (load_factor - 0.70) / 0.25
            load_adjustment = -50 + progress * 100  # -50F to +50F
        else:
            # Brief high load operation
            excess = min(load_factor - 0.95, 0.10)
            load_adjustment = 50 + excess * 300  # +50F to +80F
        
        # Excess air cooling effect
        excess_air_adjustment = -excess_air_fraction * 200  # F per fraction excess air
        
        # Fuel quality and moisture effects
        moisture_content = self._ultimate_analysis.get('Moisture', 0.0)
        moisture_adjustment = -moisture_content * 15  # F per % moisture
        
        final_flame_temp = base_flame_temp + load_adjustment + excess_air_adjustment + moisture_adjustment
        
        # Realistic bounds for coal flames
        return max(2200, min(3200, final_flame_temp))
    
    def _calculate_realistic_NOx(self, load_factor: float, flame_temp_F: float, 
                               excess_o2_pct: float) -> Tuple[float, float]:
        """Enhanced NOx calculation with realistic load dependency."""
        
        # Thermal NOx (temperature dependent)
        if flame_temp_F > 2800:
            thermal_factor = ((flame_temp_F - 2800) / 400) ** 2
        else:
            thermal_factor = 0.0
        
        # Load factor effects on NOx formation
        if load_factor < 0.70:
            load_nox_factor = 0.8  # Lower NOx at low loads
        elif load_factor > 1.0:
            load_nox_factor = 1.2  # Higher NOx at high loads
        else:
            load_nox_factor = 1.0
        
        # Base NOx formation rates
        base_thermal_nox = self._coal_lb_per_hr * 0.001 * thermal_factor * load_nox_factor
        base_fuel_nox = self._coal_lb_per_hr * 0.002 * self._ultimate_analysis.get('N', 1.5) / 1.5
        
        # NOx reduction effectiveness
        nox_reduction = min(self._NOx_eff * 0.8, 0.6)  # Cap at 60% reduction
        
        NO_thermal_lb_per_hr = base_thermal_nox * (1 - nox_reduction)
        NO_fuel_lb_per_hr = base_fuel_nox * (1 - nox_reduction * 0.5)  # Less effective on fuel NOx
        
        return NO_thermal_lb_per_hr, NO_fuel_lb_per_hr
    
    def _calculate_realistic_flue_gas(self, load_factor: float, 
                                    excess_air_fraction: float) -> float:
        """Calculate flue gas mass flow with realistic load dependency."""
        
        # Base flue gas calculation
        stoich_air_mass = self._coal_lb_per_hr * 10.2  # lb air per lb coal
        excess_air_mass = stoich_air_mass * excess_air_fraction
        total_air_mass = stoich_air_mass + excess_air_mass
        
        # Flue gas = coal + air (assuming complete combustion)
        total_flue_gas = self._coal_lb_per_hr + total_air_mass
        
        # Load-dependent corrections for leakage and measurement
        if load_factor < 0.70:
            correction_factor = 1.02  # Slight increase due to measurement uncertainty at low loads
        else:
            correction_factor = 1.0
        
        return total_flue_gas * correction_factor
    
    @property
    def load_factor(self) -> float:
        """Get load factor."""
        if self._calculated:
            return self._results['load_factor']
        return self._coal_lb_per_hr / self._design_coal_rate
    
    @property
    def combustion_efficiency(self) -> float:
        """Get combustion efficiency."""
        if self._calculated:
            return self._results['combustion_efficiency']
        return 0.0
    
    @property
    def flame_temp_F(self) -> float:
        """Get flame temperature in F."""
        if self._calculated:
            return self._results['flame_temp_F']
        return 0.0
    
    @property
    def dry_O2_pct(self) -> float:
        """Get dry O2 percentage."""
        if self._calculated:
            return self._results['dry_O2_pct']
        return 0.0
    
    @property
    def total_flue_gas_lb_per_hr(self) -> float:
        """Get total flue gas flow."""
        if self._calculated:
            return self._results['total_flue_gas_lb_per_hr']
        return 0.0


# DEPRECATED: Moved to core.fouling_and_soot_blowing.SootProductionModel
# This class has been merged with CorrectedSootProductionModel to create a unified
# soot production model that includes both combustion condition effects AND 
# realistic deposition patterns. Delete this after testing the new implementation.

# class SootProductionModel:
#     """Enhanced soot production model with realistic load focus."""
#     
#     def __init__(self, combustion_model: CoalCombustionModel):
#         """Initialize with combustion model."""
#         self.combustion_model = combustion_model
#         self.soot_data = {}
#     
#     def calculate_soot_production(self) -> Dict[str, float]:
#         """Calculate soot production with realistic load dependency."""
#         
#         if not self.combustion_model._calculated:
#             self.combustion_model.calculate()
#         
#         load_factor = self.combustion_model.load_factor
#         flame_temp = self.combustion_model.flame_temp_F
#         excess_air = self.combustion_model._results['excess_air_fraction']
#         
#         # Base soot production factors (realistic range focused)
#         if load_factor < 0.65:
#             # Higher soot at low loads (poor combustion)
#             base_soot_factor = 1.4
#         elif load_factor > 1.0:
#             # Higher soot at high loads (reduced residence time)
#             base_soot_factor = 1.2
#         else:
#             # Optimal soot production range
#             base_soot_factor = 1.0
#         
#         # Temperature effects
#         if flame_temp < 2500:
#             temp_factor = 1.3  # More soot at low temps
#         elif flame_temp > 3000:
#             temp_factor = 1.1  # Some soot increase at very high temps
#         else:
#             temp_factor = 1.0
#         
#         # Excess air effects
#         if excess_air < 0.15:
#             air_factor = 1.5  # Much more soot with insufficient air
#         elif excess_air > 0.30:
#             air_factor = 0.8  # Less soot with excess air
#         else:
#             air_factor = 1.0
#         
#         # Calculate soot production
#         base_soot_rate = self.combustion_model._coal_lb_per_hr * 0.01  # 1% of coal as potential soot
#         actual_soot_rate = base_soot_rate * base_soot_factor * temp_factor * air_factor
#         
#         # Soot distribution by boiler section (realistic physics)
#         self.soot_data = {
#             'total_soot_lb_per_hr': actual_soot_rate,
#             'furnace_soot_pct': 0.45,      # Highest in furnace (formation zone)
#             'superheater_soot_pct': 0.30,  # High in superheater (still hot)
#             'economizer_soot_pct': 0.20,   # Moderate in economizer (cooling)
#             'air_heater_soot_pct': 0.05,   # Lowest in air heater (cold)
#             'load_factor': load_factor,
#             'flame_temp_F': flame_temp,
#             'excess_air_fraction': excess_air
#         }
#         
#         return self.soot_data


class CombustionFoulingIntegrator:
    """Integration of combustion and fouling models with realistic load focus."""
    
    def __init__(self):
        """Initialize integrator."""
        self.integration_data = {}
    
    def integrate_combustion_fouling(self, combustion_model: CoalCombustionModel,
                                   fouling_timeline_hours: int = 720) -> Dict:
        """Integrate combustion and fouling with realistic load effects."""
        
        if not combustion_model._calculated:
            combustion_model.calculate()
        
        # Get soot production
        soot_model = SootProductionModel(combustion_model)
        soot_data = soot_model.calculate_soot_production()
        
        load_factor = combustion_model.load_factor
        
        # Realistic fouling buildup rates (focused on 60-105% range)
        if load_factor < 0.65:
            fouling_multiplier = 1.3  # Higher fouling at low loads (poor combustion)
        elif load_factor > 1.0:
            fouling_multiplier = 1.2  # Higher fouling at high loads (more soot)
        else:
            fouling_multiplier = 1.0  # Baseline fouling
        
        # Calculate fouling buildup over time
        base_fouling_rate = 0.001  # Base fouling rate per hour
        effective_fouling_rate = base_fouling_rate * fouling_multiplier
        
        # Fouling timeline simulation
        fouling_timeline = []
        current_fouling = {'gas': 0.002, 'water': 0.001}  # Initial clean conditions
        
        for hour in range(0, fouling_timeline_hours, 24):  # Daily intervals
            # Fouling buildup
            current_fouling['gas'] += effective_fouling_rate * 24
            current_fouling['water'] += effective_fouling_rate * 0.5 * 24
            
            # Performance degradation due to fouling
            fouling_penalty = current_fouling['gas'] * 50  # Efficiency penalty
            degraded_efficiency = combustion_model.combustion_efficiency * (1 - fouling_penalty)
            
            fouling_timeline.append({
                'hour': hour,
                'gas_fouling': current_fouling['gas'],
                'water_fouling': current_fouling['water'],
                'efficiency_penalty': fouling_penalty,
                'degraded_efficiency': degraded_efficiency
            })
        
        self.integration_data = {
            'combustion_efficiency': combustion_model.combustion_efficiency,
            'soot_production': soot_data,
            'fouling_timeline': fouling_timeline,
            'load_factor': load_factor,
            'fouling_multiplier': fouling_multiplier
        }
        
        return self.integration_data


def test_phase3_realistic_combustion():
    """Test PHASE 3 combustion model with REALISTIC load range."""
    
    print("=" * 60)
    print("PHASE 3 REALISTIC LOAD COMBUSTION MODEL TEST")
    print("=" * 60)
    print("Testing combustion efficiency across REALISTIC load range (60-105%)")
    print("REMOVED: Unrealistic extreme loads (30%, 150%)")
    print("FOCUS: Industrial boiler operating range")
    
    # Ultimate analysis for typical bituminous coal
    ultimate_analysis = {
        'C': 75.0, 'H': 5.0, 'O': 8.0, 'N': 1.5, 
        'S': 2.5, 'Ash': 8.0, 'Moisture': 2.0
    }
    
    # REALISTIC test scenarios for industrial boilers
    realistic_test_scenarios = [
        (60, 5000, 52500, "60% Load - Minimum sustained operation"),
        (70, 5833, 61246, "70% Load - Low normal operation"),
        (85, 7083, 74371, "85% Load - High normal operation"),
        (95, 7917, 83128, "95% Load - Near maximum operation"),
        (100, 8333, 87496, "100% Load - Design point"),
        (105, 8750, 91871, "105% Load - Brief peak operation")
    ]
    
    results = []
    
    for load_pct, coal_rate, air_flow, description in realistic_test_scenarios:
        print(f"\n{description}:")
        print("-" * 40)
        
        # Create combustion model
        combustion_model = CoalCombustionModel(
            ultimate_analysis=ultimate_analysis,
            coal_lb_per_hr=coal_rate,
            air_scfh=air_flow,
            design_coal_rate=8333.0  # 100% load reference
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
        print(f"  Flame Temperature: {flame_temp:.0f}F")
        print(f"  Excess O2: {excess_air:.1f}%")
        
        results.append({
            'load': load_factor,
            'efficiency': combustion_eff,
            'flame_temp': flame_temp,
            'excess_air': excess_air,
            'description': description
        })
    
    # PHASE 3 ANALYSIS: Check for enhanced variation within realistic range
    if len(results) >= 4:
        efficiencies = [r['efficiency'] for r in results]
        eff_range = max(efficiencies) - min(efficiencies)
        eff_min = min(efficiencies)
        eff_max = max(efficiencies)
        
        print(f"\n{'='*60}")
        print("PHASE 3 REALISTIC COMBUSTION EFFICIENCY VARIATION ANALYSIS:")
        print(f"{'='*60}")
        print(f"LOAD RANGE: 60% to 105% (REALISTIC INDUSTRIAL OPERATIONS)")
        print(f"  Efficiency Range: {eff_min:.1%} to {eff_max:.1%}")
        print(f"  Total Variation: {eff_range:.2%} ({eff_range/eff_min*100:.1f}% relative)")
        print(f"  PHASE 3 TARGET ACHIEVED: {'YES' if eff_range >= 0.05 else 'NO'} (target: >=5%)")
        
        # Detailed load response analysis
        print(f"\nDetailed Realistic Load Response:")
        for r in results:
            print(f"  {r['load']:.0%} load: {r['efficiency']:.1%} efficiency")
        
        # Success criteria for realistic range
        target_achieved = eff_range >= 0.05
        realistic_range = 0.88 <= eff_min <= 0.95 and 0.92 <= eff_max <= 0.98
        proper_curve = efficiencies[3] > efficiencies[0] and efficiencies[4] >= efficiencies[3]  # Efficiency improves then peaks
        load_range_realistic = all(0.55 <= r['load'] <= 1.10 for r in results)
        
        print(f"\nTarget Variation Achieved (>=5%): {target_achieved}")
        print(f"Realistic Efficiency Range (88-98%): {realistic_range}")
        print(f"Proper Load Response Curve: {proper_curve}")
        print(f"Realistic Load Range (60-105%): {load_range_realistic}")
        
        overall_success = target_achieved and realistic_range and proper_curve and load_range_realistic
        print(f"\nPHASE 3 REALISTIC COMBUSTION ENHANCEMENT SUCCESSFUL: {overall_success}")
        
        print(f"{'='*60}")
    
    print(f"\n[OK] PHASE 3 REALISTIC combustion model testing completed")
    return results


if __name__ == "__main__":
    test_phase3_realistic_combustion()